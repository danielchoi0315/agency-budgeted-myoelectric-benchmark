from __future__ import annotations

import copy
import inspect
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from . import models as models_module
from .calibration import fit_temperature, probability_temperature_scale
from .figures import save_policy_tradeoff
from .metrics import summarize_by_policy
from .policies import apply_policy_grid, apply_rate_matched_confidence_gate, entropy
from .realdata import PreparedDataset
from .schemas import PredictionTrace
from .splits import Split, assert_no_leakage


def run_dataset_benchmark(
    bundle: PreparedDataset,
    dataset_id: str,
    tau_grid: tuple[float, ...] = (0.02, 0.05, 0.10, 0.15, 0.20),
    user_model_name: str = "lda",
    assist_model_name: str = "logistic",
    extra_inputs: dict[str, Any] | None = None,
    split_family_allowlist: tuple[str, ...] | None = None,
    split_id_allowlist: tuple[str, ...] | None = None,
    max_splits_per_family: int | None = None,
    splits: list[Split] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    resolved_splits = list(splits) if splits is not None else benchmark_splits(
        bundle.metadata,
        dataset_id,
        split_family_allowlist=split_family_allowlist,
        split_id_allowlist=split_id_allowlist,
        max_splits_per_family=max_splits_per_family,
    )
    if not resolved_splits:
        raise ValueError(f"no benchmark splits registered for {dataset_id}")
    user_spec = _model_by_name(user_model_name, role="user")
    assist_spec = _model_by_name(assist_model_name, role="assist")
    resolved_inputs = extra_inputs or {}
    metric_frames: list[pd.DataFrame] = []
    for split in resolved_splits:
        split_metrics = _run_split(
            bundle,
            split,
            user_spec,
            assist_spec,
            tau_grid,
            extra_inputs=resolved_inputs,
        )
        metric_frames.append(split_metrics)
    metrics_df = pd.concat([frame for frame in metric_frames if not frame.empty], ignore_index=True)
    if metrics_df.empty:
        raise ValueError(f"all benchmark splits were skipped for {dataset_id}")
    split_level_df = (
        metrics_df.groupby(["split_family", "policy", "split_id"], as_index=False)
        .agg(
            macro_f1=("macro_f1", "mean"),
            active_macro_f1=("active_macro_f1", "mean"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            mean_ali=("mean_ali", "mean"),
            intervention_rate=("intervention_rate", "mean"),
            risk_coverage_auc=("risk_coverage_auc", "mean"),
            active_risk_coverage_auc=("active_risk_coverage_auc", "mean"),
            ece=("ece", "mean"),
            n=("n", "sum"),
            n_active=("n_active", "sum"),
        )
    )
    aggregate_df = (
        split_level_df.groupby(["split_family", "policy"], as_index=False)
        .agg(
            macro_f1=("macro_f1", "mean"),
            active_macro_f1=("active_macro_f1", "mean"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            mean_ali=("mean_ali", "mean"),
            intervention_rate=("intervention_rate", "mean"),
            risk_coverage_auc=("risk_coverage_auc", "mean"),
            active_risk_coverage_auc=("active_risk_coverage_auc", "mean"),
            ece=("ece", "mean"),
            n=("n", "sum"),
            n_active=("n_active", "sum"),
        )
        .sort_values(["mean_ali", "active_macro_f1", "macro_f1"], ascending=[True, False, False])
        .reset_index(drop=True)
    )
    return metrics_df, aggregate_df


def score_dataset_traces(
    bundle: PreparedDataset,
    dataset_id: str,
    *,
    user_model_name: str = "lda",
    assist_model_name: str = "logistic",
    extra_inputs: dict[str, Any] | None = None,
    split_family_allowlist: tuple[str, ...] | None = None,
    split_id_allowlist: tuple[str, ...] | None = None,
    max_splits_per_family: int | None = None,
    splits: list[Split] | None = None,
) -> list[PredictionTrace]:
    resolved_splits = list(splits) if splits is not None else benchmark_splits(
        bundle.metadata,
        dataset_id,
        split_family_allowlist=split_family_allowlist,
        split_id_allowlist=split_id_allowlist,
        max_splits_per_family=max_splits_per_family,
    )
    if not resolved_splits:
        raise ValueError(f"no benchmark splits registered for {dataset_id}")
    user_spec = _model_by_name(user_model_name, role="user")
    assist_spec = _model_by_name(assist_model_name, role="assist")
    resolved_inputs = extra_inputs or {}
    traces: list[PredictionTrace] = []
    for split in resolved_splits:
        traces.extend(_score_split_traces(bundle, split, user_spec, assist_spec, extra_inputs=resolved_inputs))
    return traces


def benchmark_splits(
    metadata: pd.DataFrame,
    dataset_id: str,
    *,
    split_family_allowlist: tuple[str, ...] | None = None,
    split_id_allowlist: tuple[str, ...] | None = None,
    max_splits_per_family: int | None = None,
) -> list[Split]:
    if dataset_id == "db10":
        splits = _db10_splits(metadata)
    elif dataset_id == "hyser":
        splits = _hyser_splits(metadata)
    elif dataset_id == "cemhsey":
        splits = _cemhsey_splits(metadata)
    elif dataset_id == "grabmyo":
        splits = _grabmyo_splits(metadata)
    else:
        raise ValueError(f"unsupported dataset_id: {dataset_id}")
    return _filter_splits(
        splits,
        split_family_allowlist=split_family_allowlist,
        split_id_allowlist=split_id_allowlist,
        max_splits_per_family=max_splits_per_family,
    )


def _filter_splits(
    splits: list[Split],
    *,
    split_family_allowlist: tuple[str, ...] | None,
    split_id_allowlist: tuple[str, ...] | None,
    max_splits_per_family: int | None,
) -> list[Split]:
    filtered = list(splits)
    if split_family_allowlist:
        allowed = {str(value) for value in split_family_allowlist}
        filtered = [split for split in filtered if split.split_id.split("|", 1)[0] in allowed]
    if split_id_allowlist:
        allowed_ids = {str(value) for value in split_id_allowlist}
        filtered = [split for split in filtered if split.split_id in allowed_ids]
    if max_splits_per_family is not None:
        if max_splits_per_family <= 0:
            raise ValueError("max_splits_per_family must be positive when provided")
        limited: list[Split] = []
        counts: dict[str, int] = {}
        for split in filtered:
            family = split.split_id.split("|", 1)[0]
            taken = counts.get(family, 0)
            if taken >= int(max_splits_per_family):
                continue
            counts[family] = taken + 1
            limited.append(split)
        filtered = limited
    return filtered


def _db10_splits(metadata: pd.DataFrame) -> list[Split]:
    splits: list[Split] = []
    amputees = sorted(metadata.loc[metadata["group"] == "amputee", "subject_id"].astype(str).unique())
    for subject_id in amputees:
        test_mask = metadata["subject_id"].astype(str) == subject_id
        train_able_mask = metadata["group"].astype(str) == "able_bodied"
        train_mixed_mask = ~test_mask
        train_amp_mask = (metadata["group"].astype(str) == "amputee") & (~test_mask)
        for prefix, train_mask in {
            "db10_able_to_amputee": train_able_mask,
            "db10_mixed_to_amputee": train_mixed_mask,
            "db10_amputee_loso": train_amp_mask,
        }.items():
            train_idx = metadata.index[train_mask].to_list()
            test_idx = metadata.index[test_mask].to_list()
            if not train_idx or not test_idx:
                continue
            assert_no_leakage(metadata, train_idx, test_idx, ["subject_id"])
            splits.append(
                Split(
                    split_id=f"{prefix}|{subject_id}",
                    train_index=train_idx,
                    test_index=test_idx,
                    train_groups={"subject_id": set(metadata.loc[train_idx, "subject_id"].astype(str))},
                    test_groups={"subject_id": {subject_id}},
                )
            )
    return splits


def _hyser_splits(metadata: pd.DataFrame) -> list[Split]:
    splits: list[Split] = []
    sessions = sorted(metadata["session"].astype(str).unique())
    for session_index, session in enumerate(sessions):
        if session_index == 0:
            continue
        previous_sessions = set(sessions[:session_index])
        train_idx = metadata.index[metadata["session"].astype(str).isin(previous_sessions)].to_list()
        test_idx = metadata.index[metadata["session"].astype(str) == session].to_list()
        if train_idx and test_idx:
            splits.append(
                Split(
                    split_id=f"hyser_within_subject_dayshift|{session}",
                    train_index=train_idx,
                    test_index=test_idx,
                    train_groups={"session": set(metadata.loc[train_idx, "session"].astype(str))},
                    test_groups={"session": {session}},
                )
            )
    for subject_id in sorted(metadata["subject_id"].astype(str).unique()):
        train_idx = metadata.index[metadata["subject_id"].astype(str) != subject_id].to_list()
        test_idx = metadata.index[metadata["subject_id"].astype(str) == subject_id].to_list()
        if train_idx and test_idx:
            assert_no_leakage(metadata, train_idx, test_idx, ["subject_id"])
            splits.append(
                Split(
                    split_id=f"hyser_subject_logo|{subject_id}",
                    train_index=train_idx,
                    test_index=test_idx,
                    train_groups={"subject_id": set(metadata.loc[train_idx, "subject_id"].astype(str))},
                    test_groups={"subject_id": {subject_id}},
                )
            )
    return splits


def _cemhsey_splits(metadata: pd.DataFrame) -> list[Split]:
    splits: list[Split] = []
    ordered_days = sorted(metadata["day"].astype(str).unique(), key=_day_sort_key)
    for day_index, day in enumerate(ordered_days):
        if day_index == 0:
            continue
        previous_days = set(ordered_days[:day_index])
        train_idx = metadata.index[metadata["day"].astype(str).isin(previous_days)].to_list()
        test_idx = metadata.index[metadata["day"].astype(str) == day].to_list()
        if train_idx and test_idx:
            splits.append(
                Split(
                    split_id=f"cemhsey_forward_day_logo|{day}",
                    train_index=train_idx,
                    test_index=test_idx,
                    train_groups={"day": set(metadata.loc[train_idx, "day"].astype(str))},
                    test_groups={"day": {day}},
                )
            )
    return splits


def _grabmyo_splits(metadata: pd.DataFrame) -> list[Split]:
    splits: list[Split] = []
    sessions = sorted(metadata["session"].astype(str).unique())
    for session_index, session in enumerate(sessions):
        if session_index == 0:
            continue
        previous_sessions = set(sessions[:session_index])
        train_idx = metadata.index[metadata["session"].astype(str).isin(previous_sessions)].to_list()
        test_idx = metadata.index[metadata["session"].astype(str) == session].to_list()
        if train_idx and test_idx:
            splits.append(
                Split(
                    split_id=f"grabmyo_within_subject_dayshift|{session}",
                    train_index=train_idx,
                    test_index=test_idx,
                    train_groups={
                        "session": set(metadata.loc[train_idx, "session"].astype(str)),
                    },
                    test_groups={"session": {session}},
                )
            )
    for subject_id in sorted(metadata["subject_id"].astype(str).unique()):
        train_mask = metadata["subject_id"].astype(str) != subject_id
        test_mask = metadata["subject_id"].astype(str) == subject_id
        train_idx = metadata.index[train_mask].to_list()
        test_idx = metadata.index[test_mask].to_list()
        if not train_idx or not test_idx:
            continue
        assert_no_leakage(metadata, train_idx, test_idx, ["subject_id"])
        splits.append(
            Split(
                split_id=f"grabmyo_subject_logo|{subject_id}",
                train_index=train_idx,
                test_index=test_idx,
                train_groups={"subject_id": set(metadata.loc[train_idx, "subject_id"].astype(str))},
                test_groups={"subject_id": {subject_id}},
            )
        )
    return splits


def _run_split(
    bundle: PreparedDataset,
    split: Split,
    user_spec,
    assist_spec,
    tau_grid: tuple[float, ...],
    extra_inputs: dict[str, Any],
) -> pd.DataFrame:
    traces = _score_split_traces(bundle, split, user_spec, assist_spec, extra_inputs=extra_inputs)
    if not traces:
        return pd.DataFrame()
    decisions = apply_policy_grid(traces, tau_grid=tau_grid)
    matched_decisions, _ = apply_rate_matched_confidence_gate(traces, tau_grid=tau_grid)
    decisions.extend(matched_decisions)
    metrics = summarize_by_policy(decisions)
    metrics_df = pd.DataFrame([asdict(metric) for metric in metrics])
    metrics_df["split_family"] = split.split_id.split("|", 1)[0]
    return metrics_df


def _score_split_traces(
    bundle: PreparedDataset,
    split: Split,
    user_spec,
    assist_spec,
    *,
    extra_inputs: dict[str, Any],
) -> list[PredictionTrace]:
    metadata = bundle.metadata.reset_index(drop=True)
    y = bundle.labels.astype(int)
    x_user = _resolve_model_input(bundle, extra_inputs, user_spec, role="user")
    x_assist = _resolve_model_input(bundle, extra_inputs, assist_spec, role="assist")
    train_idx = np.asarray(split.train_index, dtype=int)
    test_idx = np.asarray(split.test_index, dtype=int)
    if train_idx.size == 0 or test_idx.size == 0:
        return []
    calib_idx = _choose_calibration_indices(metadata.iloc[train_idx])
    if calib_idx.size:
        cal_idx = train_idx[calib_idx]
        cal_idx_set = set(cal_idx.tolist())
        subtrain_idx = np.asarray([idx for idx in train_idx if idx not in cal_idx_set], dtype=int)
    else:
        subtrain_idx = train_idx
        cal_idx = np.asarray([], dtype=int)
    class_labels = sorted(int(value) for value in np.unique(bundle.labels))
    if subtrain_idx.size == 0:
        return []
    if len(np.unique(y[subtrain_idx])) < 2:
        return []
    if cal_idx.size and len(np.unique(y[cal_idx])) < 2:
        cal_idx = np.asarray([], dtype=int)
        subtrain_idx = train_idx
    user_probs, user_temperature = _fit_calibrated_probs(
        user_spec,
        _slice_payload(x_user, subtrain_idx),
        y[subtrain_idx],
        _slice_payload(x_user, cal_idx),
        y[cal_idx],
        _slice_payload(x_user, test_idx),
        class_labels,
    )
    assist_probs, assist_temperature = _fit_calibrated_probs(
        assist_spec,
        _slice_payload(x_assist, subtrain_idx),
        y[subtrain_idx],
        _slice_payload(x_assist, cal_idx),
        y[cal_idx],
        _slice_payload(x_assist, test_idx),
        class_labels,
    )
    traces: list[PredictionTrace] = []
    for row_index, user_p, assist_p in zip(test_idx, user_probs, assist_probs):
        row = metadata.iloc[int(row_index)]
        traces.append(
            PredictionTrace(
                dataset_id=str(row["dataset_id"]),
                split_id=split.split_id,
                subject_id=str(row["subject_id"]),
                session=str(row["session"]),
                day=None if pd.isna(row.get("day")) else str(row["day"]),
                timestamp_s=float(row["timestamp_s"]),
                label=int(y[int(row_index)]),
                p_user=user_p,
                p_assist=assist_p,
                uncertainty=entropy(assist_p),
                calibration_temperature=float(assist_temperature),
                user_temperature=float(user_temperature),
                assist_temperature=float(assist_temperature),
                metadata={
                    "episode_id": row["episode_id"],
                    "prefix_time_s": float(row["prefix_time_s"]),
                    "label_raw": int(row["label_raw"]),
                    "group": str(row.get("group", "")),
                    "dynamic_flag": int(row.get("dynamic_flag", 0)) if not pd.isna(row.get("dynamic_flag")) else None,
                    "position_id": int(row.get("position_id", 0)) if not pd.isna(row.get("position_id")) else None,
                    "object_id": int(row.get("object_id", 0)) if not pd.isna(row.get("object_id")) else None,
                    "grasp_repetition": int(row.get("grasp_repetition", 0)) if not pd.isna(row.get("grasp_repetition")) else None,
                    "object_repetition": int(row.get("object_repetition", 0)) if not pd.isna(row.get("object_repetition")) else None,
                },
            )
        )
    return traces


def _fit_calibrated_probs(
    model_spec: object,
    x_train: Any,
    y_train: np.ndarray,
    x_cal: Any,
    y_cal: np.ndarray,
    x_test: Any,
    class_labels: list[int],
) -> tuple[np.ndarray, float]:
    model = _instantiate_model(model_spec)
    model.fit(x_train, y_train)
    probs_test = _predict_with_full_vocab(model, x_test, class_labels)
    if _payload_n_rows(x_cal) == 0 or len(np.unique(y_cal)) < 2:
        return probs_test, 1.0
    probs_cal = _predict_with_full_vocab(model, x_cal, class_labels)
    temperature = fit_temperature(probs_cal, y_cal)
    return probability_temperature_scale(probs_test, temperature), float(temperature)


def _predict_with_full_vocab(model: object, x: Any, class_labels: list[int]) -> np.ndarray:
    probs = np.asarray(model.predict_proba(x), dtype=float)
    classes_attr = getattr(model, "classes_", None)
    classes = np.asarray(classes_attr, dtype=int) if classes_attr is not None else np.asarray([], dtype=int)
    if classes.size == 0 and hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            step_classes = getattr(step, "classes_", None)
            if step_classes is not None:
                classes = np.asarray(step_classes, dtype=int)
                break
    full = np.zeros((_payload_n_rows(x), len(class_labels)), dtype=float)
    class_map = {label: idx for idx, label in enumerate(class_labels)}
    for src_index, label in enumerate(classes):
        full[:, class_map[int(label)]] = probs[:, src_index]
    full_sum = full.sum(axis=1, keepdims=True)
    full_sum[full_sum == 0] = 1.0
    return full / full_sum


def _choose_calibration_indices(train_metadata: pd.DataFrame) -> np.ndarray:
    group_key = "subject_id"
    if train_metadata[group_key].astype(str).nunique() <= 1:
        return np.asarray([], dtype=int)
    groups = sorted(train_metadata[group_key].astype(str).unique())
    n_cal_groups = max(1, len(groups) // 5)
    calibration_groups = set(groups[-n_cal_groups:])
    mask = train_metadata[group_key].astype(str).isin(calibration_groups).to_numpy()
    if mask.all():
        return np.asarray([], dtype=int)
    return np.flatnonzero(mask)


def _model_by_name(name: str, role: str):
    specs = _available_model_specs()
    if name not in specs:
        raise KeyError(f"unknown model spec: {name}")
    spec = specs[name]
    allowed_role = getattr(spec, "role", "either")
    if allowed_role not in {"either", role}:
        raise ValueError(f"model '{name}' is not valid for role '{role}'")
    return spec


def _day_sort_key(day: str) -> int:
    digits = "".join(ch for ch in str(day) if ch.isdigit())
    return int(digits) if digits else 0


def load_optional_sequence_payloads(
    prepared_root: Path,
    expected_rows: int | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    payloads: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for shared_name in ("sequence_payloads", "sequence_payloads.npz", "sequence_payloads.npy"):
        shared_path = prepared_root / shared_name
        if not shared_path.exists():
            continue
        loaded = _load_payload_source(shared_path)
        if not isinstance(loaded, dict):
            continue
        for role in ("user", "assist"):
            payload = _find_role_payload(loaded, role)
            if payload is None or role in payloads:
                continue
            _validate_payload_rows(payload, expected_rows, label=f"{role} sequence payload")
            payloads[role] = payload
            sources[role] = str(shared_path)
    for role in ("user", "assist"):
        if role in payloads:
            continue
        for stem in (
            f"{role}_sequence",
            f"{role}_sequences",
            f"{role}_sequence_payload",
            f"{role}_sequence_payloads",
            f"{role}_sequence_features",
        ):
            payload = None
            source_path: Path | None = None
            for suffix in ("", ".npy", ".npz"):
                candidate = prepared_root / f"{stem}{suffix}"
                if not candidate.exists():
                    continue
                loaded = _load_payload_source(candidate)
                payload = _find_role_payload(loaded, role, allow_direct=True)
                if payload is None and isinstance(loaded, dict) and loaded:
                    payload = loaded
                elif payload is None:
                    payload = loaded
                source_path = candidate
                break
            if payload is None or source_path is None:
                continue
            _validate_payload_rows(payload, expected_rows, label=f"{role} sequence payload")
            payloads[role] = payload
            sources[role] = str(source_path)
            break
    return payloads, sources


def _resolve_model_input(
    bundle: PreparedDataset,
    extra_inputs: dict[str, Any],
    model_spec: object,
    *,
    role: str,
) -> Any:
    tabular_input = bundle.user_features if role == "user" else bundle.assist_features
    spec_role = str(getattr(model_spec, "role", "either"))
    if spec_role not in {"either", role}:
        raise ValueError(f"model '{_model_name(model_spec)}' cannot be used as the {role} model")
    required_sequence_keys = tuple(getattr(model_spec, "required_sequence_keys", ()) or ())
    sequence_input = _lookup_sequence_payload(
        bundle,
        extra_inputs,
        role,
        required_sequence_keys=required_sequence_keys,
    )
    mode = _model_input_mode(model_spec)
    input_builder = _model_input_builder(model_spec)
    if input_builder is not None:
        built = _call_with_supported_kwargs(
            input_builder,
            {
                "role": role,
                "tabular": tabular_input,
                "sequence": sequence_input,
                "bundle": bundle,
                "extra_inputs": extra_inputs,
            },
        )
        if built is not None:
            return built
    if mode == "tabular":
        return tabular_input
    if sequence_input is None:
        raise ValueError(
            f"model '{_model_name(model_spec)}' requires sequence payloads for role '{role}', "
            "but no aligned payload was found under the prepared dataset root or bundle."
        )
    if mode == "hybrid":
        return {"tabular": tabular_input, "sequence": sequence_input}
    return sequence_input


def _lookup_sequence_payload(
    bundle: PreparedDataset,
    extra_inputs: dict[str, Any],
    role: str,
    *,
    required_sequence_keys: tuple[str, ...],
) -> Any | None:
    for container in (extra_inputs, getattr(bundle, "sequence_payloads", None)):
        payload = _find_role_payload(
            container,
            role,
            required_sequence_keys=required_sequence_keys,
        )
        if payload is not None:
            return payload
    for attr_name in (
        f"{role}_sequence",
        f"{role}_sequences",
        f"{role}_sequence_payload",
        f"{role}_sequence_payloads",
        f"{role}_sequence_features",
    ):
        if hasattr(bundle, attr_name):
            return _coerce_required_sequence_keys(getattr(bundle, attr_name), required_sequence_keys)
    return None


def _find_role_payload(
    container: Any,
    role: str,
    *,
    required_sequence_keys: tuple[str, ...] = (),
    allow_direct: bool = False,
) -> Any | None:
    if isinstance(container, dict):
        if required_sequence_keys and all(key in container for key in required_sequence_keys):
            return {key: container[key] for key in required_sequence_keys}
        for key in (
            role,
            f"{role}_sequence",
            f"{role}_sequences",
            f"{role}_sequence_payload",
            f"{role}_sequence_payloads",
            f"{role}_payload",
            f"{role}_inputs",
        ):
            if key in container:
                return _coerce_required_sequence_keys(container[key], required_sequence_keys)
        for nested_key in ("sequence", "sequences", "payloads", "sequence_payloads", "inputs"):
            nested = container.get(nested_key)
            if isinstance(nested, dict):
                payload = _find_role_payload(
                    nested,
                    role,
                    required_sequence_keys=required_sequence_keys,
                    allow_direct=False,
                )
                if payload is not None:
                    return payload
        prefixed = {
            str(key): value
            for key, value in container.items()
            if str(key).startswith(f"{role}_")
        }
        if prefixed:
            return _coerce_required_sequence_keys(prefixed, required_sequence_keys)
    if allow_direct:
        return _coerce_required_sequence_keys(container, required_sequence_keys)
    return None


def _coerce_required_sequence_keys(payload: Any, required_sequence_keys: tuple[str, ...]) -> Any:
    if not required_sequence_keys:
        return payload
    if not isinstance(payload, dict):
        raise ValueError(
            f"sequence payload for required keys {list(required_sequence_keys)} must be a mapping, "
            f"got {type(payload)!r}"
        )
    missing = [key for key in required_sequence_keys if key not in payload]
    if missing:
        raise ValueError(f"sequence payload is missing required keys: {missing}")
    return {key: payload[key] for key in required_sequence_keys}


def _load_payload_source(path: Path) -> Any:
    if path.is_dir():
        return _load_payload_dir(path)
    return _load_payload_file(path)


def _load_payload_dir(path: Path) -> dict[str, Any]:
    payloads = {
        child.stem: np.load(child, mmap_mode="r", allow_pickle=True)
        for child in sorted(path.glob("*.npy"))
    }
    if not payloads:
        raise ValueError(f"no payload arrays found under {path}")
    return payloads


def _load_payload_file(path: Path) -> Any:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=True) as handle:
            if handle.files == ["arr_0"]:
                return _unwrap_loaded_object(handle["arr_0"])
            return {key: _unwrap_loaded_object(handle[key]) for key in handle.files}
    return _unwrap_loaded_object(np.load(path, allow_pickle=True, mmap_mode="r"))


def _unwrap_loaded_object(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.shape == () and value.dtype == object:
        item = value.item()
        if isinstance(item, dict):
            return {str(key): _unwrap_loaded_object(val) for key, val in item.items()}
        return item
    return value


def _validate_payload_rows(payload: Any, expected_rows: int | None, *, label: str) -> None:
    if expected_rows is None:
        return
    n_rows = _payload_n_rows(payload)
    if n_rows != expected_rows:
        raise ValueError(f"{label} row count {n_rows} does not match metadata row count {expected_rows}")


def _slice_payload(payload: Any, index: np.ndarray) -> Any:
    slice_rows = getattr(models_module, "slice_input_rows", None)
    if callable(slice_rows):
        try:
            return slice_rows(payload, index)
        except Exception:
            pass
    if isinstance(payload, dict):
        return {key: _slice_payload(value, index) for key, value in payload.items()}
    if isinstance(payload, pd.DataFrame | pd.Series):
        return payload.iloc[index]
    if isinstance(payload, np.ndarray):
        return payload[index]
    if isinstance(payload, list):
        return [payload[int(i)] for i in index]
    if isinstance(payload, tuple):
        return tuple(payload[int(i)] for i in index)
    try:
        return payload[index]
    except Exception:
        return [payload[int(i)] for i in index]


def _payload_n_rows(payload: Any) -> int:
    row_counter = getattr(models_module, "input_row_count", None)
    if callable(row_counter):
        try:
            return int(row_counter(payload))
        except Exception:
            pass
    if isinstance(payload, dict):
        if not payload:
            return 0
        lengths = {_payload_n_rows(value) for value in payload.values()}
        if len(lengths) != 1:
            raise ValueError(f"inconsistent row counts inside payload mapping: {sorted(lengths)}")
        return int(next(iter(lengths)))
    if isinstance(payload, pd.DataFrame | pd.Series):
        return int(payload.shape[0])
    if isinstance(payload, np.ndarray):
        if payload.ndim == 0:
            return 0
        return int(payload.shape[0])
    if isinstance(payload, (list, tuple)):
        return len(payload)
    if isinstance(payload, (str, bytes)):
        raise TypeError("string payloads are not row-aligned model inputs")
    if hasattr(payload, "__len__"):
        return int(len(payload))
    raise TypeError(f"cannot infer payload row count for {type(payload)!r}")


def _instantiate_model(model_spec: object) -> object:
    for obj in _metadata_objects(model_spec):
        factory = getattr(obj, "make_estimator", None) or getattr(obj, "build_estimator", None)
        if callable(factory):
            model = _call_with_supported_kwargs(factory, {})
            if model is not None:
                return model
    base = getattr(model_spec, "estimator", model_spec)
    if callable(base) and not hasattr(base, "fit"):
        candidate = _call_with_supported_kwargs(base, {})
        if candidate is not None:
            return candidate
    try:
        return clone(base)
    except Exception:
        return copy.deepcopy(base)


def _available_model_specs() -> dict[str, object]:
    specs: dict[str, object] = {}
    for provider_name in (
        "all_model_specs",
        "benchmark_model_specs",
        "real_model_specs",
        "sequence_model_specs",
        "shallow_model_specs",
    ):
        provider = getattr(models_module, provider_name, None)
        if not callable(provider):
            continue
        try:
            provided = _call_with_supported_kwargs(provider, {})
        except ImportError:
            continue
        if not provided:
            continue
        for spec in provided:
            spec_name = getattr(spec, "name", None)
            if spec_name is None or str(spec_name) in specs:
                continue
            specs[str(spec_name)] = spec
    return specs


def _model_input_mode(model_spec: object) -> str:
    for attr_name in ("input_mode", "input_kind", "payload_kind", "data_kind", "feature_kind"):
        value = _metadata_value(model_spec, attr_name)
        normalized = _normalize_input_mode(value)
        if normalized is not None:
            return normalized
    for flag_name in (
        "requires_sequence",
        "needs_sequence",
        "use_sequence",
        "sequence_only",
        "requires_sequence_payload",
    ):
        value = _metadata_value(model_spec, flag_name)
        if value is not None:
            return "sequence" if bool(value) else "tabular"
    name = _model_name(model_spec).lower()
    if any(token in name for token in ("sequence", "_seq", "seq_", "lstm", "gru", "transformer", "tcn", "cnn1d")):
        return "sequence"
    return "tabular"


def _normalize_input_mode(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set, frozenset)):
        tokens = {str(item).strip().lower() for item in value}
        if {"tabular", "sequence"} <= tokens:
            return "hybrid"
        if "sequence" in tokens or "seq" in tokens:
            return "sequence"
        if "tabular" in tokens or "array" in tokens:
            return "tabular"
        return None
    token = str(value).strip().lower()
    if token in {"sequence", "seq", "temporal", "timeseries"}:
        return "sequence"
    if token in {"hybrid", "multimodal", "tabular+sequence", "sequence+tabular"}:
        return "hybrid"
    if token in {"tabular", "array", "features", "feature_array"}:
        return "tabular"
    return None


def _model_input_builder(model_spec: object):
    for obj in _metadata_objects(model_spec):
        for attr_name in ("prepare_benchmark_input", "build_benchmark_input", "resolve_benchmark_input"):
            builder = getattr(obj, attr_name, None)
            if callable(builder):
                return builder
    return None


def _metadata_value(model_spec: object, attr_name: str) -> Any:
    for obj in _metadata_objects(model_spec):
        value = getattr(obj, attr_name, None)
        if value is not None:
            return value
    return None


def _metadata_objects(model_spec: object) -> list[object]:
    objects: list[object] = [model_spec]
    estimator = getattr(model_spec, "estimator", None)
    if estimator is None:
        return objects
    objects.append(estimator)
    if hasattr(estimator, "named_steps"):
        objects.extend(reversed(list(estimator.named_steps.values())))
    return objects


def _model_name(model_spec: object) -> str:
    return str(getattr(model_spec, "name", getattr(model_spec, "__class__", type(model_spec)).__name__))


def _call_with_supported_kwargs(func, kwargs: dict[str, Any]) -> Any:
    signature = inspect.signature(func)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return func(**kwargs)
    supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return func(**supported)

