from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from . import models as models_module
from .stats import holm_bonferroni, paired_policy_difference_test


TAU_GRID = ("0.02", "0.05", "0.10", "0.15", "0.20")
ACTIVE_GATE_METRICS = ("active_macro_f1", "active_risk_coverage_auc")
DB10_SAFE_CANDIDATE_PAIRS = (
    ("logistic", "logistic"),
    ("extra_trees", "logistic"),
    ("random_forest", "logistic"),
    ("logistic", "extra_trees"),
    ("extra_trees", "extra_trees"),
    ("logistic", "torch_mlp"),
    ("extra_trees", "torch_mlp"),
    ("torch_mlp", "torch_mlp"),
)
DB10_SOTA_USER_ANCHORS = ("logistic", "extra_trees", "torch_mlp")
DB10_SOTA_ASSIST_ANCHORS = ("logistic", "extra_trees", "torch_mlp")
PREFERRED_MODEL_ORDER = (
    "lda",
    "logistic",
    "random_forest",
    "extra_trees",
    "torch_mlp",
    "temporal_cnn",
    "hybrid_temporal_mlp",
    "pactformer",
    "assist_hybrid_fusion",
    "assist_temporal_forest",
    "assist_stacked_fusion",
)


@dataclass(frozen=True)
class CandidateSummary:
    candidate_id: str
    candidate_lane: str
    user_model: str
    assist_model: str
    qualifying_taus: tuple[str, ...]
    qualifying_tau_count: int
    all_families_passed: bool
    beneficial_active_f1_cells: int
    beneficial_active_risk_cells: int
    split_family_count: int
    best_agency_active_macro_f1: float
    best_agency_active_risk_coverage_auc: float

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["qualifying_taus"] = "|".join(self.qualifying_taus)
        return payload


def default_db10_candidate_pairs(lanes: tuple[str, ...] | None = None) -> list[tuple[str, str]]:
    requested_lanes = _normalize_lanes(lanes)
    available_specs = _available_model_specs()
    available_names = set(available_specs)
    pairs: list[tuple[str, str]] = []
    if "safe" in requested_lanes:
        pairs.extend(_filter_known_pairs(DB10_SAFE_CANDIDATE_PAIRS, available_names))
    if "sota" in requested_lanes:
        pairs.extend(_db10_sota_candidate_pairs(available_specs))
    return _dedupe_pairs(pairs)


def default_real_candidate_pairs(*, available_sequence_keys: set[str] | None = None) -> list[tuple[str, str]]:
    specs = _available_model_specs()
    ordered_names = _ordered_model_names(specs)
    user_models = [
        name
        for name in ordered_names
        if _model_supports_role(specs.get(name), role="user")
        and _model_is_payload_compatible(specs.get(name), model_name=name, available_sequence_keys=available_sequence_keys)
    ]
    assist_models = [
        name
        for name in ordered_names
        if _model_supports_role(specs.get(name), role="assist")
        and _model_is_payload_compatible(specs.get(name), model_name=name, available_sequence_keys=available_sequence_keys)
    ]
    return [(user_model, assist_model) for user_model in user_models for assist_model in assist_models]


def infer_candidate_lane(user_model: str, assist_model: str) -> str:
    specs = _available_model_specs()
    if _model_requires_sequence(specs.get(user_model), model_name=user_model):
        return "sota"
    if _model_requires_sequence(specs.get(assist_model), model_name=assist_model):
        return "sota"
    return "safe"


def infer_db10_candidate_lane(user_model: str, assist_model: str) -> str:
    return infer_candidate_lane(user_model, assist_model)


def evaluate_candidate(
    metrics_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    *,
    user_model: str,
    assist_model: str,
    repeats: int = 2000,
) -> tuple[CandidateSummary, pd.DataFrame]:
    normalized = _normalize_unit_columns(metrics_df)
    pairwise_rows: list[dict[str, object]] = []
    for split_family, split_df in normalized.groupby("split_family", sort=True):
        unit_cols = _infer_unit_cols(split_df)
        metric_frames: list[pd.DataFrame] = []
        for metric in ACTIVE_GATE_METRICS:
            rows = []
            for tau in TAU_GRID:
                policy_a = f"agency_margin_tau_{tau}"
                policy_b = f"set_acsa_tau_{tau}"
                if policy_a not in set(split_df["policy"].astype(str)) or policy_b not in set(split_df["policy"].astype(str)):
                    continue
                estimate, ci_low, ci_high, p_value, n_units = paired_policy_difference_test(
                    split_df,
                    policy_a=policy_a,
                    policy_b=policy_b,
                    value_col=metric,
                    unit_cols=unit_cols,
                    repeats=repeats,
                )
                rows.append(
                    {
                        "split_family": str(split_family),
                        "metric": metric,
                        "policy_a": policy_a,
                        "policy_b": policy_b,
                        "tau": tau,
                        "unit_cols": "|".join(unit_cols),
                        "n_units": n_units,
                        "estimate": estimate,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "p_value": p_value,
                    }
                )
            if rows:
                metric_df = pd.DataFrame(rows)
                rejected = holm_bonferroni(metric_df["p_value"].astype(float).tolist())
                metric_df["holm_reject_0_05"] = rejected
                metric_df["beneficial"] = metric_df.apply(_is_beneficial, axis=1)
                metric_frames.append(metric_df)
        if metric_frames:
            pairwise_rows.extend(pd.concat(metric_frames, ignore_index=True).to_dict(orient="records"))
    pairwise_df = pd.DataFrame(pairwise_rows)
    summary = _summarize_candidate(pairwise_df, aggregate_df, user_model=user_model, assist_model=assist_model)
    return summary, pairwise_df


def evaluate_db10_candidate(
    metrics_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    *,
    user_model: str,
    assist_model: str,
    repeats: int = 2000,
) -> tuple[CandidateSummary, pd.DataFrame]:
    return evaluate_candidate(
        metrics_df,
        aggregate_df,
        user_model=user_model,
        assist_model=assist_model,
        repeats=repeats,
    )


def _normalize_unit_columns(data: pd.DataFrame) -> pd.DataFrame:
    normalized = data.copy()
    for col in ["dataset_id", "split_id", "split_family", "policy", "subject_id", "session", "day"]:
        if col not in normalized.columns:
            normalized[col] = "<missing>"
        normalized[col] = normalized[col].astype("object").where(normalized[col].notna(), "<missing>").astype(str)
    return normalized


def _infer_unit_cols(data: pd.DataFrame) -> list[str]:
    unit_cols: list[str] = []
    for col in ["split_id", "subject_id", "session", "day"]:
        if col not in data.columns:
            continue
        values = set(data[col].astype(str).unique())
        if values == {"<missing>"}:
            continue
        unit_cols.append(col)
    if not unit_cols:
        raise ValueError("no inferential unit columns found")
    return unit_cols


def _is_beneficial(row: pd.Series) -> bool:
    estimate = float(row["estimate"])
    if str(row["metric"]) == "active_macro_f1":
        return estimate > 0.0
    return estimate < 0.0


def _summarize_candidate(
    pairwise_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    *,
    user_model: str,
    assist_model: str,
) -> CandidateSummary:
    candidate_id = f"{user_model}__{assist_model}"
    split_families = sorted(pairwise_df["split_family"].astype(str).unique()) if not pairwise_df.empty else []
    qualifying_taus: list[str] = []
    if not pairwise_df.empty:
        for tau, tau_df in pairwise_df.groupby("tau", sort=True):
            tau_pass = True
            for metric in ACTIVE_GATE_METRICS:
                metric_df = tau_df.loc[
                    (tau_df["metric"] == metric)
                    & (tau_df["holm_reject_0_05"] == True)
                    & (tau_df["beneficial"] == True)
                ]
                if set(metric_df["split_family"].astype(str)) != set(split_families):
                    tau_pass = False
                    break
            if tau_pass:
                qualifying_taus.append(str(tau))
    agency_df = aggregate_df.loc[aggregate_df["policy"].astype(str).str.startswith("agency_margin_tau_")].copy()
    return CandidateSummary(
        candidate_id=candidate_id,
        candidate_lane=infer_candidate_lane(user_model, assist_model),
        user_model=user_model,
        assist_model=assist_model,
        qualifying_taus=tuple(qualifying_taus),
        qualifying_tau_count=len(qualifying_taus),
        all_families_passed=bool(qualifying_taus),
        beneficial_active_f1_cells=int(
            pairwise_df.loc[
                (pairwise_df["metric"] == "active_macro_f1")
                & (pairwise_df["holm_reject_0_05"] == True)
                & (pairwise_df["beneficial"] == True)
            ].shape[0]
        )
        if not pairwise_df.empty
        else 0,
        beneficial_active_risk_cells=int(
            pairwise_df.loc[
                (pairwise_df["metric"] == "active_risk_coverage_auc")
                & (pairwise_df["holm_reject_0_05"] == True)
                & (pairwise_df["beneficial"] == True)
            ].shape[0]
        )
        if not pairwise_df.empty
        else 0,
        split_family_count=len(split_families),
        best_agency_active_macro_f1=float(agency_df["active_macro_f1"].max()) if not agency_df.empty else 0.0,
        best_agency_active_risk_coverage_auc=float(agency_df["active_risk_coverage_auc"].min()) if not agency_df.empty else 1.0,
    )


def _db10_sota_candidate_pairs(available_specs: dict[str, object]) -> list[tuple[str, str]]:
    user_sequence_models = sorted(
        name
        for name, spec in available_specs.items()
        if _model_requires_sequence(spec, model_name=name) and _model_supports_role(spec, role="user")
    )
    assist_sequence_models = sorted(
        name
        for name, spec in available_specs.items()
        if _model_requires_sequence(spec, model_name=name) and _model_supports_role(spec, role="assist")
    )
    if not user_sequence_models and not assist_sequence_models:
        return []
    available_names = set(available_specs)
    pairs: list[tuple[str, str]] = []
    for assist_model in assist_sequence_models:
        for user_anchor in DB10_SOTA_USER_ANCHORS:
            if available_names and user_anchor not in available_names:
                continue
            pairs.append((user_anchor, assist_model))
    for user_model in user_sequence_models:
        for assist_anchor in DB10_SOTA_ASSIST_ANCHORS:
            if available_names and assist_anchor not in available_names:
                continue
            pairs.append((user_model, assist_anchor))
    for user_model in user_sequence_models:
        for assist_model in assist_sequence_models:
            pairs.append((user_model, assist_model))
    return _filter_known_pairs(pairs, available_names)


def _normalize_lanes(lanes: tuple[str, ...] | None) -> tuple[str, ...]:
    if not lanes:
        return ("safe", "sota")
    normalized: list[str] = []
    for lane in lanes:
        token = str(lane).strip().lower()
        if token not in {"safe", "sota"}:
            raise ValueError(f"unsupported DB10 sweep lane: {lane}")
        if token not in normalized:
            normalized.append(token)
    return tuple(normalized)


def _dedupe_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    ordered: list[tuple[str, str]] = []
    for pair in pairs:
        if pair in seen:
            continue
        seen.add(pair)
        ordered.append(pair)
    return ordered


def _filter_known_pairs(
    pairs: tuple[tuple[str, str], ...] | list[tuple[str, str]],
    available_names: set[str],
) -> list[tuple[str, str]]:
    if not available_names:
        return list(pairs)
    return [
        (user_model, assist_model)
        for user_model, assist_model in pairs
        if user_model in available_names and assist_model in available_names
    ]


def _ordered_model_names(available_specs: dict[str, object]) -> list[str]:
    preferred_rank = {name: index for index, name in enumerate(PREFERRED_MODEL_ORDER)}
    return sorted(
        available_specs,
        key=lambda name: (preferred_rank.get(name, len(preferred_rank)), str(name)),
    )


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


def _model_is_payload_compatible(
    model_spec: object | None,
    *,
    model_name: str,
    available_sequence_keys: set[str] | None,
) -> bool:
    if not _model_requires_sequence(model_spec, model_name=model_name):
        return True
    if not available_sequence_keys:
        return False
    required_keys = tuple(_metadata_value(model_spec, "required_sequence_keys") or ())
    if required_keys:
        return set(required_keys).issubset({str(key) for key in available_sequence_keys})
    return True


def _model_requires_sequence(model_spec: object | None, *, model_name: str) -> bool:
    if model_spec is None:
        return _name_suggests_sequence(model_name)
    input_kind = _metadata_value(model_spec, "input_kind")
    if input_kind is not None:
        return _normalize_input_mode(input_kind) != "tabular"
    for attr_name in (
        "requires_sequence",
        "needs_sequence",
        "use_sequence",
        "sequence_only",
        "requires_sequence_payload",
    ):
        value = _metadata_value(model_spec, attr_name)
        if value is not None:
            return bool(value)
    for attr_name in ("input_mode", "input_kind", "payload_kind", "data_kind", "feature_kind"):
        value = _metadata_value(model_spec, attr_name)
        token = _normalize_input_mode(value)
        if token is not None:
            return token != "tabular"
    return _name_suggests_sequence(model_name)


def _model_supports_role(model_spec: object | None, *, role: str) -> bool:
    if model_spec is None:
        return True
    value = _metadata_value(model_spec, "role")
    if value is None:
        return True
    token = str(value).strip().lower()
    return token in {"either", role}


def _normalize_input_mode(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set, frozenset)):
        tokens = {str(item).strip().lower() for item in value}
        if "sequence" in tokens or "seq" in tokens:
            return "sequence"
        if "tabular" in tokens or "array" in tokens:
            return "tabular"
        return None
    token = str(value).strip().lower()
    if token in {"sequence", "seq", "temporal", "timeseries", "hybrid", "multimodal"}:
        return "sequence"
    if token in {"tabular", "array", "features", "feature_array"}:
        return "tabular"
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


def _name_suggests_sequence(model_name: str) -> bool:
    name = str(model_name).strip().lower()
    return any(token in name for token in ("sequence", "_seq", "seq_", "lstm", "gru", "transformer", "tcn", "cnn1d"))


def _call_with_supported_kwargs(func, kwargs: dict[str, Any]) -> Any:
    signature = inspect.signature(func)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return func(**kwargs)
    supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return func(**supported)

