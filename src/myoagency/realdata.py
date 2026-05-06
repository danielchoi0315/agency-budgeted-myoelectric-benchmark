from __future__ import annotations

import io
import json
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.io as sio
import wfdb
from stream_unzip import stream_unzip

from .features import time_domain_features, window_signal
from .io import load_yaml
from .preprocess import bandpass_emg, notch_filter


DB10_RELEASE_PATTERN = re.compile(r"^(S\d+)_ex1\.mat$")
HYSER_SESSION_PATTERN = re.compile(r"^subject(\d{2})_session(\d+)$")
GRABMYO_RECORD_PATTERN = re.compile(
    r"^Session(?P<session>\d+)/session(?P=session)_participant(?P<participant>\d+)/"
    r"session(?P=session)_participant(?P=participant)_gesture(?P<gesture>\d+)_trial(?P<trial>\d+)$"
)
CEMHSEY_GRASP_PATTERN = re.compile(
    r"^(S\d+)/D(\d+)/(S\d+)_Day(\d+)_Session(\d+)_Task(\d+)_Trial(\d+)\.mat$"
)
DEFAULT_CEMHSEY_EXCLUSIONS_PATH = Path(__file__).resolve().parents[2] / "config" / "exclusions.yaml"
DB10_CONTEXT_MODES = {"sensor_summary", "protocol_context", "annotated_object_proxy"}
DB10_OBJECT_ID_CHOICES = tuple(range(0, 25))
DB10_OBJECT_PART_CHOICES = tuple(range(0, 25))
DB10_BINARY_CHOICES = (0, 1)
DB10_EMG_DIM = 12
DB10_GAZE_DIM = 2
DB10_PUPIL_DIM = 2
DB10_IMU_DIM = 3
DB10_SEQUENCE_USER_MAX_STEPS = 128
DB10_SEQUENCE_ASSIST_MAX_STEPS = 256
PREPARED_BUNDLE_REQUIRED_FILES = (
    "metadata.csv",
    "user_features.npy",
    "assist_features.npy",
    "labels.npy",
    "label_vocab.npy",
)
@dataclass(frozen=True)
class PreparedDataset:
    metadata: pd.DataFrame
    user_features: np.ndarray
    assist_features: np.ndarray
    labels: np.ndarray
    label_vocab: list[int]
    sequence_payloads: dict[str, np.ndarray] | None = None

    def validate(self) -> None:
        n = len(self.metadata)
        if self.user_features.shape[0] != n:
            raise ValueError("user_features row count does not match metadata")
        if self.assist_features.shape[0] != n:
            raise ValueError("assist_features row count does not match metadata")
        if self.labels.shape[0] != n:
            raise ValueError("labels row count does not match metadata")
        if not self.sequence_payloads:
            return
        for name, payload in self.sequence_payloads.items():
            arr = np.asarray(payload)
            if arr.ndim == 0:
                raise ValueError(f"sequence payload '{name}' must have at least one dimension")
            if arr.shape[0] != n:
                raise ValueError(f"sequence payload '{name}' row count does not match metadata")


def save_prepared_dataset(bundle: PreparedDataset, out_dir: Path) -> None:
    bundle.validate()
    out_dir.mkdir(parents=True, exist_ok=True)
    success_file = out_dir / "_SUCCESS"
    sequence_file = out_dir / "sequence_payloads.npz"
    legacy_sequence_file = out_dir / "sequence_payloads.npy"
    sequence_dir = out_dir / "sequence_payloads"
    if success_file.exists():
        success_file.unlink()
    if sequence_file.exists():
        sequence_file.unlink()
    if legacy_sequence_file.exists():
        legacy_sequence_file.unlink()
    if sequence_dir.exists():
        shutil.rmtree(sequence_dir)
    bundle.metadata.to_csv(out_dir / "metadata.csv", index=False)
    np.save(out_dir / "user_features.npy", bundle.user_features)
    np.save(out_dir / "assist_features.npy", bundle.assist_features)
    np.save(out_dir / "labels.npy", bundle.labels)
    np.save(out_dir / "label_vocab.npy", np.asarray(bundle.label_vocab, dtype=int))
    if bundle.sequence_payloads:
        sequence_dir.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, dict[str, Any]] = {}
        for name, payload in bundle.sequence_payloads.items():
            arr = np.asarray(payload)
            np.save(sequence_dir / f"{name}.npy", arr)
            manifest[name] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
            }
        (sequence_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    success_file.write_text("ok\n", encoding="utf-8")


def load_prepared_dataset(root: Path) -> PreparedDataset:
    sequence_payloads = load_saved_sequence_payloads(root)
    bundle = PreparedDataset(
        metadata=pd.read_csv(root / "metadata.csv"),
        user_features=np.load(root / "user_features.npy"),
        assist_features=np.load(root / "assist_features.npy"),
        labels=np.load(root / "labels.npy"),
        label_vocab=np.load(root / "label_vocab.npy").astype(int).tolist(),
        sequence_payloads=sequence_payloads,
    )
    bundle.validate()
    return bundle


def load_saved_sequence_payloads(root: Path) -> dict[str, np.ndarray] | None:
    sequence_dir = root / "sequence_payloads"
    if sequence_dir.exists():
        payloads = {
            path.stem: np.load(path, mmap_mode="r")
            for path in sorted(sequence_dir.glob("*.npy"))
        }
        return payloads or None
    sequence_file = root / "sequence_payloads.npz"
    if sequence_file.exists():
        with np.load(sequence_file) as payloads:
            return {name: payloads[name] for name in payloads.files}
    legacy_sequence_file = root / "sequence_payloads.npy"
    if legacy_sequence_file.exists():
        payload = np.load(legacy_sequence_file, allow_pickle=True)
        if isinstance(payload, np.ndarray) and payload.shape == () and payload.dtype == object:
            item = payload.item()
            if isinstance(item, dict):
                return {str(name): np.asarray(value) for name, value in item.items()}
    return None


def db10_subject_group(subject_id: str) -> str:
    match = re.search(r"S(\d+)", subject_id)
    if not match:
        raise ValueError(f"invalid DB10 subject id: {subject_id}")
    subject_num = int(match.group(1))
    return "amputee" if subject_num >= 100 else "able_bodied"


def parse_hyser_label_text(text: str) -> list[int]:
    tokens = [token.strip() for token in text.replace("\n", ",").split(",")]
    return [int(token) for token in tokens if token]


def parse_grabmyo_record_path(record_path: str) -> dict[str, str | int]:
    normalized = record_path.replace("\\", "/")
    match = GRABMYO_RECORD_PATTERN.match(normalized)
    if not match:
        raise ValueError(f"unexpected GRABMyo record path: {record_path}")
    session = int(match.group("session"))
    participant = int(match.group("participant"))
    gesture = int(match.group("gesture"))
    trial = int(match.group("trial"))
    return {
        "subject_id": f"participant{participant:02d}",
        "session": f"session{session}",
        "day": f"D{session:02d}",
        "task": "gesture",
        "gesture_label_raw": gesture,
        "trial": trial,
    }


def parse_cemhsey_grasp_member(member_name: str) -> dict[str, str | int]:
    normalized = member_name.replace("\\", "/")
    match = CEMHSEY_GRASP_PATTERN.match(normalized)
    if not match:
        raise ValueError(f"unexpected CEMHSEY GRASP member path: {member_name}")
    subject_dir, day_dir, subject_name, day_name, session, task, trial = match.groups()
    if subject_dir != subject_name or int(day_dir) != int(day_name):
        raise ValueError(f"inconsistent CEMHSEY member path: {member_name}")
    return {
        "subject_id": subject_name,
        "day": f"D{int(day_name):02d}",
        "grasp_block": f"Session{int(session)}",
        "task": f"Task{int(task)}",
        "trial": f"Trial{int(trial)}",
        "grasp_label_raw": int(session),
        "force_level": int(task),
    }


def build_db10_manifest(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("S*_ex1.mat")):
        match = DB10_RELEASE_PATTERN.match(path.name)
        if not match:
            continue
        subject_id = match.group(1)
        rows.append(
            {
                "dataset_id": "db10",
                "subject_id": subject_id,
                "group": db10_subject_group(subject_id),
                "session": "ex1",
                "day": None,
                "task": "grasp_stream",
                "source_path": str(path),
            }
        )
    return pd.DataFrame(rows)


def _coerce_metadata_column(
    metadata: pd.DataFrame,
    column: str,
    row_ids: np.ndarray,
    *,
    fill_value: str | None = None,
    prefix: str | None = None,
    default_from: str | None = None,
) -> pd.Series:
    if column in metadata.columns:
        series = metadata[column].astype("object")
    else:
        series = pd.Series([None] * len(metadata), index=metadata.index, dtype="object")
    if default_from is not None:
        fallback = metadata[default_from].astype("object")
        series = series.where(series.notna(), fallback)
    if fill_value is not None:
        series = series.fillna(fill_value)
    if prefix is not None:
        fallback = pd.Series([f"{prefix}_{row_id:06d}" for row_id in row_ids], index=metadata.index, dtype="object")
        series = series.where(series.notna(), fallback)
    normalized = series.astype(str)
    invalid = normalized.str.lower().isin({"", "nan", "none", "<na>"})
    if default_from is not None:
        fallback = metadata[default_from].astype(str)
        normalized = normalized.where(~invalid, fallback)
        invalid = normalized.str.lower().isin({"", "nan", "none", "<na>"})
    if fill_value is not None:
        normalized = normalized.where(~invalid, fill_value)
        invalid = normalized.str.lower().isin({"", "nan", "none", "<na>"})
    if prefix is not None:
        fallback = pd.Series([f"{prefix}_{row_id:06d}" for row_id in row_ids], index=metadata.index, dtype="object")
        normalized = normalized.where(~invalid, fallback.astype(str))
    return normalized.astype(str)


def _coerce_numeric_metadata_column(
    metadata: pd.DataFrame,
    column: str,
    *,
    default_from: str | None = None,
    fill_value: float = 0.0,
) -> pd.Series:
    if column in metadata.columns:
        series = pd.to_numeric(metadata[column], errors="coerce")
    else:
        series = pd.Series(np.nan, index=metadata.index, dtype=float)
    if default_from is not None:
        fallback = pd.to_numeric(metadata[default_from], errors="coerce")
        series = series.where(series.notna(), fallback)
    return series.fillna(float(fill_value)).astype(float)


def build_hyser_manifest(root: Path, task_type: str = "dynamic", signal_type: str = "preprocess") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pr_root = root / "pr_dataset"
    for session_dir in sorted(path for path in pr_root.iterdir() if path.is_dir()):
        match = HYSER_SESSION_PATTERN.match(session_dir.name)
        if not match:
            continue
        subject_id = f"subject{match.group(1)}"
        session = f"session{match.group(2)}"
        label_path = session_dir / f"label_{task_type}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"missing Hyser label file: {label_path}")
        labels = parse_hyser_label_text(label_path.read_text(encoding="utf-8"))
        record_paths = sorted(session_dir.glob(f"{task_type}_{signal_type}_sample*.hea"))
        if not record_paths:
            continue
        if len(labels) != len(record_paths):
            raise ValueError(
                f"Hyser label/record mismatch for {session_dir.name} {task_type}: "
                f"{len(labels)} labels vs {len(record_paths)} records"
            )
        for record_path in record_paths:
            sample_index = int(record_path.stem.split("sample", 1)[1])
            label_raw = labels[sample_index - 1]
            rows.append(
                {
                    "dataset_id": "hyser",
                    "subject_id": subject_id,
                    "group": "able_bodied",
                    "session": session,
                    "day": session,
                    "task": task_type,
                    "signal_type": signal_type,
                    "sample_index": sample_index,
                    "label_raw": label_raw,
                    "source_path": str(record_path.with_suffix("")),
                }
            )
    return pd.DataFrame(rows)


def build_grabmyo_manifest(root: Path) -> pd.DataFrame:
    records_path = root / "RECORDS"
    if not records_path.exists():
        raise FileNotFoundError(f"missing GRABMyo RECORDS file: {records_path}")
    unique_records = sorted({line.strip() for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()})
    rows: list[dict[str, Any]] = []
    for record in unique_records:
        parsed = parse_grabmyo_record_path(record)
        rows.append(
            {
                "dataset_id": "grabmyo",
                "subject_id": parsed["subject_id"],
                "group": "able_bodied",
                "session": parsed["session"],
                "day": parsed["day"],
                "task": parsed["task"],
                "trial": parsed["trial"],
                "label_raw": parsed["gesture_label_raw"],
                "source_path": str(root / record),
            }
        )
    return pd.DataFrame(rows)


def build_cemhsey_grasp_manifest(
    root: Path,
    exclusion_config: Path | None = None,
    include_failed_trials: bool = False,
) -> pd.DataFrame:
    manifest, _ = build_cemhsey_grasp_manifest_with_report(
        root,
        exclusion_config=exclusion_config,
        include_failed_trials=include_failed_trials,
    )
    return manifest


def build_cemhsey_grasp_manifest_with_report(
    root: Path,
    exclusion_config: Path | None = None,
    include_failed_trials: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for zip_path in sorted(root.rglob("GRASP_S*.zip")):
        with zipfile.ZipFile(zip_path) as handle:
            for member_name in handle.namelist():
                if not member_name.endswith(".mat"):
                    continue
                parsed = parse_cemhsey_grasp_member(member_name)
                rows.append(
                    {
                        "dataset_id": "cemhsey",
                        "subject_id": parsed["subject_id"],
                        "group": "able_bodied",
                        "session": "grasp_protocol",
                        "day": parsed["day"],
                        "task": "grasp",
                        "trial": parsed["trial"],
                        "grasp_block": parsed["grasp_block"],
                        "force_level": parsed["force_level"],
                        "label_raw": parsed["grasp_label_raw"],
                        "archive_path": str(zip_path),
                        "member_path": member_name,
                    }
                )
    manifest = pd.DataFrame(rows)
    return apply_cemhsey_trial_exclusions(
        manifest,
        subset="grasp",
        exclusion_config=exclusion_config,
        include_failed_trials=include_failed_trials,
    )


def apply_cemhsey_trial_exclusions(
    manifest: pd.DataFrame,
    subset: str,
    exclusion_config: Path | None = None,
    include_failed_trials: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    config_path = exclusion_config or DEFAULT_CEMHSEY_EXCLUSIONS_PATH
    configured_trials = load_cemhsey_failed_trials(config_path=config_path, subset=subset)
    report = {
        "subset": subset,
        "config_path": str(config_path),
        "config_exists": bool(config_path.exists()),
        "include_failed_trials": bool(include_failed_trials),
        "configured_failed_trials": sorted(configured_trials),
        "matched_failed_trials": [],
        "unmatched_failed_trials": [],
        "excluded_member_paths": [],
        "n_input_trials": int(len(manifest)),
        "n_excluded_trials": 0,
        "n_remaining_trials": int(len(manifest)),
    }
    if include_failed_trials:
        return manifest.reset_index(drop=True), report
    if not config_path.exists():
        raise FileNotFoundError(f"missing CEMHSEY exclusion config: {config_path}")
    if not configured_trials:
        raise ValueError(f"no configured failed trials for CEMHSEY subset '{subset}' in {config_path}")
    if manifest.empty:
        return manifest.reset_index(drop=True), report
    if "member_path" not in manifest.columns:
        raise ValueError("CEMHSEY manifest must include member_path for failed-trial exclusions")
    normalized_ids = {item.replace("\\", "/") for item in configured_trials}
    mask = manifest["member_path"].astype(str).map(
        lambda member_path: any(key in normalized_ids for key in cemhsey_exclusion_keys(member_path))
    )
    excluded = manifest.loc[mask].copy()
    matched_failed_trials = sorted({Path(str(path).replace("\\", "/")).name for path in excluded["member_path"]})
    report["matched_failed_trials"] = matched_failed_trials
    report["unmatched_failed_trials"] = sorted(
        trial_id for trial_id in configured_trials if Path(trial_id).name not in matched_failed_trials
    )
    if report["unmatched_failed_trials"]:
        raise ValueError(
            f"CEMHSEY exclusion config contains unmatched failed trials for subset '{subset}': "
            f"{report['unmatched_failed_trials']}"
        )
    report["excluded_member_paths"] = sorted(excluded["member_path"].astype(str).tolist())
    report["n_excluded_trials"] = int(mask.sum())
    report["n_remaining_trials"] = int((~mask).sum())
    return manifest.loc[~mask].reset_index(drop=True), report


def load_cemhsey_failed_trials(config_path: Path | None = None, subset: str = "grasp") -> set[str]:
    path = config_path or DEFAULT_CEMHSEY_EXCLUSIONS_PATH
    if not path.exists():
        return set()
    payload = load_yaml(path)
    if not isinstance(payload, dict):
        return set()
    cemhsey = payload.get("cemhsey", {})
    if not isinstance(cemhsey, dict):
        return set()
    key = f"{subset}_failed_trials"
    values = cemhsey.get(key, [])
    if not isinstance(values, list):
        return set()
    return {str(value).replace("\\", "/") for value in values}


def cemhsey_exclusion_keys(member_path: str) -> set[str]:
    normalized = str(member_path).replace("\\", "/")
    return {normalized, Path(normalized).name}


def load_cemhsey_zip_member_bytes(archive_path: Path, member_name: str) -> bytes:
    compress_type = None
    if archive_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive_path) as handle:
            info = handle.getinfo(member_name)
            compress_type = info.compress_type
            if compress_type != 9:
                return handle.read(member_name)
    if compress_type == 9:
        with archive_path.open("rb") as handle:
            for file_name, _, chunks in stream_unzip(handle):
                decoded = file_name.decode("utf-8")
                if decoded == member_name:
                    return b"".join(chunks)
        raise FileNotFoundError(f"{member_name} not found in {archive_path}")
    raise ValueError(f"unsupported archive type: {archive_path}")


def prepare_db10_dataset(
    root: Path,
    include_rest: bool = True,
    prefix_grid_s: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0),
    user_window_s: float = 0.2,
    min_segment_s: float = 0.25,
    limit_files: int | None = None,
    context_mode: str = "annotated_object_proxy",
    emit_sequences: bool = False,
    sequence_user_max_steps: int = DB10_SEQUENCE_USER_MAX_STEPS,
    sequence_assist_max_steps: int = DB10_SEQUENCE_ASSIST_MAX_STEPS,
) -> PreparedDataset:
    if context_mode not in DB10_CONTEXT_MODES:
        raise ValueError(f"unsupported DB10 context_mode: {context_mode}")
    if emit_sequences and sequence_user_max_steps <= 0:
        raise ValueError("sequence_user_max_steps must be positive when emit_sequences is enabled")
    if emit_sequences and sequence_assist_max_steps <= 0:
        raise ValueError("sequence_assist_max_steps must be positive when emit_sequences is enabled")
    manifest = build_db10_manifest(root)
    if limit_files is not None:
        manifest = manifest.head(limit_files).copy()
    rows: list[dict[str, Any]] = []
    user_vectors: list[np.ndarray] = []
    assist_vectors: list[np.ndarray] = []
    sequence_rows = _empty_db10_sequence_rows() if emit_sequences else None
    for _, entry in manifest.iterrows():
        source_path = Path(entry["source_path"])
        mat = sio.loadmat(
            source_path,
            squeeze_me=False,
            struct_as_record=False,
            variable_names=[
                "ts",
                "emg",
                "grasp",
                "dynamic",
                "position",
                "object",
                "grasprepetition",
                "objectrepetition",
                "objectpart",
                "gazepoint",
                "gazepoint_invalid",
                "pupildiameterleft",
                "pupildiameterleft_invalid",
                "pupildiameterright",
                "pupildiameterright_invalid",
                "tobiiacc",
                "tobiiacc_invalid",
                "tobiigyr",
                "tobiigyr_invalid",
            ],
        )
        timestamps = np.asarray(mat["ts"]).reshape(-1)
        labels = np.asarray(mat["grasp"]).reshape(-1).astype(int)
        emg = np.asarray(mat["emg"])
        fs = _estimate_fs(timestamps, default=2000.0)
        min_segment_samples = max(1, int(round(min_segment_s * fs)))
        user_window_samples = max(1, int(round(user_window_s * fs)))
        max_prefix_samples = max(1, int(round(max(prefix_grid_s) * fs)))
        dynamic = np.asarray(mat["dynamic"]).reshape(-1).astype(int)
        position = np.asarray(mat["position"]).reshape(-1).astype(int)
        object_id = np.asarray(mat["object"]).reshape(-1).astype(int)
        grasp_rep = np.asarray(mat["grasprepetition"]).reshape(-1).astype(int)
        object_rep = np.asarray(mat["objectrepetition"]).reshape(-1).astype(int)
        object_part = np.asarray(mat["objectpart"]).reshape(-1).astype(int)
        gazepoint = _safe_array(mat, "gazepoint")
        gazepoint_invalid = _safe_mask(mat, "gazepoint_invalid")
        pupil_left = _safe_array(mat, "pupildiameterleft")
        pupil_left_invalid = _safe_mask(mat, "pupildiameterleft_invalid")
        pupil_right = _safe_array(mat, "pupildiameterright")
        pupil_right_invalid = _safe_mask(mat, "pupildiameterright_invalid")
        tobiiacc = _safe_array(mat, "tobiiacc")
        tobiiacc_invalid = _safe_mask(mat, "tobiiacc_invalid")
        tobiigyr = _safe_array(mat, "tobiigyr")
        tobiigyr_invalid = _safe_mask(mat, "tobiigyr_invalid")
        for segment_index, (label_raw, start, end) in enumerate(
            _constant_label_segments(labels, min_segment_samples, include_rest)
        ):
            segment_length = end - start
            segment_id = f"{source_path.stem}_seg{segment_index:04d}"
            for prefix_s in prefix_grid_s:
                prefix_samples = int(round(prefix_s * fs))
                if prefix_samples > segment_length or prefix_samples < user_window_samples:
                    continue
                prefix_end = start + prefix_samples
                user_signal = emg[prefix_end - user_window_samples : prefix_end]
                assist_signal = emg[start:prefix_end]
                gaze_available = not np.all(gazepoint_invalid[start:prefix_end])
                pupil_left_available = not np.all(pupil_left_invalid[start:prefix_end])
                pupil_right_available = not np.all(pupil_right_invalid[start:prefix_end])
                acc_available = not np.all(tobiiacc_invalid[start:prefix_end])
                gyr_available = not np.all(tobiigyr_invalid[start:prefix_end])
                dynamic_mode = _mode(dynamic[start:prefix_end])
                position_mode = _mode(position[start:prefix_end])
                object_mode = _mode(object_id[start:prefix_end])
                object_part_mode = _mode(object_part[start:prefix_end])
                context_vector = _db10_context_features(
                    context_mode=context_mode,
                    prefix_s=float(prefix_s),
                    prefix_grid_s=prefix_grid_s,
                    dynamic_flag=dynamic_mode,
                    position_id=position_mode,
                    object_id=object_mode,
                    object_part=object_part_mode,
                    gaze_available=gaze_available,
                    pupil_left_available=pupil_left_available,
                    pupil_right_available=pupil_right_available,
                    acc_available=acc_available,
                    gyr_available=gyr_available,
                )
                user_vectors.append(
                    np.concatenate(
                        [
                            _extract_td_features(user_signal, fs=fs, apply_filter=True, notch_hz=60.0),
                            _temporal_energy_features(user_signal),
                        ]
                    )
                )
                assist_vectors.append(
                    np.concatenate(
                        [
                            _extract_td_features(assist_signal, fs=fs, apply_filter=True, notch_hz=60.0),
                            _temporal_energy_features(assist_signal),
                            _summary_features(gazepoint[start:prefix_end], gazepoint_invalid[start:prefix_end]),
                            _summary_features(pupil_left[start:prefix_end], pupil_left_invalid[start:prefix_end]),
                            _summary_features(pupil_right[start:prefix_end], pupil_right_invalid[start:prefix_end]),
                            _summary_features(tobiiacc[start:prefix_end], tobiiacc_invalid[start:prefix_end]),
                            _summary_features(tobiigyr[start:prefix_end], tobiigyr_invalid[start:prefix_end]),
                            context_vector,
                        ]
                    )
                )
                if sequence_rows is not None:
                    _append_db10_sequence_rows(
                        sequence_rows=sequence_rows,
                        user_signal=user_signal,
                        assist_signal=assist_signal,
                        user_max_steps=sequence_user_max_steps,
                        assist_max_steps=sequence_assist_max_steps,
                        user_source_max_len=user_window_samples,
                        assist_source_max_len=max_prefix_samples,
                        assist_start=start,
                        assist_end=prefix_end,
                        gazepoint=gazepoint,
                        gazepoint_invalid=gazepoint_invalid,
                        pupil_left=pupil_left,
                        pupil_left_invalid=pupil_left_invalid,
                        pupil_right=pupil_right,
                        pupil_right_invalid=pupil_right_invalid,
                        tobiiacc=tobiiacc,
                        tobiiacc_invalid=tobiiacc_invalid,
                        tobiigyr=tobiigyr,
                        tobiigyr_invalid=tobiigyr_invalid,
                        context_vector=context_vector,
                    )
                rows.append(
                    {
                        "dataset_id": "db10",
                        "subject_id": entry["subject_id"],
                        "group": entry["group"],
                        "session": entry["session"],
                        "day": entry["day"],
                        "task": entry["task"],
                        "episode_id": segment_id,
                        "record_id": source_path.stem,
                        "label_raw": int(label_raw),
                        "prefix_time_s": float(prefix_s),
                        "timestamp_s": float(timestamps[prefix_end - 1]),
                        "source_path": str(source_path),
                        "gaze_available": gaze_available,
                        "pupil_left_available": pupil_left_available,
                        "pupil_right_available": pupil_right_available,
                        "acc_available": acc_available,
                        "gyr_available": gyr_available,
                        "dynamic_flag": dynamic_mode,
                        "position_id": position_mode,
                        "object_id": object_mode,
                        "grasp_repetition": _mode(grasp_rep[start:prefix_end]),
                        "object_repetition": _mode(object_rep[start:prefix_end]),
                        "object_part": object_part_mode,
                        "assist_context_mode": context_mode,
                    }
                )
    return _finalize_prepared_dataset(rows, user_vectors, assist_vectors, sequence_rows=sequence_rows)


def prepare_hyser_dataset(
    root: Path,
    task_type: str = "dynamic",
    signal_type: str = "preprocess",
    prefix_grid_s: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0),
    user_window_s: float = 0.2,
    limit_records: int | None = None,
) -> PreparedDataset:
    if signal_type not in {"preprocess", "raw"}:
        raise ValueError("signal_type must be 'preprocess' or 'raw'")
    manifest = build_hyser_manifest(root, task_type=task_type, signal_type=signal_type)
    manifest = manifest.dropna(subset=["label_raw"]).reset_index(drop=True)
    if limit_records is not None:
        manifest = manifest.head(limit_records).copy()
    rows: list[dict[str, Any]] = []
    user_vectors: list[np.ndarray] = []
    assist_vectors: list[np.ndarray] = []
    for _, entry in manifest.iterrows():
        record = wfdb.rdrecord(str(Path(entry["source_path"])))
        signal = np.asarray(record.p_signal, dtype=float)
        fs = float(record.fs)
        user_window_samples = max(1, int(round(user_window_s * fs)))
        total_samples = signal.shape[0]
        episode_id = f"{Path(entry['source_path']).name}_{entry['sample_index']:03d}"
        apply_filter = signal_type == "raw"
        for prefix_s in prefix_grid_s:
            prefix_samples = int(round(prefix_s * fs))
            if prefix_samples > total_samples or prefix_samples < user_window_samples:
                continue
            user_signal = signal[prefix_samples - user_window_samples : prefix_samples]
            assist_signal = signal[:prefix_samples]
            user_vectors.append(_extract_td_features(user_signal, fs=fs, apply_filter=apply_filter, notch_hz=50.0))
            assist_vectors.append(
                np.concatenate(
                    [
                        _extract_td_features(assist_signal, fs=fs, apply_filter=apply_filter, notch_hz=50.0),
                        _temporal_energy_features(assist_signal),
                    ]
                )
            )
            rows.append(
                {
                    "dataset_id": "hyser",
                    "subject_id": entry["subject_id"],
                    "group": entry["group"],
                    "session": entry["session"],
                    "day": entry["day"],
                    "task": entry["task"],
                    "episode_id": f"{entry['subject_id']}_{entry['session']}_{episode_id}",
                    "record_id": f"{entry['subject_id']}_{entry['session']}_{Path(entry['source_path']).name}",
                    "label_raw": int(entry["label_raw"]),
                    "prefix_time_s": float(prefix_s),
                    "timestamp_s": float(prefix_s),
                    "source_path": entry["source_path"],
                }
            )
    return _finalize_prepared_dataset(rows, user_vectors, assist_vectors)


def prepare_grabmyo_dataset(
    root: Path,
    prefix_grid_s: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0),
    user_window_s: float = 0.2,
    limit_records: int | None = None,
) -> PreparedDataset:
    manifest = build_grabmyo_manifest(root)
    if limit_records is not None:
        manifest = manifest.head(limit_records).copy()
    rows: list[dict[str, Any]] = []
    user_vectors: list[np.ndarray] = []
    assist_vectors: list[np.ndarray] = []
    for _, entry in manifest.iterrows():
        source_path = Path(str(entry["source_path"]))
        record = wfdb.rdrecord(str(source_path))
        signal = np.asarray(record.p_signal, dtype=float)
        fs = float(record.fs)
        user_window_samples = max(1, int(round(user_window_s * fs)))
        total_samples = signal.shape[0]
        record_id = source_path.name
        for prefix_s in prefix_grid_s:
            prefix_samples = int(round(prefix_s * fs))
            if prefix_samples > total_samples or prefix_samples < user_window_samples:
                continue
            user_signal = signal[prefix_samples - user_window_samples : prefix_samples]
            assist_signal = signal[:prefix_samples]
            user_vectors.append(_extract_td_features(user_signal, fs=fs, apply_filter=True, notch_hz=50.0))
            assist_vectors.append(
                np.concatenate(
                    [
                        _extract_td_features(assist_signal, fs=fs, apply_filter=True, notch_hz=50.0),
                        _temporal_energy_features(assist_signal),
                    ]
                )
            )
            rows.append(
                {
                    "dataset_id": "grabmyo",
                    "subject_id": entry["subject_id"],
                    "group": entry["group"],
                    "session": entry["session"],
                    "day": entry["day"],
                    "task": entry["task"],
                    "episode_id": f"{entry['subject_id']}_{entry['session']}_{record_id}_{prefix_s:.2f}",
                    "record_id": record_id,
                    "label_raw": int(entry["label_raw"]),
                    "prefix_time_s": float(prefix_s),
                    "timestamp_s": float(prefix_s),
                    "source_path": str(source_path),
                }
            )
    return _finalize_prepared_dataset(rows, user_vectors, assist_vectors)


def prepare_cemhsey_dataset(
    root: Path,
    prefix_grid_s: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0),
    user_window_s: float = 1.0,
    action_start_s: float = 5.0,
    limit_records: int | None = None,
    exclusion_config: Path | None = None,
    include_failed_trials: bool = False,
) -> PreparedDataset:
    manifest = build_cemhsey_grasp_manifest(
        root,
        exclusion_config=exclusion_config,
        include_failed_trials=include_failed_trials,
    )
    if limit_records is not None:
        manifest = manifest.head(limit_records).copy()
    return prepare_cemhsey_manifest(
        manifest,
        prefix_grid_s=prefix_grid_s,
        user_window_s=user_window_s,
        action_start_s=action_start_s,
    )


def prepare_cemhsey_manifest(
    manifest: pd.DataFrame,
    prefix_grid_s: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0),
    user_window_s: float = 1.0,
    action_start_s: float = 5.0,
) -> PreparedDataset:
    rows: list[dict[str, Any]] = []
    user_vectors: list[np.ndarray] = []
    assist_vectors: list[np.ndarray] = []
    for archive_path_str, archive_rows in manifest.groupby("archive_path", sort=True):
        archive_path = Path(str(archive_path_str))
        entries = {
            str(row["member_path"]): row.to_dict()
            for _, row in archive_rows.reset_index(drop=True).iterrows()
        }
        with zipfile.ZipFile(archive_path) as handle:
            has_deflate64 = any(info.compress_type == 9 for info in handle.infolist())
        if has_deflate64:
            with archive_path.open("rb") as handle:
                for file_name, _, chunks in stream_unzip(handle):
                    member_name = file_name.decode("utf-8")
                    entry = entries.get(member_name)
                    if entry is None:
                        for _ in chunks:
                            pass
                        continue
                    _append_cemhsey_features(
                        entry=entry,
                        member_bytes=b"".join(chunks),
                        prefix_grid_s=prefix_grid_s,
                        user_window_s=user_window_s,
                        action_start_s=action_start_s,
                        rows=rows,
                        user_vectors=user_vectors,
                        assist_vectors=assist_vectors,
                    )
        else:
            with zipfile.ZipFile(archive_path) as handle:
                for member_name, entry in entries.items():
                    _append_cemhsey_features(
                        entry=entry,
                        member_bytes=handle.read(member_name),
                        prefix_grid_s=prefix_grid_s,
                        user_window_s=user_window_s,
                        action_start_s=action_start_s,
                        rows=rows,
                        user_vectors=user_vectors,
                        assist_vectors=assist_vectors,
                    )
    return _finalize_prepared_dataset(rows, user_vectors, assist_vectors)


def merge_prepared_datasets(bundles: list[PreparedDataset]) -> PreparedDataset:
    if not bundles:
        raise ValueError("cannot merge zero prepared datasets")
    expected_user_dim = int(bundles[0].user_features.shape[1])
    expected_assist_dim = int(bundles[0].assist_features.shape[1])
    metadata_frames: list[pd.DataFrame] = []
    user_blocks: list[np.ndarray] = []
    assist_blocks: list[np.ndarray] = []
    sequence_blocks: dict[str, list[np.ndarray]] | None = None
    saw_tabular_only = False
    for bundle in bundles:
        bundle.validate()
        if int(bundle.user_features.shape[1]) != expected_user_dim:
            raise ValueError(
                f"user feature width mismatch: expected {expected_user_dim}, got {bundle.user_features.shape[1]}"
            )
        if int(bundle.assist_features.shape[1]) != expected_assist_dim:
            raise ValueError(
                f"assist feature width mismatch: expected {expected_assist_dim}, got {bundle.assist_features.shape[1]}"
            )
        metadata_frames.append(bundle.metadata.drop(columns=["label"], errors="ignore").copy())
        user_blocks.append(np.asarray(bundle.user_features, dtype=np.float32))
        assist_blocks.append(np.asarray(bundle.assist_features, dtype=np.float32))
        current_sequences = bundle.sequence_payloads or {}
        if current_sequences:
            if saw_tabular_only:
                raise ValueError("cannot merge sequence-enabled and tabular-only prepared datasets")
            if sequence_blocks is None:
                sequence_blocks = {name: [np.asarray(payload)] for name, payload in current_sequences.items()}
            else:
                if set(current_sequences) != set(sequence_blocks):
                    raise ValueError("sequence payload keys must match when merging prepared datasets")
                for name, payload in current_sequences.items():
                    sequence_blocks[name].append(np.asarray(payload))
        else:
            saw_tabular_only = True
            if sequence_blocks is not None:
                raise ValueError("cannot merge sequence-enabled and tabular-only prepared datasets")
    metadata = pd.concat(metadata_frames, ignore_index=True)
    user_features = np.concatenate(user_blocks, axis=0)
    assist_features = np.concatenate(assist_blocks, axis=0)
    sequence_payloads = None
    if sequence_blocks is not None:
        sequence_payloads = {
            name: _merge_sequence_payload_blocks(blocks)
            for name, blocks in sequence_blocks.items()
        }
    return _prepared_dataset_from_arrays(
        metadata,
        user_features,
        assist_features,
        sequence_payloads=sequence_payloads,
    )


def _finalize_prepared_dataset(
    rows: list[dict[str, Any]],
    user_vectors: list[np.ndarray],
    assist_vectors: list[np.ndarray],
    sequence_rows: dict[str, list[np.ndarray]] | None = None,
) -> PreparedDataset:
    if not rows:
        raise ValueError("no rows prepared")
    metadata = pd.DataFrame(rows)
    sequence_payloads = None
    if sequence_rows is not None:
        sequence_payloads = _stack_sequence_payload_rows(sequence_rows)
    return _prepared_dataset_from_arrays(
        metadata,
        _stack_feature_rows(user_vectors),
        _stack_feature_rows(assist_vectors),
        sequence_payloads=sequence_payloads,
    )


def _empty_db10_sequence_rows() -> dict[str, list[np.ndarray]]:
    return {
        "user_emg": [],
        "user_time_mask": [],
        "assist_emg": [],
        "assist_time_mask": [],
        "assist_gaze": [],
        "assist_gaze_mask": [],
        "assist_pupil": [],
        "assist_pupil_mask": [],
        "assist_acc": [],
        "assist_acc_mask": [],
        "assist_gyr": [],
        "assist_gyr_mask": [],
        "assist_context": [],
        "assist_context_mask": [],
    }


def _append_db10_sequence_rows(
    sequence_rows: dict[str, list[np.ndarray]],
    user_signal: np.ndarray,
    assist_signal: np.ndarray,
    user_max_steps: int,
    assist_max_steps: int,
    user_source_max_len: int,
    assist_source_max_len: int,
    assist_start: int,
    assist_end: int,
    gazepoint: np.ndarray,
    gazepoint_invalid: np.ndarray,
    pupil_left: np.ndarray,
    pupil_left_invalid: np.ndarray,
    pupil_right: np.ndarray,
    pupil_right_invalid: np.ndarray,
    tobiiacc: np.ndarray,
    tobiiacc_invalid: np.ndarray,
    tobiigyr: np.ndarray,
    tobiigyr_invalid: np.ndarray,
    context_vector: np.ndarray,
) -> None:
    user_arr = _ensure_feature_width(np.asarray(user_signal, dtype=np.float32), width=DB10_EMG_DIM)
    assist_arr = _ensure_feature_width(np.asarray(assist_signal, dtype=np.float32), width=DB10_EMG_DIM)
    user_arr = _compress_sequence_to_budget(user_arr, max_len=int(user_max_steps), source_max_len=int(user_source_max_len))
    assist_arr = _compress_sequence_to_budget(
        assist_arr,
        max_len=int(assist_max_steps),
        source_max_len=int(assist_source_max_len),
    )
    assist_length = int(assist_arr.shape[0])
    sequence_rows["user_emg"].append(user_arr)
    sequence_rows["user_time_mask"].append(np.ones((user_arr.shape[0], 1), dtype=np.float32))
    sequence_rows["assist_emg"].append(assist_arr)
    sequence_rows["assist_time_mask"].append(np.ones((assist_length, 1), dtype=np.float32))
    sequence_rows["assist_gaze"].append(
        _compress_sequence_to_budget(
            _slice_sequence_signal(gazepoint, assist_start, assist_end, feature_dim=DB10_GAZE_DIM),
            max_len=int(assist_max_steps),
            source_max_len=int(assist_source_max_len),
        )
    )
    sequence_rows["assist_gaze_mask"].append(
        _compress_sequence_to_budget(
            _slice_sequence_mask(gazepoint, gazepoint_invalid, assist_start, assist_end, feature_dim=1),
            max_len=int(assist_max_steps),
            source_max_len=int(assist_source_max_len),
        )
    )
    sequence_rows["assist_pupil"].append(
        _compress_sequence_to_budget(
            np.concatenate(
                [
                    _slice_sequence_signal(pupil_left, assist_start, assist_end, feature_dim=1),
                    _slice_sequence_signal(pupil_right, assist_start, assist_end, feature_dim=1),
                ],
                axis=1,
            ),
            max_len=int(assist_max_steps),
            source_max_len=int(assist_source_max_len),
        )
    )
    sequence_rows["assist_pupil_mask"].append(
        _compress_sequence_to_budget(
            np.concatenate(
                [
                    _slice_sequence_mask(pupil_left, pupil_left_invalid, assist_start, assist_end, feature_dim=1),
                    _slice_sequence_mask(pupil_right, pupil_right_invalid, assist_start, assist_end, feature_dim=1),
                ],
                axis=1,
            ),
            max_len=int(assist_max_steps),
            source_max_len=int(assist_source_max_len),
        )
    )
    sequence_rows["assist_acc"].append(
        _compress_sequence_to_budget(
            _slice_sequence_signal(tobiiacc, assist_start, assist_end, feature_dim=DB10_IMU_DIM),
            max_len=int(assist_max_steps),
            source_max_len=int(assist_source_max_len),
        )
    )
    sequence_rows["assist_acc_mask"].append(
        _compress_sequence_to_budget(
            _slice_sequence_mask(tobiiacc, tobiiacc_invalid, assist_start, assist_end, feature_dim=1),
            max_len=int(assist_max_steps),
            source_max_len=int(assist_source_max_len),
        )
    )
    sequence_rows["assist_gyr"].append(
        _compress_sequence_to_budget(
            _slice_sequence_signal(tobiigyr, assist_start, assist_end, feature_dim=DB10_IMU_DIM),
            max_len=int(assist_max_steps),
            source_max_len=int(assist_source_max_len),
        )
    )
    sequence_rows["assist_gyr_mask"].append(
        _compress_sequence_to_budget(
            _slice_sequence_mask(tobiigyr, tobiigyr_invalid, assist_start, assist_end, feature_dim=1),
            max_len=int(assist_max_steps),
            source_max_len=int(assist_source_max_len),
        )
    )
    sequence_rows["assist_context"].append(np.asarray(context_vector, dtype=np.float32).reshape(1, -1))
    sequence_rows["assist_context_mask"].append(np.ones((1, 1), dtype=np.float32))


def _slice_sequence_signal(
    signal: np.ndarray,
    start: int,
    end: int,
    feature_dim: int | None = None,
) -> np.ndarray:
    length = max(0, int(end - start))
    arr = np.asarray(signal, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    width = int(feature_dim if feature_dim is not None else (arr.shape[1] if arr.ndim == 2 else 0))
    out = np.zeros((length, width), dtype=np.float32)
    if arr.size == 0 or width == 0:
        return out
    src_start = max(0, min(int(start), arr.shape[0]))
    src_end = max(0, min(int(end), arr.shape[0]))
    if src_end <= src_start:
        return out
    copy_width = min(width, arr.shape[1])
    dst_start = src_start - int(start)
    out[dst_start : dst_start + (src_end - src_start), :copy_width] = arr[src_start:src_end, :copy_width]
    return out


def _slice_sequence_mask(
    signal: np.ndarray,
    invalid_mask: np.ndarray | None,
    start: int,
    end: int,
    feature_dim: int,
) -> np.ndarray:
    length = max(0, int(end - start))
    out = np.zeros((length, int(feature_dim)), dtype=np.float32)
    arr = np.asarray(signal)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.size == 0:
        return out
    src_start = max(0, min(int(start), arr.shape[0]))
    src_end = max(0, min(int(end), arr.shape[0]))
    if src_end <= src_start:
        return out
    dst_start = src_start - int(start)
    if invalid_mask is None or np.asarray(invalid_mask).size == 0:
        out[dst_start : dst_start + (src_end - src_start), :] = 1.0
        return out
    invalid = np.asarray(invalid_mask).astype(bool)
    if invalid.ndim == 1:
        invalid = invalid[:, None]
    inv_start = min(src_start, invalid.shape[0])
    inv_end = min(src_end, invalid.shape[0])
    if inv_end <= inv_start:
        out[dst_start : dst_start + (src_end - src_start), :] = 1.0
        return out
    valid = (~invalid[inv_start:inv_end]).astype(np.float32)
    if valid.shape[1] == 1 and feature_dim > 1:
        valid = np.repeat(valid, feature_dim, axis=1)
    copy_width = min(int(feature_dim), valid.shape[1])
    valid_dst_start = dst_start + (inv_start - src_start)
    out[valid_dst_start : valid_dst_start + (inv_end - inv_start), :copy_width] = valid[:, :copy_width]
    if copy_width == 1 and feature_dim > 1:
        out[valid_dst_start : valid_dst_start + (inv_end - inv_start), 1:] = valid[:, :1]
    if inv_start > src_start:
        out[dst_start : dst_start + (inv_start - src_start), :] = 1.0
    if inv_end < src_end:
        tail_start = valid_dst_start + (inv_end - inv_start)
        out[tail_start : dst_start + (src_end - src_start), :] = 1.0
    return out


def _repeat_sequence_context(context_vector: np.ndarray, length: int) -> np.ndarray:
    context = np.asarray(context_vector, dtype=np.float32).reshape(1, -1)
    return np.repeat(context, max(0, int(length)), axis=0)


def _compress_sequence_to_budget(
    sequence: np.ndarray,
    *,
    max_len: int,
    source_max_len: int,
) -> np.ndarray:
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("sequence payload rows must have shape (time, features)")
    if arr.shape[0] == 0:
        return arr
    if max_len <= 0 or source_max_len <= 0:
        raise ValueError("sequence compression lengths must be positive")
    if source_max_len <= max_len:
        return arr
    target_len = max(1, int(round(arr.shape[0] * float(max_len) / float(source_max_len))))
    if target_len >= arr.shape[0]:
        return arr
    edges = np.linspace(0, arr.shape[0], num=target_len + 1, dtype=np.float64)
    out = np.zeros((target_len, arr.shape[1]), dtype=np.float32)
    for idx in range(target_len):
        start = int(np.floor(edges[idx]))
        stop = int(np.floor(edges[idx + 1]))
        start = max(0, min(start, arr.shape[0] - 1))
        stop = max(start + 1, min(stop, arr.shape[0]))
        out[idx] = arr[start:stop].mean(axis=0, dtype=np.float32)
    return out


def _ensure_feature_width(sequence: np.ndarray, *, width: int) -> np.ndarray:
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("sequence payload rows must have shape (time, features)")
    if arr.shape[1] == int(width):
        return arr
    out = np.zeros((arr.shape[0], int(width)), dtype=np.float32)
    copy_width = min(int(width), int(arr.shape[1]))
    out[:, :copy_width] = arr[:, :copy_width]
    return out


def _stack_sequence_payload_rows(sequence_rows: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    stacked: dict[str, np.ndarray] = {}
    for name, rows in sequence_rows.items():
        if rows:
            stacked[name] = _stack_sequence_rows(rows)
    return stacked


def _safe_array(mat: dict[str, Any], key: str) -> np.ndarray:
    if key not in mat:
        return np.empty((0, 0), dtype=float)
    arr = np.asarray(mat[key])
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def _safe_mask(mat: dict[str, Any], key: str) -> np.ndarray:
    if key not in mat:
        return np.zeros((0,), dtype=bool)
    return np.asarray(mat[key]).reshape(-1).astype(bool)


def _estimate_fs(timestamps: np.ndarray, default: float) -> float:
    if timestamps.size < 3:
        return default
    deltas = np.diff(timestamps[: min(4096, timestamps.size)])
    deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
    if deltas.size == 0:
        return default
    return float(1.0 / np.median(deltas))


def _constant_label_segments(
    labels: np.ndarray,
    min_segment_samples: int,
    include_rest: bool,
) -> list[tuple[int, int, int]]:
    values = np.asarray(labels, dtype=int).reshape(-1)
    if values.size == 0:
        return []
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    ends = np.r_[starts[1:], values.size]
    segments: list[tuple[int, int, int]] = []
    for start, end in zip(starts, ends):
        label = int(values[start])
        if end - start < min_segment_samples:
            continue
        if label == 0 and not include_rest:
            continue
        segments.append((label, int(start), int(end)))
    return segments


def _mode(values: np.ndarray) -> int:
    arr = np.asarray(values).reshape(-1)
    unique, counts = np.unique(arr, return_counts=True)
    return int(unique[np.argmax(counts)])


def _append_cemhsey_features(
    entry: dict[str, Any],
    member_bytes: bytes,
    prefix_grid_s: tuple[float, ...],
    user_window_s: float,
    action_start_s: float,
    rows: list[dict[str, Any]],
    user_vectors: list[np.ndarray],
    assist_vectors: list[np.ndarray],
) -> None:
    mat = sio.loadmat(io.BytesIO(member_bytes), variable_names=["data_sEMG"])
    signal = np.asarray(mat["data_sEMG"], dtype=float).T
    fs = 2048.0
    action_start = int(round(action_start_s * fs))
    user_window_samples = max(1, int(round(user_window_s * fs)))
    episode_id = (
        f"{entry['subject_id']}_{entry['day']}_{entry['grasp_block']}_"
        f"force{entry['force_level']}_{entry['trial']}"
    )
    for prefix_s in prefix_grid_s:
        prefix_end = action_start + int(round(prefix_s * fs))
        if prefix_end > signal.shape[0] or prefix_end - action_start < user_window_samples:
            continue
        user_signal = signal[prefix_end - user_window_samples : prefix_end]
        assist_signal = signal[action_start:prefix_end]
        user_vectors.append(_extract_td_features(user_signal, fs=fs, apply_filter=True, notch_hz=50.0))
        assist_vectors.append(
            np.concatenate(
                [
                    _extract_td_features(assist_signal, fs=fs, apply_filter=True, notch_hz=50.0),
                    _temporal_energy_features(assist_signal),
                ]
            )
        )
        rows.append(
            {
                "dataset_id": "cemhsey",
                "subject_id": entry["subject_id"],
                "group": entry["group"],
                "session": entry["session"],
                "day": entry["day"],
                "task": entry["task"],
                "episode_id": episode_id,
                "record_id": episode_id,
                "label_raw": int(entry["label_raw"]),
                "prefix_time_s": float(prefix_s),
                "timestamp_s": float(action_start_s + prefix_s),
                "archive_path": entry["archive_path"],
                "member_path": entry["member_path"],
                "force_level": int(entry["force_level"]),
            }
        )


def _extract_td_features(signal: np.ndarray, fs: float, apply_filter: bool, notch_hz: float) -> np.ndarray:
    arr = np.asarray(signal, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if apply_filter and arr.shape[0] > 64:
        arr = notch_filter(bandpass_emg(arr, fs=fs), fs=fs, notch_hz=notch_hz)
    windows = window_signal(arr, window_size=arr.shape[0], step_size=arr.shape[0])
    if windows.shape[0] == 0:
        return np.zeros(arr.shape[1] * 5, dtype=float)
    return time_domain_features(windows)[0]


def _summary_features(signal: np.ndarray, invalid_mask: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(signal, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.size == 0:
        return np.zeros(3, dtype=float)
    arr = arr.copy()
    if invalid_mask is not None and invalid_mask.size:
        mask = np.asarray(invalid_mask).reshape(-1).astype(bool)
        arr[mask] = np.nan
    finite = np.isfinite(arr)
    counts = finite.sum(axis=0)
    safe = np.where(finite, arr, 0.0)
    mean = np.divide(safe.sum(axis=0), counts, out=np.zeros(arr.shape[1], dtype=float), where=counts > 0)
    centered = np.where(finite, arr - mean, 0.0)
    var = np.divide(
        (centered**2).sum(axis=0),
        counts,
        out=np.zeros(arr.shape[1], dtype=float),
        where=counts > 0,
    )
    std = np.sqrt(var)
    missing = 1.0 - (counts / max(1, arr.shape[0]))
    return np.concatenate([mean, std, missing])


def _db10_context_features(
    context_mode: str,
    prefix_s: float,
    prefix_grid_s: tuple[float, ...],
    dynamic_flag: int,
    position_id: int,
    object_id: int,
    object_part: int,
    gaze_available: bool,
    pupil_left_available: bool,
    pupil_right_available: bool,
    acc_available: bool,
    gyr_available: bool,
) -> np.ndarray:
    if context_mode == "sensor_summary":
        return np.zeros(0, dtype=float)
    features: list[np.ndarray] = [
        _one_hot_choice(dynamic_flag, DB10_BINARY_CHOICES),
        _one_hot_choice(position_id, DB10_BINARY_CHOICES),
        _one_hot_choice(prefix_s, prefix_grid_s),
        np.asarray(
            [
                float(gaze_available),
                float(pupil_left_available),
                float(pupil_right_available),
                float(acc_available),
                float(gyr_available),
            ],
            dtype=float,
        ),
    ]
    if context_mode == "annotated_object_proxy":
        features.extend(
            [
                _one_hot_choice(object_id, DB10_OBJECT_ID_CHOICES),
                _one_hot_choice(object_part, DB10_OBJECT_PART_CHOICES),
            ]
        )
    return np.concatenate(features, axis=0) if features else np.zeros(0, dtype=float)


def _temporal_energy_features(signal: np.ndarray) -> np.ndarray:
    arr = np.asarray(signal, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    thirds = np.array_split(arr, 3, axis=0)
    rms = [np.sqrt(np.mean(chunk**2, axis=0)) if len(chunk) else np.zeros(arr.shape[1]) for chunk in thirds]
    return np.concatenate(rms, axis=0)


def _one_hot_choice(value: int | float, choices: tuple[int | float, ...]) -> np.ndarray:
    arr = np.zeros(len(choices), dtype=float)
    for idx, candidate in enumerate(choices):
        if isinstance(value, float) or isinstance(candidate, float):
            matched = bool(np.isclose(float(value), float(candidate)))
        else:
            matched = value == candidate
        if matched:
            arr[idx] = 1.0
            return arr
    return arr


def _stack_feature_rows(vectors: list[np.ndarray]) -> np.ndarray:
    if not vectors:
        raise ValueError("cannot stack empty feature rows")
    max_dim = max(np.asarray(vector).shape[0] for vector in vectors)
    stacked = np.zeros((len(vectors), max_dim), dtype=np.float32)
    for row_index, vector in enumerate(vectors):
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        stacked[row_index, : arr.shape[0]] = arr
    return stacked


def _stack_sequence_rows(rows: list[np.ndarray]) -> np.ndarray:
    if not rows:
        raise ValueError("cannot stack empty sequence rows")
    normalized: list[np.ndarray] = []
    widths: set[int] = set()
    max_len = 0
    for row in rows:
        arr = np.asarray(row, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise ValueError("sequence rows must have shape (time, features)")
        normalized.append(arr)
        widths.add(int(arr.shape[1]))
        max_len = max(max_len, int(arr.shape[0]))
    if len(widths) != 1:
        raise ValueError(f"sequence feature width mismatch: {sorted(widths)}")
    width = widths.pop()
    stacked = np.zeros((len(normalized), max_len, width), dtype=np.float32)
    for row_index, arr in enumerate(normalized):
        stacked[row_index, max_len - arr.shape[0] :, :] = arr
    return stacked


def _merge_sequence_payload_blocks(blocks: list[np.ndarray]) -> np.ndarray:
    if not blocks:
        raise ValueError("cannot merge empty sequence payload blocks")
    arrays = [np.asarray(block) for block in blocks]
    ndim = arrays[0].ndim
    if any(arr.ndim != ndim for arr in arrays):
        raise ValueError("sequence payload rank mismatch across prepared datasets")
    if ndim < 2:
        return np.concatenate(arrays, axis=0)
    if ndim == 2:
        trailing = arrays[0].shape[1:]
        if any(arr.shape[1:] != trailing for arr in arrays):
            raise ValueError("sequence payload width mismatch across prepared datasets")
        return np.concatenate(arrays, axis=0)
    trailing = arrays[0].shape[2:]
    if any(arr.shape[2:] != trailing for arr in arrays):
        raise ValueError("sequence payload channel mismatch across prepared datasets")
    total_rows = sum(int(arr.shape[0]) for arr in arrays)
    max_len = max(int(arr.shape[1]) for arr in arrays)
    merged = np.zeros((total_rows, max_len, *trailing), dtype=np.result_type(*arrays))
    offset = 0
    for arr in arrays:
        next_offset = offset + arr.shape[0]
        merged[offset:next_offset, max_len - arr.shape[1] :, ...] = arr
        offset = next_offset
    return merged


def _prepared_dataset_from_arrays(
    metadata: pd.DataFrame,
    user_features: np.ndarray,
    assist_features: np.ndarray,
    sequence_payloads: dict[str, np.ndarray] | None = None,
) -> PreparedDataset:
    if user_features.shape[0] != len(metadata) or assist_features.shape[0] != len(metadata):
        raise ValueError("metadata and feature arrays must have the same row count")
    label_vocab = sorted(int(label) for label in metadata["label_raw"].dropna().unique())
    label_map = {label: idx for idx, label in enumerate(label_vocab)}
    prepared_metadata = metadata.copy()
    prepared_metadata["label"] = prepared_metadata["label_raw"].map(label_map).astype(int)
    labels = prepared_metadata["label"].to_numpy(dtype=int)
    return PreparedDataset(
        metadata=prepared_metadata,
        user_features=np.asarray(user_features, dtype=np.float32),
        assist_features=np.asarray(assist_features, dtype=np.float32),
        labels=labels,
        label_vocab=label_vocab,
        sequence_payloads=(
            None
            if not sequence_payloads
            else {name: np.asarray(payload, dtype=np.float32) for name, payload in sequence_payloads.items()}
        ),
    )

