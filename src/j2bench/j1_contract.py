from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


J1_REQUIRED_FILES = (
    "metadata.csv",
    "user_features.npy",
    "assist_features.npy",
    "labels.npy",
    "label_vocab.npy",
    "_SUCCESS",
)

J1_REQUIRED_METADATA_COLUMNS = (
    "dataset_id",
    "source_dataset_id",
    "subject_id",
    "group",
    "session",
    "day",
    "task",
    "record_id",
    "episode_id",
    "label",
    "label_raw",
    "prefix_time_s",
    "timestamp_s",
)

J1_ROW_KEY_COLUMNS = (
    "source_dataset_id",
    "subject_id",
    "session",
    "day",
    "task",
    "record_id",
    "episode_id",
    "prefix_time_s",
    "timestamp_s",
)

J1_EPISODE_KEY_COLUMNS = (
    "source_dataset_id",
    "subject_id",
    "session",
    "day",
    "task",
    "record_id",
    "episode_id",
)

_INVALID_STRING_TOKENS = {"", "nan", "none", "<na>"}


def validate_j1_prepared_bundle(root: str | Path) -> dict[str, Any]:
    """Validate a strict J1 prepared bundle and return a concise manifest payload."""

    bundle_root = _resolve_bundle_root(Path(root))
    metadata = pd.read_csv(bundle_root / "metadata.csv")
    user_features = np.load(bundle_root / "user_features.npy", allow_pickle=False)
    assist_features = np.load(bundle_root / "assist_features.npy", allow_pickle=False)
    labels = np.load(bundle_root / "labels.npy", allow_pickle=False)
    label_vocab = np.load(bundle_root / "label_vocab.npy", allow_pickle=False)

    manifest = validate_j1_metadata(metadata, labels=labels, label_vocab=label_vocab)
    n_rows = int(len(metadata))

    _validate_feature_matrix(user_features, "user_features", n_rows)
    _validate_feature_matrix(assist_features, "assist_features", n_rows)
    sequence_names = _validate_sequence_payloads(bundle_root, n_rows)

    manifest["root"] = str(bundle_root)
    manifest["required_files"] = list(J1_REQUIRED_FILES)
    manifest["feature_shapes"] = {
        "user_features": [int(dim) for dim in user_features.shape],
        "assist_features": [int(dim) for dim in assist_features.shape],
    }
    manifest["sequence_payloads"] = {
        "count": int(len(sequence_names)),
        "names": sequence_names,
    }
    return manifest


def validate_j1_metadata(
    metadata: pd.DataFrame,
    *,
    labels: np.ndarray,
    label_vocab: np.ndarray | list[int],
) -> dict[str, Any]:
    """Validate strict J1 metadata without repairing or backfilling any fields."""

    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("metadata must be a pandas DataFrame")
    if metadata.empty:
        raise ValueError("J1 metadata must contain at least one row")
    if not metadata.columns.is_unique:
        raise ValueError("J1 metadata column names must be unique")

    expected_index = pd.RangeIndex(start=0, stop=len(metadata), step=1)
    if not metadata.index.equals(expected_index):
        raise ValueError(
            "J1 metadata index must be a default RangeIndex to preserve row alignment"
        )

    missing_columns = [
        column for column in J1_REQUIRED_METADATA_COLUMNS if column not in metadata.columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"J1 metadata missing required columns: {missing}")

    dataset_id = _validate_string_column(metadata, "dataset_id")
    source_dataset_id = _validate_string_column(metadata, "source_dataset_id")
    subject_id = _validate_string_column(metadata, "subject_id")
    group = _validate_string_column(metadata, "group")
    _validate_string_column(metadata, "session")
    _validate_string_column(metadata, "day")
    _validate_string_column(metadata, "task")
    _validate_string_column(metadata, "record_id")
    _validate_string_column(metadata, "episode_id")
    label = _validate_integer_series(metadata, "label", nonnegative=True)
    label_raw = _validate_integer_series(metadata, "label_raw")
    prefix_time_s = _validate_float_series(metadata, "prefix_time_s", nonnegative=True)
    timestamp_s = _validate_float_series(metadata, "timestamp_s", nonnegative=True)

    invalid_dataset = sorted({value for value in dataset_id.unique().tolist() if value != "j1"})
    if invalid_dataset:
        raise ValueError(f"J1 metadata.dataset_id must be 'j1'; observed {invalid_dataset}")

    labels_arr = _validate_integer_array(
        labels,
        "labels",
        expected_rows=len(metadata),
        nonnegative=True,
    )
    label_vocab_arr = _validate_integer_array(label_vocab, "label_vocab")
    if label_vocab_arr.size == 0:
        raise ValueError("label_vocab must contain at least one label")
    if len(set(label_vocab_arr.tolist())) != int(label_vocab_arr.size):
        raise ValueError("label_vocab must contain unique labels")
    if label_vocab_arr.size > 1 and np.any(np.diff(label_vocab_arr) <= 0):
        raise ValueError("label_vocab must be strictly increasing")

    max_index = int(label_vocab_arr.size - 1)
    if int(label.max()) > max_index:
        raise ValueError(
            "metadata.label contains indices outside label_vocab: "
            f"max={int(label.max())}, size={int(label_vocab_arr.size)}"
        )
    if int(labels_arr.max()) > max_index:
        raise ValueError(
            "labels contain indices outside label_vocab: "
            f"max={int(labels_arr.max())}, size={int(label_vocab_arr.size)}"
        )

    label_arr = label.to_numpy(dtype=np.int64)
    label_raw_arr = label_raw.to_numpy(dtype=np.int64)
    expected_label_raw = label_vocab_arr[label_arr]
    if not np.array_equal(label_arr, labels_arr):
        raise ValueError("metadata.label does not match labels")
    if not np.array_equal(label_raw_arr, expected_label_raw):
        raise ValueError("metadata.label_raw does not match label_vocab[metadata.label]")

    duplicates = metadata.duplicated(subset=list(J1_ROW_KEY_COLUMNS), keep=False)
    if bool(duplicates.any()):
        sample = (
            metadata.loc[duplicates, list(J1_ROW_KEY_COLUMNS)]
            .head(1)
            .to_dict(orient="records")[0]
        )
        raise ValueError(f"duplicate J1 records detected: {sample}")

    episode_counts = metadata.groupby(list(J1_EPISODE_KEY_COLUMNS), dropna=False, sort=False).size()
    for episode_key, episode_df in metadata.groupby(
        list(J1_EPISODE_KEY_COLUMNS),
        dropna=False,
        sort=False,
    ):
        if int(episode_df["label"].nunique(dropna=False)) != 1:
            raise ValueError(
                "metadata.label must be constant within episode "
                f"{_format_episode_key(episode_key)}"
            )
        if int(episode_df["label_raw"].nunique(dropna=False)) != 1:
            raise ValueError(
                "metadata.label_raw must be constant within episode "
                f"{_format_episode_key(episode_key)}"
            )
        prefix = episode_df["prefix_time_s"].to_numpy(dtype=float)
        timestamp = episode_df["timestamp_s"].to_numpy(dtype=float)
        if prefix.size > 1 and np.any(np.diff(prefix) <= 0):
            raise ValueError(
                "metadata.prefix_time_s must be strictly increasing within episode "
                f"{_format_episode_key(episode_key)}"
            )
        if timestamp.size > 1 and np.any(np.diff(timestamp) <= 0):
            raise ValueError(
                "metadata.timestamp_s must be strictly increasing within episode "
                f"{_format_episode_key(episode_key)}"
            )

    observed_label_raw = sorted(int(value) for value in np.unique(label_raw_arr).tolist())
    observed_label_set = set(observed_label_raw)
    return {
        "status": "PASS",
        "dataset_id": "j1",
        "n_records": int(len(metadata)),
        "n_episodes": int(episode_counts.shape[0]),
        "n_subjects": int(subject_id.nunique(dropna=False)),
        "source_dataset_counts": _series_counts(source_dataset_id),
        "group_counts": _series_counts(group),
        "label_counts": _series_counts(label_raw.astype(str)),
        "label_coverage": {
            "n_vocab": int(label_vocab_arr.size),
            "n_observed": int(len(observed_label_raw)),
            "unused_label_raw": [
                int(value)
                for value in label_vocab_arr.tolist()
                if int(value) not in observed_label_set
            ],
        },
        "prefixes_per_episode": {
            "min": int(episode_counts.min()),
            "max": int(episode_counts.max()),
            "mean": float(episode_counts.mean()),
        },
        "prefix_time_s": {
            "min": float(prefix_time_s.min()),
            "max": float(prefix_time_s.max()),
        },
        "timestamp_s": {
            "min": float(timestamp_s.min()),
            "max": float(timestamp_s.max()),
        },
    }


def _resolve_bundle_root(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"J1 prepared bundle root does not exist: {root}")
    if root.is_file():
        raise ValueError(f"J1 prepared bundle root must be a directory: {root}")
    if _has_required_files(root):
        return root

    candidates = [
        path
        for path in sorted(root.iterdir())
        if path.is_dir() and _has_required_files(path)
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(
            f"multiple J1 prepared bundles found under {root}; choose one explicitly: {names}"
        )

    missing = [name for name in J1_REQUIRED_FILES if not (root / name).is_file()]
    missing_text = ", ".join(missing)
    raise FileNotFoundError(
        f"J1 prepared bundle missing required files under {root}: {missing_text}"
    )


def _has_required_files(root: Path) -> bool:
    return all((root / name).is_file() for name in J1_REQUIRED_FILES)


def _validate_feature_matrix(values: np.ndarray, name: str, expected_rows: int) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if arr.shape[0] != int(expected_rows):
        raise ValueError(f"{name} row count does not match metadata")
    if arr.shape[1] <= 0:
        raise ValueError(f"{name} must have at least one feature column")
    if not _is_real_numeric_array(arr):
        raise ValueError(f"{name} must have a real numeric dtype")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _validate_sequence_payloads(root: Path, expected_rows: int) -> list[str]:
    sequence_dir = root / "sequence_payloads"
    if not sequence_dir.exists():
        return []
    if not sequence_dir.is_dir():
        raise ValueError(
            f"sequence_payloads must be a directory when present: {sequence_dir}"
        )

    manifest_path = sequence_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"sequence_payloads manifest missing: {manifest_path}")

    try:
        declared = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"sequence_payloads manifest is not valid JSON: {manifest_path}") from exc
    if not isinstance(declared, dict):
        raise ValueError("sequence_payloads manifest must be a JSON object")

    payload_files = sorted(sequence_dir.glob("*.npy"))
    if not payload_files:
        raise ValueError("sequence_payloads directory must contain at least one .npy payload")

    names: list[str] = []
    for path in payload_files:
        name = path.stem
        if name not in declared:
            raise ValueError(f"sequence payload {name!r} missing from manifest")
        arr = np.load(path, allow_pickle=False)
        _validate_sequence_payload(arr, name, expected_rows)
        entry = declared[name]
        if not isinstance(entry, dict):
            raise ValueError(
                f"sequence payload manifest entry for {name!r} must be a JSON object"
            )
        expected_shape = [int(dim) for dim in arr.shape]
        if entry.get("shape") != expected_shape:
            raise ValueError(f"sequence payload {name!r} shape does not match manifest")
        if entry.get("dtype") != str(arr.dtype):
            raise ValueError(f"sequence payload {name!r} dtype does not match manifest")
        names.append(name)

    extra_names = sorted(name for name in declared if name not in names)
    if extra_names:
        joined = ", ".join(extra_names)
        raise ValueError(f"sequence payload manifest declares missing payloads: {joined}")
    return names


def _validate_sequence_payload(values: np.ndarray, name: str, expected_rows: int) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim < 1:
        raise ValueError(f"sequence payload {name!r} must have at least one dimension")
    if arr.shape[0] != int(expected_rows):
        raise ValueError(f"sequence payload {name!r} row count does not match metadata")
    if not _is_real_numeric_array(arr, allow_bool=True):
        raise ValueError(
            f"sequence payload {name!r} must have a real numeric or boolean dtype"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"sequence payload {name!r} must contain only finite values")
    return arr


def _validate_string_column(metadata: pd.DataFrame, column: str) -> pd.Series:
    normalized = metadata[column].astype("string")
    stripped = normalized.str.strip()
    invalid = stripped.isna() | stripped.fillna("").str.lower().isin(_INVALID_STRING_TOKENS)
    if bool(invalid.any()):
        bad_rows = metadata.index[invalid].tolist()[:5]
        raise ValueError(f"metadata.{column} must be present and non-empty; bad rows: {bad_rows}")
    return stripped.astype(str)


def _validate_integer_series(
    metadata: pd.DataFrame,
    column: str,
    *,
    nonnegative: bool = False,
) -> pd.Series:
    numeric = pd.to_numeric(metadata[column], errors="coerce")
    if bool(numeric.isna().any()):
        bad_rows = metadata.index[numeric.isna()].tolist()[:5]
        raise ValueError(f"metadata.{column} must contain integer values; bad rows: {bad_rows}")
    values = numeric.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"metadata.{column} must contain finite values")
    if np.any(values != np.floor(values)):
        raise ValueError(f"metadata.{column} must contain integer values")
    ints = pd.Series(values.astype(np.int64), index=metadata.index, dtype=np.int64)
    if nonnegative and bool((ints < 0).any()):
        raise ValueError(f"metadata.{column} must be nonnegative")
    return ints


def _validate_float_series(
    metadata: pd.DataFrame,
    column: str,
    *,
    nonnegative: bool = False,
) -> pd.Series:
    numeric = pd.to_numeric(metadata[column], errors="coerce")
    if bool(numeric.isna().any()):
        bad_rows = metadata.index[numeric.isna()].tolist()[:5]
        raise ValueError(
            f"metadata.{column} must contain finite numeric values; bad rows: {bad_rows}"
        )
    values = numeric.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"metadata.{column} must contain finite numeric values")
    floats = pd.Series(values.astype(float), index=metadata.index, dtype=float)
    if nonnegative and bool((floats < 0.0).any()):
        raise ValueError(f"metadata.{column} must be nonnegative")
    return floats


def _validate_integer_array(
    values: np.ndarray | list[int],
    name: str,
    *,
    expected_rows: int | None = None,
    nonnegative: bool = False,
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if expected_rows is not None and arr.shape[0] != int(expected_rows):
        raise ValueError(f"{name} row count does not match metadata")
    if np.issubdtype(arr.dtype, np.bool_) or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name} must have an integer dtype")
    if nonnegative and np.any(arr < 0):
        raise ValueError(f"{name} must be nonnegative")
    return arr.astype(np.int64, copy=False)


def _is_real_numeric_array(values: np.ndarray, *, allow_bool: bool = False) -> bool:
    if np.issubdtype(values.dtype, np.bool_):
        return bool(allow_bool)
    return bool(np.issubdtype(values.dtype, np.number) and np.isrealobj(values))


def _series_counts(values: pd.Series) -> dict[str, int]:
    counts = values.value_counts(dropna=False).sort_index()
    return {str(key): int(count) for key, count in counts.items()}


def _format_episode_key(episode_key: Any) -> str:
    if not isinstance(episode_key, tuple):
        episode_key = (episode_key,)
    return ", ".join(
        f"{column}={value}"
        for column, value in zip(J1_EPISODE_KEY_COLUMNS, episode_key, strict=False)
    )

