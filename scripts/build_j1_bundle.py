from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from j2bench.io import load_yaml  # noqa: E402
from j2bench.j1_contract import validate_j1_prepared_bundle  # noqa: E402
from j2bench.realdata import (  # noqa: E402
    PreparedDataset,
    load_prepared_dataset,
    save_prepared_dataset,
    validate_j1_metadata_frame,
)


FEATURE_ARRAY_KEYS = ("user_features", "assist_features")
SPEC_TOP_LEVEL_KEYS = {"version", "bundle", "sources", "composition"}
BUNDLE_KEYS = {"dataset_id"}
SOURCE_KEYS = {"name", "prepared_root", "feature_schema", "label_map", "projection"}
FEATURE_SCHEMA_KEYS = set(FEATURE_ARRAY_KEYS)
COMPOSITION_KEYS = {"strategy", "output_feature_schema"}
PROJECTION_KEYS = set(FEATURE_ARRAY_KEYS)
ARRAY_PROJECTION_KEYS = {"from_schema", "to_schema", "columns"}
INVALID_TEXT_VALUES = {"", "nan", "none", "<na>"}


@dataclass(frozen=True)
class ArrayProjection:
    from_schema: str
    to_schema: str
    columns: tuple[int, ...]


@dataclass(frozen=True)
class SourceSpec:
    name: str
    prepared_root: Path
    feature_schema: dict[str, str]
    label_map: dict[int, int]
    projection: dict[str, ArrayProjection]


@dataclass(frozen=True)
class BuildSpec:
    spec_path: Path
    dataset_id: str
    sources: tuple[SourceSpec, ...]
    output_feature_schema: dict[str, str] | None
    composition_strategy: str | None


def build_j1_bundle(spec_path: Path) -> PreparedDataset:
    spec = load_build_spec(spec_path)
    normalized_sources = [normalize_source_bundle(source, spec.output_feature_schema) for source in spec.sources]
    if len(normalized_sources) == 1:
        return normalized_sources[0]
    return merge_source_bundles(normalized_sources)


def load_build_spec(spec_path: Path) -> BuildSpec:
    spec_path = Path(spec_path)
    raw = load_yaml(spec_path)
    if not isinstance(raw, dict):
        raise ValueError(f"J1 spec must be a mapping: {spec_path}")
    ensure_allowed_keys(raw, SPEC_TOP_LEVEL_KEYS, context="spec")

    version = raw.get("version")
    if version != 1:
        raise ValueError(f"J1 spec version must be 1, got {version!r}")

    bundle = require_mapping(raw.get("bundle"), context="spec.bundle")
    ensure_allowed_keys(bundle, BUNDLE_KEYS, context="spec.bundle")
    dataset_id = str(bundle.get("dataset_id", "")).strip()
    if dataset_id != "j1":
        raise ValueError("spec.bundle.dataset_id must be 'j1'")

    raw_sources = raw.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("spec.sources must contain at least one source")

    composition = raw.get("composition")
    if composition is None:
        output_feature_schema = None
        composition_strategy = None
    else:
        composition_mapping = require_mapping(composition, context="spec.composition")
        ensure_allowed_keys(composition_mapping, COMPOSITION_KEYS, context="spec.composition")
        composition_strategy = str(composition_mapping.get("strategy", "")).strip()
        if composition_strategy != "concatenate":
            raise ValueError("spec.composition.strategy must be 'concatenate'")
        output_feature_schema = parse_feature_schema(
            composition_mapping.get("output_feature_schema"),
            context="spec.composition.output_feature_schema",
        )

    if len(raw_sources) > 1 and composition is None:
        raise ValueError("multi-source J1 composition requires an explicit spec.composition block")

    seen_names: set[str] = set()
    seen_roots: set[str] = set()
    sources: list[SourceSpec] = []
    for index, raw_source in enumerate(raw_sources):
        context = f"spec.sources[{index}]"
        source_mapping = require_mapping(raw_source, context=context)
        ensure_allowed_keys(source_mapping, SOURCE_KEYS, context=context)

        name = require_non_empty_text(source_mapping.get("name"), context=f"{context}.name")
        casefold_name = name.casefold()
        if casefold_name in seen_names:
            raise ValueError(f"duplicate source name in spec: {name}")
        seen_names.add(casefold_name)

        prepared_root_text = require_non_empty_text(
            source_mapping.get("prepared_root"),
            context=f"{context}.prepared_root",
        )
        prepared_root = resolve_spec_path(spec_path.parent, prepared_root_text)
        root_key = str(prepared_root).casefold()
        if root_key in seen_roots:
            raise ValueError(f"duplicate prepared_root in spec: {prepared_root}")
        seen_roots.add(root_key)

        feature_schema = parse_feature_schema(source_mapping.get("feature_schema"), context=f"{context}.feature_schema")
        label_map = parse_label_map(source_mapping.get("label_map"), context=f"{context}.label_map")
        projection = parse_projection(source_mapping.get("projection"), context=f"{context}.projection")
        sources.append(
            SourceSpec(
                name=name,
                prepared_root=prepared_root,
                feature_schema=feature_schema,
                label_map=label_map,
                projection=projection,
            )
        )

    return BuildSpec(
        spec_path=spec_path,
        dataset_id=dataset_id,
        sources=tuple(sources),
        output_feature_schema=output_feature_schema,
        composition_strategy=composition_strategy,
    )


def normalize_source_bundle(source: SourceSpec, output_feature_schema: dict[str, str] | None) -> PreparedDataset:
    bundle = load_prepared_dataset(source.prepared_root)
    if len(bundle.metadata) == 0:
        raise ValueError(f"prepared root '{source.name}' is empty: {source.prepared_root}")

    metadata = bundle.metadata.drop(columns=["label"], errors="ignore").copy()
    raw_labels = resolve_raw_labels(bundle)
    mapped_labels = apply_label_map(raw_labels, source.label_map, source_name=source.name)
    metadata["source_dataset_id"] = resolve_source_dataset_id(metadata, source_name=source.name)
    if "source_label_raw" not in metadata.columns:
        metadata["source_label_raw"] = raw_labels
    metadata["label_raw"] = mapped_labels
    metadata["dataset_id"] = "j1"
    metadata["source_prepared_root"] = str(source.prepared_root)
    metadata = normalize_j1_metadata(metadata)

    target_feature_schema = resolve_target_feature_schema(source, output_feature_schema)
    user_features = project_feature_array(
        bundle.user_features,
        array_name="user_features",
        source_schema=source.feature_schema["user_features"],
        target_schema=target_feature_schema["user_features"],
        projection=source.projection.get("user_features"),
        source_name=source.name,
    )
    assist_features = project_feature_array(
        bundle.assist_features,
        array_name="assist_features",
        source_schema=source.feature_schema["assist_features"],
        target_schema=target_feature_schema["assist_features"],
        projection=source.projection.get("assist_features"),
        source_name=source.name,
    )
    sequence_payloads = None
    if bundle.sequence_payloads:
        sequence_payloads = {
            name: sanitize_sequence_payload(payload)
            for name, payload in bundle.sequence_payloads.items()
        }
    return prepared_dataset_from_arrays(
        metadata,
        user_features=user_features,
        assist_features=assist_features,
        sequence_payloads=sequence_payloads,
    )


def merge_source_bundles(bundles: list[PreparedDataset]) -> PreparedDataset:
    if not bundles:
        raise ValueError("cannot merge zero normalized J1 source bundles")
    expected_user_dim = int(np.asarray(bundles[0].user_features).shape[1])
    expected_assist_dim = int(np.asarray(bundles[0].assist_features).shape[1])
    metadata_frames: list[pd.DataFrame] = []
    user_blocks: list[np.ndarray] = []
    assist_blocks: list[np.ndarray] = []
    sequence_blocks: dict[str, list[np.ndarray]] | None = None
    saw_tabular_only = False

    for bundle in bundles:
        bundle.validate()
        user_arr = np.asarray(bundle.user_features, dtype=np.float32)
        assist_arr = np.asarray(bundle.assist_features, dtype=np.float32)
        if int(user_arr.shape[1]) != expected_user_dim:
            raise ValueError(
                f"user feature width mismatch across normalized J1 sources: "
                f"expected {expected_user_dim}, got {user_arr.shape[1]}"
            )
        if int(assist_arr.shape[1]) != expected_assist_dim:
            raise ValueError(
                f"assist feature width mismatch across normalized J1 sources: "
                f"expected {expected_assist_dim}, got {assist_arr.shape[1]}"
            )
        metadata_frames.append(bundle.metadata.drop(columns=["label"], errors="ignore").copy())
        user_blocks.append(user_arr)
        assist_blocks.append(assist_arr)

        current_sequences = bundle.sequence_payloads or {}
        if current_sequences:
            if saw_tabular_only:
                raise ValueError("cannot merge sequence-enabled and tabular-only prepared roots into one J1 bundle")
            if sequence_blocks is None:
                sequence_blocks = {
                    name: [np.asarray(payload, dtype=np.float32)]
                    for name, payload in current_sequences.items()
                }
            else:
                if set(current_sequences) != set(sequence_blocks):
                    raise ValueError("sequence payload keys must match across merged J1 sources")
                for name, payload in current_sequences.items():
                    sequence_blocks[name].append(np.asarray(payload, dtype=np.float32))
        else:
            saw_tabular_only = True
            if sequence_blocks is not None:
                raise ValueError("cannot merge sequence-enabled and tabular-only prepared roots into one J1 bundle")

    metadata = pd.concat(metadata_frames, ignore_index=True)
    user_features = np.concatenate(user_blocks, axis=0)
    assist_features = np.concatenate(assist_blocks, axis=0)
    sequence_payloads = None
    if sequence_blocks is not None:
        sequence_payloads = {
            name: merge_sequence_payload_blocks(blocks)
            for name, blocks in sequence_blocks.items()
        }
    return prepared_dataset_from_arrays(
        metadata,
        user_features=user_features,
        assist_features=assist_features,
        sequence_payloads=sequence_payloads,
    )


def prepared_dataset_from_arrays(
    metadata: pd.DataFrame,
    *,
    user_features: np.ndarray,
    assist_features: np.ndarray,
    sequence_payloads: dict[str, np.ndarray] | None,
) -> PreparedDataset:
    if len(metadata) == 0:
        raise ValueError("J1 bundle would be empty")
    user_arr = np.asarray(user_features, dtype=np.float32)
    assist_arr = np.asarray(assist_features, dtype=np.float32)
    if user_arr.ndim != 2:
        raise ValueError("user_features must be a 2D array")
    if assist_arr.ndim != 2:
        raise ValueError("assist_features must be a 2D array")
    if user_arr.shape[0] != len(metadata) or assist_arr.shape[0] != len(metadata):
        raise ValueError("metadata and feature arrays must have the same row count")

    if "label_raw" not in metadata.columns:
        raise ValueError("normalized J1 metadata must contain label_raw")
    label_raw = coerce_integer_series(metadata["label_raw"], context="normalized J1 label_raw")
    label_vocab = sorted(int(label) for label in pd.unique(label_raw))
    label_index = {label: idx for idx, label in enumerate(label_vocab)}

    prepared_metadata = metadata.copy()
    prepared_metadata["label_raw"] = label_raw
    prepared_metadata["label"] = prepared_metadata["label_raw"].map(label_index).astype(int)
    labels = prepared_metadata["label"].to_numpy(dtype=int)
    bundle = PreparedDataset(
        metadata=prepared_metadata,
        user_features=user_arr,
        assist_features=assist_arr,
        labels=labels,
        label_vocab=label_vocab,
        sequence_payloads=(
            None
            if not sequence_payloads
            else {name: np.asarray(payload, dtype=np.float32) for name, payload in sequence_payloads.items()}
        ),
    )
    bundle.validate()
    return bundle


def resolve_target_feature_schema(
    source: SourceSpec,
    output_feature_schema: dict[str, str] | None,
) -> dict[str, str]:
    if output_feature_schema is not None:
        return dict(output_feature_schema)
    resolved = dict(source.feature_schema)
    for array_name, projection in source.projection.items():
        resolved[array_name] = projection.to_schema
    return resolved


def project_feature_array(
    array: np.ndarray,
    *,
    array_name: str,
    source_schema: str,
    target_schema: str,
    projection: ArrayProjection | None,
    source_name: str,
) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{array_name} for source '{source_name}' must be a 2D array")
    if projection is None:
        if source_schema != target_schema:
            raise ValueError(
                f"{array_name} schema mismatch for source '{source_name}': "
                f"{source_schema!r} vs target {target_schema!r}; declare an explicit projection"
            )
        return arr

    if projection.from_schema != source_schema:
        raise ValueError(
            f"{array_name} projection for source '{source_name}' declares from_schema "
            f"{projection.from_schema!r}, expected {source_schema!r}"
        )
    if projection.to_schema != target_schema:
        raise ValueError(
            f"{array_name} projection for source '{source_name}' declares to_schema "
            f"{projection.to_schema!r}, expected {target_schema!r}"
        )
    if not projection.columns:
        raise ValueError(f"{array_name} projection for source '{source_name}' must select at least one column")
    if len(set(projection.columns)) != len(projection.columns):
        raise ValueError(f"{array_name} projection for source '{source_name}' contains duplicate columns")
    if min(projection.columns) < 0 or max(projection.columns) >= arr.shape[1]:
        raise ValueError(
            f"{array_name} projection for source '{source_name}' selects columns outside width {arr.shape[1]}"
        )
    return arr[:, list(projection.columns)]


def merge_sequence_payload_blocks(blocks: list[np.ndarray]) -> np.ndarray:
    if not blocks:
        raise ValueError("cannot merge empty sequence payload blocks")
    arrays = [np.asarray(block) for block in blocks]
    ndim = arrays[0].ndim
    if any(arr.ndim != ndim for arr in arrays):
        raise ValueError("sequence payload rank mismatch across merged J1 sources")
    if ndim < 2:
        return np.concatenate(arrays, axis=0)
    if ndim == 2:
        trailing = arrays[0].shape[1:]
        if any(arr.shape[1:] != trailing for arr in arrays):
            raise ValueError("sequence payload width mismatch across merged J1 sources")
        return np.concatenate(arrays, axis=0)

    trailing = arrays[0].shape[2:]
    if any(arr.shape[2:] != trailing for arr in arrays):
        raise ValueError("sequence payload channel mismatch across merged J1 sources")
    total_rows = sum(int(arr.shape[0]) for arr in arrays)
    max_len = max(int(arr.shape[1]) for arr in arrays)
    merged = np.zeros((total_rows, max_len, *trailing), dtype=np.result_type(*arrays))
    offset = 0
    for arr in arrays:
        next_offset = offset + arr.shape[0]
        merged[offset:next_offset, max_len - arr.shape[1] :, ...] = arr
        offset = next_offset
    return merged


def sanitize_sequence_payload(payload: np.ndarray) -> np.ndarray:
    arr = np.asarray(payload)
    if np.issubdtype(arr.dtype, np.bool_):
        return arr.astype(bool, copy=False)
    normalized = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(normalized).all():
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return normalized.astype(np.float32, copy=False)


def resolve_source_dataset_id(metadata: pd.DataFrame, *, source_name: str) -> pd.Series:
    if "source_dataset_id" in metadata.columns:
        series = metadata["source_dataset_id"].astype("object")
        if "dataset_id" in metadata.columns:
            series = series.where(series.notna(), metadata["dataset_id"].astype("object"))
    elif "dataset_id" in metadata.columns:
        series = metadata["dataset_id"].astype("object")
    else:
        raise ValueError(
            f"source '{source_name}' metadata must contain dataset_id or source_dataset_id for provenance"
        )
    normalized = series.astype(str)
    invalid = normalized.str.strip().str.casefold().isin(INVALID_TEXT_VALUES)
    if invalid.any():
        raise ValueError(f"source '{source_name}' has blank source_dataset_id values")
    return normalized


def resolve_raw_labels(bundle: PreparedDataset) -> pd.Series:
    if "label_raw" in bundle.metadata.columns:
        return coerce_integer_series(bundle.metadata["label_raw"], context="source label_raw")

    label_indices = np.asarray(bundle.labels, dtype=int)
    if label_indices.shape[0] != len(bundle.metadata):
        raise ValueError("source labels row count does not match metadata")
    label_vocab = np.asarray(bundle.label_vocab, dtype=int)
    if np.any(label_indices < 0) or np.any(label_indices >= label_vocab.shape[0]):
        raise ValueError("source label indices are out of bounds for label_vocab")
    raw = label_vocab[label_indices]
    return pd.Series(raw, index=bundle.metadata.index, dtype="int64")


def apply_label_map(raw_labels: pd.Series, label_map: dict[int, int], *, source_name: str) -> pd.Series:
    observed = {int(label) for label in pd.unique(raw_labels)}
    declared = set(label_map)
    if observed != declared:
        problems: list[str] = []
        missing = sorted(observed - declared)
        extra = sorted(declared - observed)
        if missing:
            problems.append(f"missing mappings for {missing}")
        if extra:
            problems.append(f"unused mappings for {extra}")
        joined = "; ".join(problems)
        raise ValueError(f"source '{source_name}' label_map must cover exactly the source labels ({joined})")
    mapped = raw_labels.map(label_map)
    if mapped.isna().any():
        raise ValueError(f"source '{source_name}' label_map left unmapped labels")
    return coerce_integer_series(mapped, context=f"mapped labels for source '{source_name}'")


def normalize_j1_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    required_columns = (
        "dataset_id",
        "source_dataset_id",
        "subject_id",
        "group",
        "session",
        "task",
        "record_id",
        "episode_id",
        "label_raw",
        "prefix_time_s",
        "timestamp_s",
    )
    missing = [column for column in required_columns if column not in metadata.columns]
    if missing:
        raise ValueError(
            "source metadata is missing required J1 fields: " + ", ".join(missing)
        )
    normalized = metadata.reset_index(drop=True).copy()
    normalized["day"] = coerce_text_column(
        normalized,
        "day",
        np.arange(len(normalized), dtype=int),
        default_from="session",
    )
    return validate_j1_metadata_frame(normalized)


def coerce_text_column(
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
    invalid = normalized.str.strip().str.casefold().isin(INVALID_TEXT_VALUES)
    if default_from is not None:
        fallback = metadata[default_from].astype(str)
        normalized = normalized.where(~invalid, fallback)
        invalid = normalized.str.strip().str.casefold().isin(INVALID_TEXT_VALUES)
    if fill_value is not None:
        normalized = normalized.where(~invalid, fill_value)
        invalid = normalized.str.strip().str.casefold().isin(INVALID_TEXT_VALUES)
    if prefix is not None:
        fallback = pd.Series([f"{prefix}_{row_id:06d}" for row_id in row_ids], index=metadata.index, dtype="object")
        normalized = normalized.where(~invalid, fallback.astype(str))
    return normalized.astype(str)


def coerce_numeric_column(
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
    if default_from is not None and default_from in metadata.columns:
        fallback = pd.to_numeric(metadata[default_from], errors="coerce")
        series = series.where(series.notna(), fallback)
    return series.fillna(float(fill_value)).astype(float)


def coerce_integer_series(values: pd.Series | np.ndarray | list[Any], *, context: str) -> pd.Series:
    series = pd.Series(values)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"{context} must contain only integer-valued labels")
    as_float = numeric.astype(float)
    if not np.allclose(as_float.to_numpy(), np.round(as_float.to_numpy())):
        raise ValueError(f"{context} must contain only integer-valued labels")
    return np.round(as_float).astype("int64")


def parse_feature_schema(value: Any, *, context: str) -> dict[str, str]:
    mapping = require_mapping(value, context=context)
    ensure_allowed_keys(mapping, FEATURE_SCHEMA_KEYS, context=context)
    parsed = {
        key: require_non_empty_text(mapping.get(key), context=f"{context}.{key}")
        for key in FEATURE_ARRAY_KEYS
    }
    return parsed


def parse_label_map(value: Any, *, context: str) -> dict[int, int]:
    mapping = require_mapping(value, context=context)
    if not mapping:
        raise ValueError(f"{context} must not be empty")
    parsed: dict[int, int] = {}
    for raw_key, raw_value in mapping.items():
        key = parse_int(raw_key, context=f"{context} key")
        if key in parsed:
            raise ValueError(f"{context} contains duplicate label {key}")
        parsed[key] = parse_int(raw_value, context=f"{context}[{key}]")
    return parsed


def parse_projection(value: Any, *, context: str) -> dict[str, ArrayProjection]:
    if value is None:
        return {}
    mapping = require_mapping(value, context=context)
    ensure_allowed_keys(mapping, PROJECTION_KEYS, context=context)
    parsed: dict[str, ArrayProjection] = {}
    for key, raw_projection in mapping.items():
        projection_mapping = require_mapping(raw_projection, context=f"{context}.{key}")
        ensure_allowed_keys(projection_mapping, ARRAY_PROJECTION_KEYS, context=f"{context}.{key}")
        columns = projection_mapping.get("columns")
        if not isinstance(columns, list) or not columns:
            raise ValueError(f"{context}.{key}.columns must be a non-empty list of column indices")
        parsed[key] = ArrayProjection(
            from_schema=require_non_empty_text(
                projection_mapping.get("from_schema"),
                context=f"{context}.{key}.from_schema",
            ),
            to_schema=require_non_empty_text(
                projection_mapping.get("to_schema"),
                context=f"{context}.{key}.to_schema",
            ),
            columns=tuple(parse_int(column, context=f"{context}.{key}.columns") for column in columns),
        )
    return parsed


def ensure_allowed_keys(mapping: dict[str, Any], allowed: set[str], *, context: str) -> None:
    unknown = set(mapping) - set(allowed)
    if unknown:
        raise ValueError(f"{context} contains unsupported keys: {sorted(unknown)}")


def require_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return value


def require_non_empty_text(value: Any, *, context: str) -> str:
    text = str(value).strip()
    if text.casefold() in INVALID_TEXT_VALUES:
        raise ValueError(f"{context} must be a non-empty string")
    return text


def parse_int(value: Any, *, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be an integer")
    text = str(value).strip()
    if text.casefold() in INVALID_TEXT_VALUES:
        raise ValueError(f"{context} must be an integer")
    try:
        parsed = int(text)
    except ValueError as exc:
        raise ValueError(f"{context} must be an integer") from exc
    return parsed


def resolve_spec_path(spec_dir: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = spec_dir / candidate
    return candidate.resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a fail-closed J1 prepared bundle from prepared roots.")
    parser.add_argument("--spec", type=Path, required=True, help="Path to the explicit J1 build spec YAML file.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for the prepared J1 bundle.")
    args = parser.parse_args(argv)

    bundle = build_j1_bundle(args.spec)
    save_prepared_dataset(bundle, args.out)
    validate_j1_prepared_bundle(args.out)
    print(f"Built J1 bundle with {len(bundle.metadata)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

