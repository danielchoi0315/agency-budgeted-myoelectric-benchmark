from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .realdata import PREPARED_BUNDLE_REQUIRED_FILES, load_saved_sequence_payloads


class ArtifactSchemaError(ValueError):
    """Raised when a J1 artifact does not match the expected schema."""


PREPARED_BUNDLE_SUMMARY_REQUIRED_FIELDS = (
    "dataset",
    "prepared_root",
    "success_file",
    "required_files",
    "metadata_columns",
    "n_rows",
    "label_vocab_size",
    "user_feature_shape",
    "assist_feature_shape",
    "has_sequence_payloads",
    "sequence_payload_keys",
)

BENCHMARK_MANIFEST_REQUIRED_FIELDS = (
    "dataset",
    "prepared_root",
    "prepared_success_file",
    "user_model",
    "assist_model",
    "assist_context_modes",
    "available_sequence_payload_roles",
    "sequence_payload_sources",
    "split_family_allowlist",
    "split_id_allowlist",
    "max_splits_per_family",
    "n_rows",
)

PUBLICATION_MANIFEST_REQUIRED_FIELDS = (
    "dataset",
    "prepared_root",
    "user_model",
    "assist_model",
    "pairwise_repeats",
    "trace_count",
    "decision_count",
    "split_families",
)

SCORECARD_REQUIRED_SECTIONS = (
    "decision banner",
    "frozen candidate and operating point",
    "primary db10 gate",
    "safety timing panel",
    "robustness panel",
    "boundary conditions",
    "readiness and reproducibility status",
    "candidate provenance",
    "decision log",
)
FRONTIER_PACKAGE_REQUIRED_SECTIONS = (
    "bottom line",
    "claim bearing families",
    "frontier summary",
    "universal tau regret",
    "comparator integrity",
    "artifact inventory",
)

_HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*$", re.MULTILINE)


def summarize_prepared_bundle(root: str | PathLike[str]) -> dict[str, Any]:
    """Build and validate a compact summary for a prepared J1 bundle."""

    bundle_root = Path(root)
    missing = [
        file_name
        for file_name in (*PREPARED_BUNDLE_REQUIRED_FILES, "_SUCCESS")
        if not (bundle_root / file_name).exists()
    ]
    if missing:
        names = ", ".join(sorted(missing))
        raise ArtifactSchemaError(f"prepared bundle root is missing required files: {names}")

    metadata = pd.read_csv(bundle_root / "metadata.csv")
    user_features = np.load(bundle_root / "user_features.npy", mmap_mode="r")
    assist_features = np.load(bundle_root / "assist_features.npy", mmap_mode="r")
    label_vocab = np.load(bundle_root / "label_vocab.npy", mmap_mode="r")
    sequence_payloads = load_saved_sequence_payloads(bundle_root) or {}

    dataset = "j1"
    if "dataset_id" in metadata.columns and not metadata.empty:
        dataset_values = metadata["dataset_id"].dropna().astype(str).unique().tolist()
        if dataset_values:
            dataset = str(dataset_values[0])

    summary = {
        "dataset": dataset,
        "prepared_root": str(bundle_root),
        "success_file": str(bundle_root / "_SUCCESS"),
        "required_files": [*PREPARED_BUNDLE_REQUIRED_FILES, "_SUCCESS"],
        "metadata_columns": [str(column) for column in metadata.columns],
        "n_rows": int(len(metadata)),
        "label_vocab_size": int(label_vocab.shape[0]),
        "user_feature_shape": [int(dim) for dim in user_features.shape],
        "assist_feature_shape": [int(dim) for dim in assist_features.shape],
        "has_sequence_payloads": bool(sequence_payloads),
        "sequence_payload_keys": sorted(str(key) for key in sequence_payloads),
    }
    return validate_prepared_bundle_summary(summary, expected_dataset=dataset)


def validate_prepared_bundle_summary(
    summary: Mapping[str, Any] | str | PathLike[str],
    *,
    expected_dataset: str | None = "j1",
) -> dict[str, Any]:
    payload = _coerce_json_mapping(summary, "prepared bundle summary")
    _require_fields(payload, PREPARED_BUNDLE_SUMMARY_REQUIRED_FIELDS, "prepared bundle summary")
    _validate_expected_dataset(payload["dataset"], "prepared bundle summary", expected_dataset)
    _require_non_empty_string(payload["prepared_root"], "prepared_root")
    success_file = _require_non_empty_string(payload["success_file"], "success_file")
    if not success_file.endswith("_SUCCESS"):
        raise ArtifactSchemaError("success_file must point to a _SUCCESS sentinel")

    required_files = _require_string_list(payload["required_files"], "required_files", allow_empty=False)
    missing_files = sorted(set((*PREPARED_BUNDLE_REQUIRED_FILES, "_SUCCESS")) - set(required_files))
    if missing_files:
        names = ", ".join(missing_files)
        raise ArtifactSchemaError(f"required_files is missing bundle entries: {names}")

    _require_string_list(payload["metadata_columns"], "metadata_columns", allow_empty=False)
    n_rows = _require_int(payload["n_rows"], "n_rows", minimum=0)
    _require_int(payload["label_vocab_size"], "label_vocab_size", minimum=1)
    user_shape = _require_shape(payload["user_feature_shape"], "user_feature_shape", dims=2)
    assist_shape = _require_shape(payload["assist_feature_shape"], "assist_feature_shape", dims=2)
    if user_shape[0] != n_rows:
        raise ArtifactSchemaError("user_feature_shape[0] must equal n_rows")
    if assist_shape[0] != n_rows:
        raise ArtifactSchemaError("assist_feature_shape[0] must equal n_rows")

    has_sequence_payloads = payload["has_sequence_payloads"]
    if not isinstance(has_sequence_payloads, bool):
        raise ArtifactSchemaError("has_sequence_payloads must be a boolean")
    sequence_payload_keys = _require_string_list(
        payload["sequence_payload_keys"],
        "sequence_payload_keys",
        allow_empty=True,
    )
    if has_sequence_payloads and not sequence_payload_keys:
        raise ArtifactSchemaError("sequence_payload_keys must be non-empty when has_sequence_payloads is true")
    if not has_sequence_payloads and sequence_payload_keys:
        raise ArtifactSchemaError("sequence_payload_keys must be empty when has_sequence_payloads is false")
    return dict(payload)


def validate_benchmark_manifest(
    manifest: Mapping[str, Any] | str | PathLike[str],
    *,
    expected_dataset: str | None = "j1",
) -> dict[str, Any]:
    payload = _coerce_json_mapping(manifest, "benchmark manifest")
    _require_fields(payload, BENCHMARK_MANIFEST_REQUIRED_FIELDS, "benchmark manifest")
    _validate_expected_dataset(payload["dataset"], "benchmark manifest", expected_dataset)
    _require_non_empty_string(payload["prepared_root"], "prepared_root")
    success_file = _require_non_empty_string(payload["prepared_success_file"], "prepared_success_file")
    if not success_file.endswith("_SUCCESS"):
        raise ArtifactSchemaError("prepared_success_file must point to a _SUCCESS sentinel")
    _require_non_empty_string(payload["user_model"], "user_model")
    _require_non_empty_string(payload["assist_model"], "assist_model")
    _require_string_list(payload["assist_context_modes"], "assist_context_modes", allow_empty=True)
    roles = _require_string_list(
        payload["available_sequence_payload_roles"],
        "available_sequence_payload_roles",
        allow_empty=True,
    )
    sequence_sources = _require_string_mapping(
        payload["sequence_payload_sources"],
        "sequence_payload_sources",
    )
    if set(sequence_sources) != set(roles):
        raise ArtifactSchemaError(
            "sequence_payload_sources keys must exactly match available_sequence_payload_roles"
        )
    _require_string_list(payload["split_family_allowlist"], "split_family_allowlist", allow_empty=True)
    _require_string_list(payload["split_id_allowlist"], "split_id_allowlist", allow_empty=True)
    _require_int(payload["max_splits_per_family"], "max_splits_per_family", allow_none=True, minimum=1)
    _require_int(payload["n_rows"], "n_rows", minimum=0)
    return dict(payload)


def validate_publication_manifest(
    manifest: Mapping[str, Any] | str | PathLike[str],
    *,
    expected_dataset: str | None = "j1",
) -> dict[str, Any]:
    payload = _coerce_json_mapping(manifest, "publication manifest")
    _require_fields(payload, PUBLICATION_MANIFEST_REQUIRED_FIELDS, "publication manifest")
    _validate_expected_dataset(payload["dataset"], "publication manifest", expected_dataset)
    _require_non_empty_string(payload["prepared_root"], "prepared_root")
    _require_non_empty_string(payload["user_model"], "user_model")
    _require_non_empty_string(payload["assist_model"], "assist_model")
    trace_count = _require_int(payload["trace_count"], "trace_count", minimum=1)
    decision_count = _require_int(payload["decision_count"], "decision_count", minimum=1)
    _require_int(payload["pairwise_repeats"], "pairwise_repeats", minimum=1)
    split_families = _require_string_list(payload["split_families"], "split_families", allow_empty=False)
    if decision_count < trace_count:
        raise ArtifactSchemaError("decision_count must be greater than or equal to trace_count")
    if len(set(split_families)) != len(split_families):
        raise ArtifactSchemaError("split_families must not contain duplicates")
    if "db10_anchor_payload_keys" in payload:
        _require_string_list(payload["db10_anchor_payload_keys"], "db10_anchor_payload_keys", allow_empty=True)
    return dict(payload)


def validate_scorecard(
    scorecard: str | PathLike[str],
    *,
    expected_title: str = "J1-open",
) -> dict[str, Any]:
    text = _coerce_text(scorecard, "scorecard")
    matches = list(_HEADING_PATTERN.finditer(text))
    if not matches:
        raise ArtifactSchemaError("scorecard must contain markdown headings")

    title = matches[0].group(1).strip()
    if _normalize_heading(expected_title) not in _normalize_heading(title):
        raise ArtifactSchemaError(f"scorecard title must contain {expected_title!r}")

    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        heading = _normalize_heading(match.group(1))
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[heading] = text[body_start:body_end].strip()

    missing_sections = [section for section in SCORECARD_REQUIRED_SECTIONS if section not in sections]
    if missing_sections:
        names = ", ".join(missing_sections)
        raise ArtifactSchemaError(f"scorecard is missing required sections: {names}")

    empty_sections = [section for section in SCORECARD_REQUIRED_SECTIONS if not sections[section]]
    if empty_sections:
        names = ", ".join(empty_sections)
        raise ArtifactSchemaError(f"scorecard sections must not be empty: {names}")

    return {
        "title": title,
        "sections": {section: sections[section] for section in SCORECARD_REQUIRED_SECTIONS},
    }


def validate_scorecard_markdown(
    scorecard: str | PathLike[str],
    *,
    expected_title: str = "J1-open",
) -> dict[str, Any]:
    return validate_scorecard(scorecard, expected_title=expected_title)


def validate_j1_prepared_bundle_summary(
    summary: Mapping[str, Any] | str | PathLike[str],
) -> dict[str, Any]:
    return validate_prepared_bundle_summary(summary, expected_dataset="j1")


def validate_j1_benchmark_manifest(
    manifest: Mapping[str, Any] | str | PathLike[str],
) -> dict[str, Any]:
    return validate_benchmark_manifest(manifest, expected_dataset="j1")


def validate_j1_publication_manifest(
    manifest: Mapping[str, Any] | str | PathLike[str],
) -> dict[str, Any]:
    return validate_publication_manifest(manifest, expected_dataset="j1")


def validate_j1_scorecard(
    scorecard: str | PathLike[str],
    *,
    expected_title: str = "J1-open",
) -> dict[str, Any]:
    return validate_scorecard(scorecard, expected_title=expected_title)


def validate_frontier_package_markdown(
    package: str | PathLike[str],
    *,
    expected_title: str = "J1-Frontier Package",
) -> dict[str, Any]:
    text = _coerce_text(package, "frontier package")
    matches = list(_HEADING_PATTERN.finditer(text))
    if not matches:
        raise ArtifactSchemaError("frontier package must contain markdown headings")

    title = matches[0].group(1).strip()
    if _normalize_heading(expected_title) not in _normalize_heading(title):
        raise ArtifactSchemaError(f"frontier package title must contain {expected_title!r}")

    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        heading = _normalize_heading(match.group(1))
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[heading] = text[body_start:body_end].strip()

    missing_sections = [
        section for section in FRONTIER_PACKAGE_REQUIRED_SECTIONS if section not in sections
    ]
    if missing_sections:
        names = ", ".join(missing_sections)
        raise ArtifactSchemaError(f"frontier package is missing required sections: {names}")

    empty_sections = [
        section for section in FRONTIER_PACKAGE_REQUIRED_SECTIONS if not sections[section]
    ]
    if empty_sections:
        names = ", ".join(empty_sections)
        raise ArtifactSchemaError(f"frontier package sections must not be empty: {names}")

    return {
        "title": title,
        "sections": {
            section: sections[section] for section in FRONTIER_PACKAGE_REQUIRED_SECTIONS
        },
    }


def validate_j1_frontier_package(
    package: str | PathLike[str],
    *,
    expected_title: str = "J1-Frontier Package",
) -> dict[str, Any]:
    return validate_frontier_package_markdown(package, expected_title=expected_title)


def _coerce_json_mapping(
    source: Mapping[str, Any] | str | PathLike[str],
    artifact_name: str,
) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)

    if isinstance(source, PathLike):
        path = Path(source)
        if not path.exists():
            raise ArtifactSchemaError(f"{artifact_name} file does not exist: {path}")
        text = path.read_text(encoding="utf-8")
    elif isinstance(source, str):
        path = Path(source)
        if path.exists():
            text = path.read_text(encoding="utf-8")
        else:
            text = source
    else:
        raise ArtifactSchemaError(f"{artifact_name} must be a mapping, JSON string, or path")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ArtifactSchemaError(f"{artifact_name} must be valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ArtifactSchemaError(f"{artifact_name} must decode to a JSON object")
    return dict(payload)


def _coerce_text(source: str | PathLike[str], artifact_name: str) -> str:
    if isinstance(source, PathLike):
        path = Path(source)
        if not path.exists():
            raise ArtifactSchemaError(f"{artifact_name} file does not exist: {path}")
        return path.read_text(encoding="utf-8")
    if isinstance(source, str):
        if "\n" in source or "\r" in source:
            return source
        try:
            path = Path(source)
            if path.exists():
                return path.read_text(encoding="utf-8")
        except OSError:
            return source
        return source
    raise ArtifactSchemaError(f"{artifact_name} must be text or a path")


def _require_fields(
    payload: Mapping[str, Any],
    required_fields: Sequence[str],
    artifact_name: str,
) -> None:
    missing = [field for field in required_fields if field not in payload]
    if missing:
        names = ", ".join(missing)
        raise ArtifactSchemaError(f"{artifact_name} is missing required fields: {names}")


def _validate_expected_dataset(value: Any, artifact_name: str, expected_dataset: str | None) -> str:
    dataset = _require_non_empty_string(value, "dataset")
    if expected_dataset is not None and dataset.lower() != expected_dataset.lower():
        raise ArtifactSchemaError(f"{artifact_name} dataset must be {expected_dataset!r}")
    return dataset


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactSchemaError(f"{field_name} must be a non-empty string")
    return value


def _require_string_list(
    value: Any,
    field_name: str,
    *,
    allow_empty: bool,
) -> list[str]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ArtifactSchemaError(f"{field_name} must be a list of strings")
    values = [_require_non_empty_string(item, f"{field_name}[]") for item in value]
    if not allow_empty and not values:
        raise ArtifactSchemaError(f"{field_name} must not be empty")
    return values


def _require_string_mapping(value: Any, field_name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ArtifactSchemaError(f"{field_name} must be an object mapping strings to strings")
    coerced: dict[str, str] = {}
    for key, item in value.items():
        coerced[_require_non_empty_string(key, f"{field_name} key")] = _require_non_empty_string(
            item,
            f"{field_name}[{key!r}]",
        )
    return coerced


def _require_int(
    value: Any,
    field_name: str,
    *,
    minimum: int | None = None,
    allow_none: bool = False,
) -> int | None:
    if value is None:
        if allow_none:
            return None
        raise ArtifactSchemaError(f"{field_name} must be an integer")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactSchemaError(f"{field_name} must be an integer")
    if minimum is not None and value < minimum:
        raise ArtifactSchemaError(f"{field_name} must be >= {minimum}")
    return value


def _require_shape(value: Any, field_name: str, *, dims: int) -> list[int]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ArtifactSchemaError(f"{field_name} must be a list of integers")
    shape = [_require_int(item, f"{field_name}[]", minimum=0) for item in value]
    if len(shape) != dims:
        raise ArtifactSchemaError(f"{field_name} must contain exactly {dims} dimensions")
    return [int(item) for item in shape]


def _normalize_heading(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


__all__ = [
    "ArtifactSchemaError",
    "BENCHMARK_MANIFEST_REQUIRED_FIELDS",
    "FRONTIER_PACKAGE_REQUIRED_SECTIONS",
    "PREPARED_BUNDLE_SUMMARY_REQUIRED_FIELDS",
    "PUBLICATION_MANIFEST_REQUIRED_FIELDS",
    "SCORECARD_REQUIRED_SECTIONS",
    "summarize_prepared_bundle",
    "validate_benchmark_manifest",
    "validate_frontier_package_markdown",
    "validate_j1_benchmark_manifest",
    "validate_j1_frontier_package",
    "validate_j1_prepared_bundle_summary",
    "validate_j1_publication_manifest",
    "validate_j1_scorecard",
    "validate_prepared_bundle_summary",
    "validate_publication_manifest",
    "validate_scorecard",
    "validate_scorecard_markdown",
]

