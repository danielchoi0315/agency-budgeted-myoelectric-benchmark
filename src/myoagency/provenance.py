from __future__ import annotations

import csv
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .audit import sha256_file

SCHEMA_VERSION = 1
DEFAULT_PACKAGE_NAMES = (
    "agency-budgeted-myoelectric-benchmark",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "statsmodels",
    "pyyaml",
    "wfdb",
)

__all__ = [
    "DEFAULT_PACKAGE_NAMES",
    "SCHEMA_VERSION",
    "build_output_manifest",
    "build_prepared_bundle_manifest",
    "build_runtime_metadata",
    "discover_package_versions",
    "manifest_json",
    "write_manifest",
]


def build_prepared_bundle_manifest(
    prepared_root: str | Path,
    *,
    dataset_id: str | None = None,
    config_paths: Sequence[str | Path] = (),
    input_paths: Sequence[str | Path] = (),
    seed_metadata: Mapping[str, Any] | None = None,
    package_names: Sequence[str] = DEFAULT_PACKAGE_NAMES,
    path_root: str | Path | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
    exclude_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Build a deterministic manifest for a prepared bundle directory."""
    root = Path(prepared_root)
    inferred_dataset_id = (
        _infer_prepared_dataset_id(root) if dataset_id is None else str(dataset_id)
    )
    return _build_manifest(
        artifact_root=root,
        artifact_type="prepared_bundle",
        dataset_id=inferred_dataset_id,
        config_paths=config_paths,
        input_paths=input_paths,
        seed_metadata=seed_metadata,
        package_names=package_names,
        path_root=path_root,
        extra_metadata=extra_metadata,
        exclude_paths=exclude_paths,
    )


def build_output_manifest(
    output_root: str | Path,
    *,
    output_kind: str,
    dataset_id: str | None = None,
    config_paths: Sequence[str | Path] = (),
    input_paths: Sequence[str | Path] = (),
    seed_metadata: Mapping[str, Any] | None = None,
    package_names: Sequence[str] = DEFAULT_PACKAGE_NAMES,
    path_root: str | Path | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
    exclude_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Build a deterministic manifest for benchmark or report output directories."""
    normalized_kind = str(output_kind).strip().replace("-", "_").replace(" ", "_")
    if not normalized_kind:
        raise ValueError("output_kind must be a non-empty string")
    return _build_manifest(
        artifact_root=Path(output_root),
        artifact_type=f"{normalized_kind}_output",
        dataset_id=None if dataset_id is None else str(dataset_id),
        config_paths=config_paths,
        input_paths=input_paths,
        seed_metadata=seed_metadata,
        package_names=package_names,
        path_root=path_root,
        extra_metadata=extra_metadata,
        exclude_paths=exclude_paths,
    )


def build_runtime_metadata(
    *,
    package_names: Sequence[str] = DEFAULT_PACKAGE_NAMES,
) -> dict[str, Any]:
    """Capture Python, platform, and discoverable package-version metadata."""
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "compiler": platform.python_compiler(),
            "executable": Path(sys.executable).resolve().as_posix(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "package_versions": discover_package_versions(package_names),
    }


def discover_package_versions(
    package_names: Sequence[str] = DEFAULT_PACKAGE_NAMES,
) -> dict[str, Any]:
    """Resolve package versions without failing when a distribution is unavailable."""
    found: dict[str, str] = {}
    missing: list[str] = []
    for name in sorted({str(item) for item in package_names if str(item).strip()}):
        try:
            found[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            missing.append(name)
    return {"found": found, "missing": missing}


def manifest_json(manifest: Mapping[str, Any]) -> str:
    """Render a manifest with stable key ordering and a trailing newline."""
    return json.dumps(_canonicalize_json(manifest), indent=2, sort_keys=True) + "\n"


def write_manifest(manifest: Mapping[str, Any], path: str | Path) -> None:
    """Persist a manifest as canonical JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(manifest_json(manifest), encoding="utf-8")


def _build_manifest(
    *,
    artifact_root: Path,
    artifact_type: str,
    dataset_id: str | None,
    config_paths: Sequence[str | Path],
    input_paths: Sequence[str | Path],
    seed_metadata: Mapping[str, Any] | None,
    package_names: Sequence[str],
    path_root: str | Path | None,
    extra_metadata: Mapping[str, Any] | None,
    exclude_paths: Sequence[str | Path],
) -> dict[str, Any]:
    root = artifact_root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"artifact root does not exist: {artifact_root}")
    if not root.is_dir():
        raise NotADirectoryError(f"artifact root must be a directory: {artifact_root}")
    path_base = Path(path_root).resolve() if path_root is not None else None
    file_records = [
        _build_file_record(path, display_path=path.relative_to(root).as_posix())
        for path in _collect_root_files(root, exclude_paths=exclude_paths)
    ]
    row_counts = {
        record["path"]: record["row_count"]
        for record in file_records
        if record["row_count"] is not None
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "hash_algorithm": "sha256",
        "artifact_type": artifact_type,
        "dataset_id": dataset_id,
        "artifact_root": _normalize_external_path(root, relative_to=path_base),
        "file_count": len(file_records),
        "files": file_records,
        "row_counts": row_counts,
        "config_files": _collect_external_path_records(config_paths, relative_to=path_base),
        "input_files": _collect_external_path_records(input_paths, relative_to=path_base),
        "runtime": build_runtime_metadata(package_names=package_names),
        "seed_metadata": _canonicalize_json(seed_metadata or {}),
        "extra_metadata": _canonicalize_json(extra_metadata or {}),
    }


def _collect_root_files(root: Path, *, exclude_paths: Sequence[str | Path]) -> list[Path]:
    excluded = {
        Path(path).resolve()
        for path in exclude_paths
    }
    files = [
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.resolve() not in excluded
    ]
    return sorted(files, key=lambda path: path.as_posix().lower())


def _collect_external_path_records(
    paths: Sequence[str | Path],
    *,
    relative_to: Path | None,
) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for raw_path in paths:
        candidate = Path(raw_path)
        if candidate.exists() and candidate.is_dir():
            expanded = sorted(
                (path.resolve() for path in candidate.rglob("*") if path.is_file()),
                key=lambda path: path.as_posix().lower(),
            )
            for path in expanded:
                display_path = _normalize_external_path(path, relative_to=relative_to)
                records[display_path] = _build_file_record(path, display_path=display_path)
            continue
        display_path = _normalize_external_path(candidate, relative_to=relative_to)
        records[display_path] = _build_file_record(candidate, display_path=display_path)
    return [records[path] for path in sorted(records)]


def _build_file_record(path: Path, *, display_path: str) -> dict[str, Any]:
    normalized_path = str(display_path).replace("\\", "/")
    kind = _detect_file_kind(path)
    if not path.exists():
        return {
            "path": normalized_path,
            "kind": kind,
            "exists": False,
            "size_bytes": None,
            "sha256": None,
            "row_count": None,
            "shape": None,
        }
    row_count, shape = _infer_row_count_and_shape(path)
    return {
        "path": normalized_path,
        "kind": kind,
        "exists": True,
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "row_count": row_count,
        "shape": shape,
    }


def _infer_row_count_and_shape(path: Path) -> tuple[int | None, list[int] | None]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _count_csv_rows(path), None
    if suffix == ".jsonl":
        return _count_jsonl_rows(path), None
    if suffix == ".npy":
        array = _load_array_header(path)
        shape = [int(dim) for dim in array.shape]
        row_count = int(array.shape[0]) if array.ndim > 0 else None
        return row_count, shape
    return None, None


def _count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            return 0
        return sum(1 for _ in reader)


def _count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _load_array_header(path: Path) -> np.ndarray:
    try:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    except ValueError:
        return np.load(path, allow_pickle=True)


def _detect_file_kind(path: Path) -> str:
    if path.name == "_SUCCESS":
        return "marker"
    kind = path.suffix.lower().lstrip(".")
    if kind == "md":
        return "markdown"
    if kind in {"yml", "yaml"}:
        return "yaml"
    return kind or "file"


def _normalize_external_path(path: Path, *, relative_to: Path | None) -> str:
    resolved = path.resolve()
    if relative_to is not None:
        try:
            return resolved.relative_to(relative_to).as_posix()
        except ValueError:
            pass
    return resolved.as_posix()


def _infer_prepared_dataset_id(root: Path) -> str | None:
    metadata_path = root / "metadata.csv"
    if not metadata_path.exists():
        return None
    try:
        dataset_ids = (
            pd.read_csv(metadata_path, usecols=["dataset_id"])["dataset_id"].dropna().astype(str)
        )
    except ValueError:
        return None
    unique = sorted(dataset_ids.unique().tolist())
    if len(unique) != 1:
        return None
    return unique[0]


def _canonicalize_json(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize_json(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, set):
        return [_canonicalize_json(item) for item in sorted(value, key=str)]
    if isinstance(value, tuple):
        return [_canonicalize_json(item) for item in value]
    if isinstance(value, list):
        return [_canonicalize_json(item) for item in value]
    return value

