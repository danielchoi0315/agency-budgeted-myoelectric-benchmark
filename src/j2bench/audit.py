from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .j1_contract import validate_j1_prepared_bundle
from .realdata import validate_j1_metadata_frame


@dataclass(frozen=True)
class FileAudit:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class DatasetAudit:
    dataset_id: str
    root: str
    generated_at: str
    status: str
    file_count: int
    total_bytes: int
    extensions: dict[str, int]
    checks: list[dict[str, str | int | bool]]
    files: list[FileAudit]
    warnings: list[str]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset_root(dataset_id: str, root: Path, hash_files: bool = False) -> DatasetAudit:
    warnings: list[str] = []
    if not root.exists():
        warnings.append(f"root does not exist: {root}")
        return DatasetAudit(
            dataset_id,
            str(root),
            _now(),
            "FAIL",
            0,
            0,
            {},
            [{"name": "root_exists", "passed": False, "detail": str(root)}],
            [],
            warnings,
        )
    files: list[FileAudit] = []
    extensions: dict[str, int] = {}
    total = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        size = path.stat().st_size
        total += size
        ext = path.suffix.lower() or "<none>"
        extensions[ext] = extensions.get(ext, 0) + 1
        digest = sha256_file(path) if hash_files else ""
        files.append(FileAudit(str(path.relative_to(root)), size, digest))
    if not files:
        warnings.append("no files found")
    checks = dataset_specific_checks(dataset_id, root, extensions, files)
    status = "PASS" if files and all(bool(check["passed"]) for check in checks) else "FAIL"
    return DatasetAudit(dataset_id, str(root), _now(), status, len(files), total, extensions, checks, files, warnings)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dataset_specific_checks(
    dataset_id: str,
    root: Path,
    extensions: dict[str, int],
    files: list[FileAudit],
) -> list[dict[str, str | int | bool]]:
    file_names = [item.path.replace("\\", "/") for item in files]
    if dataset_id == "db10":
        return [
            _check("has_mat_or_csv", extensions.get(".mat", 0) + extensions.get(".csv", 0) > 0, "DB10 should expose MDS1 data files"),
            _check(
                "has_crc_manifest",
                any(name.lower().endswith("datasetcontentcrc.sfv") for name in file_names),
                "Expected DatasetContentCRC.sfv from DB10 download",
            ),
            _check(
                "has_subject_like_files",
                sum("s" in Path(name).stem.lower() for name in file_names) >= 30,
                "Expected many subject files/directories for DB10 MDS1",
            ),
        ]
    if dataset_id == "hyser":
        hea = extensions.get(".hea", 0)
        dat = extensions.get(".dat", 0)
        return [
            _check("has_wfdb_headers", hea > 0, "Hyser WFDB headers .hea are required"),
            _check("has_wfdb_dat", dat > 0, "Hyser WFDB data .dat files are required"),
            _check("wfdb_pairs_balanced", hea == dat, f"Expected .hea/.dat pair count match, observed {hea}/{dat}"),
            _check(
                "has_day_or_session_structure",
                any(("session" in name.lower() or "day" in name.lower()) for name in file_names),
                "Expected day/session structure for two-day Hyser validation",
            ),
        ]
    if dataset_id == "cemhsey":
        zip_count = extensions.get(".zip", 0)
        required_grasp = {f"GRASP_S{i}.zip" for i in range(1, 14)}
        present_grasp = {Path(name).name for name in file_names if Path(name).name.startswith("GRASP_S")}
        return [
            _check(
                "has_archives_or_mat_files",
                extensions.get(".mat", 0) > 0 or zip_count > 0,
                "CEMHSEY may be stored as extracted MATLAB files or raw Zenodo zip archives",
            ),
            _check(
                "has_required_grasp_archives",
                required_grasp.issubset(present_grasp) or extensions.get(".mat", 0) > 0,
                f"Expected all GRASP_S1..GRASP_S13 archives, observed {sorted(present_grasp)}",
            ),
            _check(
                "has_day_structure_or_complete_archives",
                required_grasp.issubset(present_grasp)
                or any(("day" in name.lower() or "d01" in name.lower() or "d1" in name.lower()) for name in file_names),
                "Expected extracted day structure or the full GRASP archive set required by the benchmark",
            ),
        ]
    if dataset_id == "grabmyo":
        return [
            _check("has_records", any(name.endswith("RECORDS") for name in file_names), "Expected PhysioNet RECORDS"),
            _check(
                "has_sha256",
                any(name.endswith("SHA256SUMS.txt") for name in file_names),
                "Expected PhysioNet SHA256SUMS.txt",
            ),
            _check(
                "has_subject_records_or_metadata",
                extensions.get(".mat", 0) > 0 or extensions.get(".dat", 0) > 0 or any(name.endswith("RECORDS") for name in file_names),
                "Expected GRABMyo data records or at least metadata manifest",
            ),
        ]
    if dataset_id == "capgmyo_dba":
        return [
            _check("has_mat_files", extensions.get(".mat", 0) > 0, "CapgMyo DB-a should contain MATLAB files"),
            _check(
                "has_subject_structure",
                any(("db-a" in name.lower() or "subject" in name.lower() or "s1" in name.lower()) for name in file_names),
                "Expected DB-a/subject structure",
            ),
        ]
    if dataset_id == "putemg":
        return [
            _check(
                "has_data_files",
                extensions.get(".h5", 0) + extensions.get(".hdf5", 0) + extensions.get(".csv", 0) > 0,
                "putEMG should contain HDF5 or CSV data files",
            ),
            _check(
                "has_session_structure",
                any(("session" in name.lower() or "emg" in name.lower()) for name in file_names),
                "Expected session/EMG structure",
            ),
        ]
    if dataset_id == "j1":
        metadata_path = root / "metadata.csv"
        bundle_ok = False
        bundle_detail = "J1 source bundle must satisfy the strict prepared-bundle contract"
        metadata_ok = False
        metadata_detail = "J1 metadata.csv must satisfy the strict handoff schema"
        prefix_ok = False
        prefix_detail = "J1 metadata.csv must have monotone prefix/timestamp ordering within episodes"
        if all((root / name).exists() for name in ("metadata.csv", "user_features.npy", "assist_features.npy", "labels.npy", "label_vocab.npy", "_SUCCESS")):
            try:
                manifest = validate_j1_prepared_bundle(root)
            except Exception as exc:
                bundle_detail = f"J1 prepared-bundle validation failed: {exc}"
            else:
                bundle_ok = True
                metadata_ok = True
                prefix_ok = True
                metadata_detail = f"validated {manifest['n_records']} metadata rows with strict J1 schema"
                prefix_detail = f"validated monotone prefix/timestamp ordering across {manifest['n_episodes']} episodes"
                bundle_detail = f"validated strict J1 prepared bundle with {manifest['n_records']} rows"
        elif metadata_path.exists():
            try:
                validated = validate_j1_metadata_frame(pd.read_csv(metadata_path))
            except Exception as exc:
                metadata_detail = f"J1 metadata validation failed: {exc}"
                prefix_detail = metadata_detail
            else:
                metadata_ok = True
                prefix_ok = True
                metadata_detail = f"validated {len(validated)} metadata rows with strict J1 schema"
                prefix_detail = f"validated monotone prefix/timestamp ordering across {validated['episode_id'].nunique()} episodes"
        return [
            _check(
                "has_metadata_csv",
                any(Path(name).name == "metadata.csv" for name in file_names),
                "J1 expects an already-prepared benchmark bundle with metadata.csv",
            ),
            _check(
                "has_required_feature_arrays",
                all(any(Path(name).name == required for name in file_names) for required in [
                    "user_features.npy",
                    "assist_features.npy",
                    "labels.npy",
                    "label_vocab.npy",
                ]),
                "J1 expects user_features.npy, assist_features.npy, labels.npy, and label_vocab.npy",
            ),
            _check(
                "has_success_marker",
                any(Path(name).name == "_SUCCESS" for name in file_names),
                "J1 prepared bundles should include _SUCCESS",
            ),
            _check("prepared_bundle_contract_valid", bundle_ok, bundle_detail),
            _check("metadata_schema_valid", metadata_ok, metadata_detail),
            _check("causal_prefix_order_valid", prefix_ok, prefix_detail),
        ]
    return [_check("known_dataset_id", False, f"No dataset-specific checks registered for {dataset_id}")]


def _check(name: str, passed: bool, detail: str) -> dict[str, str | int | bool]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def write_audit(audit: DatasetAudit, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(audit), indent=2), encoding="utf-8")

