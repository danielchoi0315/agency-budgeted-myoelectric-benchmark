from __future__ import annotations

import argparse
import hashlib
import inspect
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from j2bench.io import read_json, write_json  # noqa: E402
from j2bench.realdata import (  # noqa: E402
    _append_cemhsey_features,
    _extract_td_features,
    _temporal_energy_features,
    build_cemhsey_grasp_manifest_with_report,
    load_prepared_dataset,
    merge_prepared_datasets,
    prepare_cemhsey_manifest,
    save_prepared_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CEMHSEY in resumable per-archive chunks, then merge.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parts-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit-archives", type=int)
    parser.add_argument("--audit", type=Path, default=Path("results/audits/cemhsey_audit.json"))
    args = parser.parse_args()

    manifest, exclusion_report = build_cemhsey_grasp_manifest_with_report(args.root)
    archive_paths = sorted(str(path) for path in manifest["archive_path"].astype(str).unique())
    if args.limit_archives is not None:
        archive_paths = archive_paths[: args.limit_archives]
        exclusion_report = trim_exclusion_report(exclusion_report, archive_paths)
    args.parts_dir.mkdir(parents=True, exist_ok=True)
    audit_sha = load_archive_sha_index(args.audit) if args.audit.exists() else {}

    progress: list[dict[str, object]] = []
    for archive_path in archive_paths:
        archive_name = Path(archive_path).stem
        part_dir = args.parts_dir / archive_name
        ok_file = part_dir / "_SUCCESS"
        fingerprint_file = part_dir / "_fingerprint.json"
        archive_manifest = manifest[manifest["archive_path"].astype(str) == archive_path].reset_index(drop=True)
        expected_fingerprint = make_part_fingerprint(archive_path=Path(archive_path), archive_manifest=archive_manifest, audit_sha=audit_sha)
        cached_fingerprint = read_json(fingerprint_file) if fingerprint_file.exists() else None
        if ok_file.exists() and cached_fingerprint == expected_fingerprint:
            print(f"skip complete part {archive_name}")
        else:
            if part_dir.exists():
                shutil.rmtree(part_dir)
            part_dir.mkdir(parents=True, exist_ok=True)
            print(f"prepare part {archive_name} ({len(archive_manifest)} trials)")
            bundle = prepare_cemhsey_manifest(archive_manifest)
            save_prepared_dataset(bundle, part_dir)
            write_json(expected_fingerprint, fingerprint_file)
        progress.append(
            {
                "archive_name": archive_name,
                "archive_path": archive_path,
                "n_trials": int(len(archive_manifest)),
                "complete": ok_file.exists(),
                "fingerprint": expected_fingerprint,
            }
        )
        write_json(progress, args.parts_dir / "progress.json")

    bundles = [load_prepared_dataset(args.parts_dir / Path(archive_path).stem) for archive_path in archive_paths]
    merged = merge_prepared_datasets(bundles)
    save_prepared_dataset(merged, args.out)
    write_json(exclusion_report, args.out / "excluded_trials.json")
    summary = {
        "parts_dir": str(args.parts_dir),
        "out": str(args.out),
        "n_parts": len(bundles),
        "n_rows": int(len(merged.metadata)),
        "n_excluded_trials": int(exclusion_report["n_excluded_trials"]),
    }
    write_json(summary, args.out / "merge_summary.json")
    print(f"Prepared CEMHSEY merged dataset: {summary['n_rows']} rows from {summary['n_parts']} parts -> {args.out}")


def load_archive_sha_index(audit_path: Path) -> dict[str, str]:
    audit = read_json(audit_path)
    return {Path(item["path"]).name: str(item["sha256"]) for item in audit.get("files", [])}


def make_part_fingerprint(archive_path: Path, archive_manifest, audit_sha: dict[str, str]) -> dict[str, object]:
    source_hash = hashlib.sha256()
    for func in [_append_cemhsey_features, _extract_td_features, _temporal_energy_features]:
        source_hash.update(inspect.getsource(func).encode("utf-8"))
    stat = archive_path.stat()
    return {
        "archive_name": archive_path.name,
        "archive_size_bytes": int(stat.st_size),
        "archive_mtime_ns": int(stat.st_mtime_ns),
        "archive_sha256": audit_sha.get(archive_path.name, ""),
        "n_trials": int(len(archive_manifest)),
        "member_paths_sha256": hashlib.sha256(
            "\n".join(archive_manifest["member_path"].astype(str).tolist()).encode("utf-8")
        ).hexdigest(),
        "prefix_grid_s": [5.0, 10.0, 15.0, 20.0],
        "user_window_s": 1.0,
        "action_start_s": 5.0,
        "source_hash": source_hash.hexdigest(),
    }


def trim_exclusion_report(report: dict[str, object], archive_paths: list[str]) -> dict[str, object]:
    selected_archives = {Path(path).name for path in archive_paths}
    trimmed = dict(report)
    excluded_member_paths = [str(path) for path in report.get("excluded_member_paths", [])]
    selected_member_paths = [
        member_path
        for member_path in excluded_member_paths
        if infer_archive_name_from_member_path(member_path) in selected_archives
    ]
    trimmed["excluded_member_paths"] = selected_member_paths
    trimmed["matched_failed_trials"] = sorted({Path(path).name for path in selected_member_paths})
    trimmed["unmatched_failed_trials"] = sorted(
        trial_id
        for trial_id in report.get("configured_failed_trials", [])
        if Path(str(trial_id)).name not in set(trimmed["matched_failed_trials"])
    )
    trimmed["n_excluded_trials"] = len(selected_member_paths)
    return trimmed


def infer_archive_name_from_member_path(member_path: str) -> str:
    normalized = str(member_path).replace("\\", "/")
    subject = normalized.split("/", 1)[0]
    return f"GRASP_{subject}.zip"


if __name__ == "__main__":
    main()

