from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from j2bench.provenance import build_output_manifest, write_manifest  # noqa: E402
from j2bench.real_benchmark import (  # noqa: E402
    _composite_group_series,
    _metadata_string_series,
    _valid_group_values,
    benchmark_splits,
)
from j2bench.splits import Split  # noqa: E402


COMPOSITE_UNIT_COLUMNS = frozenset({"subject_id", "session", "day", "episode_id", "record_id"})
OVERLAP_COLUMNS = ("source_dataset_id", "subject_id", "day", "session", "episode_id")
COVERAGE_COLUMNS = ("source_dataset_id", "subject_id", "session", "day", "episode_id", "label_raw")
FORWARD_ORDER_COLUMNS = {
    "j1_forward_day_logo": "day",
    "j1_forward_session_logo": "session",
}
CRITICAL_BOUNDARY_COLUMNS = {
    "j1_source_dataset_holdout": "source_dataset_id",
    "j1_able_to_amputee": "subject_id",
    "j1_mixed_to_amputee": "subject_id",
    "j1_amputee_loso": "subject_id",
    "j1_subject_logo": "subject_id",
    "j1_forward_day_logo": "day",
    "j1_forward_session_logo": "session",
}
DISPLAY_SAMPLE_LIMIT = 4


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a leakage-oriented split report for the J1 benchmark.")
    parser.add_argument("--dataset", choices=["j1"], default="j1")
    parser.add_argument("--prepared-root", type=Path, help="Prepared dataset root containing metadata.csv.")
    parser.add_argument("--metadata", type=Path, help="Path to a prepared metadata.csv file.")
    parser.add_argument("--out-root", type=Path, default=Path("results/reports/j1_open"))
    parser.add_argument("--report-path", type=Path, help="Optional explicit markdown output path.")
    parser.add_argument("--summary-path", type=Path, help="Optional explicit JSON summary path.")
    parser.add_argument("--split-family", action="append", default=[], help="Optional split-family allowlist.")
    parser.add_argument("--split-id", action="append", default=[], help="Optional exact split-id allowlist.")
    parser.add_argument("--max-splits-per-family", type=int, help="Optional cap on split count per family.")
    args = parser.parse_args()

    metadata_path, prepared_root = resolve_input_paths(args.prepared_root, args.metadata)
    metadata = pd.read_csv(metadata_path)
    payload = build_split_report_payload(
        metadata,
        dataset_id=args.dataset,
        metadata_path=metadata_path,
        prepared_root=prepared_root,
        split_family_allowlist=tuple(args.split_family) or None,
        split_id_allowlist=tuple(args.split_id) or None,
        max_splits_per_family=args.max_splits_per_family,
    )

    report_path = args.report_path or args.out_root / "split_report.md"
    summary_path = args.summary_path or args.out_root / "split_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path.write_text(build_markdown(payload), encoding="utf-8")
    report_provenance = build_output_manifest(
        report_path.parent,
        output_kind="split_report",
        dataset_id=args.dataset,
        config_paths=(Path("config/config.yaml"),),
        input_paths=tuple(
            path
            for path in (metadata_path, prepared_root)
            if path is not None
        ),
        extra_metadata={
            "summary_path": str(summary_path),
            "report_path": str(report_path),
        },
        exclude_paths=(report_path.parent / "split_report_provenance.json",),
    )
    write_manifest(report_provenance, report_path.parent / "split_report_provenance.json")
    print(f"Wrote split report to {report_path}")


def resolve_input_paths(
    prepared_root: Path | None,
    metadata_path: Path | None,
) -> tuple[Path, Path | None]:
    if prepared_root is None and metadata_path is None:
        raise ValueError("provide either --prepared-root or --metadata")
    if prepared_root is not None and metadata_path is not None:
        raise ValueError("provide only one of --prepared-root or --metadata")
    if metadata_path is not None:
        return metadata_path, None
    assert prepared_root is not None
    return prepared_root / "metadata.csv", prepared_root


def build_split_report_payload(
    metadata: pd.DataFrame,
    *,
    dataset_id: str = "j1",
    metadata_path: Path | None = None,
    prepared_root: Path | None = None,
    split_family_allowlist: tuple[str, ...] | None = None,
    split_id_allowlist: tuple[str, ...] | None = None,
    max_splits_per_family: int | None = None,
) -> dict[str, Any]:
    resolved_metadata = metadata.reset_index(drop=True).copy()
    splits = benchmark_splits(
        resolved_metadata,
        dataset_id,
        split_family_allowlist=split_family_allowlist,
        split_id_allowlist=split_id_allowlist,
        max_splits_per_family=max_splits_per_family,
    )
    if not splits:
        raise ValueError(f"no benchmark splits resolved for {dataset_id}")

    split_rows = [summarize_split(resolved_metadata, split) for split in splits]
    family_rows = summarize_families(resolved_metadata, splits, split_rows)
    critical_failures = [row["split_id"] for row in split_rows if not row["critical_overlap_passed"]]
    forward_failures = [
        row["split_id"]
        for row in split_rows
        if row["forward_ordering_applicable"] and not row["forward_ordering_passed"]
    ]

    summary = {
        "n_splits": len(split_rows),
        "n_families": len(family_rows),
        "critical_overlap_checks_passed": not critical_failures,
        "forward_order_checks_applicable": int(sum(bool(row["forward_ordering_applicable"]) for row in split_rows)),
        "forward_order_checks_passed": not forward_failures,
        "critical_failures": critical_failures,
        "forward_order_failures": forward_failures,
    }
    return {
        "dataset_id": dataset_id,
        "status": "ok" if not critical_failures and not forward_failures else "violations_detected",
        "inputs": {
            "metadata_path": str(metadata_path) if metadata_path is not None else None,
            "prepared_root": str(prepared_root) if prepared_root is not None else None,
            "prepared_success_file_present": (
                (prepared_root / "_SUCCESS").exists() if prepared_root is not None else None
            ),
            "split_family_allowlist": list(split_family_allowlist or ()),
            "split_id_allowlist": list(split_id_allowlist or ()),
            "max_splits_per_family": max_splits_per_family,
        },
        "summary": summary,
        "metadata": summarize_metadata(resolved_metadata),
        "families": family_rows,
        "splits": split_rows,
    }


def summarize_metadata(metadata: pd.DataFrame) -> dict[str, Any]:
    sources = unit_values(metadata, "source_dataset_id")
    subjects = unit_values(metadata, "subject_id")
    sessions = unit_values(metadata, "session")
    days = unit_values(metadata, "day")
    episodes = unit_values(metadata, "episode_id")
    groups = raw_values(metadata, "group")
    labels = raw_values(metadata, "label_raw")
    tasks = raw_values(metadata, "task")
    return {
        "rows": int(len(metadata)),
        "source_dataset_count": len(sorted_values(sources)),
        "subject_count": len(sorted_values(subjects)),
        "session_count": len(sorted_values(sessions)),
        "day_count": len(sorted_values(days)),
        "episode_count": len(sorted_values(episodes)),
        "label_count": len(sorted_values(labels)),
        "group_values": sorted_values(groups),
        "source_dataset_values": sorted_values(sources),
        "task_values": sorted_values(tasks),
    }


def summarize_split(metadata: pd.DataFrame, split: Split) -> dict[str, Any]:
    family = split.split_id.split("|", 1)[0]
    boundary_key = next(iter(split.test_groups.keys()), "")
    train_idx = list(split.train_index)
    test_idx = list(split.test_index)
    overlap_checks = {
        key: overlap_summary(values_for_index(metadata, key, train_idx), values_for_index(metadata, key, test_idx))
        for key in OVERLAP_COLUMNS
    }
    if boundary_key and boundary_key not in overlap_checks:
        overlap_checks[boundary_key] = overlap_summary(
            values_for_index(metadata, boundary_key, train_idx),
            values_for_index(metadata, boundary_key, test_idx),
        )
    critical_overlap_keys = dedupe(
        [
            CRITICAL_BOUNDARY_COLUMNS.get(family) or boundary_key,
            "episode_id" if "episode_id" in metadata.columns else "",
        ]
    )
    forward_order = summarize_forward_order(metadata, split, family)
    return {
        "split_id": split.split_id,
        "split_family": family,
        "boundary_key": boundary_key,
        "boundary_train_values": sorted_values(pd.Series(list(split.train_groups.get(boundary_key, set())), dtype=object)),
        "boundary_test_values": sorted_values(pd.Series(list(split.test_groups.get(boundary_key, set())), dtype=object)),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_subjects": count_values(values_for_index(metadata, "subject_id", train_idx)),
        "test_subjects": count_values(values_for_index(metadata, "subject_id", test_idx)),
        "train_sessions": count_values(values_for_index(metadata, "session", train_idx)),
        "test_sessions": count_values(values_for_index(metadata, "session", test_idx)),
        "train_days": count_values(values_for_index(metadata, "day", train_idx)),
        "test_days": count_values(values_for_index(metadata, "day", test_idx)),
        "train_episodes": count_values(values_for_index(metadata, "episode_id", train_idx)),
        "test_episodes": count_values(values_for_index(metadata, "episode_id", test_idx)),
        "train_group_values": sorted_values(raw_values_for_index(metadata, "group", train_idx)),
        "test_group_values": sorted_values(raw_values_for_index(metadata, "group", test_idx)),
        "train_source_values": sorted_values(values_for_index(metadata, "source_dataset_id", train_idx)),
        "test_source_values": sorted_values(values_for_index(metadata, "source_dataset_id", test_idx)),
        "overlap_checks": overlap_checks,
        "critical_overlap_keys": critical_overlap_keys,
        "critical_overlap_passed": all(overlap_checks[key]["count"] == 0 for key in critical_overlap_keys if key),
        "forward_ordering_applicable": bool(forward_order["applicable"]),
        "forward_ordering_passed": forward_order["passed"],
        "forward_order": forward_order,
    }


def summarize_families(
    metadata: pd.DataFrame,
    splits: list[Split],
    split_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    detail_by_id = {row["split_id"]: row for row in split_rows}
    totals = {
        "rows": int(len(metadata)),
        **{column: count_values(unit_values(metadata, column)) for column in COVERAGE_COLUMNS},
    }
    grouped: dict[str, list[Split]] = {}
    for split in splits:
        family = split.split_id.split("|", 1)[0]
        grouped.setdefault(family, []).append(split)

    family_rows: list[dict[str, Any]] = []
    for family, family_splits in grouped.items():
        details = [detail_by_id[split.split_id] for split in family_splits]
        test_index = sorted({idx for split in family_splits for idx in split.test_index})
        boundary_key = single_value([str(detail["boundary_key"]) for detail in details if detail["boundary_key"]])
        applicable_forward = [detail for detail in details if detail["forward_ordering_applicable"]]
        family_rows.append(
            {
                "split_family": family,
                "boundary_key": boundary_key,
                "n_splits": int(len(family_splits)),
                "split_ids": [split.split_id for split in family_splits],
                "test_boundaries": sort_group_values(
                    boundary_key,
                    [
                        value
                        for detail in details
                        for value in detail["boundary_test_values"]
                    ],
                ),
                "train_rows_min": int(min(len(split.train_index) for split in family_splits)),
                "train_rows_max": int(max(len(split.train_index) for split in family_splits)),
                "test_rows_min": int(min(len(split.test_index) for split in family_splits)),
                "test_rows_max": int(max(len(split.test_index) for split in family_splits)),
                "covered_test_rows": int(len(test_index)),
                "covered_test_rows_pct": ratio(len(test_index), totals["rows"]),
                "covered_test_subjects": coverage_count(metadata, test_index, "subject_id"),
                "covered_test_subjects_pct": ratio(
                    coverage_count(metadata, test_index, "subject_id"),
                    totals["subject_id"],
                ),
                "covered_test_sessions": coverage_count(metadata, test_index, "session"),
                "covered_test_sessions_pct": ratio(
                    coverage_count(metadata, test_index, "session"),
                    totals["session"],
                ),
                "covered_test_days": coverage_count(metadata, test_index, "day"),
                "covered_test_days_pct": ratio(
                    coverage_count(metadata, test_index, "day"),
                    totals["day"],
                ),
                "covered_test_episodes": coverage_count(metadata, test_index, "episode_id"),
                "covered_test_episodes_pct": ratio(
                    coverage_count(metadata, test_index, "episode_id"),
                    totals["episode_id"],
                ),
                "covered_test_sources": coverage_count(metadata, test_index, "source_dataset_id"),
                "covered_test_sources_pct": ratio(
                    coverage_count(metadata, test_index, "source_dataset_id"),
                    totals["source_dataset_id"],
                ),
                "critical_overlap_passed": all(bool(detail["critical_overlap_passed"]) for detail in details),
                "forward_ordering_applicable": bool(applicable_forward),
                "forward_ordering_passed": (
                    all(bool(detail["forward_ordering_passed"]) for detail in applicable_forward)
                    if applicable_forward
                    else None
                ),
            }
        )
    return family_rows


def summarize_forward_order(metadata: pd.DataFrame, split: Split, family: str) -> dict[str, Any]:
    boundary_key = FORWARD_ORDER_COLUMNS.get(family)
    if boundary_key is None:
        return {"applicable": False, "passed": None}
    ordered_values = sorted_values(unit_values(metadata, boundary_key))
    position = {value: idx for idx, value in enumerate(ordered_values)}
    train_values = sorted_values(values_for_index(metadata, boundary_key, split.train_index))
    test_values = sorted_values(values_for_index(metadata, boundary_key, split.test_index))
    if not train_values or not test_values:
        return {
            "applicable": True,
            "passed": False,
            "ordered_values": ordered_values,
            "train_values": train_values,
            "test_values": test_values,
            "train_last": None,
            "test_first": None,
        }
    train_last = train_values[-1]
    test_first = test_values[0]
    passed = max(position[value] for value in train_values) < min(position[value] for value in test_values)
    return {
        "applicable": True,
        "passed": bool(passed),
        "ordered_values": ordered_values,
        "train_values": train_values,
        "test_values": test_values,
        "train_last": train_last,
        "test_first": test_first,
    }


def unit_values(metadata: pd.DataFrame, column: str) -> pd.Series:
    if column in COMPOSITE_UNIT_COLUMNS:
        return _composite_group_series(metadata, column)
    return _metadata_string_series(metadata, column)


def raw_values(metadata: pd.DataFrame, column: str) -> pd.Series:
    return _metadata_string_series(metadata, column)


def values_for_index(metadata: pd.DataFrame, column: str, index: list[int]) -> pd.Series:
    return unit_values(metadata, column).iloc[list(index)]


def raw_values_for_index(metadata: pd.DataFrame, column: str, index: list[int]) -> pd.Series:
    return raw_values(metadata, column).iloc[list(index)]


def overlap_summary(train_values: pd.Series, test_values: pd.Series) -> dict[str, Any]:
    train_set = set(sorted_values(train_values))
    test_set = set(sorted_values(test_values))
    shared = sorted(train_set & test_set)
    return {
        "count": int(len(shared)),
        "sample": shared[:DISPLAY_SAMPLE_LIMIT],
    }


def coverage_count(metadata: pd.DataFrame, index: list[int], column: str) -> int:
    return count_values(values_for_index(metadata, column, index))


def count_values(series: pd.Series) -> int:
    return int(len(sorted_values(series)))


def sorted_values(series: pd.Series) -> list[str]:
    return _valid_group_values(series.astype(object))


def sort_group_values(column: str, values: list[str]) -> list[str]:
    if not values:
        return []
    if column == "source_dataset_id":
        return sorted({str(value) for value in values if str(value)})
    return sorted_values(pd.Series(values, dtype=object))


def ratio(count: int, total: int) -> float | None:
    if total <= 0:
        return None
    return float(count) / float(total)


def dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys([value for value in values if value]))


def single_value(values: list[str]) -> str:
    unique = dedupe(values)
    if not unique:
        return ""
    return unique[0] if len(unique) == 1 else "|".join(unique)


def build_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    metadata = payload["metadata"]
    family_rows = payload["families"]
    split_rows = payload["splits"]
    forward_rows = [row for row in split_rows if row["forward_ordering_applicable"]]
    lines = [
        "# J1 Split Leakage Report",
        "",
        "## Summary",
        f"- Status: {payload['status']}",
        f"- Splits: {summary['n_splits']} across {summary['n_families']} families",
        (
            "- Critical overlap checks: PASS"
            if summary["critical_overlap_checks_passed"]
            else f"- Critical overlap checks: FAIL ({', '.join(summary['critical_failures'])})"
        ),
        (
            "- Forward-order checks: PASS"
            if summary["forward_order_checks_passed"] or summary["forward_order_checks_applicable"] == 0
            else f"- Forward-order checks: FAIL ({', '.join(summary['forward_order_failures'])})"
        ),
        "",
        "## Inputs",
        f"- Metadata path: {payload['inputs']['metadata_path'] or '<in-memory>'}",
        f"- Prepared root: {payload['inputs']['prepared_root'] or '<not provided>'}",
        (
            "- Prepared success file present: "
            f"{payload['inputs']['prepared_success_file_present']}"
            if payload["inputs"]["prepared_success_file_present"] is not None
            else "- Prepared success file present: <not checked>"
        ),
        "",
        "## Dataset Summary",
    ]
    dataset_frame = pd.DataFrame(
        [
            {
                "rows": metadata["rows"],
                "source_datasets": f"{metadata['source_dataset_count']} ({display_values(metadata['source_dataset_values'])})",
                "subjects": metadata["subject_count"],
                "sessions": metadata["session_count"],
                "days": metadata["day_count"],
                "episodes": metadata["episode_count"],
                "labels": metadata["label_count"],
                "groups": display_values(metadata["group_values"]),
                "tasks": display_values(metadata["task_values"]),
            }
        ]
    )
    lines.extend(render_table(dataset_frame))
    lines.extend(
        [
            "",
            "## Family Coverage",
            *render_table(pd.DataFrame([display_family_row(row) for row in family_rows])),
            "",
            "## Split Detail",
            *render_table(pd.DataFrame([display_split_row(row) for row in split_rows])),
            "",
            "## Forward Order Checks",
            *render_table(pd.DataFrame([display_forward_row(row) for row in forward_rows])),
            "",
        ]
    )
    return "\n".join(lines)


def display_family_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "split_family": row["split_family"],
        "boundary_key": row["boundary_key"] or "<none>",
        "n_splits": row["n_splits"],
        "test_boundaries": display_values(row["test_boundaries"]),
        "train_rows": display_range(row["train_rows_min"], row["train_rows_max"]),
        "test_rows": display_range(row["test_rows_min"], row["test_rows_max"]),
        "test_subjects": display_ratio(row["covered_test_subjects"], row["covered_test_subjects_pct"]),
        "test_days": display_ratio(row["covered_test_days"], row["covered_test_days_pct"]),
        "test_episodes": display_ratio(row["covered_test_episodes"], row["covered_test_episodes_pct"]),
        "overlap": display_pass_fail(row["critical_overlap_passed"]),
        "forward_order": display_pass_fail(row["forward_ordering_passed"], na_ok=True),
    }


def display_split_row(row: dict[str, Any]) -> dict[str, Any]:
    boundary_key = row["boundary_key"] or "<none>"
    boundary_overlap = row["overlap_checks"].get(row["boundary_key"], {"count": 0, "sample": []})
    return {
        "split_id": row["split_id"],
        "boundary_key": boundary_key,
        "train_boundary": display_values(row["boundary_train_values"]),
        "test_boundary": display_values(row["boundary_test_values"]),
        "train_groups": display_values(row["train_group_values"]),
        "test_groups": display_values(row["test_group_values"]),
        "train_rows": row["train_rows"],
        "test_rows": row["test_rows"],
        "boundary_overlap": display_overlap(boundary_overlap),
        "episode_overlap": display_overlap(row["overlap_checks"].get("episode_id", {"count": 0, "sample": []})),
        "critical": display_pass_fail(row["critical_overlap_passed"]),
        "forward_order": display_pass_fail(row["forward_ordering_passed"], na_ok=True),
    }


def display_forward_row(row: dict[str, Any]) -> dict[str, Any]:
    details = row["forward_order"]
    return {
        "split_id": row["split_id"],
        "boundary_key": row["boundary_key"],
        "train_values": display_values(details.get("train_values", [])),
        "test_values": display_values(details.get("test_values", [])),
        "train_last": details.get("train_last") or "",
        "test_first": details.get("test_first") or "",
        "status": display_pass_fail(details.get("passed"), na_ok=True),
    }


def render_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["_No data_"]
    headers = [str(column) for column in frame.columns]
    rows = [[format_cell(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    line = "| " + " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    sep = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)) + " |"
        for row in rows
    ]
    return [line, sep, *body]


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return text.replace("\n", " ").replace("|", "\\|")


def display_values(values: list[str]) -> str:
    if not values:
        return "<none>"
    if len(values) <= DISPLAY_SAMPLE_LIMIT:
        return ", ".join(values)
    head = ", ".join(values[:DISPLAY_SAMPLE_LIMIT])
    return f"{head}, +{len(values) - DISPLAY_SAMPLE_LIMIT} more"


def display_range(min_value: int, max_value: int) -> str:
    return str(min_value) if min_value == max_value else f"{min_value}-{max_value}"


def display_ratio(count: int, pct: float | None) -> str:
    if pct is None:
        return str(count)
    return f"{count} ({pct:.0%})"


def display_pass_fail(value: bool | None, *, na_ok: bool = False) -> str:
    if value is None and na_ok:
        return "n/a"
    return "PASS" if bool(value) else "FAIL"


def display_overlap(overlap: dict[str, Any]) -> str:
    if int(overlap.get("count", 0)) == 0:
        return "0"
    sample = overlap.get("sample", [])
    if sample:
        return f"{overlap['count']} ({display_values(sample)})"
    return str(overlap["count"])


if __name__ == "__main__":
    main()

