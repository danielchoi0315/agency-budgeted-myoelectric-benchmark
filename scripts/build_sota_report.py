from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from j2bench.io import load_yaml, read_json  # noqa: E402
from j2bench.literature import (  # noqa: E402
    build_comparability_rows,
    evaluate_target,
    summarize_comparability_matrix,
    summarize_overall_status,
)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build a claim-safe SOTA comparison report for J2.")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--anchors", type=Path, default=Path("config/literature_anchors.yaml"))
    parser.add_argument("--comparability", type=Path, default=Path("config/literature_comparability.yaml"))
    parser.add_argument(
        "--anchor-status",
        type=Path,
        default=Path("results/reports/publication_clean/anchor_status.json"),
    )
    parser.add_argument("--out-root", type=Path, default=Path("results/reports/publication_clean"))
    args = parser.parse_args()

    config_path = resolve_repo_path(repo_root, args.config)
    anchors_path = resolve_repo_path(repo_root, args.anchors)
    comparability_path = resolve_repo_path(repo_root, args.comparability)
    anchor_status_path = resolve_repo_path(repo_root, args.anchor_status)
    out_root = resolve_repo_path(repo_root, args.out_root)

    config = load_yaml(config_path)
    literature_cfg = load_yaml(anchors_path)
    comparability_cfg = load_yaml(comparability_path)
    anchor_status = read_json(anchor_status_path)
    payload = build_payload(
        config=config,
        literature_cfg=literature_cfg,
        comparability_cfg=comparability_cfg,
        anchor_status=anchor_status,
    )

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "sota_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_root / "sota_report.md").write_text(build_markdown(payload), encoding="utf-8")
    print(f"Wrote SOTA report to {out_root}")


def build_payload(
    *,
    config: dict[str, Any],
    literature_cfg: dict[str, Any],
    comparability_cfg: dict[str, Any],
    anchor_status: dict[str, Any],
) -> dict[str, Any]:
    observations = {
        (str(row["dataset_id"]), str(row["split_family"])): row
        for row in anchor_status.get("datasets", {}).get("benchmark_observations", [])
    }
    targets = list(literature_cfg.get("targets", []))
    anchors = list(literature_cfg.get("anchors", []))
    comparison_rows = build_comparability_rows(comparability_cfg, targets=targets)
    comparison_summary = summarize_comparability_matrix(comparison_rows)
    target_results = [
        evaluate_target(
            target=target,
            observation_row=observations.get((str(target["dataset_id"]), str(target["split_family"]))),
            anchors=anchors,
            comparison_rows=comparison_rows,
        )
        for target in targets
    ]
    overall = summarize_overall_status(
        target_results,
        anchor_status=str(anchor_status.get("status", "unknown")),
        guardrails=dict(literature_cfg.get("guardrails", {})),
    )
    executed_datasets = sorted({str(row["dataset_id"]) for row in anchor_status.get("datasets", {}).get("benchmark_observations", [])})
    declared_external = list(config.get("workflow", {}).get("external_datasets", []))
    expansion_candidates = [dataset for dataset in declared_external if dataset not in executed_datasets]
    return {
        "status": overall["status"],
        "recommendation": overall["recommendation"],
        "benchmark_prerequisite_status": str(anchor_status.get("status", "unknown")),
        "disallowed_terms": overall["disallowed_terms"],
        "preferred_terms": overall["preferred_terms"],
        "targets": target_results,
        "literature_anchors": anchors,
        "comparability_summary": comparison_summary,
        "comparability_matrix": comparison_rows,
        "executed_datasets": executed_datasets,
        "expansion_candidates": expansion_candidates,
    }


def build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# J2 SOTA Status",
        "",
        "## Overall Status",
        f"- Status: {payload['status']}",
        f"- Benchmark prerequisite status: {payload['benchmark_prerequisite_status']}",
        f"- Recommendation: {payload['recommendation']}",
        "",
        "## Claim Guardrails",
        f"- Disallowed without direct support: {', '.join(payload.get('disallowed_terms', [])) or '<none>'}",
        f"- Preferred phrasing: {', '.join(payload.get('preferred_terms', [])) or '<none>'}",
        "",
        "## Target Evaluations",
    ]
    target_frame = pd.DataFrame(payload.get("targets", []))
    if not target_frame.empty:
        target_frame = target_frame[
            [
                "target_id",
                "dataset_id",
                "split_family",
                "priority",
                "claim_surface",
                "direct_sota_eligible",
                "local_metric_name",
                "local_metric_value",
                "baseline_delta",
                "local_policy",
                "direct_anchor_count",
                "partial_anchor_count",
                "contextual_anchor_count",
                "boundary_anchor_count",
                "status",
                "detail",
            ]
        ].copy()
    lines.extend(render_table(target_frame))
    lines.append("")
    lines.append("## Literature Anchor Ledger")
    anchor_frame = pd.DataFrame(payload.get("literature_anchors", []))
    if not anchor_frame.empty:
        anchor_frame = anchor_frame[
            [
                "anchor_id",
                "dataset_id",
                "citation",
                "comparability",
                "protocol_family",
                "metric_name",
                "metric_value",
                "spread_value",
                "unit",
                "url",
                "targets",
                "note",
            ]
        ].copy()
        anchor_frame["targets"] = anchor_frame["targets"].apply(lambda items: "; ".join(items))
    lines.extend(render_table(anchor_frame))
    lines.append("")
    lines.append("## Direct Comparability Summary")
    summary_frame = pd.DataFrame(payload.get("comparability_summary", []))
    if not summary_frame.empty:
        summary_frame = summary_frame[
            [
                "target_id",
                "target_dataset_id",
                "target_split_family",
                "target_priority",
                "target_claim_surface",
                "target_direct_sota_eligible",
                "n_direct",
                "n_partial",
                "n_not_comparable",
                "n_contextual",
                "n_boundary",
            ]
        ].copy()
    lines.extend(render_table(summary_frame))
    lines.append("")
    lines.append("## Direct Comparability Matrix")
    matrix_frame = pd.DataFrame(payload.get("comparability_matrix", []))
    if not matrix_frame.empty:
        matrix_frame = matrix_frame[
            [
                "target_id",
                "target_claim_surface",
                "target_direct_sota_eligible",
                "paper_id",
                "citation",
                "role",
                "derived_comparability",
                "dataset",
                "task_family",
                "split_shift",
                "population",
                "adaptation",
                "decision_object",
                "metric",
                "anchor_id",
                "blocking_fields",
            ]
        ].copy()
    lines.extend(render_table(matrix_frame))
    lines.append("")
    lines.append("## Dataset Coverage")
    lines.append(f"- Executed datasets: {', '.join(payload.get('executed_datasets', [])) or '<none>'}")
    lines.append(f"- Expansion candidates: {', '.join(payload.get('expansion_candidates', [])) or '<none>'}")
    lines.append("")
    lines.append("## Release Gate")
    lines.append("- Only primary targets marked `direct_sota_eligible = True` can ever release protocol-specific SOTA wording.")
    lines.append("- Comparator-only and benchmark-only targets remain benchmark language even if a direct numeric win appears later.")
    lines.append("")
    lines.append("## Interpretation")
    if payload["status"] == "direct_comparable_sota_supported":
        lines.append("- A protocol-specific SOTA statement is supportable only for the target set encoded above.")
        lines.append("- Keep the claim narrow and tie it to the exact dataset/split/metric definitions registered here.")
    elif payload["status"] in {"blocked_benchmark_not_ready", "blocked_primary_evidence_incomplete"}:
        lines.append("- The current evidence package is incomplete for manuscript-ready benchmarking claims.")
        lines.append("- Resolve the blocking readiness or primary-evidence gaps before making novelty or submission claims.")
    elif payload["status"] in {"direct_numeric_win_pending_stats", "direct_numeric_win_boundary_limited"}:
        lines.append("- A numeric win exists somewhere in the comparison set, but the claim gate is still blocked.")
        lines.append("- Keep the language benchmark-scoped and avoid SOTA wording until the blocking conditions clear.")
    else:
        lines.append("- The current repository supports a bounded benchmark contribution, not a blanket best-published claim.")
        lines.append("- Use the literature anchors to frame novelty around policy-layer benchmarking, matched-budget tradeoffs, and reproducibility.")
    lines.append("")
    return "\n".join(lines)


def render_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["_No data_"]
    return frame.fillna("").to_markdown(index=False).splitlines()


def resolve_repo_path(repo_root: Path, candidate: Path) -> Path:
    return candidate if candidate.is_absolute() else repo_root / candidate


if __name__ == "__main__":
    main()

