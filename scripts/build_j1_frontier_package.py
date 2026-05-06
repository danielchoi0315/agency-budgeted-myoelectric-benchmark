from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from j2bench.figures import save_j1_frontier_atlas, save_j1_safety_timing_frontier  # noqa: E402
from j2bench.j1_frontier import build_j1_frontier_package  # noqa: E402
from j2bench.provenance import build_output_manifest, write_manifest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the paper-grade J1 frontier package from scorecard, pairwise, publication, "
            "and earliest-safe artifacts."
        )
    )
    parser.add_argument(
        "--scorecard-path",
        type=Path,
        default=Path("results/reports/j1_open/j1_scorecard.json"),
    )
    parser.add_argument(
        "--primary-pairwise-path",
        type=Path,
        default=Path("results/reports/j1_open/stats/primary_pairwise.csv"),
    )
    parser.add_argument(
        "--publication-aggregate-path",
        type=Path,
        default=Path("results/reports/j1_open/publication_clean/j1_publication_aggregate_policy_metrics.csv"),
    )
    parser.add_argument(
        "--publication-unit-path",
        type=Path,
        default=Path("results/reports/j1_open/publication_clean/j1_publication_metrics_by_policy_unit.csv"),
    )
    parser.add_argument(
        "--earliest-summary-path",
        type=Path,
        default=Path("results/reports/j1_open/publication_clean/j1_earliest_safe_summary.csv"),
    )
    parser.add_argument(
        "--earliest-unit-path",
        type=Path,
        default=Path("results/reports/j1_open/publication_clean/j1_earliest_safe_by_unit.csv"),
    )
    parser.add_argument(
        "--confidence-iso-budget-pairwise-path",
        type=Path,
        default=Path("results/reports/j1_open/publication_clean/j1_confidence_gate_iso_budget_pairwise.csv"),
    )
    parser.add_argument(
        "--earliest-iso-budget-pairwise-path",
        type=Path,
        default=Path("results/reports/j1_open/publication_clean/j1_earliest_safe_iso_budget_pairwise.csv"),
    )
    parser.add_argument(
        "--confidence-exact-budget-pairwise-path",
        type=Path,
        default=Path("results/reports/j1_open/publication_clean/j1_confidence_gate_exact_budget_pairwise.csv"),
    )
    parser.add_argument(
        "--earliest-exact-budget-pairwise-path",
        type=Path,
        default=Path("results/reports/j1_open/publication_clean/j1_earliest_safe_exact_budget_pairwise.csv"),
    )
    parser.add_argument(
        "--claims-config-path",
        type=Path,
        default=Path("config/j1_claims.yaml"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("results/reports/j1_open/frontier"),
    )
    parser.add_argument("--summary-path", type=Path)
    parser.add_argument("--markdown-path", type=Path)
    args = parser.parse_args(argv)

    package = build_j1_frontier_package(
        scorecard_path=args.scorecard_path,
        primary_pairwise_path=args.primary_pairwise_path,
        publication_aggregate_path=args.publication_aggregate_path,
        publication_unit_path=args.publication_unit_path,
        earliest_summary_path=args.earliest_summary_path,
        earliest_unit_path=args.earliest_unit_path,
        confidence_iso_budget_pairwise_path=args.confidence_iso_budget_pairwise_path,
        earliest_iso_budget_pairwise_path=args.earliest_iso_budget_pairwise_path,
        confidence_exact_budget_pairwise_path=args.confidence_exact_budget_pairwise_path,
        earliest_exact_budget_pairwise_path=args.earliest_exact_budget_pairwise_path,
        claims_config_path=args.claims_config_path,
    )

    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary_path or out_root / "j1_frontier_package.json"
    markdown_path = args.markdown_path or out_root / "j1_frontier_package.md"

    frontier_points_path = out_root / "frontier_points.csv"
    pareto_path = out_root / "pareto_frontier.csv"
    overlap_path = out_root / "frontier_overlap.csv"
    non_dominated_path = out_root / "non_dominated_summary.csv"
    regret_path = out_root / "universal_tau_regret.csv"
    unit_deltas_path = out_root / "frontier_unit_deltas.csv"
    budget_summary_path = out_root / "budget_match_summary.csv"
    iso_budget_claim_summary_path = out_root / "iso_budget_claim_summary.csv"
    exact_budget_claim_summary_path = out_root / "exact_budget_claim_summary.csv"
    instability_path = out_root / "operating_point_instability.json"
    atlas_path = out_root / "frontier_atlas.png"
    safety_path = out_root / "safety_timing_frontier.png"
    provenance_path = out_root / "j1_frontier_package_provenance.json"

    package["frontier_points"].to_csv(frontier_points_path, index=False)
    package["pareto_frontier"].to_csv(pareto_path, index=False)
    package["frontier_overlap"].to_csv(overlap_path, index=False)
    package["non_dominated_summary"].to_csv(non_dominated_path, index=False)
    package["universal_tau_regret"].to_csv(regret_path, index=False)
    package["frontier_unit_deltas"].to_csv(unit_deltas_path, index=False)
    package["budget_match_summary"].to_csv(budget_summary_path, index=False)
    package["iso_budget_claim_summary"].to_csv(iso_budget_claim_summary_path, index=False)
    package["exact_budget_claim_summary"].to_csv(exact_budget_claim_summary_path, index=False)
    instability_path.write_text(
        json.dumps(package["operating_point_instability"], indent=2) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(package["summary"], indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(package["markdown"], encoding="utf-8")

    split_families = package["summary"].get("observed_split_families", [])
    save_j1_frontier_atlas(
        package["frontier_points"],
        atlas_path,
        split_families=split_families,
        title_prefix="J1",
    )
    save_j1_safety_timing_frontier(
        package["frontier_points"],
        safety_path,
        split_families=split_families,
        title_prefix="J1",
    )

    provenance = build_output_manifest(
        out_root,
        output_kind="frontier_package",
        dataset_id="j1",
        config_paths=(args.claims_config_path, Path("config/config.yaml")),
        input_paths=(
            args.scorecard_path,
            args.primary_pairwise_path,
            args.publication_aggregate_path,
            args.publication_unit_path,
            args.earliest_summary_path,
            args.earliest_unit_path,
            args.confidence_iso_budget_pairwise_path,
            args.earliest_iso_budget_pairwise_path,
            args.confidence_exact_budget_pairwise_path,
            args.earliest_exact_budget_pairwise_path,
        ),
        extra_metadata={
            "summary_path": str(summary_path),
            "markdown_path": str(markdown_path),
        },
        exclude_paths=(provenance_path,),
    )
    write_manifest(provenance, provenance_path)

    print(f"Wrote J1 frontier package to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

