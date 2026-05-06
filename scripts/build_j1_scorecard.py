from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from j2bench.j1_scorecard import build_j1_scorecard_from_roots, render_j1_scorecard  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble a final J1 decision scorecard from benchmark, stats, comparator, "
            "provenance, and report artifacts."
        )
    )
    parser.add_argument("--benchmark-root", type=Path, default=Path("results/real/j1"))
    parser.add_argument("--stats-root", type=Path, default=Path("results/reports/j1_open/stats"))
    parser.add_argument(
        "--comparator-root",
        type=Path,
        default=Path("results/reports/j1_open/publication_clean"),
    )
    parser.add_argument("--report-root", type=Path, default=Path("results/reports/j1_open"))
    parser.add_argument("--benchmark-provenance", type=Path)
    parser.add_argument("--stats-provenance", type=Path)
    parser.add_argument("--comparator-provenance", type=Path)
    parser.add_argument("--report-provenance", type=Path)
    parser.add_argument("--out-root", type=Path, default=Path("results/reports/j1_open"))
    parser.add_argument("--scorecard-path", type=Path)
    parser.add_argument("--summary-path", type=Path)
    args = parser.parse_args(argv)

    payload = build_j1_scorecard_from_roots(
        benchmark_root=args.benchmark_root,
        stats_root=args.stats_root,
        comparator_root=args.comparator_root,
        report_root=args.report_root,
        benchmark_provenance=args.benchmark_provenance,
        stats_provenance=args.stats_provenance,
        comparator_provenance=args.comparator_provenance,
        report_provenance=args.report_provenance,
    )
    markdown = render_j1_scorecard(payload)

    scorecard_path = args.scorecard_path or args.out_root / "j1_scorecard.md"
    summary_path = args.summary_path or args.out_root / "j1_scorecard.json"
    scorecard_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    scorecard_path.write_text(markdown, encoding="utf-8")
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote J1 scorecard ({payload['decision']}) to {scorecard_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

