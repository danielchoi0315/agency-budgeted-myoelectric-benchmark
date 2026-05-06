from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from j2bench.j1_frontier import (  # noqa: E402
    build_j1_frontier_readout_payload,
    render_j1_frontier_readout,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a J1-Frontier readout from the universal-winner scorecard and "
            "primary pairwise statistics."
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
    parser.add_argument("--out-root", type=Path, default=Path("results/reports/j1_open"))
    parser.add_argument("--readout-path", type=Path)
    parser.add_argument("--summary-path", type=Path)
    args = parser.parse_args(argv)

    payload = build_j1_frontier_readout_payload(
        scorecard_path=args.scorecard_path,
        primary_pairwise_path=args.primary_pairwise_path,
    )
    markdown = render_j1_frontier_readout(payload)

    readout_path = args.readout_path or args.out_root / "j1_frontier_readout.md"
    summary_path = args.summary_path or args.out_root / "j1_frontier_readout.json"
    readout_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    readout_path.write_text(markdown, encoding="utf-8")
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote J1 frontier readout to {readout_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

