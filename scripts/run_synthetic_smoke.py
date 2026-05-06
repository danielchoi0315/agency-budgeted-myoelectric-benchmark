from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from j2bench.figures import save_policy_tradeoff
from j2bench.metrics import summarize_by_policy
from j2bench.policies import apply_policy_grid
from j2bench.synthetic import make_synthetic_metadata, make_synthetic_traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a synthetic end-to-end benchmark smoke test.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260415)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    metadata = make_synthetic_metadata()
    traces = make_synthetic_traces(metadata, seed=args.seed)
    decisions = apply_policy_grid(traces, tau_grid=[0.02, 0.05, 0.10, 0.15, 0.20, 0.30])
    metrics = summarize_by_policy(decisions)

    metadata.to_csv(args.out / "metadata.csv", index=False)
    decision_rows = []
    for decision in decisions:
        row = asdict(decision)
        row["decision_probs"] = json.dumps(list(map(float, decision.decision_probs)))
        decision_rows.append(row)
    pd.DataFrame(decision_rows).to_csv(args.out / "policy_decisions.csv", index=False)
    metrics_df = pd.DataFrame([asdict(metric) for metric in metrics])
    metrics_df.to_csv(args.out / "metrics_by_policy_subject.csv", index=False)
    aggregate = (
        metrics_df.groupby("policy", as_index=False)
        .agg(
            macro_f1=("macro_f1", "mean"),
            mean_ali=("mean_ali", "mean"),
            intervention_rate=("intervention_rate", "mean"),
            action_change_rate=("action_change_rate", "mean"),
            risk_coverage_auc=("risk_coverage_auc", "mean"),
            ece=("ece", "mean"),
        )
        .sort_values(["mean_ali", "macro_f1"], ascending=[True, False])
    )
    aggregate.to_csv(args.out / "aggregate_policy_metrics.csv", index=False)
    save_policy_tradeoff(aggregate, args.out / "policy_tradeoff.png")
    print(f"Synthetic smoke test complete -> {args.out}")


if __name__ == "__main__":
    main()

