from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from j2bench.realdata import load_prepared_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Upgrade existing benchmark outputs to the current schema.")
    parser.add_argument("--dataset", choices=["db10", "hyser", "cemhsey"], required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--user-model", required=True)
    parser.add_argument("--assist-model", required=True)
    args = parser.parse_args()

    metrics_path = args.out / "metrics_by_policy_unit.csv"
    aggregate_path = args.out / "aggregate_policy_metrics.csv"
    if not metrics_path.exists() or not aggregate_path.exists():
        raise FileNotFoundError(f"missing benchmark outputs under {args.out}")

    metrics_df = pd.read_csv(metrics_path)
    aggregate_df = pd.read_csv(aggregate_path)

    if "active_macro_f1" not in metrics_df.columns:
        if args.dataset == "db10":
            raise ValueError("DB10 outputs require a full rerun to compute active_macro_f1 correctly")
        metrics_df["active_macro_f1"] = metrics_df["macro_f1"]
    if "active_risk_coverage_auc" not in metrics_df.columns:
        if args.dataset == "db10":
            raise ValueError("DB10 outputs require a full rerun to compute active_risk_coverage_auc correctly")
        metrics_df["active_risk_coverage_auc"] = metrics_df["risk_coverage_auc"]
    if "n_active" not in metrics_df.columns:
        if args.dataset == "db10":
            raise ValueError("DB10 outputs require a full rerun to compute n_active correctly")
        metrics_df["n_active"] = metrics_df["n"]

    if "active_macro_f1" not in aggregate_df.columns:
        aggregate_df["active_macro_f1"] = aggregate_df["macro_f1"]
    if "active_risk_coverage_auc" not in aggregate_df.columns:
        aggregate_df["active_risk_coverage_auc"] = aggregate_df["risk_coverage_auc"]
    if "n_active" not in aggregate_df.columns:
        aggregate_df["n_active"] = aggregate_df["n"]

    metrics_df.to_csv(metrics_path, index=False)
    aggregate_df.to_csv(aggregate_path, index=False)

    bundle = load_prepared_dataset(args.prepared_root)
    context_modes = []
    if "assist_context_mode" in bundle.metadata.columns:
        context_modes = sorted(bundle.metadata["assist_context_mode"].dropna().astype(str).unique().tolist())
    manifest = {
        "dataset": args.dataset,
        "prepared_root": str(args.prepared_root),
        "user_model": args.user_model,
        "assist_model": args.assist_model,
        "assist_context_modes": context_modes,
        "n_rows": int(len(bundle.metadata)),
        "upgraded_legacy_outputs": True,
    }
    (args.out / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Upgraded benchmark outputs under {args.out}")


if __name__ == "__main__":
    main()

