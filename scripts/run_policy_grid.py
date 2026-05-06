from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from j2bench.metrics import summarize_by_policy
from j2bench.policies import apply_policy_grid
from j2bench.schemas import PredictionTrace


def _parse_vector(value: str) -> np.ndarray:
    parsed = json.loads(value)
    return np.asarray(parsed, dtype=float)


def load_prediction_traces(path: Path) -> list[PredictionTrace]:
    frame = pd.read_csv(path)
    required = {
        "dataset_id",
        "split_id",
        "subject_id",
        "session",
        "timestamp_s",
        "label",
        "p_user",
        "p_assist",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"prediction trace file missing columns: {sorted(missing)}")
    traces: list[PredictionTrace] = []
    for row in frame.to_dict(orient="records"):
        traces.append(
            PredictionTrace(
                dataset_id=str(row["dataset_id"]),
                split_id=str(row["split_id"]),
                subject_id=str(row["subject_id"]),
                session=str(row["session"]),
                day=None if pd.isna(row.get("day")) else str(row.get("day")),
                timestamp_s=float(row["timestamp_s"]),
                label=int(row["label"]),
                p_user=_parse_vector(str(row["p_user"])),
                p_assist=_parse_vector(str(row["p_assist"])),
                uncertainty=float(row.get("uncertainty", np.nan)),
                calibration_temperature=float(row.get("calibration_temperature", 1.0)),
            )
        )
    return traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply shared policy grid to calibrated prediction traces.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tau-grid", nargs="+", type=float, default=[0.02, 0.05, 0.10, 0.15, 0.20, 0.30])
    args = parser.parse_args()

    traces = load_prediction_traces(args.predictions)
    decisions = apply_policy_grid(traces, tau_grid=args.tau_grid)
    metrics = summarize_by_policy(decisions)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for decision in decisions:
        row = asdict(decision)
        row["decision_probs"] = json.dumps(list(map(float, decision.decision_probs)))
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.out_dir / "policy_decisions.csv", index=False)
    pd.DataFrame([asdict(metric) for metric in metrics]).to_csv(args.out_dir / "metrics_by_policy_subject.csv", index=False)
    print(f"Wrote policy decisions and metrics to {args.out_dir}")


if __name__ == "__main__":
    main()

