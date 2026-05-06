from __future__ import annotations

import numpy as np
import pandas as pd

from .schemas import PredictionTrace


def make_synthetic_metadata(
    n_subjects: int = 12,
    n_sessions: int = 2,
    n_days: int = 2,
    n_trials_per_cell: int = 30,
    dataset_id: str = "synthetic",
) -> pd.DataFrame:
    rows = []
    for subject in range(n_subjects):
        group = "transradial_amputee" if subject % 4 == 0 else "able_bodied"
        for session in range(n_sessions):
            for day in range(n_days):
                for trial in range(n_trials_per_cell):
                    rows.append(
                        {
                            "dataset_id": dataset_id,
                            "subject_id": f"S{subject:02d}",
                            "group": group,
                            "session": f"session_{session + 1}",
                            "day": f"day_{day + 1}",
                            "trial": trial,
                        }
                    )
    return pd.DataFrame(rows)


def make_synthetic_traces(
    metadata: pd.DataFrame,
    n_classes: int = 6,
    seed: int = 20260415,
    split_id: str = "synthetic_split",
) -> list[PredictionTrace]:
    rng = np.random.default_rng(seed)
    traces: list[PredictionTrace] = []
    for i, row in metadata.reset_index(drop=True).iterrows():
        label = int(rng.integers(0, n_classes))
        group_penalty = 0.12 if row["group"] == "transradial_amputee" else 0.0
        user_strength = max(0.35, 0.70 - group_penalty + rng.normal(0, 0.05))
        assist_strength = max(0.35, 0.78 + rng.normal(0, 0.04))
        p_user = _noisy_prob(label, n_classes, user_strength, rng)
        p_assist = _noisy_prob(label, n_classes, assist_strength, rng)
        if rng.random() < 0.18:
            wrong = int((label + rng.integers(1, n_classes)) % n_classes)
            p_assist = _noisy_prob(wrong, n_classes, 0.72, rng)
        traces.append(
            PredictionTrace(
                dataset_id=str(row["dataset_id"]),
                split_id=split_id,
                subject_id=str(row["subject_id"]),
                session=str(row["session"]),
                day=str(row["day"]),
                timestamp_s=float(i) * 0.2,
                label=label,
                p_user=p_user,
                p_assist=p_assist,
                uncertainty=float(1.0 - np.max(p_user)),
                calibration_temperature=1.0,
                metadata={"trial": int(row["trial"]), "group": str(row["group"])},
            )
        )
    return traces


def _noisy_prob(label: int, n_classes: int, strength: float, rng: np.random.Generator) -> np.ndarray:
    base = np.full(n_classes, (1.0 - strength) / (n_classes - 1))
    base[label] = strength
    noise = rng.dirichlet(np.ones(n_classes)) * 0.08
    return base * 0.92 + noise


