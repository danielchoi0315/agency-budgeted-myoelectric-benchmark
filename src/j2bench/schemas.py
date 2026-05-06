from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _probability_vector(values: np.ndarray | list[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D probability vector")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    if np.any(arr < -1e-12):
        raise ValueError(f"{name} contains negative probabilities")
    total = float(arr.sum())
    if total <= 0:
        raise ValueError(f"{name} must have positive mass")
    return arr / total


@dataclass(frozen=True)
class Episode:
    dataset_id: str
    subject_id: str
    group: str
    session: str
    day: str | None
    task: str
    start_time_s: float
    end_time_s: float
    labels: dict[str, Any]
    sensor_refs: dict[str, str | Path]
    modality_available: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        item = asdict(self)
        item["sensor_refs"] = {k: str(v) for k, v in self.sensor_refs.items()}
        return item


@dataclass(frozen=True)
class WindowBatch:
    dataset_id: str
    split_id: str
    subject_ids: list[str]
    groups: list[str]
    sessions: list[str]
    days: list[str | None]
    labels: np.ndarray
    features: np.ndarray
    timestamps_s: np.ndarray
    time_to_event_s: np.ndarray
    modality_mask: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        n = int(self.labels.shape[0])
        if self.features.shape[0] != n:
            raise ValueError("features and labels have different row counts")
        if self.timestamps_s.shape[0] != n or self.time_to_event_s.shape[0] != n:
            raise ValueError("timestamp arrays must align with labels")
        for name, mask in self.modality_mask.items():
            if np.asarray(mask).shape[0] != n:
                raise ValueError(f"modality mask {name!r} does not align with labels")


@dataclass(frozen=True)
class PredictionTrace:
    dataset_id: str
    split_id: str
    subject_id: str
    session: str
    day: str | None
    timestamp_s: float
    label: int
    p_user: np.ndarray
    p_assist: np.ndarray
    uncertainty: float
    calibration_temperature: float = 1.0
    user_temperature: float = 1.0
    assist_temperature: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        p_user = _probability_vector(self.p_user, "p_user")
        p_assist = _probability_vector(self.p_assist, "p_assist")
        if p_user.shape != p_assist.shape:
            raise ValueError("p_user and p_assist must have the same class count")
        object.__setattr__(self, "p_user", p_user)
        object.__setattr__(self, "p_assist", p_assist)

    @property
    def n_classes(self) -> int:
        return int(self.p_user.shape[0])


@dataclass(frozen=True)
class PolicyDecision:
    dataset_id: str
    split_id: str
    subject_id: str
    session: str
    day: str | None
    timestamp_s: float
    label: int
    policy: str
    selected_action: int
    user_action: int
    assist_action: int
    decision_probs: np.ndarray
    selected_confidence: float
    coverage_confidence: float
    control_source: str
    action_changed: bool
    autonomy_used: bool
    intervene: bool
    defer: bool
    abstain: bool
    expected_utility: float
    expected_ali: float
    tau: float
    agency_margin: float
    decision_time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        probs = _probability_vector(self.decision_probs, "decision_probs")
        n_classes = probs.shape[0]
        for name, value in {
            "label": self.label,
            "selected_action": self.selected_action,
            "user_action": self.user_action,
            "assist_action": self.assist_action,
        }.items():
            if not 0 <= int(value) < n_classes:
                raise ValueError(f"{name}={value} outside decision_probs class axis")
        for name, value in {
            "expected_utility": self.expected_utility,
            "expected_ali": self.expected_ali,
            "tau": self.tau,
            "agency_margin": self.agency_margin,
            "decision_time_ms": self.decision_time_ms,
            "selected_confidence": self.selected_confidence,
            "coverage_confidence": self.coverage_confidence,
        }.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.tau < 0 or self.expected_ali < 0:
            raise ValueError("tau and expected_ali must be nonnegative")
        if not np.isclose(self.agency_margin, self.tau - self.expected_ali):
            raise ValueError("agency_margin must equal tau - expected_ali")
        if not self.abstain and int(np.argmax(probs)) != int(self.selected_action):
            raise ValueError("selected_action must equal argmax(decision_probs) for non-abstain decisions")
        if not np.isclose(self.selected_confidence, probs[int(self.selected_action)]):
            raise ValueError("selected_confidence must equal decision_probs[selected_action]")
        if not 0.0 <= float(self.coverage_confidence) <= 1.0:
            raise ValueError("coverage_confidence must be in [0, 1]")
        object.__setattr__(self, "decision_probs", probs)

    @property
    def correct(self) -> bool:
        return int(self.selected_action) == int(self.label)


@dataclass(frozen=True)
class MetricRecord:
    dataset_id: str
    split_id: str
    policy: str
    subject_id: str | None
    session: str | None
    day: str | None
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    active_macro_f1: float
    brier: float
    nll: float
    ece: float
    risk_coverage_auc: float
    active_risk_coverage_auc: float
    intervention_rate: float
    action_change_rate: float
    mean_ali: float
    median_decision_time_ms: float
    n: int
    n_active: int

