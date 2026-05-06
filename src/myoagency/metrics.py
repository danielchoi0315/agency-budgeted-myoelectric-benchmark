from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

from .calibration import expected_calibration_error, multiclass_nll, normalize_probabilities
from .schemas import MetricRecord, PolicyDecision


def multiclass_brier(probs: np.ndarray, labels: np.ndarray, n_classes: int | None = None) -> float:
    arr = normalize_probabilities(probs)
    y = np.asarray(labels, dtype=int)
    if n_classes is None:
        n_classes = arr.shape[1]
    one_hot = np.eye(n_classes)[y]
    return float(np.mean(np.sum((arr - one_hot) ** 2, axis=1)))


def risk_coverage_auc(confidence: np.ndarray, correct: np.ndarray) -> float:
    conf = np.asarray(confidence, dtype=float)
    cor = np.asarray(correct, dtype=float)
    if conf.shape[0] == 0:
        raise ValueError("cannot compute risk-coverage on empty arrays")
    order = np.argsort(-conf)
    sorted_correct = cor[order]
    coverage_counts = np.arange(1, sorted_correct.shape[0] + 1, dtype=float)
    risks = 1.0 - np.cumsum(sorted_correct) / coverage_counts
    return float(np.mean(risks))


def decisions_to_arrays(decisions: list[PolicyDecision]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray([d.label for d in decisions], dtype=int)
    preds = np.asarray([d.selected_action for d in decisions], dtype=int)
    confidence = np.asarray([d.coverage_confidence for d in decisions], dtype=float)
    return labels, preds, confidence


def summarize_decisions(
    decisions: Iterable[PolicyDecision],
    dataset_id: str,
    split_id: str,
    policy: str,
    subject_id: str | None = None,
    session: str | None = None,
    day: str | None = None,
) -> MetricRecord:
    rows = list(decisions)
    if not rows:
        raise ValueError("cannot summarize empty decisions")
    labels, preds, confidence = decisions_to_arrays(rows)
    decision_probs = normalize_probabilities(np.vstack([d.decision_probs for d in rows]))
    correct = preds == labels
    labels_axis = list(range(decision_probs.shape[1]))
    active_mask, active_axis = _active_subset(dataset_id, labels, labels_axis)
    active_macro_f1, n_active = _active_macro_f1(labels, preds, active_mask, active_axis)
    active_risk_auc = risk_coverage_auc(confidence[active_mask], correct[active_mask]) if np.any(active_mask) else 0.0
    return MetricRecord(
        dataset_id=dataset_id,
        split_id=split_id,
        policy=policy,
        subject_id=subject_id,
        session=session,
        day=day,
        accuracy=float(accuracy_score(labels, preds)),
        balanced_accuracy=float(recall_score(labels, preds, labels=labels_axis, average="macro", zero_division=0)),
        macro_f1=float(f1_score(labels, preds, labels=labels_axis, average="macro", zero_division=0)),
        active_macro_f1=active_macro_f1,
        brier=multiclass_brier(decision_probs, labels),
        nll=multiclass_nll(decision_probs, labels),
        ece=expected_calibration_error(decision_probs, labels),
        risk_coverage_auc=risk_coverage_auc(confidence, correct),
        active_risk_coverage_auc=float(active_risk_auc),
        intervention_rate=float(np.mean([d.autonomy_used for d in rows])),
        action_change_rate=float(np.mean([d.action_changed for d in rows])),
        mean_ali=float(np.mean([d.expected_ali for d in rows])),
        median_decision_time_ms=float(np.median([d.decision_time_ms for d in rows])),
        n=len(rows),
        n_active=n_active,
    )


def summarize_by_policy(decisions: Iterable[PolicyDecision]) -> list[MetricRecord]:
    grouped: dict[tuple, list[PolicyDecision]] = defaultdict(list)
    for decision in decisions:
        key = (
            decision.dataset_id,
            decision.split_id,
            decision.policy,
            decision.subject_id,
            decision.session,
            decision.day,
        )
        grouped[key].append(decision)
    records: list[MetricRecord] = []
    for (dataset_id, split_id, policy, subject_id, session, day), rows in grouped.items():
        records.append(summarize_decisions(rows, dataset_id, split_id, policy, subject_id, session, day))
    return records


def _active_subset(
    dataset_id: str,
    labels: np.ndarray,
    labels_axis: list[int],
) -> tuple[np.ndarray, list[int]]:
    if dataset_id == "db10" and 0 in labels_axis:
        active_mask = labels != 0
        active_axis = [label for label in labels_axis if label != 0]
    else:
        active_mask = np.ones(labels.shape[0], dtype=bool)
        active_axis = labels_axis
    return active_mask, active_axis


def _active_macro_f1(
    labels: np.ndarray,
    preds: np.ndarray,
    active_mask: np.ndarray,
    active_axis: list[int],
) -> tuple[float, int]:
    if not np.any(active_mask):
        return 0.0, 0
    return (
        float(
            f1_score(
                labels[active_mask],
                preds[active_mask],
                labels=active_axis,
                average="macro",
                zero_division=0,
            )
        ),
        int(np.count_nonzero(active_mask)),
    )

