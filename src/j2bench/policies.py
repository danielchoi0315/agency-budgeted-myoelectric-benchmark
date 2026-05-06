from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .schemas import PolicyDecision, PredictionTrace


def entropy(probs: np.ndarray) -> float:
    p = np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)) / np.log(p.shape[0]))


def expected_ali(p_user: np.ndarray, candidate_action: int) -> float:
    """Agency loss relative to preserving the user's most likely intended action."""
    p = np.asarray(p_user, dtype=float)
    user_action = int(np.argmax(p))
    return float(max(0.0, p[user_action] - p[int(candidate_action)]))


def _decision(
    trace: PredictionTrace,
    policy: str,
    selected_action: int,
    decision_probs: np.ndarray,
    tau: float,
    expected_utility: float,
    decision_time_ms: float,
    control_source: str,
    defer: bool = False,
    abstain: bool = False,
    metadata: dict | None = None,
    coverage_confidence: float | None = None,
) -> PolicyDecision:
    user_action = int(np.argmax(trace.p_user))
    assist_action = int(np.argmax(trace.p_assist))
    ali = expected_ali(trace.p_user, selected_action)
    action_changed = int(selected_action) != user_action
    normalized_probs = np.asarray(decision_probs, dtype=float)
    normalized_probs = normalized_probs / np.sum(normalized_probs)
    selected_confidence = float(normalized_probs[int(selected_action)])
    merged_metadata = dict(trace.metadata)
    if metadata:
        merged_metadata.update(metadata)
    return PolicyDecision(
        dataset_id=trace.dataset_id,
        split_id=trace.split_id,
        subject_id=trace.subject_id,
        session=trace.session,
        day=trace.day,
        timestamp_s=trace.timestamp_s,
        label=int(trace.label),
        policy=policy,
        selected_action=int(selected_action),
        user_action=user_action,
        assist_action=assist_action,
        decision_probs=normalized_probs,
        selected_confidence=selected_confidence,
        coverage_confidence=selected_confidence if coverage_confidence is None else float(coverage_confidence),
        control_source=control_source,
        action_changed=action_changed,
        autonomy_used=action_changed,
        intervene=action_changed,
        defer=defer,
        abstain=abstain,
        expected_utility=float(expected_utility),
        expected_ali=ali,
        tau=float(tau),
        agency_margin=float(tau - ali),
        decision_time_ms=float(decision_time_ms),
        metadata=merged_metadata,
    )


def user_only(trace: PredictionTrace, tau: float = 0.0, decision_time_ms: float = 0.0) -> PolicyDecision:
    action = int(np.argmax(trace.p_user))
    return _decision(trace, "user_only", action, trace.p_user, tau, trace.p_user[action], decision_time_ms, "user")


def assist_only(trace: PredictionTrace, tau: float = 1.0, decision_time_ms: float = 0.0) -> PolicyDecision:
    action = int(np.argmax(trace.p_assist))
    return _decision(trace, "assist_only", action, trace.p_assist, tau, trace.p_assist[action], decision_time_ms, "assist")


def confidence_blend(
    trace: PredictionTrace,
    alpha: float = 0.5,
    tau: float = 1.0,
    decision_time_ms: float = 0.0,
) -> PolicyDecision:
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    blended = (1.0 - alpha) * trace.p_user + alpha * trace.p_assist
    action = int(np.argmax(blended))
    return _decision(
        trace,
        f"confidence_blend_alpha_{alpha:.2f}",
        action,
        blended,
        tau,
        blended[action],
        decision_time_ms,
        "blend",
        metadata={"alpha": alpha},
    )


def selective_prediction(
    trace: PredictionTrace,
    user_confidence_threshold: float = 0.70,
    assist_confidence_threshold: float = 0.70,
    tau: float = 1.0,
    decision_time_ms: float = 0.0,
) -> PolicyDecision:
    user_action = int(np.argmax(trace.p_user))
    assist_action = int(np.argmax(trace.p_assist))
    user_conf = float(trace.p_user[user_action])
    assist_conf = float(trace.p_assist[assist_action])
    if user_conf >= user_confidence_threshold:
        action = user_action
        decision_probs = trace.p_user
        control_source = "user"
        abstain = False
        defer = True
    elif assist_conf >= assist_confidence_threshold:
        action = assist_action
        decision_probs = trace.p_assist
        control_source = "assist"
        abstain = False
        defer = False
    else:
        action = user_action
        decision_probs = trace.p_user
        control_source = "user"
        abstain = False
        defer = True
    return _decision(
        trace,
        "selective_prediction",
        action,
        decision_probs,
        tau,
        float(decision_probs[action]),
        decision_time_ms,
        control_source,
        defer=defer,
        abstain=abstain,
        metadata={
            "user_confidence_threshold": user_confidence_threshold,
            "assist_confidence_threshold": assist_confidence_threshold,
        },
    )


def plain_confidence_threshold_policy(
    trace: PredictionTrace,
    threshold: float,
    tau: float = 1.0,
    decision_time_ms: float = 0.0,
    policy_name: str | None = None,
) -> PolicyDecision:
    if not 0.0 <= threshold <= 1.01:
        raise ValueError("threshold must be in [0, 1.01]")
    user_action = int(np.argmax(trace.p_user))
    assist_action = int(np.argmax(trace.p_assist))
    user_conf = float(trace.p_user[user_action])
    assist_conf = float(trace.p_assist[assist_action])
    if assist_conf >= threshold:
        action = assist_action
        decision_probs = trace.p_assist
        control_source = "assist_confidence"
        coverage_confidence = assist_conf
        defer = False
    else:
        action = user_action
        decision_probs = trace.p_user
        control_source = "user"
        coverage_confidence = user_conf
        defer = True
    return _decision(
        trace,
        policy_name or f"plain_conf_threshold_{threshold:.2f}",
        action,
        decision_probs,
        tau,
        float(decision_probs[action]),
        decision_time_ms,
        control_source,
        defer=defer,
        metadata={"assist_confidence_threshold": float(threshold)},
        coverage_confidence=coverage_confidence,
    )


def set_acsa(trace: PredictionTrace, tau: float, decision_time_ms: float = 0.0) -> PolicyDecision:
    user_action = int(np.argmax(trace.p_user))
    assist_action = int(np.argmax(trace.p_assist))
    user_utility = float(trace.p_user[user_action])
    assist_utility = float(trace.p_assist[assist_action])
    assist_ali = expected_ali(trace.p_user, assist_action)
    if assist_utility > user_utility and assist_ali <= tau:
        action = assist_action
        utility = assist_utility
        decision_probs = trace.p_assist
        control_source = "set_acsa"
    else:
        action = user_action
        utility = user_utility
        decision_probs = trace.p_user
        control_source = "user"
    return _decision(trace, f"set_acsa_tau_{tau:.2f}", action, decision_probs, tau, utility, decision_time_ms, control_source)


def agency_margin_policy(
    trace: PredictionTrace,
    tau: float,
    utility_gain_epsilon: float = 0.0,
    decision_time_ms: float = 0.0,
) -> PolicyDecision:
    """Choose the best assistive action allowed by the agency budget."""
    user_action = int(np.argmax(trace.p_user))
    user_utility = float(trace.p_user[user_action])
    allowed: list[tuple[int, float, float]] = []
    for action in range(trace.n_classes):
        ali = expected_ali(trace.p_user, action)
        margin = tau - ali
        if margin >= -1e-12:
            utility = float(trace.p_assist[action])
            allowed.append((action, utility, margin))
    if not allowed:
        return _decision(
            trace,
            f"agency_margin_tau_{tau:.2f}",
            user_action,
            trace.p_user,
            tau,
            user_utility,
            decision_time_ms,
            "user",
        )
    action, utility, margin = max(allowed, key=lambda item: (item[1], item[2]))
    if utility <= user_utility + utility_gain_epsilon:
        action = user_action
        utility = user_utility
        decision_probs = trace.p_user
        control_source = "user"
        coverage_confidence = float(trace.p_user[action])
    else:
        decision_probs = _masked_distribution(trace.p_assist, {item[0] for item in allowed}, action)
        control_source = "agency_margin"
        coverage_confidence = float(np.clip(trace.p_assist[action], 0.0, 1.0))
    return _decision(
        trace,
        f"agency_margin_tau_{tau:.2f}",
        action,
        decision_probs,
        tau,
        utility,
        decision_time_ms,
        control_source,
        metadata={"utility_gain_epsilon": utility_gain_epsilon, "selected_margin": margin},
        coverage_confidence=coverage_confidence,
    )


def _masked_distribution(probs: np.ndarray, allowed_actions: set[int], selected_action: int) -> np.ndarray:
    arr = np.asarray(probs, dtype=float).copy()
    mask = np.zeros_like(arr, dtype=bool)
    mask[list(allowed_actions)] = True
    arr[~mask] = 0.0
    if arr.sum() <= 0:
        arr[:] = 0.0
        arr[selected_action] = 1.0
    else:
        arr /= arr.sum()
    if int(np.argmax(arr)) != int(selected_action):
        arr[selected_action] = arr.max() + 1e-9
        arr /= arr.sum()
    return arr


def apply_rate_matched_confidence_gate(
    traces: Iterable[PredictionTrace],
    tau_grid: Iterable[float],
) -> tuple[list[PolicyDecision], list[dict[str, float]]]:
    trace_rows = list(traces)
    if not trace_rows:
        return [], []
    assist_conf = np.asarray([float(np.max(trace.p_assist)) for trace in trace_rows], dtype=float)
    candidate_thresholds = np.unique(np.concatenate([assist_conf, np.array([0.0, 1.01], dtype=float)]))
    candidate_rates = np.asarray([float(np.mean(assist_conf >= threshold)) for threshold in candidate_thresholds], dtype=float)
    decisions: list[PolicyDecision] = []
    summaries: list[dict[str, float]] = []
    for tau in tau_grid:
        tau_value = float(tau)
        agency_rate = float(
            np.mean(
                [
                    agency_margin_policy(
                        trace,
                        tau=tau_value,
                        decision_time_ms=trace_decision_time_ms(trace),
                    ).autonomy_used
                    for trace in trace_rows
                ]
            )
        )
        threshold = _match_intervention_rate_threshold(candidate_thresholds, candidate_rates, agency_rate)
        matched_rate = float(np.mean(assist_conf >= threshold))
        policy_name = f"plain_conf_threshold_matched_tau_{tau_value:.2f}"
        for trace in trace_rows:
            decisions.append(
                plain_confidence_threshold_policy(
                    trace,
                    threshold=threshold,
                    tau=tau_value,
                    decision_time_ms=trace_decision_time_ms(trace),
                    policy_name=policy_name,
                )
            )
        summaries.append(
            {
                "tau": tau_value,
                "threshold": float(threshold),
                "target_intervention_rate": agency_rate,
                "matched_intervention_rate": matched_rate,
                "intervention_rate_gap": abs(matched_rate - agency_rate),
            }
        )
    return decisions, summaries


def _match_intervention_rate_threshold(
    candidate_thresholds: np.ndarray,
    candidate_rates: np.ndarray,
    target_rate: float,
) -> float:
    order = np.lexsort(
        (
            -candidate_thresholds,
            (candidate_rates > target_rate).astype(int),
            np.abs(candidate_rates - target_rate),
        )
    )
    return float(candidate_thresholds[int(order[0])])


def select_plain_confidence_threshold_for_target_intervention_rate(
    traces: Iterable[PredictionTrace],
    *,
    target_rate: float,
    candidate_thresholds: np.ndarray | None = None,
) -> dict[str, float | int]:
    trace_rows = list(traces)
    if not trace_rows:
        raise ValueError("cannot select a confidence threshold from an empty trace set")
    assist_conf = np.asarray([float(np.max(trace.p_assist)) for trace in trace_rows], dtype=float)
    disagree_mask = np.asarray(
        [
            int(np.argmax(trace.p_assist)) != int(np.argmax(trace.p_user))
            for trace in trace_rows
        ],
        dtype=bool,
    )
    if candidate_thresholds is None:
        threshold_bank = np.unique(np.concatenate([assist_conf, np.array([0.0, 1.01], dtype=float)]))
    else:
        threshold_bank = np.unique(
            np.concatenate([np.asarray(candidate_thresholds, dtype=float), np.array([0.0, 1.01], dtype=float)])
        )
    candidate_rates = np.asarray(
        [
            float(np.mean((assist_conf >= threshold) & disagree_mask))
            for threshold in threshold_bank
        ],
        dtype=float,
    )
    threshold = _match_intervention_rate_threshold(
        threshold_bank,
        candidate_rates,
        float(target_rate),
    )
    matched_rate = float(np.mean((assist_conf >= threshold) & disagree_mask))
    return {
        "threshold": float(threshold),
        "matched_intervention_rate": matched_rate,
        "intervention_rate_gap": float(abs(matched_rate - float(target_rate))),
        "candidate_threshold_count": int(threshold_bank.size),
        "disagreement_trace_count": int(disagree_mask.sum()),
        "trace_count": int(len(trace_rows)),
        "max_attainable_intervention_rate": float(candidate_rates.max()) if candidate_rates.size else 0.0,
    }


def trace_decision_time_ms(trace: PredictionTrace) -> float:
    prefix_time_s = trace.metadata.get("prefix_time_s")
    if prefix_time_s is None:
        return float(trace.timestamp_s) * 1000.0
    return float(prefix_time_s) * 1000.0


def apply_policy_grid(
    traces: Iterable[PredictionTrace],
    tau_grid: Iterable[float],
    alpha_grid: Iterable[float] = (0.25, 0.50, 0.75),
    threshold_grid: Iterable[float] = (0.60, 0.70, 0.80),
) -> list[PolicyDecision]:
    decisions: list[PolicyDecision] = []
    for trace in traces:
        decision_time_ms = trace_decision_time_ms(trace)
        decisions.append(user_only(trace, decision_time_ms=decision_time_ms))
        decisions.append(assist_only(trace, decision_time_ms=decision_time_ms))
        for alpha in alpha_grid:
            decisions.append(confidence_blend(trace, alpha=alpha, decision_time_ms=decision_time_ms))
        for threshold in threshold_grid:
            decisions.append(
                selective_prediction(
                    trace,
                    user_confidence_threshold=threshold,
                    assist_confidence_threshold=threshold,
                    decision_time_ms=decision_time_ms,
                )
            )
        for tau in tau_grid:
            decisions.append(set_acsa(trace, tau=tau, decision_time_ms=decision_time_ms))
            decisions.append(agency_margin_policy(trace, tau=tau, decision_time_ms=decision_time_ms))
    return decisions

