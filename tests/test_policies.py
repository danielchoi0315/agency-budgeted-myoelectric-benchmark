import numpy as np

from myoagency.policies import (
    agency_margin_policy,
    apply_rate_matched_confidence_gate,
    expected_ali,
    plain_confidence_threshold_policy,
    set_acsa,
)
from myoagency.schemas import PredictionTrace


def make_trace(p_user, p_assist, label=1, *, episode_id="ep0", prefix_time_s=0.2):
    return PredictionTrace(
        dataset_id="synthetic",
        split_id="s0",
        subject_id="S01",
        session="session_1",
        day="day_1",
        timestamp_s=0.0,
        label=label,
        p_user=np.asarray(p_user, dtype=float),
        p_assist=np.asarray(p_assist, dtype=float),
        uncertainty=0.1,
        metadata={"episode_id": episode_id, "prefix_time_s": prefix_time_s},
    )


def test_expected_ali_is_zero_for_user_argmax():
    assert expected_ali(np.array([0.8, 0.1, 0.1]), 0) == 0.0


def test_expected_ali_penalizes_deviation_from_user_intent():
    assert np.isclose(expected_ali(np.array([0.8, 0.1, 0.1]), 1), 0.7)


def test_set_acsa_blocks_assist_action_outside_tau():
    trace = make_trace([0.8, 0.1, 0.1], [0.1, 0.85, 0.05], label=1)
    decision = set_acsa(trace, tau=0.2)
    assert decision.selected_action == 0
    assert decision.agency_margin >= 0


def test_agency_margin_policy_intervenes_when_inside_budget():
    trace = make_trace([0.45, 0.40, 0.15], [0.10, 0.80, 0.10], label=1)
    decision = agency_margin_policy(trace, tau=0.10)
    assert decision.selected_action == 1
    assert decision.intervene
    assert decision.agency_margin >= 0


def test_agency_margin_masks_disallowed_assist_argmax():
    trace = make_trace([0.45, 0.40, 0.15], [0.01, 0.49, 0.50], label=1)
    decision = agency_margin_policy(trace, tau=0.10)
    assert decision.selected_action == 1
    assert np.argmax(decision.decision_probs) == decision.selected_action
    assert decision.decision_probs[2] == 0.0
    assert np.isclose(decision.selected_confidence, decision.decision_probs[decision.selected_action])


def test_set_acsa_does_not_count_agreement_as_intervention():
    trace = make_trace([0.70, 0.20, 0.10], [0.75, 0.15, 0.10], label=0)
    decision = set_acsa(trace, tau=0.20)
    assert decision.selected_action == decision.user_action
    assert not decision.intervene
    assert not decision.autonomy_used


def test_plain_confidence_threshold_policy_uses_assist_when_threshold_passed():
    trace = make_trace([0.55, 0.35, 0.10], [0.05, 0.90, 0.05], label=1)
    decision = plain_confidence_threshold_policy(trace, threshold=0.80, tau=0.20)
    assert decision.selected_action == 1
    assert decision.intervene
    assert decision.coverage_confidence == 0.90


def test_decision_preserves_trace_metadata():
    trace = make_trace([0.55, 0.35, 0.10], [0.05, 0.90, 0.05], label=1, episode_id="keep_me", prefix_time_s=0.4)
    decision = agency_margin_policy(trace, tau=0.20)
    assert decision.metadata["episode_id"] == "keep_me"
    assert np.isclose(decision.metadata["prefix_time_s"], 0.4)


def test_apply_rate_matched_confidence_gate_returns_tau_named_policies():
    traces = [
        make_trace([0.60, 0.30, 0.10], [0.55, 0.35, 0.10], label=0, episode_id="ep1"),
        make_trace([0.42, 0.38, 0.20], [0.10, 0.80, 0.10], label=1, episode_id="ep2"),
        make_trace([0.40, 0.35, 0.25], [0.20, 0.30, 0.50], label=2, episode_id="ep3"),
    ]
    decisions, rows = apply_rate_matched_confidence_gate(traces, tau_grid=(0.10, 0.20))
    assert len(rows) == 2
    assert {row["tau"] for row in rows} == {0.10, 0.20}
    assert all("plain_conf_threshold_matched_tau_" in decision.policy for decision in decisions)

