from __future__ import annotations

import pytest

from j2bench.j1_selection import (
    J1_PRIMARY_COMPARATOR,
    J1_REQUIRED_COMPARATORS,
    J1Candidate,
    J1CandidateOutcome,
    assess_j1_candidate_promotion,
    freeze_j1_candidate_registry,
    generate_j1_challenger_manifest,
    promote_j1_winners,
    select_j1_tau,
)


def _candidate(
    candidate_id: str,
    *,
    lane: str = "canonical",
    allowed_taus: tuple[str, ...] = ("0.05", "0.10"),
    priority: int | None = None,
    evidence_tier: str = "",
    sensor_budget: str = "matched",
) -> J1Candidate:
    return J1Candidate(
        candidate_id=candidate_id,
        user_model=f"{candidate_id}_user",
        assist_model=f"{candidate_id}_assist",
        lane=lane,
        allowed_taus=allowed_taus,
        priority=priority,
        evidence_tier=evidence_tier,
        sensor_budget=sensor_budget,
    )


def test_freeze_registry_requires_primary_comparator() -> None:
    with pytest.raises(ValueError, match="primary comparator"):
        freeze_j1_candidate_registry(
            [_candidate("tabular_anchor")],
            comparator_policies=("user_only", "assist_only", "set_acsa", "agency_margin"),
            primary_comparator=J1_PRIMARY_COMPARATOR,
        )


def test_select_j1_tau_uses_frozen_candidate_order() -> None:
    candidate = _candidate("tabular_anchor", allowed_taus=("0.05", "0.10", "0.20"))

    selection = select_j1_tau(candidate, ("0.20", "0.10"))

    assert selection.status == "selected"
    assert selection.selected_tau == "0.10"
    assert selection.eligible_taus == ("0.10", "0.20")


def test_assess_j1_candidate_promotion_rejects_unfrozen_comparators() -> None:
    registry = freeze_j1_candidate_registry([_candidate("tabular_anchor")])
    outcome = J1CandidateOutcome(
        candidate_id="tabular_anchor",
        comparator_policies=("user_only", "assist_only", "agency_margin"),
        qualifying_taus=("0.05",),
        primary_gate_passed=True,
        safety_gate_passed=True,
    )

    assessment = assess_j1_candidate_promotion(registry, outcome)
    report = promote_j1_winners(registry, (outcome,))

    assert assessment.status == "comparator_mismatch"
    assert not assessment.promotable
    assert report.canonical.outcome == "no_winner"
    assert report.canonical.candidate_id is None


def test_promote_j1_winners_fails_closed_when_multiple_candidates_pass() -> None:
    registry = freeze_j1_candidate_registry(
        [
            _candidate("tabular_a"),
            _candidate("tabular_b"),
        ]
    )
    outcomes = (
        J1CandidateOutcome(
            candidate_id="tabular_a",
            comparator_policies=J1_REQUIRED_COMPARATORS,
            qualifying_taus=("0.05",),
            primary_gate_passed=True,
            safety_gate_passed=True,
        ),
        J1CandidateOutcome(
            candidate_id="tabular_b",
            comparator_policies=J1_REQUIRED_COMPARATORS,
            qualifying_taus=("0.05",),
            primary_gate_passed=True,
            safety_gate_passed=True,
        ),
    )

    report = promote_j1_winners(registry, outcomes)

    assert report.canonical.outcome == "no_winner"
    assert report.canonical.reason.startswith("multiple canonical candidates")


def test_generate_j1_challenger_manifest_carries_governance_and_winner() -> None:
    registry = freeze_j1_candidate_registry(
        [
            _candidate("tabular_anchor"),
            _candidate(
                "sequence_edge",
                lane="challenger",
                allowed_taus=("0.10", "0.20"),
                priority=0,
                evidence_tier="changed_input_budget",
                sensor_budget="sequence_plus_context",
            ),
        ]
    )
    outcomes = (
        J1CandidateOutcome(
            candidate_id="tabular_anchor",
            comparator_policies=J1_REQUIRED_COMPARATORS,
            qualifying_taus=(),
            primary_gate_passed=False,
            safety_gate_passed=True,
        ),
        J1CandidateOutcome(
            candidate_id="sequence_edge",
            comparator_policies=J1_REQUIRED_COMPARATORS,
            qualifying_taus=("0.20", "0.10"),
            primary_gate_passed=True,
            safety_gate_passed=True,
        ),
    )

    report = promote_j1_winners(registry, outcomes)
    manifest = generate_j1_challenger_manifest(registry, outcomes, promotion_report=report)

    assert report.challenger.outcome == "promoted"
    assert report.challenger.candidate_id == "sequence_edge"
    assert report.challenger.selected_tau == "0.10"
    assert manifest["comparator_policies"] == list(J1_REQUIRED_COMPARATORS)
    assert manifest["primary_comparator"] == J1_PRIMARY_COMPARATOR
    assert manifest["winner"]["candidate_id"] == "sequence_edge"
    assert manifest["winner"]["selected_tau"] == "0.10"
    assert manifest["candidates"][0]["evidence_tier"] == "changed_input_budget"
    assert manifest["candidates"][0]["sensor_budget"] == "sequence_plus_context"
    assert manifest["candidates"][0]["assessment"]["status"] == "promotable"

