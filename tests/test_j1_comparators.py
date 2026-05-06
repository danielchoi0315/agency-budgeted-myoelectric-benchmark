from __future__ import annotations

import pytest

from j2bench.j1_comparators import (
    J1_PRIMARY_COMPARATOR,
    J1_REQUIRED_COMPARATORS,
    freeze_j1_primary_comparator_set,
    freeze_matched_confidence_tau_manifest,
    matches_frozen_j1_comparator_set,
    select_j1_primary_comparator,
)


def _matched_row(
    tau: str,
    *,
    threshold: float = 0.75,
    target_rate: float = 0.30,
    matched_rate: float = 0.31,
    gap: float | None = None,
) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "tau": tau,
        "threshold": threshold,
        "target_intervention_rate": target_rate,
        "matched_intervention_rate": matched_rate,
    }
    if gap is not None:
        row["intervention_rate_gap"] = gap
    return row


def test_freeze_primary_comparator_requires_explicit_tau_manifest() -> None:
    with pytest.raises(ValueError, match="explicit tau manifest"):
        freeze_j1_primary_comparator_set(
            [],
            intervention_rate_tolerance=0.02,
        )


def test_freeze_matched_confidence_tau_manifest_normalizes_tau_rows() -> None:
    tau_order, manifest = freeze_matched_confidence_tau_manifest(
        [
            _matched_row("tau_0.1", threshold=0.72, target_rate=0.30, matched_rate=0.31),
            _matched_row("0.20", threshold=0.84, target_rate=0.18, matched_rate=0.18),
        ]
    )

    assert tau_order == ("0.10", "0.20")
    assert manifest["0.10"].policy_name == "plain_conf_threshold_matched_tau_0.10"
    assert manifest["0.10"].intervention_rate_gap == pytest.approx(0.01)
    assert manifest["0.20"].threshold == pytest.approx(0.84)


def test_freeze_primary_comparator_rejects_non_primary_family() -> None:
    with pytest.raises(ValueError, match="primary comparator must be the matched"):
        freeze_j1_primary_comparator_set(
            [_matched_row("0.10")],
            primary_comparator="set_acsa",
            intervention_rate_tolerance=0.02,
        )


def test_select_primary_comparator_uses_frozen_tau_manifest_when_within_tolerance() -> None:
    comparator_lock = freeze_j1_primary_comparator_set(
        [
            _matched_row("0.05", threshold=0.68, target_rate=0.34, matched_rate=0.33),
            _matched_row("0.10", threshold=0.74, target_rate=0.30, matched_rate=0.31),
        ],
        intervention_rate_tolerance=0.02,
    )

    selection = select_j1_primary_comparator(
        comparator_lock,
        tau="tau_0.10",
        comparator_policies=J1_REQUIRED_COMPARATORS,
    )

    assert selection.status == "selected"
    assert selection.comparator_policy == "plain_conf_threshold_matched_tau_0.10"
    assert selection.threshold == pytest.approx(0.74)
    assert selection.intervention_rate_gap == pytest.approx(0.01)
    assert selection.reason.startswith("selected the frozen matched confidence-threshold comparator")


def test_select_primary_comparator_fails_closed_for_comparator_set_mismatch() -> None:
    comparator_lock = freeze_j1_primary_comparator_set(
        [_matched_row("0.10")],
        intervention_rate_tolerance=0.02,
    )

    selection = select_j1_primary_comparator(
        comparator_lock,
        tau="0.10",
        comparator_policies=(
            "user_only",
            "assist_only",
            "agency_margin",
            J1_PRIMARY_COMPARATOR,
        ),
    )

    assert not matches_frozen_j1_comparator_set(
        comparator_lock,
        ("user_only", "assist_only", "agency_margin", J1_PRIMARY_COMPARATOR),
    )
    assert selection.status == "no_selection"
    assert selection.comparator_policy is None
    assert "does not match the frozen J1 comparator lock" in selection.reason


def test_select_primary_comparator_fails_closed_when_tau_is_missing() -> None:
    comparator_lock = freeze_j1_primary_comparator_set(
        [_matched_row("0.10")],
        intervention_rate_tolerance=0.02,
    )

    selection = select_j1_primary_comparator(
        comparator_lock,
        tau="0.20",
        comparator_policies=J1_REQUIRED_COMPARATORS,
    )

    assert selection.status == "no_selection"
    assert selection.comparator_policy is None
    assert "absent from the frozen primary comparator manifest" in selection.reason


def test_select_primary_comparator_fails_closed_when_rate_gap_exceeds_tolerance() -> None:
    comparator_lock = freeze_j1_primary_comparator_set(
        [_matched_row("0.10", target_rate=0.30, matched_rate=0.34)],
        intervention_rate_tolerance=0.02,
    )

    selection = select_j1_primary_comparator(
        comparator_lock,
        tau="0.10",
        comparator_policies=J1_REQUIRED_COMPARATORS,
    )

    assert selection.status == "no_selection"
    assert selection.comparator_policy == "plain_conf_threshold_matched_tau_0.10"
    assert selection.intervention_rate_gap == pytest.approx(0.04)
    assert "exceeds the declared intervention-rate tolerance" in selection.reason

