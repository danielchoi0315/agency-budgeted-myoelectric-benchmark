from __future__ import annotations

import pytest

from j2bench.literature import (
    build_comparability_rows,
    derive_comparison_classification,
    evaluate_target,
    summarize_comparability_matrix,
    summarize_overall_status,
)


def test_derive_comparison_classification_detects_partial_vs_boundary() -> None:
    partial = derive_comparison_classification(
        {
            "role": "comparator",
            "criteria": {
                "dataset": "exact",
                "task_family": "related",
                "split_shift": "different",
                "population": "related",
                "adaptation": "exact",
                "decision_object": "different",
                "metric": "different",
            },
        }
    )
    boundary = derive_comparison_classification({"role": "boundary", "criteria": {}})

    assert partial == "partial"
    assert boundary == "boundary"


def test_build_comparability_rows_and_summary() -> None:
    cfg = {
        "papers": [{"paper_id": "paper_a", "citation": "Paper A"}],
        "comparisons": [
            {
                "paper_id": "paper_a",
                "target_id": "db10_cross_population_policy",
                "role": "comparator",
                "criteria": {
                    "dataset": "exact",
                    "task_family": "related",
                    "split_shift": "different",
                    "population": "related",
                    "adaptation": "exact",
                    "decision_object": "different",
                    "metric": "different",
                },
            }
        ],
    }
    rows = build_comparability_rows(
        cfg,
        targets=[
            {
                "target_id": "db10_cross_population_policy",
                "dataset_id": "db10",
                "split_family": "db10_mixed_to_amputee",
                "priority": "primary",
            }
        ],
    )
    summary = summarize_comparability_matrix(rows)

    assert rows[0]["derived_comparability"] == "partial"
    assert rows[0]["target_split_family"] == "db10_mixed_to_amputee"
    assert rows[0]["target_direct_sota_eligible"] is False
    assert summary == [
        {
            "target_id": "db10_cross_population_policy",
            "target_dataset_id": "db10",
            "target_split_family": "db10_mixed_to_amputee",
            "target_priority": "primary",
            "target_claim_surface": "benchmark_only",
            "target_direct_sota_eligible": False,
            "n_direct": 0,
            "n_partial": 1,
            "n_not_comparable": 0,
            "n_contextual": 0,
            "n_boundary": 0,
        }
    ]


def test_evaluate_target_reports_missing_direct_anchor() -> None:
    target = {
        "target_id": "db10_cross_population_policy",
        "dataset_id": "db10",
        "split_family": "db10_mixed_to_amputee",
        "local_metric_field": "best_agency_active_macro_f1",
        "baseline_metric_field": "user_only_active_macro_f1",
        "policy_field": "best_agency_policy",
        "priority": "primary",
    }
    observation = {
        "dataset_id": "db10",
        "split_family": "db10_mixed_to_amputee",
        "best_agency_active_macro_f1": 0.69,
        "user_only_active_macro_f1": 0.08,
        "best_agency_policy": "agency_margin_tau_0.20",
    }
    anchors = [
        {
            "anchor_id": "db10_context_only",
            "comparability": "partial",
            "targets": ["db10_cross_population_policy"],
            "metric_value": 0.95,
        }
    ]

    result = evaluate_target(target=target, observation_row=observation, anchors=anchors)

    assert result["status"] == "not_established_no_direct_anchor"
    assert result["claim_ready"] is False
    assert result["baseline_delta"] == pytest.approx(0.61)


def test_evaluate_target_reports_candidate_direct_sota() -> None:
    target = {
        "target_id": "hyser_cross_day_policy",
        "dataset_id": "hyser",
        "split_family": "hyser_within_subject_dayshift",
        "local_metric_field": "best_agency_active_macro_f1",
        "baseline_metric_field": "user_only_active_macro_f1",
        "policy_field": "best_agency_policy",
        "priority": "primary",
        "direct_sota_eligible": True,
        "statistical_support_ready": True,
    }
    observation = {
        "dataset_id": "hyser",
        "split_family": "hyser_within_subject_dayshift",
        "best_agency_active_macro_f1": 0.44,
        "user_only_active_macro_f1": 0.38,
        "best_agency_policy": "agency_margin_tau_0.20",
    }
    anchors = [
        {
            "anchor_id": "hyser_direct_anchor",
            "comparability": "direct",
            "targets": ["hyser_cross_day_policy"],
            "metric_value": 0.41,
        }
    ]

    result = evaluate_target(target=target, observation_row=observation, anchors=anchors)

    assert result["status"] == "candidate_direct_sota"
    assert result["claim_ready"] is True
    assert result["best_direct_anchor_id"] == "hyser_direct_anchor"


def test_evaluate_target_blocks_direct_win_for_benchmark_only_target() -> None:
    target = {
        "target_id": "hyser_cross_day_policy",
        "dataset_id": "hyser",
        "split_family": "hyser_within_subject_dayshift",
        "local_metric_field": "best_agency_active_macro_f1",
        "priority": "primary",
        "direct_sota_eligible": False,
        "statistical_support_ready": True,
    }
    observation = {"best_agency_active_macro_f1": 0.44}
    anchors = [
        {
            "anchor_id": "hyser_direct_anchor",
            "comparability": "direct",
            "targets": ["hyser_cross_day_policy"],
            "metric_value": 0.41,
        }
    ]

    result = evaluate_target(target=target, observation_row=observation, anchors=anchors)

    assert result["status"] == "candidate_direct_numeric_win_benchmark_only"
    assert result["claim_ready"] is False


def test_evaluate_target_uses_comparability_matrix_for_direct_anchor_counts() -> None:
    target = {
        "target_id": "hyser_cross_day_policy",
        "dataset_id": "hyser",
        "split_family": "hyser_within_subject_dayshift",
        "local_metric_field": "best_agency_active_macro_f1",
        "baseline_metric_field": "user_only_active_macro_f1",
        "policy_field": "best_agency_policy",
        "priority": "primary",
    }
    observation = {
        "dataset_id": "hyser",
        "split_family": "hyser_within_subject_dayshift",
        "best_agency_active_macro_f1": 0.44,
        "user_only_active_macro_f1": 0.38,
        "best_agency_policy": "agency_margin_tau_0.20",
    }
    anchors = [
        {
            "anchor_id": "hyser_direct_anchor",
            "comparability": "direct",
            "targets": ["hyser_cross_day_policy"],
            "metric_value": 0.41,
        }
    ]
    comparison_rows = [
        {
            "target_id": "hyser_cross_day_policy",
            "derived_comparability": "partial",
            "anchor_id": "hyser_direct_anchor",
        }
    ]

    result = evaluate_target(target=target, observation_row=observation, anchors=anchors, comparison_rows=comparison_rows)

    assert result["direct_anchor_count"] == 0
    assert result["partial_anchor_count"] == 1
    assert result["status"] == "not_established_no_direct_anchor"


def test_evaluate_target_reports_missing_local_result() -> None:
    target = {
        "target_id": "cemhsey_longitudinal_policy",
        "dataset_id": "cemhsey",
        "split_family": "cemhsey_forward_day_logo",
        "local_metric_field": "best_agency_active_macro_f1",
        "priority": "primary",
    }

    result = evaluate_target(target=target, observation_row=None, anchors=[])

    assert result["status"] == "missing_local_result"
    assert result["claim_ready"] is False


def test_evaluate_target_reports_direct_anchor_not_beaten() -> None:
    target = {
        "target_id": "hyser_cross_day_policy",
        "dataset_id": "hyser",
        "split_family": "hyser_within_subject_dayshift",
        "local_metric_field": "best_agency_active_macro_f1",
        "priority": "primary",
        "statistical_support_ready": True,
    }
    observation = {"best_agency_active_macro_f1": 0.40}
    anchors = [
        {
            "anchor_id": "hyser_direct_anchor",
            "comparability": "direct",
            "targets": ["hyser_cross_day_policy"],
            "metric_value": 0.41,
        }
    ]

    result = evaluate_target(target=target, observation_row=observation, anchors=anchors)

    assert result["status"] == "not_established_direct_anchor_not_beaten"
    assert result["claim_ready"] is False


def test_evaluate_target_requires_statistical_support_for_sota_release() -> None:
    target = {
        "target_id": "hyser_cross_day_policy",
        "dataset_id": "hyser",
        "split_family": "hyser_within_subject_dayshift",
        "local_metric_field": "best_agency_active_macro_f1",
        "priority": "primary",
        "direct_sota_eligible": True,
    }
    observation = {"best_agency_active_macro_f1": 0.44}
    anchors = [
        {
            "anchor_id": "hyser_direct_anchor",
            "comparability": "direct",
            "targets": ["hyser_cross_day_policy"],
            "metric_value": 0.41,
        }
    ]

    result = evaluate_target(target=target, observation_row=observation, anchors=anchors)

    assert result["status"] == "candidate_direct_numeric_win_pending_stats"
    assert result["claim_ready"] is False


def test_summarize_overall_status_blocks_sota_without_direct_support() -> None:
    summary = summarize_overall_status(
        [
            {
                "target_id": "db10_cross_population_policy",
                "priority": "primary",
                "direct_sota_eligible": False,
                "status": "not_established_no_direct_anchor",
                "claim_ready": False,
            }
        ],
        anchor_status="provisionally_ready",
        guardrails={"disallowed_terms_without_direct_support": ["SOTA"]},
    )

    assert summary["status"] == "claim_safe_benchmark_only"
    assert summary["disallowed_terms"] == ["SOTA"]


def test_summarize_overall_status_blocks_incomplete_primary_evidence() -> None:
    summary = summarize_overall_status(
        [{"target_id": "db10_cross_population_policy", "priority": "primary", "direct_sota_eligible": False, "status": "missing_local_result", "claim_ready": False}],
        anchor_status="provisionally_ready",
    )

    assert summary["status"] == "blocked_primary_evidence_incomplete"


def test_summarize_overall_status_ignores_supporting_target_failures_for_primary_gate() -> None:
    summary = summarize_overall_status(
        [
            {"target_id": "db10_cross_population_policy", "priority": "primary", "direct_sota_eligible": False, "status": "not_established_no_direct_anchor", "claim_ready": False},
            {"target_id": "hyser_cross_day_policy", "priority": "supporting", "status": "missing_local_result", "claim_ready": False},
        ],
        anchor_status="provisionally_ready",
    )

    assert summary["status"] == "claim_safe_benchmark_only"

