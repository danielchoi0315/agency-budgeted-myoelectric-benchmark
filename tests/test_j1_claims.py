from __future__ import annotations

from j2bench.j1_claims import evaluate_j1_claim_governance, load_j1_claims


def _passing_evidence() -> dict[str, object]:
    return {
        "primary_splits": {
            "j1_amputee_loso": {
                "comparison_family": "agency_vs_plain_conf",
                "baseline_policy": "plain_conf_threshold_matched_tau_0.10",
                "metrics": {
                    "active_macro_f1": {"ci_lower": 0.01},
                    "stable_safe_episode_rate": {"delta": 0.03},
                    "earliest_safe_time_s": {"delta": -0.08},
                    "intervention_rate_gap": {"abs": 0.01},
                },
            },
            "j1_mixed_to_amputee": {
                "comparison_family": "agency_vs_plain_conf",
                "baseline_policy": "plain_conf_threshold_matched_tau_0.10",
                "metrics": {
                    "active_macro_f1": {"ci_lower": 0.02},
                    "stable_safe_episode_rate": {"delta": 0.05},
                    "earliest_safe_time_s": {"delta": -0.04},
                    "intervention_rate_gap": {"abs": 0.02},
                },
            },
        },
        "supporting_datasets": {
            "hyser": {
                "metrics": {
                    "active_macro_f1": {"delta": 0.03},
                    "active_risk_coverage_auc": {"delta": 0.02},
                    "stable_safe_episode_rate": {"delta": 0.01},
                }
            },
            "cemhsey": {
                "metrics": {
                    "active_macro_f1": {"delta": 0.02},
                    "active_risk_coverage_auc": {"delta": 0.01},
                    "stable_safe_episode_rate": {"delta": 0.02},
                }
            },
        },
    }


def test_load_j1_claims_exposes_machine_readable_classes() -> None:
    config = load_j1_claims()

    assert config["program_id"] == "j1_open"
    assert config["claim_surface"] == "benchmark_only"
    assert config["split_families"]["primary"] == [
        "j1_amputee_loso",
        "j1_mixed_to_amputee",
    ]
    assert config["claim_classes"]["matched_budget_performance"]["allowed"] is True
    assert config["claim_classes"]["translational"]["allowed"] is False
    assert "clinically validated" in config["claim_classes"]["translational"]["disallowed_terms"]
    assert "trust" in config["claim_classes"]["human_factors"]["disallowed_terms"]


def test_evaluate_j1_claim_governance_supports_benchmark_safe_claims() -> None:
    result = evaluate_j1_claim_governance(
        _passing_evidence(),
        requested_claim_classes=[
            "matched_budget_performance",
            "safe_intervention_timing",
            "robustness_under_shift",
        ],
    )

    assert result["overall_status"] == "GO"
    assert result["gate_results"]["primary_split_success"]["status"] == "pass"
    assert result["gate_results"]["comparator_support"]["status"] == "pass"
    assert result["gate_results"]["robustness_direction"]["status"] == "pass"
    assert result["claim_results"]["matched_budget_performance"]["status"] == "allowed"
    assert result["claim_results"]["safe_intervention_timing"]["status"] == "allowed"
    assert result["claim_results"]["robustness_under_shift"]["status"] == "allowed"


def test_evaluate_j1_claim_governance_blocks_primary_claim_without_plain_conf_support() -> None:
    evidence = _passing_evidence()
    evidence["primary_splits"]["j1_mixed_to_amputee"]["comparison_family"] = "agency_vs_set_acsa"
    evidence["primary_splits"]["j1_mixed_to_amputee"]["baseline_policy"] = "set_acsa_tau_0.10"

    result = evaluate_j1_claim_governance(
        evidence,
        requested_claim_classes=["matched_budget_performance"],
    )

    assert result["overall_status"] == "HOLD"
    assert result["gate_results"]["comparator_support"]["status"] == "fail"
    assert "j1_mixed_to_amputee" in result["gate_results"]["comparator_support"]["failing_splits"]
    assert result["claim_results"]["matched_budget_performance"]["status"] == "blocked"


def test_evaluate_j1_claim_governance_requires_two_supporting_datasets_in_direction() -> None:
    evidence = _passing_evidence()
    evidence["supporting_datasets"]["cemhsey"]["metrics"]["active_risk_coverage_auc"]["delta"] = -0.01

    result = evaluate_j1_claim_governance(
        evidence,
        requested_claim_classes=["robustness_under_shift"],
    )

    assert result["overall_status"] == "HOLD"
    assert result["gate_results"]["robustness_direction"]["status"] == "fail"
    assert result["gate_results"]["robustness_direction"]["passing_datasets"] == ["hyser"]
    assert "cemhsey" in result["gate_results"]["robustness_direction"]["failing_datasets"]
    assert result["claim_results"]["robustness_under_shift"]["status"] == "blocked"


def test_evaluate_j1_claim_governance_rejects_translational_and_human_factor_claims() -> None:
    result = evaluate_j1_claim_governance(
        _passing_evidence(),
        requested_claim_classes=["translational", "human_factors"],
    )

    assert result["overall_status"] == "NO_GO"
    assert result["claim_results"]["translational"]["status"] == "disallowed"
    assert "deployment ready" in result["claim_results"]["translational"]["disallowed_terms"]
    assert result["claim_results"]["human_factors"]["status"] == "disallowed"
    assert "workload" in result["claim_results"]["human_factors"]["disallowed_terms"]

