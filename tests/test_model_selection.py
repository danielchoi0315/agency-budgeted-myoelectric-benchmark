from __future__ import annotations

import pandas as pd

from myoagency.model_selection import default_db10_candidate_pairs, default_real_candidate_pairs, evaluate_candidate, evaluate_db10_candidate


def test_evaluate_db10_candidate_finds_qualifying_tau() -> None:
    rows = []
    for split_family in ["db10_able_to_amputee", "db10_amputee_loso", "db10_mixed_to_amputee"]:
        for subject_index in range(1, 9):
            subject_id = f"S{subject_index}"
            agency_f1 = 0.75 + (subject_index * 0.002)
            set_f1 = 0.55 + (subject_index * 0.001)
            agency_risk = 0.10 + (subject_index * 0.001)
            set_risk = 0.23 + (subject_index * 0.001)
            rows.extend(
                [
                    {
                        "dataset_id": "db10",
                        "split_id": f"{split_family}|{subject_id}",
                        "split_family": split_family,
                        "policy": "agency_margin_tau_0.05",
                        "subject_id": subject_id,
                        "session": "ex1",
                        "day": "<missing>",
                        "active_macro_f1": agency_f1,
                        "active_risk_coverage_auc": agency_risk,
                    },
                    {
                        "dataset_id": "db10",
                        "split_id": f"{split_family}|{subject_id}",
                        "split_family": split_family,
                        "policy": "set_acsa_tau_0.05",
                        "subject_id": subject_id,
                        "session": "ex1",
                        "day": "<missing>",
                        "active_macro_f1": set_f1,
                        "active_risk_coverage_auc": set_risk,
                    },
                ]
            )
    metrics_df = pd.DataFrame(rows)
    aggregate_df = pd.DataFrame(
        [
            {
                "split_family": split_family,
                "policy": "agency_margin_tau_0.05",
                "active_macro_f1": 0.70,
                "active_risk_coverage_auc": 0.11,
            }
            for split_family in ["db10_able_to_amputee", "db10_amputee_loso", "db10_mixed_to_amputee"]
        ]
    )
    summary, pairwise = evaluate_db10_candidate(metrics_df, aggregate_df, user_model="logistic", assist_model="torch_mlp", repeats=256)
    assert "0.05" in summary.qualifying_taus
    assert summary.all_families_passed
    assert pairwise["beneficial"].all()


def test_default_db10_candidates_include_new_assist_fusion_lane_when_available() -> None:
    pairs = default_db10_candidate_pairs(lanes=("sota",))
    assert ("logistic", "assist_hybrid_fusion") in pairs
    assert ("logistic", "assist_temporal_forest") in pairs
    assert ("logistic", "assist_stacked_fusion") in pairs


def test_default_real_candidates_stay_tabular_without_sequence_payloads() -> None:
    pairs = default_real_candidate_pairs(available_sequence_keys=set())
    assert ("logistic", "logistic") in pairs
    assert ("torch_mlp", "torch_mlp") in pairs
    assert ("temporal_cnn", "torch_mlp") not in pairs
    assert ("logistic", "assist_temporal_forest") not in pairs


def test_default_real_candidates_admit_user_sequence_models_when_payloads_exist() -> None:
    pairs = default_real_candidate_pairs(available_sequence_keys={"user_emg", "user_time_mask", "assist_emg", "assist_time_mask"})
    assert ("temporal_cnn", "torch_mlp") in pairs
    assert ("hybrid_temporal_mlp", "torch_mlp") in pairs
    assert ("logistic", "assist_temporal_forest") not in pairs


def test_evaluate_candidate_works_for_non_db10_split_families() -> None:
    rows = []
    for split_family in ["hyser_subject_logo", "hyser_within_subject_dayshift"]:
        for subject_index in range(1, 11):
            subject_id = f"S{subject_index}"
            rows.extend(
                [
                    {
                        "dataset_id": "hyser",
                        "split_id": f"{split_family}|{subject_id}",
                        "split_family": split_family,
                        "policy": "agency_margin_tau_0.10",
                        "subject_id": subject_id,
                        "session": "session1",
                        "day": "day1",
                        "active_macro_f1": 0.70 + (subject_index * 0.003),
                        "active_risk_coverage_auc": 0.14 + (subject_index * 0.001),
                    },
                    {
                        "dataset_id": "hyser",
                        "split_id": f"{split_family}|{subject_id}",
                        "split_family": split_family,
                        "policy": "set_acsa_tau_0.10",
                        "subject_id": subject_id,
                        "session": "session1",
                        "day": "day1",
                        "active_macro_f1": 0.50 + (subject_index * 0.001),
                        "active_risk_coverage_auc": 0.26 + (subject_index * 0.001),
                    },
                ]
            )
    metrics_df = pd.DataFrame(rows)
    aggregate_df = pd.DataFrame(
        [
            {
                "split_family": split_family,
                "policy": "agency_margin_tau_0.10",
                "active_macro_f1": 0.64,
                "active_risk_coverage_auc": 0.17,
            }
            for split_family in ["hyser_subject_logo", "hyser_within_subject_dayshift"]
        ]
    )
    summary, pairwise = evaluate_candidate(
        metrics_df,
        aggregate_df,
        user_model="torch_mlp",
        assist_model="torch_mlp",
        repeats=256,
    )
    assert "0.10" in summary.qualifying_taus
    assert summary.split_family_count == 2
    assert pairwise["beneficial"].all()

