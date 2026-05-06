from __future__ import annotations

import numpy as np
import pandas as pd

from myoagency.publication import (
    EXACT_BUDGET_PLAIN_CONF_POLICY_PREFIX,
    annotate_exact_budget_pairwise_diagnostics,
    build_dense_plain_confidence_threshold_bank,
    build_wang_db10_splits,
    compute_iso_budget_pairwise_stats,
    interpolate_metric_curve,
    materialize_exact_budget_plain_confidence_units,
    matched_policy_pairs,
    merge_budget_into_unit_frame,
    summarize_earliest_safe_aggregate,
    summarize_earliest_safe_by_episode,
    summarize_earliest_safe_by_unit,
)
from myoagency.schemas import PredictionTrace


def test_summarize_earliest_safe_prefers_stable_future_correct_prefix() -> None:
    decisions = pd.DataFrame(
        [
            {
                "dataset_id": "db10",
                "split_id": "db10_mixed_to_amputee|S1",
                "split_family": "db10_mixed_to_amputee",
                "policy": "agency_margin_tau_0.10",
                "subject_id": "S1",
                "session": "ex1",
                "day": None,
                "episode_id": "ep1",
                "prefix_time_s": 0.2,
                "timestamp_s": 0.2,
                "selected_action": 1,
                "label": 1,
                "expected_ali": 0.05,
                "tau": 0.10,
                "autonomy_used": True,
                "correct": True,
            },
            {
                "dataset_id": "db10",
                "split_id": "db10_mixed_to_amputee|S1",
                "split_family": "db10_mixed_to_amputee",
                "policy": "agency_margin_tau_0.10",
                "subject_id": "S1",
                "session": "ex1",
                "day": None,
                "episode_id": "ep1",
                "prefix_time_s": 0.4,
                "timestamp_s": 0.4,
                "selected_action": 2,
                "label": 2,
                "expected_ali": 0.04,
                "tau": 0.10,
                "autonomy_used": True,
                "correct": True,
            },
            {
                "dataset_id": "db10",
                "split_id": "db10_mixed_to_amputee|S1",
                "split_family": "db10_mixed_to_amputee",
                "policy": "agency_margin_tau_0.10",
                "subject_id": "S1",
                "session": "ex1",
                "day": None,
                "episode_id": "ep1",
                "prefix_time_s": 0.6,
                "timestamp_s": 0.6,
                "selected_action": 2,
                "label": 2,
                "expected_ali": 0.03,
                "tau": 0.10,
                "autonomy_used": True,
                "correct": True,
            },
        ]
    )

    episode_frame = summarize_earliest_safe_by_episode(decisions)
    assert np.isclose(episode_frame["earliest_safe_s"].iloc[0], 0.2)
    assert np.isclose(episode_frame["earliest_stable_safe_s"].iloc[0], 0.4)

    unit_frame = summarize_earliest_safe_by_unit(episode_frame)
    assert np.isclose(unit_frame["stable_safe_episode_rate"].iloc[0], 1.0)

    summary = summarize_earliest_safe_aggregate(unit_frame)
    assert np.isclose(summary["median_earliest_stable_safe_s"].iloc[0], 0.4)


def test_build_wang_db10_splits_creates_four_repetition_holds() -> None:
    metadata = pd.DataFrame(
        [
            {
                "episode_id": f"ep{rep}_{idx}",
                "group": "able_bodied",
                "label_raw": 1,
                "dynamic_flag": 0,
                "position_id": 1,
                "object_repetition": rep,
            }
            for rep in [5, 6, 7, 8]
            for idx in range(3)
        ]
    )
    splits = build_wang_db10_splits(metadata)
    assert len(splits) == 4
    held_out_sizes = sorted(len(split.test_index) for split in splits)
    assert held_out_sizes == [3, 3, 3, 3]


def test_matched_policy_pairs_matches_tau_suffixes() -> None:
    pairs = matched_policy_pairs(
        ["agency_margin_tau_0.10", "plain_conf_threshold_matched_tau_0.10", "agency_margin_tau_0.20"],
        left_prefix="agency_margin_tau_",
        right_prefix="plain_conf_threshold_matched_tau_",
        tau_grid=(0.10, 0.20),
    )
    assert pairs == [("agency_margin_tau_0.10", "plain_conf_threshold_matched_tau_0.10")]


def test_interpolate_metric_curve_averages_duplicate_budgets_and_marks_unsupported_range() -> None:
    curve = pd.DataFrame(
        [
            {"intervention_rate": 0.4, "active_macro_f1": 0.5},
            {"intervention_rate": 0.6, "active_macro_f1": 0.7},
            {"intervention_rate": 0.6, "active_macro_f1": 0.9},
        ]
    )
    interior = interpolate_metric_curve(
        curve,
        target_budget=0.5,
        metric="active_macro_f1",
        allow_extrapolation=False,
        min_points=2,
    )
    assert np.isclose(interior["value"], 0.65)
    assert np.isclose(interior["nearest_gap"], 0.1)
    assert interior["extrapolated"] is False
    assert interior["supported"] is True
    assert interior["n_points"] == 2

    outside = interpolate_metric_curve(
        curve,
        target_budget=0.75,
        metric="active_macro_f1",
        allow_extrapolation=False,
        min_points=2,
    )
    assert np.isnan(outside["value"])
    assert np.isclose(outside["nearest_gap"], 0.15)
    assert outside["extrapolated"] is True
    assert outside["supported"] is False


def test_compute_iso_budget_pairwise_stats_interpolates_within_support_and_drops_out_of_support() -> None:
    frame = pd.DataFrame(
        [
            {
                "dataset_id": "db10",
                "split_family": "db10_amputee_loso",
                "split_id": "fold1",
                "subject_id": "S1",
                "session": "day1",
                "day": None,
                "policy": "plain_conf_threshold_matched_tau_0.10",
                "intervention_rate": 0.4,
                "active_macro_f1": 0.5,
                "mean_ali": 0.08,
            },
            {
                "dataset_id": "db10",
                "split_family": "db10_amputee_loso",
                "split_id": "fold1",
                "subject_id": "S1",
                "session": "day1",
                "day": None,
                "policy": "plain_conf_threshold_matched_tau_0.20",
                "intervention_rate": 0.6,
                "active_macro_f1": 0.7,
                "mean_ali": 0.04,
            },
            {
                "dataset_id": "db10",
                "split_family": "db10_amputee_loso",
                "split_id": "fold1",
                "subject_id": "S1",
                "session": "day1",
                "day": None,
                "policy": "agency_margin_tau_0.10",
                "intervention_rate": 0.5,
                "active_macro_f1": 0.72,
                "mean_ali": 0.03,
            },
            {
                "dataset_id": "db10",
                "split_family": "db10_amputee_loso",
                "split_id": "fold1",
                "subject_id": "S1",
                "session": "day1",
                "day": None,
                "policy": "agency_margin_tau_0.20",
                "intervention_rate": 0.75,
                "active_macro_f1": 0.8,
                "mean_ali": 0.02,
            },
            {
                "dataset_id": "db10",
                "split_family": "db10_amputee_loso",
                "split_id": "fold1",
                "subject_id": "S2",
                "session": "day1",
                "day": None,
                "policy": "plain_conf_threshold_matched_tau_0.10",
                "intervention_rate": 0.2,
                "active_macro_f1": 0.3,
                "mean_ali": 0.09,
            },
            {
                "dataset_id": "db10",
                "split_family": "db10_amputee_loso",
                "split_id": "fold1",
                "subject_id": "S2",
                "session": "day1",
                "day": None,
                "policy": "plain_conf_threshold_matched_tau_0.20",
                "intervention_rate": 0.4,
                "active_macro_f1": 0.5,
                "mean_ali": 0.05,
            },
            {
                "dataset_id": "db10",
                "split_family": "db10_amputee_loso",
                "split_id": "fold1",
                "subject_id": "S2",
                "session": "day1",
                "day": None,
                "policy": "agency_margin_tau_0.10",
                "intervention_rate": 0.3,
                "active_macro_f1": 0.52,
                "mean_ali": 0.04,
            },
            {
                "dataset_id": "db10",
                "split_family": "db10_amputee_loso",
                "split_id": "fold1",
                "subject_id": "S2",
                "session": "day1",
                "day": None,
                "policy": "agency_margin_tau_0.20",
                "intervention_rate": 0.55,
                "active_macro_f1": 0.56,
                "mean_ali": 0.03,
            },
        ]
    )
    unit_deltas, pairwise = compute_iso_budget_pairwise_stats(
        frame,
        policy_pairs=[
            ("agency_margin_tau_0.10", "plain_conf_threshold_matched_tau_0.10"),
            ("agency_margin_tau_0.20", "plain_conf_threshold_matched_tau_0.20"),
        ],
        curve_policy_prefix="plain_conf_threshold_matched_tau_",
        metrics=("active_macro_f1", "mean_ali"),
        repeats=256,
        lower_is_better={"mean_ali"},
    )

    inside = unit_deltas.loc[
        (unit_deltas["tau"].astype(str) == "0.10")
        & (unit_deltas["metric"].astype(str) == "active_macro_f1")
    ].sort_values("subject_id")
    assert np.isclose(inside.iloc[0]["plain_conf_iso_budget_value"], 0.6)
    assert np.isclose(inside.iloc[1]["plain_conf_iso_budget_value"], 0.4)
    assert inside["curve_supported"].astype(bool).all()

    outside = unit_deltas.loc[
        (unit_deltas["tau"].astype(str) == "0.20")
        & (unit_deltas["metric"].astype(str) == "active_macro_f1")
    ]
    assert outside["curve_budget_extrapolated"].astype(bool).all()
    assert outside["curve_supported"].astype(bool).eq(False).all()
    assert outside["plain_conf_iso_budget_value"].isna().all()

    assert pairwise["tau"].tolist() == ["0.10", "0.10"]
    macro_row = pairwise.loc[pairwise["metric"].astype(str) == "active_macro_f1"].iloc[0]
    ali_row = pairwise.loc[pairwise["metric"].astype(str) == "mean_ali"].iloc[0]
    assert macro_row["unit_cols"] == "split_id|subject_id|session"
    assert int(macro_row["n_units"]) == 2
    assert int(macro_row["n_extrapolated_units"]) == 0
    assert np.isclose(macro_row["estimate"], 0.12)
    assert bool(ali_row["beneficial"]) is True
    assert np.isclose(ali_row["estimate"], -0.03)


def test_merge_budget_into_unit_frame_attaches_intervention_rate() -> None:
    unit_frame = pd.DataFrame(
        [
            {
                "dataset_id": "db10",
                "split_family": "db10_amputee_loso",
                "split_id": "fold1",
                "policy": "agency_margin_tau_0.10",
                "subject_id": "S1",
                "session": "ex1",
                "day": "day1",
                "stable_safe_episode_rate": 0.5,
            }
        ]
    )
    budget_frame = pd.DataFrame(
        [
            {
                "dataset_id": "db10",
                "split_family": "db10_amputee_loso",
                "split_id": "fold1",
                "policy": "agency_margin_tau_0.10",
                "subject_id": "S1",
                "session": "ex1",
                "day": "day1",
                "intervention_rate": 0.3,
            }
        ]
    )
    merged = merge_budget_into_unit_frame(unit_frame, budget_frame=budget_frame)
    assert np.isclose(merged["intervention_rate"].iloc[0], 0.3)


def test_merge_budget_into_unit_frame_normalizes_join_key_types() -> None:
    unit_frame = pd.DataFrame(
        [
            {
                "dataset_id": "db10",
                "split_family": "db10_mixed_to_amputee",
                "split_id": "fold1",
                "policy": "agency_margin_tau_0.10",
                "subject_id": "S1",
                "session": "ex1",
                "day": float("nan"),
                "stable_safe_episode_rate": 0.5,
            }
        ]
    )
    budget_frame = pd.DataFrame(
        [
            {
                "dataset_id": "db10",
                "split_family": "db10_mixed_to_amputee",
                "split_id": "fold1",
                "policy": "agency_margin_tau_0.10",
                "subject_id": "S1",
                "session": "ex1",
                "day": "<missing>",
                "intervention_rate": 0.4,
            }
        ]
    )
    merged = merge_budget_into_unit_frame(unit_frame, budget_frame=budget_frame)
    assert np.isclose(merged["intervention_rate"].iloc[0], 0.4)
    assert merged["day"].iloc[0] == "<missing>"


def test_build_dense_plain_confidence_threshold_bank_caps_size_and_keeps_bounds() -> None:
    traces = [
        PredictionTrace(
            dataset_id="db10",
            split_id="db10_mixed_to_amputee|fold1",
            subject_id="S1",
            session="ex1",
            day="day1",
            timestamp_s=float(index),
            label=1,
            p_user=np.asarray([0.7, 0.3]),
            p_assist=np.asarray([1.0 - value, value]),
            uncertainty=0.0,
            metadata={"episode_id": f"ep{index}", "prefix_time_s": float(index)},
        )
        for index, value in enumerate(np.linspace(0.05, 0.95, 20), start=1)
    ]
    bank = build_dense_plain_confidence_threshold_bank(traces, max_thresholds=5)
    assert bank[0] == 0.0
    assert bank[-1] == 1.01
    assert bank.size <= 7


def test_materialize_exact_budget_plain_confidence_units_uses_global_bank_and_adds_pairwise_diagnostics() -> None:
    traces = [
        PredictionTrace(
            dataset_id="db10",
            split_id="db10_mixed_to_amputee|fold1",
            subject_id="S1",
            session="ex1",
            day="day1",
            timestamp_s=0.1,
            label=1,
            p_user=np.asarray([0.8, 0.2]),
            p_assist=np.asarray([0.2, 0.8]),
            uncertainty=0.0,
            metadata={"episode_id": "ep1", "prefix_time_s": 0.1},
        ),
        PredictionTrace(
            dataset_id="db10",
            split_id="db10_mixed_to_amputee|fold1",
            subject_id="S1",
            session="ex1",
            day="day1",
            timestamp_s=0.2,
            label=0,
            p_user=np.asarray([0.9, 0.1]),
            p_assist=np.asarray([0.85, 0.15]),
            uncertainty=0.0,
            metadata={"episode_id": "ep1", "prefix_time_s": 0.2},
        ),
        PredictionTrace(
            dataset_id="db10",
            split_id="db10_mixed_to_amputee|fold2",
            subject_id="S2",
            session="ex1",
            day="day1",
            timestamp_s=0.1,
            label=1,
            p_user=np.asarray([0.75, 0.25]),
            p_assist=np.asarray([0.1, 0.9]),
            uncertainty=0.0,
            metadata={"episode_id": "ep2", "prefix_time_s": 0.1},
        ),
        PredictionTrace(
            dataset_id="db10",
            split_id="db10_mixed_to_amputee|fold2",
            subject_id="S2",
            session="ex1",
            day="day1",
            timestamp_s=0.2,
            label=0,
            p_user=np.asarray([0.85, 0.15]),
            p_assist=np.asarray([0.4, 0.6]),
            uncertainty=0.0,
            metadata={"episode_id": "ep2", "prefix_time_s": 0.2},
        ),
        PredictionTrace(
            dataset_id="db10",
            split_id="db10_mixed_to_amputee|fold2",
            subject_id="S2",
            session="ex1",
            day="day1",
            timestamp_s=0.3,
            label=0,
            p_user=np.asarray([0.9, 0.1]),
            p_assist=np.asarray([0.8, 0.2]),
            uncertainty=0.0,
            metadata={"episode_id": "ep2", "prefix_time_s": 0.3},
        ),
    ]
    budget_frame = pd.DataFrame(
        [
            {
                "dataset_id": "db10",
                "split_family": "db10_mixed_to_amputee",
                "split_id": "db10_mixed_to_amputee|fold1",
                "policy": "agency_margin_tau_0.10",
                "subject_id": "S1",
                "session": "ex1",
                "day": "day1",
                "intervention_rate": 0.5,
            },
            {
                "dataset_id": "db10",
                "split_family": "db10_mixed_to_amputee",
                "split_id": "db10_mixed_to_amputee|fold2",
                "policy": "agency_margin_tau_0.10",
                "subject_id": "S2",
                "session": "ex1",
                "day": "day1",
                "intervention_rate": 0.5,
            },
        ]
    )

    selection, metrics_frame, earliest_frame = materialize_exact_budget_plain_confidence_units(
        traces=traces,
        budget_frame=budget_frame,
        policy_pairs=[("agency_margin_tau_0.10", "plain_conf_threshold_matched_tau_0.10")],
        threshold_bank=np.asarray([0.0, 0.55, 0.75, 1.01]),
    )

    assert selection["policy_b"].astype(str).eq(
        f"{EXACT_BUDGET_PLAIN_CONF_POLICY_PREFIX}0.10"
    ).all()
    s1 = selection.loc[selection["subject_id"].astype(str) == "S1"].iloc[0]
    s2 = selection.loc[selection["subject_id"].astype(str) == "S2"].iloc[0]
    assert np.isclose(float(s1["selected_threshold"]), 0.75)
    assert bool(s1["exact_budget_match"]) is True
    assert bool(s2["exact_budget_match"]) is False
    assert metrics_frame["policy"].astype(str).eq(
        f"{EXACT_BUDGET_PLAIN_CONF_POLICY_PREFIX}0.10"
    ).all()
    assert not earliest_frame.empty

    pairwise = pd.DataFrame(
        [
            {
                "split_family": "db10_mixed_to_amputee",
                "comparison_family": "agency_vs_plain_conf_exact_budget",
                "metric": "active_macro_f1",
                "policy_a": "agency_margin_tau_0.10",
                "policy_b": f"{EXACT_BUDGET_PLAIN_CONF_POLICY_PREFIX}0.10",
                "estimate": 0.05,
                "beneficial": True,
            }
        ]
    )
    annotated = annotate_exact_budget_pairwise_diagnostics(
        pairwise,
        selection_frame=selection,
    )
    assert annotated["tau"].tolist() == ["0.10"]
    assert annotated["policy_b_same_tau"].tolist() == ["plain_conf_threshold_matched_tau_0.10"]
    assert np.isclose(float(annotated["exact_match_fraction"].iloc[0]), 0.5)

