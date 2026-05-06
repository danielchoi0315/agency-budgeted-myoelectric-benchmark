from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from j2bench.stats import (
    annotate_holm_bonferroni,
    holm_bonferroni,
    paired_delta_test,
    paired_policy_difference_test,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from postprocess_real_stats import infer_unit_cols, resolve_metrics_paths, should_include_discovered_metrics_path


def test_paired_policy_difference_test() -> None:
    data = pd.DataFrame(
        [
            {"subject_id": "S1", "policy": "agency", "macro_f1": 0.60},
            {"subject_id": "S1", "policy": "baseline", "macro_f1": 0.50},
            {"subject_id": "S2", "policy": "agency", "macro_f1": 0.55},
            {"subject_id": "S2", "policy": "baseline", "macro_f1": 0.45},
            {"subject_id": "S3", "policy": "agency", "macro_f1": 0.62},
            {"subject_id": "S3", "policy": "baseline", "macro_f1": 0.52},
            {"subject_id": "S4", "policy": "agency", "macro_f1": 0.58},
            {"subject_id": "S4", "policy": "baseline", "macro_f1": 0.48},
        ]
    )
    estimate, ci_low, ci_high, p_value, n_units = paired_policy_difference_test(
        data,
        policy_a="agency",
        policy_b="baseline",
        value_col="macro_f1",
        unit_cols=["subject_id"],
        repeats=1000,
        seed=7,
    )
    assert n_units == 4
    assert estimate > 0.0
    assert ci_low > 0.0
    assert 0.0 <= p_value <= 1.0


def test_holm_bonferroni() -> None:
    assert holm_bonferroni([0.001, 0.02, 0.5]) == [True, True, False]


def test_annotate_holm_bonferroni_groups_by_metric() -> None:
    data = pd.DataFrame(
        [
            {"split_family": "db10", "comparison_family": "agency_vs_plain_conf", "metric": "active_macro_f1", "p_value": 0.001},
            {"split_family": "db10", "comparison_family": "agency_vs_plain_conf", "metric": "active_macro_f1", "p_value": 0.02},
            {"split_family": "db10", "comparison_family": "agency_vs_plain_conf", "metric": "active_macro_f1", "p_value": 0.5},
            {"split_family": "db10", "comparison_family": "agency_vs_plain_conf", "metric": "ece", "p_value": 0.03},
        ]
    )
    annotated = annotate_holm_bonferroni(
        data,
        group_cols=["split_family", "comparison_family", "metric"],
    )
    macro = annotated.loc[annotated["metric"] == "active_macro_f1"].sort_values("holm_rank")
    assert macro["holm_reject"].tolist() == [True, True, False]
    assert macro["holm_n_tests"].tolist() == [3, 3, 3]
    ece = annotated.loc[annotated["metric"] == "ece"].iloc[0]
    assert bool(ece["holm_reject"]) is True
    assert int(ece["holm_n_tests"]) == 1


def test_paired_policy_difference_test_single_pair_p_value() -> None:
    data = pd.DataFrame(
        [
            {"split_id": "fold1", "policy": "agency", "macro_f1": 0.7},
            {"split_id": "fold1", "policy": "baseline", "macro_f1": 0.4},
        ]
    )
    _, _, _, p_value, n_units = paired_policy_difference_test(
        data,
        policy_a="agency",
        policy_b="baseline",
        value_col="macro_f1",
        unit_cols=["split_id"],
        repeats=256,
        seed=5,
    )
    assert n_units == 1
    assert p_value == 1.0


def test_paired_delta_test_reuses_paired_inference_core() -> None:
    estimate, ci_low, ci_high, p_value, n_units = paired_delta_test(
        [0.10, 0.12, 0.08, 0.11],
        repeats=512,
        seed=11,
    )
    assert n_units == 4
    assert estimate > 0.0
    assert ci_low > 0.0
    assert 0.0 <= p_value <= 1.0


def test_paired_policy_difference_test_requires_complete_pairs() -> None:
    data = pd.DataFrame(
        [
            {"split_id": "fold1", "policy": "agency", "macro_f1": 0.7},
            {"split_id": "fold1", "policy": "baseline", "macro_f1": 0.4},
            {"split_id": "fold2", "policy": "agency", "macro_f1": 0.6},
        ]
    )
    try:
        paired_policy_difference_test(
            data,
            policy_a="agency",
            policy_b="baseline",
            value_col="macro_f1",
            unit_cols=["split_id"],
            repeats=256,
            seed=5,
        )
    except ValueError as exc:
        assert "missing paired policy values" in str(exc)
    else:
        raise AssertionError("expected incomplete policy pairs to raise")


def test_infer_unit_cols_uses_subject_session_day_when_available() -> None:
    data = pd.DataFrame(
        [
            {"split_id": "fold1", "subject_id": "S1", "session": "day1", "day": "<missing>"},
            {"split_id": "fold1", "subject_id": "S1", "session": "day2", "day": "<missing>"},
        ]
    )
    assert infer_unit_cols(data) == ["split_id", "subject_id", "session"]


def test_resolve_metrics_paths_discovers_immediate_dataset_children(tmp_path) -> None:
    (tmp_path / "db10").mkdir()
    (tmp_path / "db10" / "metrics_by_policy_unit.csv").write_text("dataset_id\n", encoding="utf-8")
    (tmp_path / "db10_archive").mkdir()
    (tmp_path / "db10_archive" / "metrics_by_policy_unit.csv").write_text("dataset_id\n", encoding="utf-8")

    paths = resolve_metrics_paths(tmp_path, explicit_metrics=[])

    assert paths == sorted(
        [
            tmp_path / "db10" / "metrics_by_policy_unit.csv",
            tmp_path / "db10_archive" / "metrics_by_policy_unit.csv",
        ]
    )


def test_should_include_discovered_metrics_path_rejects_archive_dir_name() -> None:
    assert should_include_discovered_metrics_path(
        dataset_dir=Path("results/real/db10"),
        dataset_id="db10",
        explicit_metrics=False,
    )
    assert not should_include_discovered_metrics_path(
        dataset_dir=Path("results/real/db10_archive"),
        dataset_id="db10",
        explicit_metrics=False,
    )

