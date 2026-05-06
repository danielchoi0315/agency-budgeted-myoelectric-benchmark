from dataclasses import asdict

import numpy as np
import pandas as pd

from j2bench.calibration import expected_calibration_error, probability_temperature_scale
from j2bench.metrics import multiclass_brier, risk_coverage_auc, summarize_by_policy
from j2bench.schemas import PredictionTrace


def test_temperature_scaling_preserves_probability_rows():
    probs = np.array([[0.8, 0.2], [0.4, 0.6]])
    scaled = probability_temperature_scale(probs, temperature=2.0)
    assert scaled.shape == probs.shape
    assert np.allclose(scaled.sum(axis=1), 1.0)


def test_brier_is_zero_for_perfect_predictions():
    probs = np.eye(3)
    labels = np.array([0, 1, 2])
    assert np.isclose(multiclass_brier(probs, labels), 0.0)


def test_ece_is_low_for_confident_correct_predictions():
    probs = np.array([[0.99, 0.01], [0.01, 0.99]])
    labels = np.array([0, 1])
    assert expected_calibration_error(probs, labels, n_bins=5) < 0.02


def test_risk_coverage_auc_bounds():
    confidence = np.array([0.9, 0.8, 0.2, 0.1])
    correct = np.array([1, 1, 0, 0])
    auc = risk_coverage_auc(confidence, correct)
    assert 0.0 <= auc <= 1.0


def test_risk_coverage_auc_perfect_is_zero_for_any_n():
    assert np.isclose(risk_coverage_auc(np.array([0.9]), np.array([1])), 0.0)
    assert np.isclose(risk_coverage_auc(np.array([0.9, 0.8, 0.7]), np.array([1, 1, 1])), 0.0)


def test_risk_coverage_auc_all_wrong_is_one():
    assert np.isclose(risk_coverage_auc(np.array([0.9, 0.8]), np.array([0, 0])), 1.0)


def test_db10_active_macro_f1_excludes_rest():
    traces = [
        PredictionTrace(
            dataset_id="db10",
            split_id="fold1",
            subject_id="S101",
            session="ex1",
            day=None,
            timestamp_s=0.0,
            label=0,
            p_user=np.array([0.9, 0.1, 0.0]),
            p_assist=np.array([0.9, 0.1, 0.0]),
            uncertainty=0.0,
        ),
        PredictionTrace(
            dataset_id="db10",
            split_id="fold1",
            subject_id="S101",
            session="ex1",
            day=None,
            timestamp_s=0.1,
            label=1,
            p_user=np.array([0.6, 0.3, 0.1]),
            p_assist=np.array([0.1, 0.8, 0.1]),
            uncertainty=0.0,
        ),
        PredictionTrace(
            dataset_id="db10",
            split_id="fold1",
            subject_id="S101",
            session="ex1",
            day=None,
            timestamp_s=0.2,
            label=2,
            p_user=np.array([0.6, 0.1, 0.3]),
            p_assist=np.array([0.2, 0.7, 0.1]),
            uncertainty=0.0,
        ),
    ]
    from j2bench.policies import assist_only

    metrics = summarize_by_policy([assist_only(trace) for trace in traces])
    frame = pd.DataFrame([asdict(metric) for metric in metrics])
    assert np.isclose(frame["active_macro_f1"].iloc[0], 1.0 / 3.0)
    assert frame["n_active"].iloc[0] == 2
    active_auc = risk_coverage_auc(np.array([0.8, 0.7]), np.array([1, 0]))
    assert np.isclose(frame["active_risk_coverage_auc"].iloc[0], active_auc)

