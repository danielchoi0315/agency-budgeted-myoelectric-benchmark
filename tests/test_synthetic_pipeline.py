from dataclasses import asdict

import pandas as pd

from j2bench.metrics import summarize_by_policy
from j2bench.policies import apply_policy_grid
from j2bench.synthetic import make_synthetic_metadata, make_synthetic_traces


def test_synthetic_pipeline_produces_metrics():
    metadata = make_synthetic_metadata(n_subjects=4, n_trials_per_cell=4)
    traces = make_synthetic_traces(metadata, n_classes=4)
    decisions = apply_policy_grid(traces, tau_grid=[0.05, 0.10])
    metrics = summarize_by_policy(decisions)
    frame = pd.DataFrame([asdict(metric) for metric in metrics])
    assert not frame.empty
    policies = set(frame["policy"])
    assert {"user_only", "assist_only"}.issubset(policies)
    assert any(policy.startswith("agency_margin") for policy in policies)
    assert any(policy.startswith("set_acsa") for policy in policies)
    assert frame["macro_f1"].between(0, 1).all()
    assert frame["mean_ali"].ge(0).all()

