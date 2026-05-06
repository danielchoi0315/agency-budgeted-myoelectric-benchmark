from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from j2bench.metrics import summarize_decisions
from j2bench.schemas import PolicyDecision


def _load_script_module(script_name: str):
    module_path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_decision(*, label: int, selected_action: int, episode_id: str) -> PolicyDecision:
    probs = np.zeros(3, dtype=float)
    probs[selected_action] = 1.0
    return PolicyDecision(
        dataset_id="j1",
        split_id="j1_amputee_loso|S101",
        subject_id="S101",
        session="session1",
        day="D01",
        timestamp_s=0.2,
        label=label,
        policy="agency_margin_tau_0.10",
        selected_action=selected_action,
        user_action=selected_action,
        assist_action=selected_action,
        decision_probs=probs,
        selected_confidence=1.0,
        coverage_confidence=1.0,
        control_source="agency_margin",
        action_changed=False,
        autonomy_used=True,
        intervene=True,
        defer=False,
        abstain=False,
        expected_utility=1.0,
        expected_ali=0.0,
        tau=0.10,
        agency_margin=0.10,
        decision_time_ms=200.0,
        metadata={"episode_id": episode_id, "prefix_time_s": 0.2},
    )


def test_j1_active_metrics_exclude_rest_label_zero() -> None:
    record = summarize_decisions(
        [
            _make_decision(label=0, selected_action=0, episode_id="rest"),
            _make_decision(label=1, selected_action=1, episode_id="active_correct"),
            _make_decision(label=2, selected_action=0, episode_id="active_incorrect"),
        ],
        dataset_id="j1",
        split_id="j1_amputee_loso|S101",
        policy="agency_margin_tau_0.10",
        subject_id="S101",
        session="session1",
        day="D01",
    )
    assert record.n_active == 2
    assert np.isclose(record.active_macro_f1, 0.5)


def test_j1_primary_pairs_use_plain_confidence_gate() -> None:
    module = _load_script_module("postprocess_real_stats.py")
    policies = pd.Series(
        [
            "agency_margin_tau_0.10",
            "plain_conf_threshold_matched_tau_0.10",
            "set_acsa_tau_0.10",
        ]
    )
    assert module.primary_comparison_family("j1") == "agency_vs_plain_conf"
    assert module.primary_pairs_for_dataset("j1", policies) == [
        ("agency_margin_tau_0.10", "plain_conf_threshold_matched_tau_0.10")
    ]
    assert module.primary_comparison_family("db10") == "matched_tau"
    assert module.primary_pairs_for_dataset("db10", policies) == [
        ("agency_margin_tau_0.10", "set_acsa_tau_0.10")
    ]


def test_infer_available_sequence_keys_flattens_nested_payload_mappings() -> None:
    module = _load_script_module("sweep_real_models.py")
    payloads = {
        "user": {
            "user_emg": np.ones((2, 4, 3), dtype=np.float32),
            "user_time_mask": np.ones((2, 4, 1), dtype=np.float32),
        },
        "assist": {
            "assist_emg": np.ones((2, 8, 3), dtype=np.float32),
            "assist_time_mask": np.ones((2, 8, 1), dtype=np.float32),
        },
    }
    keys = module.infer_available_sequence_keys(payloads)
    assert {"user", "assist", "user_emg", "user_time_mask", "assist_emg", "assist_time_mask"}.issubset(keys)

