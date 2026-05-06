from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "build_exact_budget_instability_report.py"
    spec = importlib.util.spec_from_file_location("build_exact_budget_instability_report", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_exact_budget_instability_payload_summarizes_required_scope(tmp_path: Path) -> None:
    module = _load_module()
    publication_root = tmp_path / "publication_clean"
    j1_root = tmp_path / "j1_open"
    publication_root.mkdir(parents=True, exist_ok=True)
    (j1_root / "publication_clean").mkdir(parents=True, exist_ok=True)

    db10_confidence = [
        {
            "split_family": split_family,
            "comparison_family": "agency_vs_plain_conf_exact_budget",
            "metric": metric,
            "tau": "0.10",
            "estimate": 0.1 if metric == "active_macro_f1" else -0.1,
            "beneficial": True,
            "holm_reject": True,
            "beneficial_and_holm_significant": True,
            "mean_abs_intervention_rate_gap": 0.01,
            "exact_match_fraction": 1.0,
        }
        for split_family in ["db10_shift_a", "db10_shift_b"]
        for metric in ["active_macro_f1", "active_risk_coverage_auc"]
    ]
    db10_earliest = [
        {
            "split_family": split_family,
            "comparison_family": "agency_vs_plain_conf_exact_budget",
            "metric": metric,
            "tau": "0.10",
            "estimate": 0.1 if metric != "median_earliest_stable_safe_s" else -0.1,
            "beneficial": True,
            "holm_reject": True,
            "beneficial_and_holm_significant": True,
            "mean_abs_intervention_rate_gap": 0.01,
            "exact_match_fraction": 1.0,
        }
        for split_family in ["db10_shift_a", "db10_shift_b"]
        for metric in [
            "stable_safe_episode_rate",
            "final_correct_rate",
            "median_earliest_stable_safe_s",
        ]
    ]
    _write_csv(publication_root / "db10_confidence_gate_exact_budget_pairwise.csv", db10_confidence)
    _write_csv(publication_root / "db10_earliest_safe_exact_budget_pairwise.csv", db10_earliest)

    j1_confidence = [
        {
            "split_family": split_family,
            "comparison_family": "agency_vs_plain_conf_exact_budget",
            "metric": "active_macro_f1",
            "tau": "0.10",
            "estimate": 0.05,
            "beneficial": True,
            "holm_reject": True,
            "beneficial_and_holm_significant": True,
            "mean_abs_intervention_rate_gap": 0.02,
            "exact_match_fraction": 0.5,
        }
        for split_family in ["j1_primary_a", "j1_primary_b", "j1_descriptive"]
    ] + [
        {
            "split_family": split_family,
            "comparison_family": "agency_vs_plain_conf_exact_budget",
            "metric": "active_risk_coverage_auc",
            "tau": "0.10",
            "estimate": -0.01,
            "beneficial": True,
            "holm_reject": False,
            "beneficial_and_holm_significant": False,
            "mean_abs_intervention_rate_gap": 0.02,
            "exact_match_fraction": 0.5,
        }
        for split_family in ["j1_primary_a", "j1_primary_b", "j1_descriptive"]
    ]
    j1_earliest = [
        {
            "split_family": split_family,
            "comparison_family": "agency_vs_plain_conf_exact_budget",
            "metric": metric,
            "tau": "0.10",
            "estimate": 0.02 if metric != "median_earliest_stable_safe_s" else -0.02,
            "beneficial": True,
            "holm_reject": False,
            "beneficial_and_holm_significant": False,
            "mean_abs_intervention_rate_gap": 0.02,
            "exact_match_fraction": 0.5,
        }
        for split_family in ["j1_primary_a", "j1_primary_b", "j1_descriptive"]
        for metric in [
            "stable_safe_episode_rate",
            "final_correct_rate",
            "median_earliest_stable_safe_s",
        ]
    ]
    _write_csv(j1_root / "publication_clean" / "j1_confidence_gate_exact_budget_pairwise.csv", j1_confidence)
    _write_csv(j1_root / "publication_clean" / "j1_earliest_safe_exact_budget_pairwise.csv", j1_earliest)
    (j1_root / "j1_scorecard.json").write_text(
        json.dumps(
            {
                "required_split_families": ["j1_primary_a", "j1_primary_b"],
            }
        ),
        encoding="utf-8",
    )

    payload = module.build_payload(publication_root=publication_root, j1_root=j1_root)
    overview = {row["dataset_id"]: row for row in payload["dataset_overview"]}

    assert payload["status"] == "mixed_fixed_tau_support"
    assert overview["db10"]["fixed_tau_supported"] is True
    assert overview["db10"]["universal_exact_budget_taus"] == "0.10"
    assert overview["j1"]["fixed_tau_supported"] is False
    assert overview["j1"]["required_split_family_count"] == 2
    assert overview["j1"]["required_scope"] == "scorecard_required"
    assert "j1_descriptive" not in overview["j1"]["required_split_families"]

