from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_build_anchor_report_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "build_anchor_report.py"
    spec = importlib.util.spec_from_file_location("build_anchor_report_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_checks_detects_db10_qualifying_tau() -> None:
    module = _load_build_anchor_report_module()
    primary_pairwise = pd.DataFrame(
        [
            {
                "dataset_id": "db10",
                "split_family": split_family,
                "comparison_family": "matched_tau",
                "metric": metric,
                "policy_a": "agency_margin_tau_0.20",
                "policy_b": "set_acsa_tau_0.20",
                "estimate": estimate,
                "holm_reject_0_05": True,
            }
            for split_family in ["db10_able_to_amputee", "db10_amputee_loso", "db10_mixed_to_amputee"]
            for metric, estimate in [("active_macro_f1", 0.01), ("active_risk_coverage_auc", -0.01)]
        ]
    )
    checks = module.build_checks(
        exclusion_report={"matched_failed_trials": [module.EXPECTED_CEMHSEY_GRASP_FAILURE]},
        observations=pd.DataFrame(columns=["dataset_id"]),
        primary_pairwise=primary_pairwise,
    )
    by_name = {check["name"]: check for check in checks}
    assert by_name["db10_primary_pairwise_significance"]["passed"] is True
    assert "0.20" in by_name["db10_primary_pairwise_significance"]["detail"]


def test_build_checks_reports_manifest_alignment() -> None:
    module = _load_build_anchor_report_module()
    checks = module.build_checks(
        exclusion_report={"matched_failed_trials": [module.EXPECTED_CEMHSEY_GRASP_FAILURE]},
        observations=pd.DataFrame(columns=["dataset_id"]),
        primary_pairwise=pd.DataFrame(),
        manifest_rows=pd.DataFrame(
            [
                {
                    "dataset_id": "db10",
                    "user_model": "logistic",
                    "assist_model": "assist_temporal_forest",
                    "config_user_model": "logistic",
                    "config_assist_model": "assist_temporal_forest",
                    "matches_config": True,
                }
            ]
        ),
    )
    by_name = {check["name"]: check for check in checks}
    assert by_name["canonical_model_manifest_alignment"]["passed"] is True

