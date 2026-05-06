from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

from j2bench.artifact_schemas import validate_j1_frontier_package
from j2bench.j1_frontier import (
    build_j1_frontier_package,
    build_j1_frontier_readout_payload,
    render_j1_frontier_package,
    render_j1_frontier_readout,
)


def _load_module(script_name: str, module_name: str):
    module_path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_pairwise(path: Path) -> Path:
    rows = [
        {
            "dataset_id": "j1",
            "split_family": "j1_amputee_loso",
            "comparison_family": "agency_vs_plain_conf",
            "metric": "active_macro_f1",
            "policy_a": "agency_margin_tau_0.10",
            "policy_b": "plain_conf_threshold_matched_tau_0.10",
            "estimate": 0.04,
            "holm_reject_0_05": True,
        },
        {
            "dataset_id": "j1",
            "split_family": "j1_amputee_loso",
            "comparison_family": "agency_vs_plain_conf",
            "metric": "active_macro_f1",
            "policy_a": "agency_margin_tau_0.20",
            "policy_b": "plain_conf_threshold_matched_tau_0.20",
            "estimate": 0.07,
            "holm_reject_0_05": True,
        },
        {
            "dataset_id": "j1",
            "split_family": "j1_amputee_loso",
            "comparison_family": "agency_vs_plain_conf",
            "metric": "active_risk_coverage_auc",
            "policy_a": "agency_margin_tau_0.10",
            "policy_b": "plain_conf_threshold_matched_tau_0.10",
            "estimate": -0.01,
            "holm_reject_0_05": False,
        },
        {
            "dataset_id": "j1",
            "split_family": "j1_amputee_loso",
            "comparison_family": "agency_vs_plain_conf",
            "metric": "active_risk_coverage_auc",
            "policy_a": "agency_margin_tau_0.20",
            "policy_b": "plain_conf_threshold_matched_tau_0.20",
            "estimate": -0.03,
            "holm_reject_0_05": True,
        },
        {
            "dataset_id": "j1",
            "split_family": "j1_mixed_to_amputee",
            "comparison_family": "agency_vs_plain_conf",
            "metric": "active_macro_f1",
            "policy_a": "agency_margin_tau_0.10",
            "policy_b": "plain_conf_threshold_matched_tau_0.10",
            "estimate": 0.05,
            "holm_reject_0_05": True,
        },
        {
            "dataset_id": "j1",
            "split_family": "j1_mixed_to_amputee",
            "comparison_family": "agency_vs_plain_conf",
            "metric": "active_macro_f1",
            "policy_a": "agency_margin_tau_0.20",
            "policy_b": "plain_conf_threshold_matched_tau_0.20",
            "estimate": 0.06,
            "holm_reject_0_05": True,
        },
        {
            "dataset_id": "j1",
            "split_family": "j1_mixed_to_amputee",
            "comparison_family": "agency_vs_plain_conf",
            "metric": "active_risk_coverage_auc",
            "policy_a": "agency_margin_tau_0.10",
            "policy_b": "plain_conf_threshold_matched_tau_0.10",
            "estimate": -0.04,
            "holm_reject_0_05": True,
        },
        {
            "dataset_id": "j1",
            "split_family": "j1_mixed_to_amputee",
            "comparison_family": "agency_vs_plain_conf",
            "metric": "active_risk_coverage_auc",
            "policy_a": "agency_margin_tau_0.20",
            "policy_b": "plain_conf_threshold_matched_tau_0.20",
            "estimate": -0.01,
            "holm_reject_0_05": False,
        },
    ]
    return _write_csv(path, rows)


def _write_claims_config(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "version: 1",
                "program_id: j1_open",
                "claim_surface: benchmark_only",
                "split_families:",
                "  primary:",
                "    - j1_amputee_loso",
                "    - j1_mixed_to_amputee",
                "  boundary:",
                "    - j1_able_to_amputee",
                "claim_classes:",
                "  matched_budget_performance:",
                "    allowed: true",
                "    requires: [primary_split_success, comparator_support]",
                "gates:",
                "  primary_split_success:",
                "    required_split_families: [j1_amputee_loso, j1_mixed_to_amputee]",
                "    min_passing_splits: 2",
                "    primary_metric:",
                "      name: active_macro_f1",
                "      field: ci_lower",
                "      operator: gt",
                "      value: 0.0",
                "  comparator_support:",
                "    required_split_families: [j1_amputee_loso, j1_mixed_to_amputee]",
                "    min_supported_splits: 2",
                "    comparison_family: agency_vs_plain_conf",
                "    baseline_policy_prefix: plain_conf_threshold_matched_tau_",
                "  robustness_direction:",
                "    eligible_datasets: [hyser]",
                "    min_supporting_datasets: 1",
                "    metrics:",
                "      active_macro_f1:",
                "        field: delta",
                "        operator: gte",
                "        value: 0.0",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_frontier_inputs(tmp_path: Path) -> dict[str, Path]:
    scorecard_path = _write_json(
        tmp_path / "scorecard.json",
        {
            "dataset": "j1",
            "decision": "NO GO",
            "blocking_reasons": ["no tau cleared the primary universal-winner gate"],
            "required_split_families": ["j1_amputee_loso", "j1_mixed_to_amputee"],
            "gates": {
                "boundary_conditions": {"report_status": "ok", "passed": True},
                "readiness_reproducibility": {"status": "PASS", "passed": True},
            },
        },
    )
    pairwise_path = _write_pairwise(tmp_path / "primary_pairwise.csv")
    aggregate_rows = []
    earliest_summary_rows = []
    unit_rows = []
    earliest_unit_rows = []
    confidence_iso_rows = []
    earliest_iso_rows = []
    confidence_exact_rows = []
    earliest_exact_rows = []

    family_specs = {
        "j1_amputee_loso": {
            "agency_010": (0.60, 0.36, 0.04, 0.47, 0.56, 0.63, 0.34),
            "agency_020": (0.76, 0.24, 0.08, 0.65, 0.72, 0.79, 0.24),
            "plain_010": (0.54, 0.41, 0.03, 0.40, 0.51, 0.58, 0.38),
            "plain_020": (0.71, 0.22, 0.09, 0.70, 0.68, 0.74, 0.27),
            "set_010": (0.57, 0.39, 0.03, 0.45, 0.53, 0.60, 0.37),
            "user": (0.22, 0.82, 0.00, 0.00, 0.18, 0.23, 0.70),
        },
        "j1_mixed_to_amputee": {
            "agency_010": (0.58, 0.33, 0.05, 0.43, 0.54, 0.61, 0.32),
            "agency_020": (0.73, 0.34, 0.08, 0.68, 0.67, 0.74, 0.29),
            "plain_010": (0.51, 0.39, 0.03, 0.38, 0.49, 0.55, 0.36),
            "plain_020": (0.70, 0.25, 0.10, 0.72, 0.69, 0.75, 0.26),
            "set_010": (0.55, 0.37, 0.04, 0.41, 0.50, 0.57, 0.35),
            "user": (0.24, 0.80, 0.00, 0.00, 0.20, 0.24, 0.68),
        },
        "j1_able_to_amputee": {
            "agency_010": (0.46, 0.44, 0.04, 0.40, 0.42, 0.50, 0.42),
            "agency_020": (0.55, 0.37, 0.08, 0.64, 0.50, 0.57, 0.31),
            "plain_010": (0.44, 0.42, 0.03, 0.34, 0.39, 0.46, 0.44),
            "plain_020": (0.54, 0.34, 0.10, 0.69, 0.48, 0.55, 0.30),
            "set_010": (0.45, 0.43, 0.03, 0.36, 0.40, 0.47, 0.43),
            "user": (0.18, 0.85, 0.00, 0.00, 0.16, 0.21, 0.74),
        },
    }

    for family, spec in family_specs.items():
        split_id = f"{family}|S001"
        for policy, values in {
            "agency_margin_tau_0.10": spec["agency_010"],
            "agency_margin_tau_0.20": spec["agency_020"],
            "plain_conf_threshold_matched_tau_0.10": spec["plain_010"],
            "plain_conf_threshold_matched_tau_0.20": spec["plain_020"],
            "set_acsa_tau_0.10": spec["set_010"],
            "user_only": spec["user"],
        }.items():
            active_macro_f1, risk_auc, mean_ali, intervention_rate, stable_safe, final_correct, earliest = values
            aggregate_rows.append(
                {
                    "split_family": family,
                    "policy": policy,
                    "accuracy": 0.0,
                    "macro_f1": 0.0,
                    "active_macro_f1": active_macro_f1,
                    "balanced_accuracy": 0.0,
                    "mean_ali": mean_ali,
                    "intervention_rate": intervention_rate,
                    "risk_coverage_auc": risk_auc,
                    "active_risk_coverage_auc": risk_auc,
                    "ece": 0.0,
                    "median_decision_time_ms": 600.0,
                    "n": 100,
                    "n_active": 50,
                }
            )
            earliest_summary_rows.append(
                {
                    "dataset_id": "j1",
                    "split_family": family,
                    "policy": policy,
                    "safe_episode_rate": stable_safe,
                    "stable_safe_episode_rate": stable_safe,
                    "final_correct_rate": final_correct,
                    "median_earliest_intervene_s": earliest,
                    "median_earliest_correct_s": earliest,
                    "median_earliest_safe_s": earliest,
                    "median_earliest_stable_safe_s": earliest,
                }
            )
            unit_rows.append(
                {
                    "dataset_id": "j1",
                    "split_id": split_id,
                    "policy": policy,
                    "subject_id": "S001",
                    "session": "ex1",
                    "day": "ex1",
                    "accuracy": 0.0,
                    "balanced_accuracy": 0.0,
                    "macro_f1": 0.0,
                    "active_macro_f1": active_macro_f1,
                    "brier": 0.0,
                    "nll": 0.0,
                    "ece": 0.0,
                    "risk_coverage_auc": risk_auc,
                    "active_risk_coverage_auc": risk_auc,
                    "intervention_rate": intervention_rate,
                    "action_change_rate": intervention_rate,
                    "mean_ali": mean_ali,
                    "median_decision_time_ms": 600.0,
                    "n": 100,
                    "n_active": 50,
                    "split_family": family,
                }
            )
            earliest_unit_rows.append(
                {
                    "dataset_id": "j1",
                    "split_family": family,
                    "split_id": split_id,
                    "policy": policy,
                    "subject_id": "S001",
                    "session": "ex1",
                    "day": "ex1",
                    "n_episodes": 50,
                    "safe_episode_rate": stable_safe,
                    "stable_safe_episode_rate": stable_safe,
                    "final_correct_rate": final_correct,
                    "median_earliest_intervene_s": earliest,
                    "median_earliest_correct_s": earliest,
                    "median_earliest_safe_s": earliest,
                    "median_earliest_stable_safe_s": earliest,
                }
            )
        for tau in ("0.10", "0.20"):
            policy_a = f"agency_margin_tau_{tau}"
            confidence_iso_rows.extend(
                [
                    {
                        "split_family": family,
                        "comparison_family": "agency_vs_plain_conf_iso_budget",
                        "metric": "active_macro_f1",
                        "tau": tau,
                        "policy_a": policy_a,
                        "policy_b": "plain_conf_threshold_iso_budget_curve",
                        "n_units": 1,
                        "n_units_total": 1,
                        "n_extrapolated_units": 0 if tau == "0.20" else 1,
                        "extrapolated_fraction": 0.0 if tau == "0.20" else 1.0,
                        "mean_nearest_curve_budget_gap": 0.01 if tau == "0.20" else 0.12,
                        "beneficial": tau == "0.20",
                        "holm_reject": tau == "0.20",
                        "beneficial_and_holm_significant": tau == "0.20",
                    },
                    {
                        "split_family": family,
                        "comparison_family": "agency_vs_plain_conf_iso_budget",
                        "metric": "active_risk_coverage_auc",
                        "tau": tau,
                        "policy_a": policy_a,
                        "policy_b": "plain_conf_threshold_iso_budget_curve",
                        "n_units": 1,
                        "n_units_total": 1,
                        "n_extrapolated_units": 0 if tau == "0.20" else 1,
                        "extrapolated_fraction": 0.0 if tau == "0.20" else 1.0,
                        "mean_nearest_curve_budget_gap": 0.01 if tau == "0.20" else 0.12,
                        "beneficial": tau == "0.20",
                        "holm_reject": tau == "0.20",
                        "beneficial_and_holm_significant": tau == "0.20",
                    },
                ]
            )
            earliest_iso_rows.extend(
                [
                    {
                        "split_family": family,
                        "comparison_family": "agency_vs_plain_conf_iso_budget",
                        "metric": "stable_safe_episode_rate",
                        "tau": tau,
                        "policy_a": policy_a,
                        "policy_b": "plain_conf_threshold_iso_budget_curve",
                        "n_units": 1,
                        "n_units_total": 1,
                        "n_extrapolated_units": 0 if tau == "0.20" else 1,
                        "extrapolated_fraction": 0.0 if tau == "0.20" else 1.0,
                        "mean_nearest_curve_budget_gap": 0.02 if tau == "0.20" else 0.15,
                        "beneficial": True,
                        "holm_reject": tau == "0.20",
                        "beneficial_and_holm_significant": tau == "0.20",
                    },
                    {
                        "split_family": family,
                        "comparison_family": "agency_vs_plain_conf_iso_budget",
                        "metric": "final_correct_rate",
                        "tau": tau,
                        "policy_a": policy_a,
                        "policy_b": "plain_conf_threshold_iso_budget_curve",
                        "n_units": 1,
                        "n_units_total": 1,
                        "n_extrapolated_units": 0 if tau == "0.20" else 1,
                        "extrapolated_fraction": 0.0 if tau == "0.20" else 1.0,
                        "mean_nearest_curve_budget_gap": 0.02 if tau == "0.20" else 0.15,
                        "beneficial": True,
                        "holm_reject": tau == "0.20",
                        "beneficial_and_holm_significant": tau == "0.20",
                    },
                    {
                        "split_family": family,
                        "comparison_family": "agency_vs_plain_conf_iso_budget",
                        "metric": "median_earliest_stable_safe_s",
                        "tau": tau,
                        "policy_a": policy_a,
                        "policy_b": "plain_conf_threshold_iso_budget_curve",
                        "n_units": 1,
                        "n_units_total": 1,
                        "n_extrapolated_units": 0 if tau == "0.20" else 1,
                        "extrapolated_fraction": 0.0 if tau == "0.20" else 1.0,
                        "mean_nearest_curve_budget_gap": 0.02 if tau == "0.20" else 0.15,
                        "beneficial": True,
                        "holm_reject": tau == "0.20",
                        "beneficial_and_holm_significant": tau == "0.20",
                    },
                ]
            )
            confidence_exact_rows.extend(
                [
                    {
                        "split_family": family,
                        "comparison_family": "agency_vs_plain_conf_exact_budget",
                        "metric": "active_macro_f1",
                        "tau": tau,
                        "policy_a": policy_a,
                        "policy_b": f"plain_conf_threshold_exact_budget_tau_{tau}",
                        "mean_abs_intervention_rate_gap": 0.01 if tau == "0.20" else 0.08,
                        "exact_match_fraction": 1.0 if tau == "0.20" else 0.5,
                        "beneficial": tau == "0.20",
                        "holm_reject": tau == "0.20",
                        "beneficial_and_holm_significant": tau == "0.20",
                    },
                    {
                        "split_family": family,
                        "comparison_family": "agency_vs_plain_conf_exact_budget",
                        "metric": "active_risk_coverage_auc",
                        "tau": tau,
                        "policy_a": policy_a,
                        "policy_b": f"plain_conf_threshold_exact_budget_tau_{tau}",
                        "mean_abs_intervention_rate_gap": 0.01 if tau == "0.20" else 0.08,
                        "exact_match_fraction": 1.0 if tau == "0.20" else 0.5,
                        "beneficial": tau == "0.20",
                        "holm_reject": tau == "0.20",
                        "beneficial_and_holm_significant": tau == "0.20",
                    },
                ]
            )
            earliest_exact_rows.extend(
                [
                    {
                        "split_family": family,
                        "comparison_family": "agency_vs_plain_conf_exact_budget",
                        "metric": "stable_safe_episode_rate",
                        "tau": tau,
                        "policy_a": policy_a,
                        "policy_b": f"plain_conf_threshold_exact_budget_tau_{tau}",
                        "mean_abs_intervention_rate_gap": 0.01 if tau == "0.20" else 0.08,
                        "exact_match_fraction": 1.0 if tau == "0.20" else 0.5,
                        "beneficial": True,
                        "holm_reject": tau == "0.20",
                        "beneficial_and_holm_significant": tau == "0.20",
                    },
                    {
                        "split_family": family,
                        "comparison_family": "agency_vs_plain_conf_exact_budget",
                        "metric": "final_correct_rate",
                        "tau": tau,
                        "policy_a": policy_a,
                        "policy_b": f"plain_conf_threshold_exact_budget_tau_{tau}",
                        "mean_abs_intervention_rate_gap": 0.01 if tau == "0.20" else 0.08,
                        "exact_match_fraction": 1.0 if tau == "0.20" else 0.5,
                        "beneficial": True,
                        "holm_reject": tau == "0.20",
                        "beneficial_and_holm_significant": tau == "0.20",
                    },
                    {
                        "split_family": family,
                        "comparison_family": "agency_vs_plain_conf_exact_budget",
                        "metric": "median_earliest_stable_safe_s",
                        "tau": tau,
                        "policy_a": policy_a,
                        "policy_b": f"plain_conf_threshold_exact_budget_tau_{tau}",
                        "mean_abs_intervention_rate_gap": 0.01 if tau == "0.20" else 0.08,
                        "exact_match_fraction": 1.0 if tau == "0.20" else 0.5,
                        "beneficial": True,
                        "holm_reject": tau == "0.20",
                        "beneficial_and_holm_significant": tau == "0.20",
                    },
                ]
            )

    return {
        "scorecard": scorecard_path,
        "pairwise": pairwise_path,
        "aggregate": _write_csv(tmp_path / "aggregate.csv", aggregate_rows),
        "unit_metrics": _write_csv(tmp_path / "unit_metrics.csv", unit_rows),
        "earliest_summary": _write_csv(tmp_path / "earliest_summary.csv", earliest_summary_rows),
        "earliest_unit": _write_csv(tmp_path / "earliest_unit.csv", earliest_unit_rows),
        "confidence_iso_pairwise": _write_csv(tmp_path / "confidence_iso_pairwise.csv", confidence_iso_rows),
        "earliest_iso_pairwise": _write_csv(tmp_path / "earliest_iso_pairwise.csv", earliest_iso_rows),
        "confidence_exact_pairwise": _write_csv(tmp_path / "confidence_exact_pairwise.csv", confidence_exact_rows),
        "earliest_exact_pairwise": _write_csv(tmp_path / "earliest_exact_pairwise.csv", earliest_exact_rows),
        "claims": _write_claims_config(tmp_path / "j1_claims.yaml"),
    }


def test_build_j1_frontier_readout_payload_detects_no_universal_tau(tmp_path: Path) -> None:
    scorecard_path = _write_json(
        tmp_path / "scorecard.json",
        {
            "dataset": "j1",
            "decision": "NO GO",
            "blocking_reasons": ["no tau cleared the primary universal-winner gate"],
            "required_split_families": ["j1_amputee_loso", "j1_mixed_to_amputee"],
            "boundary_conditions": {"status": "PASS", "passed": True},
            "readiness": {"status": "PASS", "passed": True},
        },
    )
    pairwise_path = _write_pairwise(tmp_path / "primary_pairwise.csv")

    payload = build_j1_frontier_readout_payload(
        scorecard_path=scorecard_path,
        primary_pairwise_path=pairwise_path,
    )

    assert payload["universal_winner_supported"] is False
    assert payload["frontier_story_supported"] is True
    assert payload["universal_taus"] == []
    assert payload["unique_best_taus"] == ["0.10", "0.20"]
    assert payload["metric_disagreement_families"] == ["j1_mixed_to_amputee"]

    family_metric = {
        (entry["split_family"], entry["metric"]): entry for entry in payload["family_metric_summary"]
    }
    assert family_metric[("j1_amputee_loso", "active_macro_f1")]["best_tau"] == "0.20"
    assert family_metric[("j1_amputee_loso", "active_risk_coverage_auc")]["best_tau"] == "0.20"
    assert family_metric[("j1_mixed_to_amputee", "active_macro_f1")]["best_tau"] == "0.20"
    assert family_metric[("j1_mixed_to_amputee", "active_risk_coverage_auc")]["best_tau"] == "0.10"

    markdown = render_j1_frontier_readout(payload)
    assert "Universal tau candidates: <none>" in markdown
    assert "J1-Frontier Readout" in markdown


def test_build_j1_frontier_package_emits_frontier_and_budget_outputs(tmp_path: Path) -> None:
    paths = _write_frontier_inputs(tmp_path)
    package = build_j1_frontier_package(
        scorecard_path=paths["scorecard"],
        primary_pairwise_path=paths["pairwise"],
        publication_aggregate_path=paths["aggregate"],
        publication_unit_path=paths["unit_metrics"],
        earliest_summary_path=paths["earliest_summary"],
        earliest_unit_path=paths["earliest_unit"],
        confidence_iso_budget_pairwise_path=paths["confidence_iso_pairwise"],
        earliest_iso_budget_pairwise_path=paths["earliest_iso_pairwise"],
        confidence_exact_budget_pairwise_path=paths["confidence_exact_pairwise"],
        earliest_exact_budget_pairwise_path=paths["earliest_exact_pairwise"],
        claims_config_path=paths["claims"],
    )

    summary = package["summary"]
    assert summary["primary_split_families"] == ["j1_amputee_loso", "j1_mixed_to_amputee"]
    assert summary["boundary_split_families"] == ["j1_able_to_amputee"]
    assert summary["primary_frontier_supported"] is True
    assert summary["primary_universal_taus"] == []
    assert summary["primary_iso_budget_supported_taus"] == ["0.20"]
    assert summary["primary_exact_budget_audit_taus"] == ["0.20"]
    assert package["pareto_frontier"].shape[0] > 0
    assert package["budget_match_summary"]["mean_abs_raw_intervention_rate_gap"].max() > 0.0
    assert package["iso_budget_claim_summary"]["confidence_all_primary_metrics_sig_beneficial"].any()
    assert package["exact_budget_claim_summary"]["confidence_all_primary_metrics_sig_beneficial"].any()
    assert "0.20" in package["universal_tau_regret"]["tau"].tolist()
    assert package["operating_point_instability"]["primary_family_metric_disagreement_families"] == [
        "j1_amputee_loso",
        "j1_mixed_to_amputee",
    ]

    markdown = render_j1_frontier_package(
        summary=summary,
        non_dominated_summary=package["non_dominated_summary"],
        universal_tau_regret=package["universal_tau_regret"],
        budget_match_summary=package["budget_match_summary"],
        iso_budget_claim_summary=package["iso_budget_claim_summary"],
        exact_budget_claim_summary=package["exact_budget_claim_summary"],
    )
    validated = validate_j1_frontier_package(markdown)
    assert "bottom line" in validated["sections"]
    assert "artifact inventory" in validated["sections"]


def test_build_j1_frontier_scripts_write_outputs(tmp_path: Path) -> None:
    readout_module = _load_module(
        "build_j1_frontier_readout.py",
        "build_j1_frontier_readout_module",
    )
    package_module = _load_module(
        "build_j1_frontier_package.py",
        "build_j1_frontier_package_module",
    )
    paths = _write_frontier_inputs(tmp_path)
    readout_root = tmp_path / "readout"
    frontier_root = tmp_path / "frontier"

    readout_exit = readout_module.main(
        [
            "--scorecard-path",
            str(paths["scorecard"]),
            "--primary-pairwise-path",
            str(paths["pairwise"]),
            "--out-root",
            str(readout_root),
        ]
    )
    package_exit = package_module.main(
        [
            "--scorecard-path",
            str(paths["scorecard"]),
            "--primary-pairwise-path",
            str(paths["pairwise"]),
            "--publication-aggregate-path",
            str(paths["aggregate"]),
            "--publication-unit-path",
            str(paths["unit_metrics"]),
            "--earliest-summary-path",
            str(paths["earliest_summary"]),
            "--earliest-unit-path",
            str(paths["earliest_unit"]),
            "--confidence-iso-budget-pairwise-path",
            str(paths["confidence_iso_pairwise"]),
            "--earliest-iso-budget-pairwise-path",
            str(paths["earliest_iso_pairwise"]),
            "--confidence-exact-budget-pairwise-path",
            str(paths["confidence_exact_pairwise"]),
            "--earliest-exact-budget-pairwise-path",
            str(paths["earliest_exact_pairwise"]),
            "--claims-config-path",
            str(paths["claims"]),
            "--out-root",
            str(frontier_root),
        ]
    )

    assert readout_exit == 0
    assert package_exit == 0
    assert (readout_root / "j1_frontier_readout.json").exists()
    assert (frontier_root / "frontier_points.csv").exists()
    assert (frontier_root / "pareto_frontier.csv").exists()
    assert (frontier_root / "universal_tau_regret.csv").exists()
    assert (frontier_root / "iso_budget_claim_summary.csv").exists()
    assert (frontier_root / "exact_budget_claim_summary.csv").exists()
    assert (frontier_root / "frontier_atlas.png").exists()
    assert (frontier_root / "safety_timing_frontier.png").exists()
    assert (frontier_root / "j1_frontier_package_provenance.json").exists()

    summary = json.loads((frontier_root / "j1_frontier_package.json").read_text(encoding="utf-8"))
    markdown = (frontier_root / "j1_frontier_package.md").read_text(encoding="utf-8")
    assert summary["primary_frontier_supported"] is True
    assert summary["primary_iso_budget_supported_taus"] == ["0.20"]
    assert summary["primary_exact_budget_audit_taus"] == ["0.20"]
    assert "Comparator Integrity" in markdown

