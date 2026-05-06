from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

from j2bench.artifact_schemas import validate_scorecard
from j2bench.j1_scorecard import build_j1_scorecard_from_roots, render_j1_scorecard


REQUIRED_FAMILIES = ["j1_able_to_amputee", "j1_subject_logo"]
SELECTED_TAU = "0.10"
SELECTED_POLICY = f"agency_margin_tau_{SELECTED_TAU}"


def _load_build_j1_scorecard_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "build_j1_scorecard.py"
    spec = importlib.util.spec_from_file_location("build_j1_scorecard_module", module_path)
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


def _provenance_payload(name: str) -> dict[str, object]:
    return {
        "artifact_type": name,
        "artifact_root": f"C:/tmp/{name}",
        "files": [
            {
                "path": f"{name}/artifact.csv",
                "exists": True,
            }
        ],
    }


def _write_scorecard_inputs(
    tmp_path: Path,
    *,
    primary_gate_passes: bool = True,
) -> dict[str, Path]:
    benchmark_root = tmp_path / "benchmark"
    stats_root = tmp_path / "stats"
    comparator_root = tmp_path / "comparator"
    report_root = tmp_path / "report"
    provenance_root = tmp_path / "provenance"
    prepared_root = tmp_path / "prepared"

    _write_json(
        benchmark_root / "benchmark_manifest.json",
        {
            "dataset": "j1",
            "prepared_root": str(prepared_root),
            "prepared_success_file": str(prepared_root / "_SUCCESS"),
            "user_model": "logistic",
            "assist_model": "logistic",
            "assist_context_modes": [],
            "available_sequence_payload_roles": [],
            "sequence_payload_sources": {},
            "split_family_allowlist": [],
            "split_id_allowlist": [],
            "max_splits_per_family": None,
            "n_rows": 24,
        },
    )
    _write_csv(
        benchmark_root / "aggregate_policy_metrics.csv",
        [
            {
                "split_family": split_family,
                "policy": SELECTED_POLICY,
                "active_macro_f1": 0.74,
                "active_risk_coverage_auc": 0.12,
                "mean_ali": 0.03,
                "intervention_rate": 0.18,
                "median_decision_time_ms": 240.0,
            }
            for split_family in REQUIRED_FAMILIES
        ]
        + [
            {
                "split_family": split_family,
                "policy": "user_only",
                "active_macro_f1": 0.66,
                "active_risk_coverage_auc": 0.15,
                "mean_ali": 0.05,
                "intervention_rate": 0.0,
                "median_decision_time_ms": 210.0,
            }
            for split_family in REQUIRED_FAMILIES
        ],
    )

    holm_reject = bool(primary_gate_passes)
    estimate_macro = 0.03 if primary_gate_passes else -0.01
    estimate_risk = -0.02 if primary_gate_passes else 0.02
    _write_csv(
        stats_root / "primary_pairwise.csv",
        [
            {
                "dataset_id": "j1",
                "split_family": split_family,
                "comparison_family": "matched_tau",
                "metric": metric,
                "policy_a": SELECTED_POLICY,
                "policy_b": f"set_acsa_tau_{SELECTED_TAU}",
                "estimate": estimate_macro if metric == "active_macro_f1" else estimate_risk,
                "holm_reject_0_05": holm_reject,
            }
            for split_family in REQUIRED_FAMILIES
            for metric in ("active_macro_f1", "active_risk_coverage_auc")
        ],
    )

    _write_json(
        comparator_root / "j1_publication_extension_manifest.json",
        {
            "dataset": "j1",
            "prepared_root": str(prepared_root),
            "user_model": "logistic",
            "assist_model": "logistic",
            "pairwise_repeats": 2000,
            "trace_count": 48,
            "decision_count": 480,
            "split_families": REQUIRED_FAMILIES,
        },
    )
    _write_csv(
        comparator_root / "j1_confidence_gate_match_summary.csv",
        [
            {
                "tau": 0.10,
                "matched_intervention_rate": 0.18,
                "intervention_rate_gap": 0.01,
            }
        ],
    )
    _write_csv(
        comparator_root / "j1_confidence_gate_pairwise.csv",
        [
            {
                "split_family": split_family,
                "comparison_family": comparison_family,
                "metric": metric,
                "policy_a": SELECTED_POLICY,
                "policy_b": (
                    f"set_acsa_tau_{SELECTED_TAU}"
                    if comparison_family == "agency_vs_set_acsa"
                    else f"plain_conf_threshold_matched_tau_{SELECTED_TAU}"
                ),
                "estimate": 0.02 if metric == "active_macro_f1" else -0.01,
                "holm_reject": True,
                "beneficial_and_holm_significant": True,
            }
            for split_family in REQUIRED_FAMILIES
            for comparison_family in ("agency_vs_set_acsa", "agency_vs_plain_conf")
            for metric in ("active_macro_f1", "active_risk_coverage_auc")
        ],
    )
    _write_csv(
        comparator_root / "j1_earliest_safe_summary.csv",
        [
            {
                "dataset_id": "j1",
                "split_family": split_family,
                "policy": SELECTED_POLICY,
                "stable_safe_episode_rate": 0.72,
                "final_correct_rate": 0.93,
                "median_earliest_stable_safe_s": 0.42,
            }
            for split_family in REQUIRED_FAMILIES
        ],
    )
    _write_csv(
        comparator_root / "j1_earliest_safe_pairwise.csv",
        [
            {
                "split_family": split_family,
                "comparison_family": comparison_family,
                "metric": metric,
                "policy_a": SELECTED_POLICY,
                "policy_b": (
                    f"set_acsa_tau_{SELECTED_TAU}"
                    if comparison_family == "agency_vs_set_acsa"
                    else f"plain_conf_threshold_matched_tau_{SELECTED_TAU}"
                ),
                "estimate": (
                    0.04
                    if metric in {"stable_safe_episode_rate", "final_correct_rate"}
                    else -0.08
                ),
                "holm_reject": True,
            }
            for split_family in REQUIRED_FAMILIES
            for comparison_family in ("agency_vs_set_acsa", "agency_vs_plain_conf")
            for metric in (
                "stable_safe_episode_rate",
                "final_correct_rate",
                "median_earliest_stable_safe_s",
            )
        ],
    )

    _write_json(
        report_root / "split_report.json",
        {
            "dataset_id": "j1",
            "status": "ok",
            "summary": {
                "critical_overlap_checks_passed": True,
                "forward_order_checks_applicable": 0,
                "forward_order_checks_passed": True,
            },
            "families": [{"split_family": split_family} for split_family in REQUIRED_FAMILIES],
            "splits": [{"split_id": f"{split_family}|fold1"} for split_family in REQUIRED_FAMILIES],
        },
    )

    benchmark_provenance = _write_json(
        provenance_root / "benchmark_provenance.json",
        _provenance_payload("benchmark_output"),
    )
    stats_provenance = _write_json(
        provenance_root / "stats_provenance.json",
        _provenance_payload("stats_output"),
    )
    comparator_provenance = _write_json(
        provenance_root / "comparator_provenance.json",
        _provenance_payload("comparator_output"),
    )
    report_provenance = _write_json(
        provenance_root / "report_provenance.json",
        _provenance_payload("report_output"),
    )

    return {
        "benchmark_root": benchmark_root,
        "stats_root": stats_root,
        "comparator_root": comparator_root,
        "report_root": report_root,
        "benchmark_provenance": benchmark_provenance,
        "stats_provenance": stats_provenance,
        "comparator_provenance": comparator_provenance,
        "report_provenance": report_provenance,
    }


def test_build_j1_scorecard_returns_go_for_complete_evidence(tmp_path: Path) -> None:
    paths = _write_scorecard_inputs(tmp_path, primary_gate_passes=True)

    payload = build_j1_scorecard_from_roots(
        benchmark_root=paths["benchmark_root"],
        stats_root=paths["stats_root"],
        comparator_root=paths["comparator_root"],
        report_root=paths["report_root"],
        benchmark_provenance=paths["benchmark_provenance"],
        stats_provenance=paths["stats_provenance"],
        comparator_provenance=paths["comparator_provenance"],
        report_provenance=paths["report_provenance"],
    )
    markdown = render_j1_scorecard(payload)

    assert payload["decision"] == "GO"
    assert payload["candidate"]["selected_tau"] == SELECTED_TAU
    assert payload["candidate"]["selected_policy"] == SELECTED_POLICY
    assert payload["gates"]["primary_db10_gate"]["passed"] is True
    assert payload["gates"]["robustness_panel"]["passed"] is True
    assert payload["gates"]["safety_timing_panel"]["passed"] is True
    assert "Decision: GO" in markdown
    assert "agency_margin_tau_0.10" in markdown
    validate_scorecard(markdown)


def test_build_j1_scorecard_fails_closed_on_missing_critical_evidence(tmp_path: Path) -> None:
    paths = _write_scorecard_inputs(tmp_path, primary_gate_passes=True)
    (paths["stats_root"] / "primary_pairwise.csv").unlink()

    payload = build_j1_scorecard_from_roots(
        benchmark_root=paths["benchmark_root"],
        stats_root=paths["stats_root"],
        comparator_root=paths["comparator_root"],
        report_root=paths["report_root"],
        benchmark_provenance=paths["benchmark_provenance"],
        stats_provenance=paths["stats_provenance"],
        comparator_provenance=paths["comparator_provenance"],
        report_provenance=paths["report_provenance"],
    )
    markdown = render_j1_scorecard(payload)

    assert payload["decision"] == "NO GO"
    assert any("stats_primary_pairwise" in item for item in payload["missing_critical_evidence"])
    assert "Decision: NO GO" in markdown
    assert "Missing critical evidence" in markdown
    validate_scorecard(markdown)


def test_build_j1_scorecard_script_writes_explicit_no_go(tmp_path: Path) -> None:
    module = _load_build_j1_scorecard_module()
    paths = _write_scorecard_inputs(tmp_path, primary_gate_passes=False)
    out_root = tmp_path / "scorecard_out"

    assert (
        module.main(
            [
                "--benchmark-root",
                str(paths["benchmark_root"]),
                "--stats-root",
                str(paths["stats_root"]),
                "--comparator-root",
                str(paths["comparator_root"]),
                "--report-root",
                str(paths["report_root"]),
                "--benchmark-provenance",
                str(paths["benchmark_provenance"]),
                "--stats-provenance",
                str(paths["stats_provenance"]),
                "--comparator-provenance",
                str(paths["comparator_provenance"]),
                "--report-provenance",
                str(paths["report_provenance"]),
                "--out-root",
                str(out_root),
            ]
        )
        == 0
    )

    summary = json.loads((out_root / "j1_scorecard.json").read_text(encoding="utf-8"))
    markdown = (out_root / "j1_scorecard.md").read_text(encoding="utf-8")

    assert summary["decision"] == "NO GO"
    assert "no tau cleared Holm-significant primary-comparison improvements" in "; ".join(
        summary["blocking_reasons"]
    )
    assert "Decision: NO GO" in markdown
    validate_scorecard(markdown)

