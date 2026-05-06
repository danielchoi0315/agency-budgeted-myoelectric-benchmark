from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .artifact_schemas import validate_j1_benchmark_manifest, validate_j1_publication_manifest


PRIMARY_GATE_METRICS = ("active_macro_f1", "active_risk_coverage_auc")
PRIMARY_COMPARISON_FAMILIES = ("agency_vs_plain_conf", "matched_tau")
ROBUSTNESS_COMPARISON_FAMILIES = ("agency_vs_set_acsa", "agency_vs_plain_conf")
SAFETY_GATE_METRICS = (
    "stable_safe_episode_rate",
    "final_correct_rate",
    "median_earliest_stable_safe_s",
)
LOWER_IS_BETTER_METRICS = {
    "active_risk_coverage_auc",
    "median_earliest_safe_s",
    "median_earliest_stable_safe_s",
}
MATCHED_BUDGET_GAP_TOLERANCE = 0.05
TAU_PATTERN = re.compile(r"tau[_-]?([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def build_j1_scorecard_from_roots(
    *,
    benchmark_root: str | Path,
    stats_root: str | Path,
    comparator_root: str | Path,
    report_root: str | Path,
    benchmark_provenance: str | Path | None = None,
    stats_provenance: str | Path | None = None,
    comparator_provenance: str | Path | None = None,
    report_provenance: str | Path | None = None,
) -> dict[str, Any]:
    benchmark_dir = Path(benchmark_root)
    stats_dir = Path(stats_root)
    comparator_dir = Path(comparator_root)
    report_dir = Path(report_root)
    return build_j1_scorecard_payload(
        benchmark_manifest_path=benchmark_dir / "benchmark_manifest.json",
        benchmark_aggregate_path=benchmark_dir / "aggregate_policy_metrics.csv",
        stats_primary_pairwise_path=stats_dir / "primary_pairwise.csv",
        comparator_manifest_path=comparator_dir / "j1_publication_extension_manifest.json",
        comparator_match_summary_path=comparator_dir / "j1_confidence_gate_match_summary.csv",
        comparator_pairwise_path=comparator_dir / "j1_confidence_gate_pairwise.csv",
        earliest_safe_summary_path=comparator_dir / "j1_earliest_safe_summary.csv",
        earliest_safe_pairwise_path=comparator_dir / "j1_earliest_safe_pairwise.csv",
        split_report_path=report_dir / "split_report.json",
        benchmark_provenance_path=benchmark_provenance,
        stats_provenance_path=stats_provenance,
        comparator_provenance_path=comparator_provenance,
        report_provenance_path=report_provenance,
    )


def build_j1_scorecard_payload(
    *,
    benchmark_manifest_path: str | Path,
    benchmark_aggregate_path: str | Path,
    stats_primary_pairwise_path: str | Path,
    comparator_manifest_path: str | Path,
    comparator_match_summary_path: str | Path,
    comparator_pairwise_path: str | Path,
    earliest_safe_summary_path: str | Path,
    earliest_safe_pairwise_path: str | Path,
    split_report_path: str | Path,
    benchmark_provenance_path: str | Path | None = None,
    stats_provenance_path: str | Path | None = None,
    comparator_provenance_path: str | Path | None = None,
    report_provenance_path: str | Path | None = None,
) -> dict[str, Any]:
    benchmark_manifest, benchmark_manifest_evidence = _load_json_artifact(
        benchmark_manifest_path,
        label="benchmark_manifest",
        validator=validate_j1_benchmark_manifest,
    )
    benchmark_aggregate, benchmark_aggregate_evidence = _load_csv_artifact(
        benchmark_aggregate_path,
        label="benchmark_aggregate",
        required_columns=(
            "split_family",
            "policy",
            "active_macro_f1",
            "active_risk_coverage_auc",
            "mean_ali",
            "intervention_rate",
        ),
    )
    stats_primary_pairwise, stats_primary_pairwise_evidence = _load_csv_artifact(
        stats_primary_pairwise_path,
        label="stats_primary_pairwise",
        required_columns=(
            "dataset_id",
            "split_family",
            "comparison_family",
            "metric",
            "policy_a",
            "estimate",
            "holm_reject_0_05",
        ),
    )
    comparator_manifest, comparator_manifest_evidence = _load_json_artifact(
        comparator_manifest_path,
        label="comparator_manifest",
        validator=validate_j1_publication_manifest,
    )
    comparator_match_summary, comparator_match_summary_evidence = _load_csv_artifact(
        comparator_match_summary_path,
        label="comparator_match_summary",
        required_columns=(
            "tau",
            "matched_intervention_rate",
            "intervention_rate_gap",
        ),
    )
    comparator_pairwise, comparator_pairwise_evidence = _load_csv_artifact(
        comparator_pairwise_path,
        label="comparator_pairwise",
        required_columns=(
            "split_family",
            "comparison_family",
            "metric",
            "policy_a",
            "estimate",
            "holm_reject",
            "beneficial_and_holm_significant",
        ),
    )
    earliest_safe_summary, earliest_safe_summary_evidence = _load_csv_artifact(
        earliest_safe_summary_path,
        label="earliest_safe_summary",
        required_columns=(
            "dataset_id",
            "split_family",
            "policy",
            "stable_safe_episode_rate",
            "final_correct_rate",
            "median_earliest_stable_safe_s",
        ),
    )
    earliest_safe_pairwise, earliest_safe_pairwise_evidence = _load_csv_artifact(
        earliest_safe_pairwise_path,
        label="earliest_safe_pairwise",
        required_columns=(
            "split_family",
            "comparison_family",
            "metric",
            "policy_a",
            "estimate",
            "holm_reject",
        ),
    )
    split_report, split_report_evidence = _load_json_artifact(
        split_report_path,
        label="split_report",
        validator=_validate_split_report,
    )

    provenance_specs = {
        "benchmark_provenance": benchmark_provenance_path,
        "stats_provenance": stats_provenance_path,
        "comparator_provenance": comparator_provenance_path,
        "report_provenance": report_provenance_path,
    }
    provenance_payloads: dict[str, dict[str, Any] | None] = {}
    provenance_evidence: dict[str, dict[str, Any]] = {}
    for label, path in provenance_specs.items():
        payload, evidence = _load_json_artifact(
            path,
            label=label,
            validator=_validate_provenance_manifest,
            allow_missing=True,
        )
        provenance_payloads[label] = payload
        provenance_evidence[label] = evidence

    evidence = {
        benchmark_manifest_evidence["label"]: benchmark_manifest_evidence,
        benchmark_aggregate_evidence["label"]: benchmark_aggregate_evidence,
        stats_primary_pairwise_evidence["label"]: stats_primary_pairwise_evidence,
        comparator_manifest_evidence["label"]: comparator_manifest_evidence,
        comparator_match_summary_evidence["label"]: comparator_match_summary_evidence,
        comparator_pairwise_evidence["label"]: comparator_pairwise_evidence,
        earliest_safe_summary_evidence["label"]: earliest_safe_summary_evidence,
        earliest_safe_pairwise_evidence["label"]: earliest_safe_pairwise_evidence,
        split_report_evidence["label"]: split_report_evidence,
        **provenance_evidence,
    }

    required_families = _resolve_required_families(
        comparator_manifest=comparator_manifest,
        benchmark_aggregate=benchmark_aggregate,
        split_report=split_report,
    )
    candidate = _build_candidate_summary(
        benchmark_manifest=benchmark_manifest,
        comparator_manifest=comparator_manifest,
    )
    primary_gate = _evaluate_primary_gate(
        stats_primary_pairwise,
        required_families=required_families,
    )
    candidate["selected_tau"] = primary_gate["selected_tau"]
    candidate["selected_policy"] = primary_gate["selected_policy"]
    benchmark_selection = _evaluate_benchmark_selection(
        benchmark_aggregate,
        selected_policy=primary_gate["selected_policy"],
        required_families=required_families,
    )
    robustness_gate = _evaluate_robustness_gate(
        comparator_pairwise,
        comparator_match_summary,
        selected_policy=primary_gate["selected_policy"],
        selected_tau=primary_gate["selected_tau"],
        required_families=required_families,
    )
    safety_gate = _evaluate_safety_gate(
        earliest_safe_summary,
        earliest_safe_pairwise,
        selected_policy=primary_gate["selected_policy"],
        required_families=required_families,
    )
    boundary_gate = _evaluate_boundary_gate(
        split_report,
        required_families=required_families,
    )
    readiness_gate = _evaluate_readiness_gate(
        candidate=candidate,
        provenance_payloads=provenance_payloads,
    )

    missing_critical_evidence = _collect_missing_critical_evidence(
        evidence=evidence,
        primary_gate=primary_gate,
        benchmark_selection=benchmark_selection,
        robustness_gate=robustness_gate,
        safety_gate=safety_gate,
        boundary_gate=boundary_gate,
        readiness_gate=readiness_gate,
    )
    blocking_reasons = _collect_blocking_reasons(
        primary_gate=primary_gate,
        benchmark_selection=benchmark_selection,
        robustness_gate=robustness_gate,
        safety_gate=safety_gate,
        boundary_gate=boundary_gate,
        readiness_gate=readiness_gate,
    )
    go = (
        not missing_critical_evidence
        and primary_gate["passed"]
        and benchmark_selection["passed"]
        and robustness_gate["passed"]
        and safety_gate["passed"]
        and boundary_gate["passed"]
        and readiness_gate["passed"]
    )

    payload = {
        "dataset": "j1",
        "decision": "GO" if go else "NO GO",
        "go": bool(go),
        "required_split_families": required_families,
        "candidate": candidate,
        "critical_evidence": evidence,
        "missing_critical_evidence": missing_critical_evidence,
        "blocking_reasons": blocking_reasons,
        "gates": {
            "primary_db10_gate": primary_gate,
            "benchmark_selection": benchmark_selection,
            "robustness_panel": robustness_gate,
            "safety_timing_panel": safety_gate,
            "boundary_conditions": boundary_gate,
            "readiness_reproducibility": readiness_gate,
        },
        "decision_log": _build_decision_log(
            go=go,
            primary_gate=primary_gate,
            benchmark_selection=benchmark_selection,
            robustness_gate=robustness_gate,
            safety_gate=safety_gate,
            boundary_gate=boundary_gate,
            readiness_gate=readiness_gate,
            missing_critical_evidence=missing_critical_evidence,
            blocking_reasons=blocking_reasons,
        ),
    }
    return payload


def render_j1_scorecard(payload: Mapping[str, Any]) -> str:
    candidate = dict(payload["candidate"])
    primary_gate = dict(payload["gates"]["primary_db10_gate"])
    benchmark_selection = dict(payload["gates"]["benchmark_selection"])
    robustness_gate = dict(payload["gates"]["robustness_panel"])
    safety_gate = dict(payload["gates"]["safety_timing_panel"])
    boundary_gate = dict(payload["gates"]["boundary_conditions"])
    readiness_gate = dict(payload["gates"]["readiness_reproducibility"])
    missing_critical = list(payload.get("missing_critical_evidence", []))
    blocking_reasons = list(payload.get("blocking_reasons", []))
    selected_policy = candidate.get("selected_policy") or "<unselected>"
    selected_tau = candidate.get("selected_tau") or "<none>"

    lines = [
        "# J1-open Scorecard",
        "",
        "## Decision Banner",
        f"- Decision: {payload['decision']}",
        f"- Candidate: {candidate.get('user_model', '<unknown>')} / {candidate.get('assist_model', '<unknown>')}",
        f"- Operating point: {selected_policy} (tau {selected_tau})",
        f"- Families in scope: {', '.join(payload.get('required_split_families', [])) or '<none>'}",
    ]
    if missing_critical:
        lines.append(f"- Missing critical evidence: {'; '.join(missing_critical)}")
    if blocking_reasons:
        lines.append(f"- Blocking reasons: {'; '.join(blocking_reasons)}")
    if not missing_critical and not blocking_reasons:
        lines.append("- Every blocking gate is currently satisfied.")
    lines.extend(
        [
            "",
            "## Frozen Candidate and Operating Point",
            f"- Prepared root: {candidate.get('prepared_root', '<unknown>')}",
            f"- Benchmark rows: {candidate.get('benchmark_rows', '<unknown>')}",
            f"- Comparator trace count: {candidate.get('trace_count', '<unknown>')}",
            f"- Pairwise repeats: {candidate.get('pairwise_repeats', '<unknown>')}",
        ]
    )
    if benchmark_selection["family_metrics"]:
        lines.append("- Selected policy aggregate metrics:")
        for row in benchmark_selection["family_metrics"]:
            lines.append(
                "  - "
                f"{row['split_family']}: active_macro_f1={_fmt_float(row['active_macro_f1'])}, "
                f"active_risk_coverage_auc={_fmt_float(row['active_risk_coverage_auc'])}, "
                f"mean_ali={_fmt_float(row['mean_ali'])}, "
                f"intervention_rate={_fmt_float(row['intervention_rate'])}, "
                f"median_decision_time_ms={_fmt_float(row.get('median_decision_time_ms'))}"
            )
    else:
        lines.append("- Selected policy aggregate metrics are unavailable.")

    lines.extend(
        [
            "",
            "## Primary DB10 Gate",
            f"- Status: {'PASS' if primary_gate['passed'] else 'FAIL'}",
            f"- Qualifying taus: {', '.join(primary_gate['qualifying_taus']) or '<none>'}",
            f"- Selected tau: {primary_gate.get('selected_tau') or '<none>'}",
        ]
    )
    if primary_gate["missing_evidence"]:
        lines.append(f"- Missing evidence: {'; '.join(primary_gate['missing_evidence'])}")
    if primary_gate["failures"]:
        lines.append(f"- Failures: {'; '.join(primary_gate['failures'])}")

    lines.extend(
        [
            "",
            "## Safety/Timing Panel",
            f"- Status: {'PASS' if safety_gate['passed'] else 'FAIL'}",
            (
                "- Mean stable-safe episode rate: "
                f"{_fmt_float(safety_gate['mean_stable_safe_episode_rate'])}"
                if safety_gate["mean_stable_safe_episode_rate"] is not None
                else "- Mean stable-safe episode rate: <unavailable>"
            ),
            (
                "- Mean final correct rate: "
                f"{_fmt_float(safety_gate['mean_final_correct_rate'])}"
                if safety_gate["mean_final_correct_rate"] is not None
                else "- Mean final correct rate: <unavailable>"
            ),
            (
                "- Median earliest stable-safe time: "
                f"{_fmt_float(safety_gate['median_earliest_stable_safe_s'])} s"
                if safety_gate["median_earliest_stable_safe_s"] is not None
                else "- Median earliest stable-safe time: <unavailable>"
            ),
        ]
    )
    if safety_gate["missing_evidence"]:
        lines.append(f"- Missing evidence: {'; '.join(safety_gate['missing_evidence'])}")
    if safety_gate["failures"]:
        lines.append(f"- Failures: {'; '.join(safety_gate['failures'])}")

    lines.extend(
        [
            "",
            "## Robustness Panel",
            f"- Status: {'PASS' if robustness_gate['passed'] else 'FAIL'}",
            (
                "- Matched-budget intervention-rate gap: "
                f"{_fmt_float(robustness_gate['matched_budget_gap'])}"
                if robustness_gate["matched_budget_gap"] is not None
                else "- Matched-budget intervention-rate gap: <unavailable>"
            ),
        ]
    )
    if robustness_gate["missing_evidence"]:
        lines.append(f"- Missing evidence: {'; '.join(robustness_gate['missing_evidence'])}")
    if robustness_gate["failures"]:
        lines.append(f"- Failures: {'; '.join(robustness_gate['failures'])}")

    lines.extend(
        [
            "",
            "## Boundary Conditions",
            f"- Status: {'PASS' if boundary_gate['passed'] else 'FAIL'}",
            f"- Split report status: {boundary_gate['report_status']}",
            f"- Covered families: {', '.join(boundary_gate['covered_families']) or '<none>'}",
        ]
    )
    if boundary_gate["missing_evidence"]:
        lines.append(f"- Missing evidence: {'; '.join(boundary_gate['missing_evidence'])}")
    if boundary_gate["failures"]:
        lines.append(f"- Failures: {'; '.join(boundary_gate['failures'])}")

    lines.extend(
        [
            "",
            "## Readiness and Reproducibility Status",
            f"- Status: {'PASS' if readiness_gate['passed'] else 'FAIL'}",
            f"- Model alignment: {'PASS' if readiness_gate['manifest_alignment_passed'] else 'FAIL'}",
            f"- Valid provenance manifests: {readiness_gate['valid_provenance_count']} / {readiness_gate['expected_provenance_count']}",
        ]
    )
    if readiness_gate["missing_evidence"]:
        lines.append(f"- Missing evidence: {'; '.join(readiness_gate['missing_evidence'])}")
    if readiness_gate["failures"]:
        lines.append(f"- Failures: {'; '.join(readiness_gate['failures'])}")

    lines.extend(
        [
            "",
            "## Candidate Provenance",
            f"- Benchmark manifest path: {payload['critical_evidence']['benchmark_manifest']['path']}",
            f"- Comparator manifest path: {payload['critical_evidence']['comparator_manifest']['path']}",
            f"- Split report path: {payload['critical_evidence']['split_report']['path']}",
            f"- Benchmark prepared root: {candidate.get('benchmark_prepared_root', '<unknown>')}",
            f"- Comparator prepared root: {candidate.get('comparator_prepared_root', '<unknown>')}",
        ]
    )

    lines.extend(
        [
            "",
            "## Decision Log",
        ]
    )
    for entry in payload.get("decision_log", []):
        lines.append(f"- {entry}")
    lines.append("")
    return "\n".join(lines)


def _build_candidate_summary(
    *,
    benchmark_manifest: Mapping[str, Any] | None,
    comparator_manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    user_model = ""
    assist_model = ""
    benchmark_prepared_root = ""
    comparator_prepared_root = ""
    benchmark_rows: int | None = None
    trace_count: int | None = None
    pairwise_repeats: int | None = None

    if benchmark_manifest is not None:
        user_model = str(benchmark_manifest.get("user_model", "")).strip()
        assist_model = str(benchmark_manifest.get("assist_model", "")).strip()
        benchmark_prepared_root = str(benchmark_manifest.get("prepared_root", "")).strip()
        n_rows = benchmark_manifest.get("n_rows")
        benchmark_rows = int(n_rows) if isinstance(n_rows, int) else None
    if comparator_manifest is not None:
        user_model = user_model or str(comparator_manifest.get("user_model", "")).strip()
        assist_model = assist_model or str(comparator_manifest.get("assist_model", "")).strip()
        comparator_prepared_root = str(comparator_manifest.get("prepared_root", "")).strip()
        trace = comparator_manifest.get("trace_count")
        repeats = comparator_manifest.get("pairwise_repeats")
        trace_count = int(trace) if isinstance(trace, int) else None
        pairwise_repeats = int(repeats) if isinstance(repeats, int) else None

    prepared_root = benchmark_prepared_root or comparator_prepared_root or "<unknown>"
    return {
        "user_model": user_model or "<unknown>",
        "assist_model": assist_model or "<unknown>",
        "prepared_root": prepared_root,
        "benchmark_prepared_root": benchmark_prepared_root or "<unknown>",
        "comparator_prepared_root": comparator_prepared_root or "<unknown>",
        "benchmark_rows": benchmark_rows,
        "trace_count": trace_count,
        "pairwise_repeats": pairwise_repeats,
        "selected_tau": None,
        "selected_policy": None,
    }


def _evaluate_primary_gate(
    frame: pd.DataFrame | None,
    *,
    required_families: list[str],
) -> dict[str, Any]:
    missing: list[str] = []
    failures: list[str] = []
    qualifying_taus: list[str] = []

    if frame is None:
        missing.append("primary pairwise statistics are unavailable")
    elif not required_families:
        missing.append("required split families could not be resolved for the primary gate")
    else:
        filtered = frame.copy()
        filtered = filtered.loc[filtered["dataset_id"].astype(str).str.lower() == "j1"].copy()
        filtered = filtered.loc[
            filtered["comparison_family"].astype(str).isin(PRIMARY_COMPARISON_FAMILIES)
        ].copy()
        filtered["tau"] = filtered.apply(_extract_tau_from_pairwise_row, axis=1)
        filtered = filtered.loc[filtered["tau"].notna()].copy()
        if filtered.empty:
            missing.append("no J1 primary-comparison rows were found in primary_pairwise.csv")
        else:
            filtered["passed"] = filtered.apply(_primary_pairwise_row_passes, axis=1)
            required_family_set = set(required_families)
            for tau, tau_df in filtered.groupby("tau", sort=True):
                metrics_complete = True
                for metric in PRIMARY_GATE_METRICS:
                    metric_df = tau_df.loc[
                        (tau_df["metric"].astype(str) == metric) & tau_df["passed"].astype(bool)
                    ]
                    observed = set(metric_df["split_family"].astype(str))
                    if not required_family_set.issubset(observed):
                        metrics_complete = False
                        break
                if metrics_complete:
                    qualifying_taus.append(str(tau))
            if not qualifying_taus:
                failures.append(
                    "no tau cleared Holm-significant primary-comparison improvements for both active_macro_f1 "
                    "and active_risk_coverage_auc across every required split family"
                )

    selected_tau = qualifying_taus[0] if qualifying_taus else None
    return {
        "passed": selected_tau is not None and not missing and not failures,
        "required_families": list(required_families),
        "qualifying_taus": qualifying_taus,
        "selected_tau": selected_tau,
        "selected_policy": _agency_policy_for_tau(selected_tau) if selected_tau is not None else None,
        "missing_evidence": missing,
        "failures": failures,
    }


def _evaluate_benchmark_selection(
    frame: pd.DataFrame | None,
    *,
    selected_policy: str | None,
    required_families: list[str],
) -> dict[str, Any]:
    missing: list[str] = []
    failures: list[str] = []
    family_metrics: list[dict[str, Any]] = []

    if selected_policy is None:
        failures.append("no operating point was selected from the primary gate")
    elif frame is None:
        missing.append("benchmark aggregate metrics are unavailable")
    else:
        subset = frame.loc[frame["policy"].astype(str) == selected_policy].copy()
        observed_families = set(subset["split_family"].astype(str))
        required_family_set = set(required_families)
        if not required_family_set.issubset(observed_families):
            missing.append(
                "benchmark aggregate metrics are incomplete for the selected policy across the required families"
            )
        else:
            subset = subset.loc[subset["split_family"].astype(str).isin(required_families)].copy()
            family_metrics = [
                {
                    "split_family": str(row["split_family"]),
                    "active_macro_f1": float(row["active_macro_f1"]),
                    "active_risk_coverage_auc": float(row["active_risk_coverage_auc"]),
                    "mean_ali": float(row["mean_ali"]),
                    "intervention_rate": float(row["intervention_rate"]),
                    "median_decision_time_ms": (
                        float(row["median_decision_time_ms"])
                        if "median_decision_time_ms" in subset.columns and pd.notna(row.get("median_decision_time_ms"))
                        else None
                    ),
                }
                for _, row in subset.sort_values("split_family").iterrows()
            ]

    return {
        "passed": not missing and not failures and bool(family_metrics),
        "selected_policy": selected_policy,
        "family_metrics": family_metrics,
        "missing_evidence": missing,
        "failures": failures,
    }


def _evaluate_robustness_gate(
    pairwise: pd.DataFrame | None,
    match_summary: pd.DataFrame | None,
    *,
    selected_policy: str | None,
    selected_tau: str | None,
    required_families: list[str],
) -> dict[str, Any]:
    missing: list[str] = []
    failures: list[str] = []
    matched_budget_gap: float | None = None

    if selected_policy is None or selected_tau is None:
        failures.append("no operating point was selected for comparator review")
    if match_summary is None:
        missing.append("matched-budget comparator summary is unavailable")
    elif selected_tau is not None:
        summary = match_summary.copy()
        summary["tau"] = summary["tau"].map(_normalize_tau_value)
        row_df = summary.loc[summary["tau"] == selected_tau].copy()
        if row_df.empty:
            missing.append(f"matched-budget comparator summary is missing tau {selected_tau}")
        else:
            matched_budget_gap = float(row_df.iloc[0]["intervention_rate_gap"])
            if abs(matched_budget_gap) > MATCHED_BUDGET_GAP_TOLERANCE:
                failures.append(
                    f"matched-budget intervention-rate gap {matched_budget_gap:.4f} exceeds "
                    f"{MATCHED_BUDGET_GAP_TOLERANCE:.4f}"
                )

    if pairwise is None:
        missing.append("confidence-gate pairwise comparator stats are unavailable")
    elif selected_policy is not None:
        required_family_set = set(required_families)
        policy_rows = pairwise.loc[pairwise["policy_a"].astype(str) == selected_policy].copy()
        if policy_rows.empty:
            missing.append(f"confidence-gate pairwise stats are missing rows for {selected_policy}")
        for comparison_family in ROBUSTNESS_COMPARISON_FAMILIES:
            for metric in PRIMARY_GATE_METRICS:
                subset = policy_rows.loc[
                    (policy_rows["comparison_family"].astype(str) == comparison_family)
                    & (policy_rows["metric"].astype(str) == metric)
                ].copy()
                observed = set(subset["split_family"].astype(str))
                if not required_family_set.issubset(observed):
                    missing.append(
                        f"confidence-gate pairwise stats are incomplete for {comparison_family}/{metric}"
                    )
                    continue
                subset = subset.loc[subset["split_family"].astype(str).isin(required_families)].copy()
                if not subset["beneficial_and_holm_significant"].astype(bool).all():
                    failures.append(
                        f"{comparison_family} does not show beneficial Holm-significant {metric} "
                        f"for every required split family at {selected_policy}"
                    )

    return {
        "passed": not missing and not failures,
        "selected_policy": selected_policy,
        "selected_tau": selected_tau,
        "matched_budget_gap": matched_budget_gap,
        "missing_evidence": _dedupe(missing),
        "failures": _dedupe(failures),
    }


def _evaluate_safety_gate(
    summary: pd.DataFrame | None,
    pairwise: pd.DataFrame | None,
    *,
    selected_policy: str | None,
    required_families: list[str],
) -> dict[str, Any]:
    missing: list[str] = []
    failures: list[str] = []
    mean_stable_safe_episode_rate: float | None = None
    mean_final_correct_rate: float | None = None
    median_earliest_stable_safe_s: float | None = None

    if selected_policy is None:
        failures.append("no operating point was selected for safety review")
    if summary is None:
        missing.append("earliest-safe summary is unavailable")
    elif selected_policy is not None:
        selected_rows = summary.loc[
            (summary["dataset_id"].astype(str).str.lower() == "j1")
            & (summary["policy"].astype(str) == selected_policy)
        ].copy()
        observed_families = set(selected_rows["split_family"].astype(str))
        required_family_set = set(required_families)
        if not required_family_set.issubset(observed_families):
            missing.append(
                f"earliest-safe summary is incomplete for the selected policy {selected_policy}"
            )
        else:
            selected_rows = selected_rows.loc[
                selected_rows["split_family"].astype(str).isin(required_families)
            ].copy()
            mean_stable_safe_episode_rate = float(
                selected_rows["stable_safe_episode_rate"].astype(float).mean()
            )
            mean_final_correct_rate = float(
                selected_rows["final_correct_rate"].astype(float).mean()
            )
            median_earliest_stable_safe_s = _safe_median(
                selected_rows["median_earliest_stable_safe_s"].astype(float)
            )

    if pairwise is None:
        missing.append("earliest-safe pairwise stats are unavailable")
    elif selected_policy is not None:
        required_family_set = set(required_families)
        policy_rows = pairwise.loc[pairwise["policy_a"].astype(str) == selected_policy].copy()
        if policy_rows.empty:
            missing.append(f"earliest-safe pairwise stats are missing rows for {selected_policy}")
        for comparison_family in ROBUSTNESS_COMPARISON_FAMILIES:
            for metric in SAFETY_GATE_METRICS:
                subset = policy_rows.loc[
                    (policy_rows["comparison_family"].astype(str) == comparison_family)
                    & (policy_rows["metric"].astype(str) == metric)
                ].copy()
                observed = set(subset["split_family"].astype(str))
                if not required_family_set.issubset(observed):
                    missing.append(
                        f"earliest-safe pairwise stats are incomplete for {comparison_family}/{metric}"
                    )
                    continue
                subset = subset.loc[subset["split_family"].astype(str).isin(required_families)].copy()
                harmful = subset.apply(_pairwise_row_is_harmful_regression, axis=1)
                significant = subset["holm_reject"].astype(bool)
                if bool((harmful & significant).any()):
                    failures.append(
                        f"{comparison_family} shows a Holm-significant safety regression on {metric} "
                        f"for {selected_policy}"
                    )

    return {
        "passed": not missing and not failures,
        "selected_policy": selected_policy,
        "mean_stable_safe_episode_rate": mean_stable_safe_episode_rate,
        "mean_final_correct_rate": mean_final_correct_rate,
        "median_earliest_stable_safe_s": median_earliest_stable_safe_s,
        "missing_evidence": _dedupe(missing),
        "failures": _dedupe(failures),
    }


def _evaluate_boundary_gate(
    split_report: Mapping[str, Any] | None,
    *,
    required_families: list[str],
) -> dict[str, Any]:
    missing: list[str] = []
    failures: list[str] = []
    report_status = "missing"
    covered_families: list[str] = []

    if split_report is None:
        missing.append("split leakage report is unavailable")
    else:
        report_status = str(split_report.get("status", "unknown"))
        summary = dict(split_report.get("summary", {}))
        family_rows = list(split_report.get("families", []))
        covered_families = sorted(
            str(row.get("split_family", "")).strip()
            for row in family_rows
            if str(row.get("split_family", "")).strip()
        )
        if set(required_families) and not set(required_families).issubset(set(covered_families)):
            missing.append("split leakage report does not cover every required split family")
        if report_status != "ok":
            failures.append(f"split report status is {report_status}")
        if not bool(summary.get("critical_overlap_checks_passed")):
            failures.append("critical overlap checks did not pass in the split report")
        forward_applicable = int(summary.get("forward_order_checks_applicable", 0) or 0)
        forward_passed = bool(summary.get("forward_order_checks_passed"))
        if forward_applicable > 0 and not forward_passed:
            failures.append("forward-order checks failed in the split report")

    return {
        "passed": not missing and not failures,
        "report_status": report_status,
        "covered_families": covered_families,
        "missing_evidence": _dedupe(missing),
        "failures": _dedupe(failures),
    }


def _evaluate_readiness_gate(
    *,
    candidate: Mapping[str, Any],
    provenance_payloads: Mapping[str, Mapping[str, Any] | None],
) -> dict[str, Any]:
    missing: list[str] = []
    failures: list[str] = []

    benchmark_root = str(candidate.get("benchmark_prepared_root", "")).strip()
    comparator_root = str(candidate.get("comparator_prepared_root", "")).strip()
    manifest_alignment_passed = bool(
        benchmark_root
        and comparator_root
        and benchmark_root == comparator_root
    )
    if not manifest_alignment_passed:
        failures.append("benchmark and comparator manifests do not agree on the prepared root")

    valid_provenance_count = 0
    for label, payload in provenance_payloads.items():
        if payload is None:
            missing.append(f"{label} is missing")
            continue
        valid_provenance_count += 1
        missing_files = [
            str(file_record.get("path", "<unknown>"))
            for file_record in payload.get("files", [])
            if file_record.get("exists") is False
        ]
        if missing_files:
            failures.append(f"{label} records missing files: {', '.join(missing_files)}")

    return {
        "passed": not missing and not failures and manifest_alignment_passed,
        "manifest_alignment_passed": manifest_alignment_passed,
        "expected_provenance_count": int(len(provenance_payloads)),
        "valid_provenance_count": int(valid_provenance_count),
        "missing_evidence": _dedupe(missing),
        "failures": _dedupe(failures),
    }


def _collect_missing_critical_evidence(
    *,
    evidence: Mapping[str, Mapping[str, Any]],
    primary_gate: Mapping[str, Any],
    benchmark_selection: Mapping[str, Any],
    robustness_gate: Mapping[str, Any],
    safety_gate: Mapping[str, Any],
    boundary_gate: Mapping[str, Any],
    readiness_gate: Mapping[str, Any],
) -> list[str]:
    missing: list[str] = []
    for label, artifact in evidence.items():
        if not artifact.get("valid", False):
            error = str(artifact.get("error", "invalid artifact"))
            missing.append(f"{label}: {error}")
    for gate in (
        primary_gate,
        benchmark_selection,
        robustness_gate,
        safety_gate,
        boundary_gate,
        readiness_gate,
    ):
        missing.extend(str(item) for item in gate.get("missing_evidence", []))
    return _dedupe(missing)


def _collect_blocking_reasons(
    *,
    primary_gate: Mapping[str, Any],
    benchmark_selection: Mapping[str, Any],
    robustness_gate: Mapping[str, Any],
    safety_gate: Mapping[str, Any],
    boundary_gate: Mapping[str, Any],
    readiness_gate: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    for gate in (
        primary_gate,
        benchmark_selection,
        robustness_gate,
        safety_gate,
        boundary_gate,
        readiness_gate,
    ):
        reasons.extend(str(item) for item in gate.get("failures", []))
    return _dedupe(reasons)


def _build_decision_log(
    *,
    go: bool,
    primary_gate: Mapping[str, Any],
    benchmark_selection: Mapping[str, Any],
    robustness_gate: Mapping[str, Any],
    safety_gate: Mapping[str, Any],
    boundary_gate: Mapping[str, Any],
    readiness_gate: Mapping[str, Any],
    missing_critical_evidence: Sequence[str],
    blocking_reasons: Sequence[str],
) -> list[str]:
    entries = [
        (
            f"Primary gate selected tau {primary_gate['selected_tau']}."
            if primary_gate.get("selected_tau")
            else "Primary gate did not release an operating point."
        ),
        (
            f"Benchmark aggregate coverage is {'complete' if benchmark_selection['passed'] else 'incomplete'} "
            f"for {benchmark_selection.get('selected_policy') or '<unselected>'}."
        ),
        (
            f"Robustness comparator review is {'clear' if robustness_gate['passed'] else 'blocked'}."
        ),
        (
            f"Safety/timing review is {'clear' if safety_gate['passed'] else 'blocked'}."
        ),
        (
            f"Boundary report status is {boundary_gate['report_status']}."
        ),
        (
            f"Readiness and reproducibility are {'aligned' if readiness_gate['passed'] else 'not aligned'}."
        ),
    ]
    if missing_critical_evidence:
        entries.append(f"Missing critical evidence forced a closed decision: {'; '.join(missing_critical_evidence)}")
    if blocking_reasons:
        entries.append(f"Blocking findings: {'; '.join(blocking_reasons)}")
    entries.append(f"Final release decision: {'GO' if go else 'NO GO'}.")
    return entries


def _resolve_required_families(
    *,
    comparator_manifest: Mapping[str, Any] | None,
    benchmark_aggregate: pd.DataFrame | None,
    split_report: Mapping[str, Any] | None,
) -> list[str]:
    if comparator_manifest is not None:
        families = [
            str(item).strip()
            for item in comparator_manifest.get("split_families", [])
            if str(item).strip()
        ]
        if families:
            return sorted(dict.fromkeys(families))
    if benchmark_aggregate is not None and not benchmark_aggregate.empty:
        families = sorted(
            str(item).strip()
            for item in benchmark_aggregate["split_family"].astype(str).unique().tolist()
            if str(item).strip()
        )
        if families:
            return families
    if split_report is not None:
        families = [
            str(row.get("split_family", "")).strip()
            for row in split_report.get("families", [])
            if str(row.get("split_family", "")).strip()
        ]
        if families:
            return sorted(dict.fromkeys(families))
    return []


def _load_json_artifact(
    path: str | Path | None,
    *,
    label: str,
    validator,
    allow_missing: bool = False,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    resolved_path = None if path is None else Path(path)
    evidence = {
        "label": label,
        "path": str(resolved_path) if resolved_path is not None else "",
        "present": False,
        "valid": False,
        "error": None,
    }
    if resolved_path is None:
        if allow_missing:
            evidence["error"] = "no path provided"
            return None, evidence
        evidence["error"] = "path is required"
        return None, evidence
    if not resolved_path.is_file():
        evidence["error"] = "file does not exist"
        return None, evidence
    evidence["present"] = True
    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("JSON payload must be an object")
        validated = validator(dict(payload))
    except Exception as exc:  # pragma: no cover - defensive wrapper
        evidence["error"] = str(exc)
        return None, evidence
    evidence["valid"] = True
    return dict(validated), evidence


def _load_csv_artifact(
    path: str | Path,
    *,
    label: str,
    required_columns: Sequence[str],
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    resolved_path = Path(path)
    evidence = {
        "label": label,
        "path": str(resolved_path),
        "present": False,
        "valid": False,
        "error": None,
    }
    if not resolved_path.is_file():
        evidence["error"] = "file does not exist"
        return None, evidence
    evidence["present"] = True
    try:
        frame = pd.read_csv(resolved_path)
    except Exception as exc:  # pragma: no cover - defensive wrapper
        evidence["error"] = str(exc)
        return None, evidence
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        evidence["error"] = f"missing required columns: {', '.join(missing)}"
        return None, evidence
    evidence["valid"] = True
    return frame, evidence


def _validate_split_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    dataset_id = str(payload.get("dataset_id", "")).strip().lower()
    if dataset_id != "j1":
        raise ValueError("split report dataset_id must be 'j1'")
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("split report must contain a summary object")
    families = payload.get("families")
    if not isinstance(families, list):
        raise ValueError("split report must contain a families list")
    splits = payload.get("splits")
    if not isinstance(splits, list):
        raise ValueError("split report must contain a splits list")
    return dict(payload)


def _validate_provenance_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact_type = str(payload.get("artifact_type", "")).strip()
    if not artifact_type:
        raise ValueError("provenance manifest must contain artifact_type")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("provenance manifest must contain a non-empty files list")
    for record in files:
        if not isinstance(record, Mapping):
            raise ValueError("provenance manifest file entries must be objects")
        if "path" not in record or "exists" not in record:
            raise ValueError("provenance manifest file entries must include path and exists")
    return dict(payload)


def _extract_tau_from_pairwise_row(row: pd.Series) -> str | None:
    for column in ("policy_a", "policy_b"):
        if column in row and isinstance(row[column], str):
            tau = _extract_tau(row[column])
            if tau is not None:
                return tau
    return None


def _primary_pairwise_row_passes(row: pd.Series) -> bool:
    metric = str(row["metric"])
    estimate = float(row["estimate"])
    rejected = bool(row["holm_reject_0_05"])
    return rejected and _estimate_is_beneficial(metric, estimate)


def _pairwise_row_is_harmful_regression(row: pd.Series) -> bool:
    metric = str(row["metric"])
    estimate = float(row["estimate"])
    if metric in LOWER_IS_BETTER_METRICS:
        return estimate > 0.0
    return estimate < 0.0


def _estimate_is_beneficial(metric: str, estimate: float) -> bool:
    if metric in LOWER_IS_BETTER_METRICS:
        return estimate < 0.0
    return estimate > 0.0


def _agency_policy_for_tau(tau: str | None) -> str | None:
    if tau is None:
        return None
    return f"agency_margin_tau_{tau}"


def _extract_tau(value: str) -> str | None:
    match = TAU_PATTERN.search(str(value))
    if match is None:
        return None
    return _normalize_tau_value(match.group(1))


def _normalize_tau_value(value: object) -> str:
    try:
        numeric = float(str(value).strip())
    except ValueError as exc:  # pragma: no cover - defensive wrapper
        raise ValueError(f"invalid tau value: {value}") from exc
    return f"{numeric:.2f}"


def _safe_median(values: pd.Series) -> float | None:
    arr = values.to_numpy(dtype=float)
    if arr.size == 0:
        return None
    finite = arr[~pd.isna(arr)]
    if finite.size == 0:
        return None
    return float(pd.Series(finite).median())


def _fmt_float(value: object) -> str:
    if value is None:
        return "<none>"
    number = float(value)
    if math.isnan(number):
        return "nan"
    return f"{number:.4f}"


def _dedupe(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value).strip()))


__all__ = [
    "MATCHED_BUDGET_GAP_TOLERANCE",
    "PRIMARY_GATE_METRICS",
    "ROBUSTNESS_COMPARISON_FAMILIES",
    "SAFETY_GATE_METRICS",
    "build_j1_scorecard_from_roots",
    "build_j1_scorecard_payload",
    "render_j1_scorecard",
]

