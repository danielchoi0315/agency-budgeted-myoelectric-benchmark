from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .j1_claims import load_j1_claims
from .j1_scorecard import LOWER_IS_BETTER_METRICS


PRIMARY_METRICS = ("active_macro_f1", "active_risk_coverage_auc")
PRIMARY_COMPARISON_FAMILIES = ("agency_vs_plain_conf", "matched_tau")
ISO_BUDGET_COMPARISON_FAMILY = "agency_vs_plain_conf_iso_budget"
EXACT_BUDGET_COMPARISON_FAMILY = "agency_vs_plain_conf_exact_budget"
SAFETY_COMPARATOR_METRICS = (
    "stable_safe_episode_rate",
    "final_correct_rate",
    "median_earliest_stable_safe_s",
)
FRONTIER_OBJECTIVES: dict[str, str] = {
    "active_macro_f1": "max",
    "active_risk_coverage_auc": "min",
    "stable_safe_episode_rate": "max",
    "median_earliest_stable_safe_s": "min",
}
COMPARATOR_OBJECTIVES: dict[str, str] = {
    "active_macro_f1": "max",
    "active_risk_coverage_auc": "min",
    "stable_safe_episode_rate": "max",
    "mean_ali": "min",
}
REGRET_METRICS = (
    "active_macro_f1",
    "active_risk_coverage_auc",
    "stable_safe_episode_rate",
    "mean_ali",
)
UNIT_DELTA_METRICS = (
    "active_macro_f1",
    "active_risk_coverage_auc",
    "mean_ali",
    "stable_safe_episode_rate",
    "final_correct_rate",
    "median_earliest_stable_safe_s",
)
PACKAGE_LOWER_IS_BETTER_METRICS = set(LOWER_IS_BETTER_METRICS).union(
    {
        "mean_ali",
        "intervention_rate",
        "median_earliest_safe_s",
        "median_earliest_stable_safe_s",
    }
)
J1_FRONTIER_PACKAGE_ARTIFACTS = (
    "frontier_points.csv",
    "pareto_frontier.csv",
    "frontier_overlap.csv",
    "non_dominated_summary.csv",
    "universal_tau_regret.csv",
    "frontier_unit_deltas.csv",
    "budget_match_summary.csv",
    "iso_budget_claim_summary.csv",
    "exact_budget_claim_summary.csv",
    "operating_point_instability.json",
    "frontier_atlas.png",
    "safety_timing_frontier.png",
    "j1_frontier_package.json",
    "j1_frontier_package.md",
    "j1_frontier_package_provenance.json",
)
TAU_PATTERN = re.compile(r"tau[_-]?([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def build_j1_frontier_readout_payload(
    *,
    scorecard_path: str | Path,
    primary_pairwise_path: str | Path,
) -> dict[str, Any]:
    scorecard = _load_json_mapping(scorecard_path)
    if str(scorecard.get("dataset", "")).lower() != "j1":
        raise ValueError("frontier readout requires a J1 scorecard payload")

    required_families = [
        str(family)
        for family in scorecard.get("required_split_families", [])
        if str(family).strip()
    ]
    if not required_families:
        raise ValueError("scorecard did not declare any required J1 split families")

    pairwise = _load_primary_pairwise(primary_pairwise_path, required_families=required_families)
    rows = _summarize_pairwise_rows(pairwise)
    universal_taus = _resolve_universal_taus(rows, required_families=required_families)
    family_summary = _build_family_summary(rows, required_families=required_families)

    boundary_passed, readiness_passed, boundary_status, readiness_status = _extract_scorecard_statuses(
        scorecard
    )
    best_tau_set = {
        entry["best_tau"]
        for entry in family_summary
        if isinstance(entry.get("best_tau"), str) and str(entry["best_tau"]).strip()
    }

    metric_best_tau_by_family: dict[str, dict[str, str | None]] = {
        metric: {
            entry["split_family"]: entry["best_tau"]
            for entry in family_summary
            if entry["metric"] == metric
        }
        for metric in PRIMARY_METRICS
    }
    metric_disagreement_families = sorted(
        family
        for family in required_families
        if len(
            {
                metric_best_tau_by_family[metric].get(family)
                for metric in PRIMARY_METRICS
                if metric_best_tau_by_family[metric].get(family) is not None
            }
        )
        > 1
    )

    interpretation = _build_interpretation(
        scorecard_decision=str(scorecard.get("decision") or "UNKNOWN"),
        universal_taus=universal_taus,
        best_tau_set=best_tau_set,
        metric_disagreement_families=metric_disagreement_families,
        boundary_passed=boundary_passed,
        readiness_passed=readiness_passed,
    )
    return {
        "dataset": "j1",
        "scorecard_decision": str(scorecard.get("decision") or "UNKNOWN"),
        "blocking_reasons": [str(reason) for reason in scorecard.get("blocking_reasons", [])],
        "required_split_families": required_families,
        "universal_winner_supported": bool(universal_taus),
        "frontier_story_supported": (not universal_taus) and boundary_passed and readiness_passed,
        "universal_taus": universal_taus,
        "n_required_families": len(required_families),
        "n_unique_best_taus": len(best_tau_set),
        "unique_best_taus": sorted(best_tau_set, key=_sort_tau_key),
        "metric_disagreement_families": metric_disagreement_families,
        "boundary_conditions": {
            "status": boundary_status,
            "passed": boundary_passed,
        },
        "readiness": {
            "status": readiness_status,
            "passed": readiness_passed,
        },
        "family_metric_summary": family_summary,
        "interpretation": interpretation,
    }


def render_j1_frontier_readout(payload: dict[str, Any]) -> str:
    family_rows = payload.get("family_metric_summary", [])
    table_lines = [
        "| split_family | metric | best_tau | estimate | significant_beneficial_taus | beneficial_taus |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for row in family_rows:
        significant = ", ".join(row["significant_beneficial_taus"]) or "<none>"
        beneficial = ", ".join(row["beneficial_taus"]) or "<none>"
        estimate = _format_float(row["best_estimate"])
        table_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["split_family"]),
                    str(row["metric"]),
                    str(row["best_tau"] or "<none>"),
                    estimate,
                    significant,
                    beneficial,
                ]
            )
            + " |"
        )

    universal_taus = ", ".join(payload.get("universal_taus", [])) or "<none>"
    best_taus = ", ".join(payload.get("unique_best_taus", [])) or "<none>"
    disagreements = ", ".join(payload.get("metric_disagreement_families", [])) or "<none>"
    blocking_reasons = payload.get("blocking_reasons", [])
    interpretation = payload.get("interpretation", [])

    lines = [
        "# J1-Frontier Readout",
        "",
        "## Bottom Line",
        f"- Universal winner supported: {'YES' if payload.get('universal_winner_supported') else 'NO'}",
        f"- Frontier story supported: {'YES' if payload.get('frontier_story_supported') else 'NO'}",
        f"- Scorecard decision: {payload.get('scorecard_decision', 'UNKNOWN')}",
        f"- Universal tau candidates: {universal_taus}",
        f"- Unique best taus across primary families/metrics: {best_taus}",
        f"- Families with metric-level best-tau disagreement: {disagreements}",
        "",
        "## Why The Old Story Failed",
        f"- Boundary conditions: {payload['boundary_conditions']['status']}",
        f"- Readiness and reproducibility: {payload['readiness']['status']}",
    ]
    for reason in blocking_reasons:
        lines.append(f"- {reason}")

    lines.extend(
        [
            "",
            "## Family-Metric Frontier Summary",
            *table_lines,
            "",
            "## Reframed Contribution",
        ]
    )
    for point in interpretation:
        lines.append(f"- {point}")

    lines.extend(
        [
            "",
            "## Release Direction",
            "- Publish J1 as a causal-prefix frontier benchmark under shift, not as a universal policy-win paper.",
            "- Make operating-point instability, matched-budget comparator strength, and family-dependent tradeoffs first-class results.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_j1_frontier_package(
    *,
    scorecard_path: str | Path,
    primary_pairwise_path: str | Path,
    publication_aggregate_path: str | Path,
    publication_unit_path: str | Path,
    earliest_summary_path: str | Path,
    earliest_unit_path: str | Path,
    confidence_iso_budget_pairwise_path: str | Path | None = None,
    earliest_iso_budget_pairwise_path: str | Path | None = None,
    confidence_exact_budget_pairwise_path: str | Path | None = None,
    earliest_exact_budget_pairwise_path: str | Path | None = None,
    claims_config_path: str | Path | None = None,
) -> dict[str, Any]:
    scorecard = _load_json_mapping(scorecard_path)
    if str(scorecard.get("dataset", "")).lower() != "j1":
        raise ValueError("frontier package requires a J1 scorecard payload")

    claims = load_j1_claims(claims_config_path)
    split_families = claims.get("split_families", {})
    primary_families = _normalize_string_list(split_families.get("primary", []))
    boundary_families = _normalize_string_list(split_families.get("boundary", []))

    aggregate = _load_csv_artifact(
        publication_aggregate_path,
        required_columns=(
            "split_family",
            "policy",
            "active_macro_f1",
            "active_risk_coverage_auc",
            "mean_ali",
            "intervention_rate",
        ),
        artifact_name="publication aggregate",
    )
    unit_metrics = _load_csv_artifact(
        publication_unit_path,
        required_columns=(
            "dataset_id",
            "split_family",
            "split_id",
            "policy",
            "subject_id",
            "session",
            "day",
            "active_macro_f1",
            "active_risk_coverage_auc",
            "mean_ali",
            "intervention_rate",
        ),
        artifact_name="publication unit metrics",
    )
    earliest_summary = _load_csv_artifact(
        earliest_summary_path,
        required_columns=(
            "dataset_id",
            "split_family",
            "policy",
            "stable_safe_episode_rate",
            "final_correct_rate",
            "median_earliest_stable_safe_s",
        ),
        artifact_name="earliest summary",
    )
    earliest_unit = _load_csv_artifact(
        earliest_unit_path,
        required_columns=(
            "dataset_id",
            "split_family",
            "split_id",
            "policy",
            "subject_id",
            "session",
            "day",
            "stable_safe_episode_rate",
            "final_correct_rate",
            "median_earliest_stable_safe_s",
        ),
        artifact_name="earliest unit summary",
    )
    confidence_iso_budget_pairwise = _load_optional_csv_artifact(
        confidence_iso_budget_pairwise_path,
        required_columns=(
            "split_family",
            "comparison_family",
            "metric",
            "tau",
            "policy_a",
            "policy_b",
            "n_units",
            "n_units_total",
            "n_extrapolated_units",
            "extrapolated_fraction",
            "mean_nearest_curve_budget_gap",
            "beneficial",
        ),
        artifact_name="confidence iso-budget pairwise",
    )
    earliest_iso_budget_pairwise = _load_optional_csv_artifact(
        earliest_iso_budget_pairwise_path,
        required_columns=(
            "split_family",
            "comparison_family",
            "metric",
            "tau",
            "policy_a",
            "policy_b",
            "n_units",
            "n_units_total",
            "n_extrapolated_units",
            "extrapolated_fraction",
            "mean_nearest_curve_budget_gap",
            "beneficial",
        ),
        artifact_name="earliest iso-budget pairwise",
    )
    confidence_exact_budget_pairwise = _load_optional_csv_artifact(
        confidence_exact_budget_pairwise_path,
        required_columns=(
            "split_family",
            "comparison_family",
            "metric",
            "tau",
            "policy_a",
            "policy_b",
            "mean_abs_intervention_rate_gap",
            "exact_match_fraction",
            "beneficial",
        ),
        artifact_name="confidence exact-budget pairwise",
    )
    earliest_exact_budget_pairwise = _load_optional_csv_artifact(
        earliest_exact_budget_pairwise_path,
        required_columns=(
            "split_family",
            "comparison_family",
            "metric",
            "tau",
            "policy_a",
            "policy_b",
            "mean_abs_intervention_rate_gap",
            "exact_match_fraction",
            "beneficial",
        ),
        artifact_name="earliest exact-budget pairwise",
    )

    scorecard_required_families = _normalize_string_list(scorecard.get("required_split_families", []))
    readout = build_j1_frontier_readout_payload(
        scorecard_path=scorecard_path,
        primary_pairwise_path=primary_pairwise_path,
    )
    all_pairwise = _load_primary_pairwise(
        primary_pairwise_path,
        required_families=scorecard_required_families or sorted(aggregate["split_family"].astype(str).unique()),
    )

    points = _build_frontier_points(
        aggregate=aggregate,
        earliest_summary=earliest_summary,
        primary_families=primary_families,
        boundary_families=boundary_families,
    )
    points = _annotate_frontier_membership(points)
    pareto_frontier = points.loc[points["is_non_dominated"].astype(bool)].copy()

    primary_pairwise_rows = _summarize_pairwise_rows(
        _filter_pairwise_families(all_pairwise, families=primary_families)
    )
    primary_universal_taus = _resolve_universal_taus(
        primary_pairwise_rows,
        required_families=primary_families,
    ) if primary_families else []

    frontier_overlap = _build_frontier_overlap(
        pareto_frontier,
        primary_families=primary_families,
        boundary_families=boundary_families,
    )
    non_dominated_summary = _build_non_dominated_summary(
        points,
        primary_families=primary_families,
        boundary_families=boundary_families,
    )
    frontier_unit_deltas = _build_frontier_unit_deltas(
        unit_metrics=unit_metrics,
        earliest_unit=earliest_unit,
        primary_families=primary_families,
        boundary_families=boundary_families,
    )
    budget_match_summary = _build_budget_match_summary(frontier_unit_deltas)
    iso_budget_claim_summary = _build_iso_budget_claim_summary(
        confidence_pairwise=confidence_iso_budget_pairwise,
        earliest_pairwise=earliest_iso_budget_pairwise,
        primary_families=primary_families,
        boundary_families=boundary_families,
    )
    exact_budget_claim_summary = _build_exact_budget_claim_summary(
        confidence_pairwise=confidence_exact_budget_pairwise,
        earliest_pairwise=earliest_exact_budget_pairwise,
        primary_families=primary_families,
        boundary_families=boundary_families,
    )
    universal_tau_regret = _build_universal_tau_regret(
        points=points,
        pairwise_rows=primary_pairwise_rows,
        primary_families=primary_families,
        iso_budget_claim_summary=iso_budget_claim_summary,
    )
    operating_point_instability = _build_operating_point_instability(
        points=points,
        primary_families=primary_families,
        universal_taus=primary_universal_taus,
    )

    observed_split_families = sorted(points["split_family"].astype(str).unique().tolist())
    descriptive_families = sorted(
        family
        for family in observed_split_families
        if family not in set(primary_families).union(boundary_families)
    )
    boundary_passed, readiness_passed, boundary_status, readiness_status = _extract_scorecard_statuses(
        scorecard
    )

    comparator_integrity = _summarize_comparator_integrity(
        budget_match_summary=budget_match_summary,
        iso_budget_claim_summary=iso_budget_claim_summary,
        exact_budget_claim_summary=exact_budget_claim_summary,
        primary_families=primary_families,
    )
    summary = {
        "dataset": "j1",
        "title": "J1-Frontier Package",
        "scorecard_decision": str(scorecard.get("decision") or "UNKNOWN"),
        "readout_frontier_story_supported": bool(readout.get("frontier_story_supported")),
        "primary_frontier_supported": (not primary_universal_taus) and boundary_passed and readiness_passed,
        "boundary_conditions": {
            "status": boundary_status,
            "passed": boundary_passed,
        },
        "readiness": {
            "status": readiness_status,
            "passed": readiness_passed,
        },
        "scorecard_required_families": scorecard_required_families,
        "primary_split_families": primary_families,
        "boundary_split_families": boundary_families,
        "descriptive_split_families": descriptive_families,
        "observed_split_families": observed_split_families,
        "frontier_point_count": int(points.shape[0]),
        "pareto_point_count": int(pareto_frontier.shape[0]),
        "non_dominated_policy_count": int(pareto_frontier["policy"].astype(str).nunique()),
        "readout_universal_taus": list(readout.get("universal_taus", [])),
        "primary_universal_taus": list(primary_universal_taus),
        "primary_iso_budget_supported_taus": list(
            comparator_integrity.get("primary_iso_budget_supported_taus", [])
        ),
        "primary_exact_budget_audit_taus": list(
            comparator_integrity.get("primary_exact_budget_audit_taus", [])
        ),
        "unique_best_taus_scorecard_scope": list(readout.get("unique_best_taus", [])),
        "operating_point_instability": operating_point_instability,
        "comparator_integrity": comparator_integrity,
        "artifact_inventory": list(J1_FRONTIER_PACKAGE_ARTIFACTS),
    }
    markdown = render_j1_frontier_package(
        summary=summary,
        non_dominated_summary=non_dominated_summary,
        universal_tau_regret=universal_tau_regret,
        budget_match_summary=budget_match_summary,
        iso_budget_claim_summary=iso_budget_claim_summary,
        exact_budget_claim_summary=exact_budget_claim_summary,
    )
    return {
        "summary": summary,
        "markdown": markdown,
        "frontier_points": points,
        "pareto_frontier": pareto_frontier,
        "frontier_overlap": frontier_overlap,
        "non_dominated_summary": non_dominated_summary,
        "universal_tau_regret": universal_tau_regret,
        "frontier_unit_deltas": frontier_unit_deltas,
        "budget_match_summary": budget_match_summary,
        "iso_budget_claim_summary": iso_budget_claim_summary,
        "exact_budget_claim_summary": exact_budget_claim_summary,
        "operating_point_instability": operating_point_instability,
    }


def render_j1_frontier_package(
    *,
    summary: dict[str, Any],
    non_dominated_summary: pd.DataFrame,
    universal_tau_regret: pd.DataFrame,
    budget_match_summary: pd.DataFrame,
    iso_budget_claim_summary: pd.DataFrame,
    exact_budget_claim_summary: pd.DataFrame,
) -> str:
    primary = ", ".join(summary.get("primary_split_families", [])) or "<none>"
    boundary = ", ".join(summary.get("boundary_split_families", [])) or "<none>"
    descriptive = ", ".join(summary.get("descriptive_split_families", [])) or "<none>"
    universal_taus = ", ".join(summary.get("primary_universal_taus", [])) or "<none>"
    iso_budget_taus = ", ".join(summary.get("primary_iso_budget_supported_taus", [])) or "<none>"
    exact_budget_taus = ", ".join(summary.get("primary_exact_budget_audit_taus", [])) or "<none>"
    scorecard_taus = ", ".join(summary.get("readout_universal_taus", [])) or "<none>"
    comparator_integrity = summary.get("comparator_integrity", {})

    lines = [
        "# J1-Frontier Package",
        "",
        "## Bottom Line",
        f"- Scorecard decision: {summary.get('scorecard_decision', 'UNKNOWN')}",
        f"- Primary frontier supported: {'YES' if summary.get('primary_frontier_supported') else 'NO'}",
        f"- Scorecard-scope frontier supported: {'YES' if summary.get('readout_frontier_story_supported') else 'NO'}",
        f"- Primary universal tau candidates: {universal_taus}",
        f"- Primary iso-budget comparator taus: {iso_budget_taus}",
        f"- Primary exact-budget audit taus: {exact_budget_taus}",
        f"- Scorecard-scope universal tau candidates: {scorecard_taus}",
        f"- Pareto points: {summary.get('pareto_point_count', 0)} / {summary.get('frontier_point_count', 0)}",
        f"- Boundary conditions: {summary['boundary_conditions']['status']}",
        f"- Readiness and reproducibility: {summary['readiness']['status']}",
        "",
        "## Claim-Bearing Families",
        f"- Primary: {primary}",
        f"- Boundary: {boundary}",
        f"- Descriptive only: {descriptive}",
        "",
        "## Frontier Summary",
        *_frame_to_markdown(
            non_dominated_summary,
            columns=[
                "split_family",
                "split_tier",
                "non_dominated_point_count",
                "non_dominated_policy_count",
                "best_agency_tau_active_macro_f1",
                "best_agency_tau_active_risk_coverage_auc",
                "best_agency_tau_stable_safe_episode_rate",
            ],
            max_rows=12,
        ),
        "",
        "## Universal Tau Regret",
        *_frame_to_markdown(
            universal_tau_regret,
            columns=[
                "tau",
                "n_primary_families_non_dominated",
                "n_primary_families_sig_both_primary",
                "n_primary_families_iso_budget_sig_both_primary",
                "n_primary_families_iso_budget_harmful_safety",
                "mean_regret_active_macro_f1",
                "mean_regret_active_risk_coverage_auc",
                "mean_regret_stable_safe_episode_rate",
                "mean_regret_mean_ali",
            ],
            max_rows=10,
        ),
        "",
        "## Comparator Integrity",
        f"- Worst primary-family mean absolute raw intervention-rate gap: {_format_float(comparator_integrity.get('worst_primary_mean_abs_raw_gap'))}",
        f"- Worst primary-family max absolute raw intervention-rate gap: {_format_float(comparator_integrity.get('worst_primary_max_abs_raw_gap'))}",
        f"- Mean primary-family nearest-curve budget gap: {_format_float(comparator_integrity.get('mean_primary_nearest_curve_gap'))}",
        f"- Primary-family iso-budget support taus: {iso_budget_taus}",
        f"- Worst primary-family confidence extrapolated fraction: {_format_float(comparator_integrity.get('worst_primary_confidence_extrapolated_fraction'))}",
        f"- Worst primary-family earliest-safe extrapolated fraction: {_format_float(comparator_integrity.get('worst_primary_earliest_extrapolated_fraction'))}",
        f"- Primary-family exact-budget audit taus: {exact_budget_taus}",
        f"- Worst primary-family exact-budget confidence gap: {_format_float(comparator_integrity.get('worst_primary_exact_budget_confidence_gap'))}",
        f"- Worst primary-family exact-budget earliest-safe gap: {_format_float(comparator_integrity.get('worst_primary_exact_budget_earliest_gap'))}",
        f"- Lowest primary-family exact-budget confidence match fraction: {_format_float(comparator_integrity.get('lowest_primary_exact_budget_confidence_match_fraction'))}",
        f"- Lowest primary-family exact-budget earliest-safe match fraction: {_format_float(comparator_integrity.get('lowest_primary_exact_budget_earliest_match_fraction'))}",
        "- The package keeps geometry and iso-budget inference separate: the frontier remains geometric, and comparator support is reported from within-unit iso-budget tests.",
        *_frame_to_markdown(
            budget_match_summary.loc[
                budget_match_summary["split_tier"].astype(str) == "primary"
            ].copy(),
            columns=[
                "split_family",
                "tau",
                "n_units",
                "mean_abs_raw_intervention_rate_gap",
                "max_abs_raw_intervention_rate_gap",
                "mean_nearest_curve_budget_gap",
                "mean_iso_budget_delta_active_macro_f1",
                "mean_iso_budget_delta_active_risk_coverage_auc",
            ],
            max_rows=12,
        ),
        "",
        *_frame_to_markdown(
            iso_budget_claim_summary.loc[
                iso_budget_claim_summary["split_tier"].astype(str) == "primary"
            ].copy(),
            columns=[
                "split_family",
                "tau",
                "confidence_all_primary_metrics_sig_beneficial",
                "safety_harmful_sig_metric_count",
                "confidence_mean_nearest_curve_budget_gap",
                "confidence_extrapolated_fraction",
            ],
            max_rows=12,
        ),
        "",
        *_frame_to_markdown(
            exact_budget_claim_summary.loc[
                exact_budget_claim_summary["split_tier"].astype(str) == "primary"
            ].copy(),
            columns=[
                "split_family",
                "tau",
                "confidence_all_primary_metrics_sig_beneficial",
                "safety_harmful_sig_metric_count",
                "confidence_mean_abs_intervention_rate_gap",
                "confidence_exact_match_fraction",
            ],
            max_rows=12,
        ),
        "",
        "## Artifact Inventory",
    ]
    for artifact_name in summary.get("artifact_inventory", []):
        lines.append(f"- {artifact_name}")
    return "\n".join(lines) + "\n"


def _load_json_mapping(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {source}")
    return payload


def _load_csv_artifact(
    path: str | Path,
    *,
    required_columns: tuple[str, ...],
    artifact_name: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(set(required_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{artifact_name} is missing required columns: {', '.join(missing)}")
    return frame


def _load_optional_csv_artifact(
    path: str | Path | None,
    *,
    required_columns: tuple[str, ...],
    artifact_name: str,
) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=list(required_columns))
    source = Path(path)
    if not source.exists():
        return pd.DataFrame(columns=list(required_columns))
    return _load_csv_artifact(
        source,
        required_columns=required_columns,
        artifact_name=artifact_name,
    )


def _build_frontier_points(
    *,
    aggregate: pd.DataFrame,
    earliest_summary: pd.DataFrame,
    primary_families: list[str],
    boundary_families: list[str],
) -> pd.DataFrame:
    earliest_subset = earliest_summary[
        [
            "split_family",
            "policy",
            "stable_safe_episode_rate",
            "final_correct_rate",
            "median_earliest_safe_s",
            "median_earliest_stable_safe_s",
        ]
    ].copy()
    merged = aggregate.merge(
        earliest_subset,
        on=["split_family", "policy"],
        how="left",
        validate="one_to_one",
    )
    merged["dataset_id"] = "j1"
    merged["policy_family"] = merged["policy"].map(_infer_policy_family)
    merged["tau"] = merged["policy"].map(_extract_tau)
    merged["split_tier"] = merged["split_family"].map(
        lambda family: _split_tier(str(family), primary_families, boundary_families)
    )
    merged["is_primary_split"] = merged["split_tier"].astype(str) == "primary"
    merged["is_boundary_split"] = merged["split_tier"].astype(str) == "boundary"
    merged["is_descriptive_split"] = merged["split_tier"].astype(str) == "descriptive"
    merged["frontier_label"] = merged.apply(
        lambda row: _frontier_label(str(row["policy_family"]), row["tau"]),
        axis=1,
    )
    merged["is_non_dominated"] = False
    merged["frontier_valid"] = False
    return merged.sort_values(["split_family", "policy_family", "policy"]).reset_index(drop=True)


def _annotate_frontier_membership(points: pd.DataFrame) -> pd.DataFrame:
    annotated = points.copy()
    objective_columns = list(FRONTIER_OBJECTIVES)
    annotated["frontier_valid"] = annotated[objective_columns].notna().all(axis=1)
    annotated["is_non_dominated"] = False
    for split_family, family_df in annotated.groupby("split_family", sort=True):
        valid_index = family_df.index[family_df["frontier_valid"].astype(bool)]
        if len(valid_index) == 0:
            continue
        mask = _pareto_mask(annotated.loc[valid_index, objective_columns], FRONTIER_OBJECTIVES)
        annotated.loc[valid_index, "is_non_dominated"] = mask
    return annotated


def _build_frontier_overlap(
    pareto_frontier: pd.DataFrame,
    *,
    primary_families: list[str],
    boundary_families: list[str],
) -> pd.DataFrame:
    if pareto_frontier.empty:
        return pd.DataFrame(
            columns=[
                "policy",
                "policy_family",
                "tau",
                "split_family_count_non_dominated",
                "primary_split_family_count_non_dominated",
                "boundary_split_family_count_non_dominated",
                "descriptive_split_family_count_non_dominated",
                "non_dominated_split_families",
            ]
        )

    rows: list[dict[str, Any]] = []
    primary_set = set(primary_families)
    boundary_set = set(boundary_families)
    for keys, policy_df in pareto_frontier.groupby(["policy", "policy_family", "tau"], dropna=False, sort=True):
        split_families = sorted(policy_df["split_family"].astype(str).unique().tolist())
        rows.append(
            {
                "policy": str(keys[0]),
                "policy_family": str(keys[1]),
                "tau": None if pd.isna(keys[2]) else str(keys[2]),
                "split_family_count_non_dominated": len(split_families),
                "primary_split_family_count_non_dominated": sum(f in primary_set for f in split_families),
                "boundary_split_family_count_non_dominated": sum(f in boundary_set for f in split_families),
                "descriptive_split_family_count_non_dominated": sum(
                    f not in primary_set and f not in boundary_set for f in split_families
                ),
                "non_dominated_split_families": "|".join(split_families),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(
        [
            "primary_split_family_count_non_dominated",
            "split_family_count_non_dominated",
            "policy_family",
            "policy",
        ],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def _build_non_dominated_summary(
    points: pd.DataFrame,
    *,
    primary_families: list[str],
    boundary_families: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_family, family_df in points.groupby("split_family", sort=True):
        pareto_df = family_df.loc[family_df["is_non_dominated"].astype(bool)].copy()
        rows.append(
            {
                "split_family": str(split_family),
                "split_tier": _split_tier(str(split_family), primary_families, boundary_families),
                "point_count": int(family_df.shape[0]),
                "non_dominated_point_count": int(pareto_df.shape[0]),
                "policy_count": int(family_df["policy"].astype(str).nunique()),
                "non_dominated_policy_count": int(pareto_df["policy"].astype(str).nunique()),
                "non_dominated_policy_families": "|".join(
                    sorted(pareto_df["policy_family"].astype(str).unique().tolist())
                ),
                "best_agency_tau_active_macro_f1": _best_agency_tau_for_metric(
                    family_df, metric="active_macro_f1"
                ),
                "best_agency_tau_active_risk_coverage_auc": _best_agency_tau_for_metric(
                    family_df, metric="active_risk_coverage_auc"
                ),
                "best_agency_tau_stable_safe_episode_rate": _best_agency_tau_for_metric(
                    family_df, metric="stable_safe_episode_rate"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["split_tier", "split_family"]).reset_index(drop=True)


def _build_frontier_unit_deltas(
    *,
    unit_metrics: pd.DataFrame,
    earliest_unit: pd.DataFrame,
    primary_families: list[str],
    boundary_families: list[str],
) -> pd.DataFrame:
    earliest_subset = earliest_unit[
        [
            "split_family",
            "split_id",
            "policy",
            "subject_id",
            "session",
            "day",
            "stable_safe_episode_rate",
            "final_correct_rate",
            "median_earliest_stable_safe_s",
        ]
    ].copy()
    merged = unit_metrics.merge(
        earliest_subset,
        on=["split_family", "split_id", "policy", "subject_id", "session", "day"],
        how="left",
        validate="one_to_one",
    )
    merged["policy_family"] = merged["policy"].map(_infer_policy_family)
    merged["tau"] = merged["policy"].map(_extract_tau)
    merged["split_tier"] = merged["split_family"].map(
        lambda family: _split_tier(str(family), primary_families, boundary_families)
    )

    rows: list[dict[str, Any]] = []
    group_cols = ["split_family", "split_id", "subject_id", "session", "day"]
    for keys, unit_df in merged.groupby(group_cols, dropna=False, sort=True):
        plain_df = unit_df.loc[unit_df["policy_family"].astype(str) == "plain_conf_threshold_matched"].copy()
        agency_df = unit_df.loc[unit_df["policy_family"].astype(str) == "agency_margin"].copy()
        if plain_df.empty or agency_df.empty:
            continue
        same_tau_plain = {
            str(row["tau"]): row
            for _, row in plain_df.loc[plain_df["tau"].notna()].iterrows()
        }
        for _, agency_row in agency_df.loc[agency_df["tau"].notna()].iterrows():
            tau = str(agency_row["tau"])
            target_budget = float(agency_row["intervention_rate"])
            curve_payload = _interpolate_policy_curve(
                curve_df=plain_df,
                target_budget=target_budget,
                metrics=UNIT_DELTA_METRICS,
            )
            matched_plain = same_tau_plain.get(tau)
            raw_gap = (
                float(agency_row["intervention_rate"]) - float(matched_plain["intervention_rate"])
                if matched_plain is not None
                else float("nan")
            )
            row = {
                "dataset_id": "j1",
                "split_family": str(keys[0]),
                "split_tier": _split_tier(str(keys[0]), primary_families, boundary_families),
                "split_id": str(keys[1]),
                "subject_id": str(keys[2]),
                "session": str(keys[3]),
                "day": str(keys[4]),
                "tau": tau,
                "agency_policy": str(agency_row["policy"]),
                "plain_conf_policy": str(matched_plain["policy"]) if matched_plain is not None else "",
                "agency_intervention_rate": float(agency_row["intervention_rate"]),
                "plain_conf_same_tau_intervention_rate": (
                    float(matched_plain["intervention_rate"]) if matched_plain is not None else float("nan")
                ),
                "raw_intervention_rate_gap": raw_gap,
                "nearest_curve_budget_gap": float(curve_payload["nearest_gap"]),
                "curve_budget_extrapolated": bool(curve_payload["extrapolated"]),
            }
            for metric in UNIT_DELTA_METRICS:
                agency_value = _safe_float(agency_row.get(metric))
                same_tau_value = _safe_float(matched_plain.get(metric)) if matched_plain is not None else float("nan")
                iso_budget_value = _safe_float(curve_payload["values"].get(metric))
                row[f"agency_{metric}"] = agency_value
                row[f"plain_conf_same_tau_{metric}"] = same_tau_value
                row[f"plain_conf_iso_budget_{metric}"] = iso_budget_value
                row[f"delta_same_tau_{metric}"] = _delta_or_nan(agency_value, same_tau_value)
                row[f"delta_iso_budget_{metric}"] = _delta_or_nan(agency_value, iso_budget_value)
                row[f"directional_delta_same_tau_{metric}"] = _directional_delta_or_nan(
                    metric,
                    agency_value,
                    same_tau_value,
                )
                row[f"directional_delta_iso_budget_{metric}"] = _directional_delta_or_nan(
                    metric,
                    agency_value,
                    iso_budget_value,
                )
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["split_family", "subject_id", "tau"]).reset_index(drop=True)


def _build_budget_match_summary(unit_deltas: pd.DataFrame) -> pd.DataFrame:
    if unit_deltas.empty:
        return pd.DataFrame(
            columns=[
                "split_family",
                "split_tier",
                "tau",
                "n_units",
                "mean_abs_raw_intervention_rate_gap",
                "max_abs_raw_intervention_rate_gap",
                "mean_nearest_curve_budget_gap",
                "max_nearest_curve_budget_gap",
                "mean_iso_budget_delta_active_macro_f1",
                "mean_iso_budget_delta_active_risk_coverage_auc",
                "mean_iso_budget_delta_stable_safe_episode_rate",
                "mean_iso_budget_delta_mean_ali",
            ]
        )

    rows: list[dict[str, Any]] = []
    group_cols = ["split_family", "split_tier", "tau"]
    for keys, group_df in unit_deltas.groupby(group_cols, dropna=False, sort=True):
        rows.append(
            {
                "split_family": str(keys[0]),
                "split_tier": str(keys[1]),
                "tau": str(keys[2]),
                "n_units": int(group_df.shape[0]),
                "mean_abs_raw_intervention_rate_gap": float(
                    group_df["raw_intervention_rate_gap"].abs().mean()
                ),
                "max_abs_raw_intervention_rate_gap": float(
                    group_df["raw_intervention_rate_gap"].abs().max()
                ),
                "mean_nearest_curve_budget_gap": float(group_df["nearest_curve_budget_gap"].mean()),
                "max_nearest_curve_budget_gap": float(group_df["nearest_curve_budget_gap"].max()),
                "mean_iso_budget_delta_active_macro_f1": float(
                    group_df["directional_delta_iso_budget_active_macro_f1"].mean()
                ),
                "mean_iso_budget_delta_active_risk_coverage_auc": float(
                    group_df["directional_delta_iso_budget_active_risk_coverage_auc"].mean()
                ),
                "mean_iso_budget_delta_stable_safe_episode_rate": float(
                    group_df["directional_delta_iso_budget_stable_safe_episode_rate"].mean()
                ),
                "mean_iso_budget_delta_mean_ali": float(
                    group_df["directional_delta_iso_budget_mean_ali"].mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["split_family", "tau"]).reset_index(drop=True)


def _build_iso_budget_claim_summary(
    *,
    confidence_pairwise: pd.DataFrame,
    earliest_pairwise: pd.DataFrame,
    primary_families: list[str],
    boundary_families: list[str],
) -> pd.DataFrame:
    columns = [
        "split_family",
        "split_tier",
        "tau",
        "confidence_primary_metric_count",
        "confidence_primary_sig_beneficial_count",
        "confidence_all_primary_metrics_sig_beneficial",
        "confidence_harmful_sig_metric_count",
        "safety_metric_count",
        "safety_sig_beneficial_count",
        "safety_harmful_sig_metric_count",
        "confidence_mean_nearest_curve_budget_gap",
        "confidence_extrapolated_fraction",
        "earliest_mean_nearest_curve_budget_gap",
        "earliest_extrapolated_fraction",
    ]
    normalized_confidence = _normalize_iso_budget_pairwise_frame(confidence_pairwise)
    normalized_earliest = _normalize_iso_budget_pairwise_frame(earliest_pairwise)
    if normalized_confidence.empty and normalized_earliest.empty:
        return pd.DataFrame(columns=columns)

    split_values = sorted(
        set(normalized_confidence["split_family"].astype(str)).union(
            normalized_earliest["split_family"].astype(str)
        )
    )
    tau_values = sorted(
        {
            str(tau)
            for tau in pd.concat(
                [
                    normalized_confidence["tau"].astype(str),
                    normalized_earliest["tau"].astype(str),
                ],
                ignore_index=True,
            )
            if str(tau).strip() and str(tau) != "nan"
        },
        key=_sort_tau_key,
    )

    rows: list[dict[str, Any]] = []
    for split_family in split_values:
        split_tier = _split_tier(split_family, primary_families, boundary_families)
        for tau in tau_values:
            conf_subset = normalized_confidence.loc[
                (normalized_confidence["split_family"].astype(str) == split_family)
                & (normalized_confidence["tau"].astype(str) == tau)
                & normalized_confidence["metric"].astype(str).isin(PRIMARY_METRICS)
            ].copy()
            early_subset = normalized_earliest.loc[
                (normalized_earliest["split_family"].astype(str) == split_family)
                & (normalized_earliest["tau"].astype(str) == tau)
                & normalized_earliest["metric"].astype(str).isin(SAFETY_COMPARATOR_METRICS)
            ].copy()
            if conf_subset.empty and early_subset.empty:
                continue
            confidence_sig_metrics = set(
                conf_subset.loc[
                    conf_subset["beneficial_and_holm_significant"].astype(bool),
                    "metric",
                ].astype(str)
            )
            confidence_harmful_count = int(
                (
                    conf_subset["holm_reject"].astype(bool)
                    & ~conf_subset["beneficial"].astype(bool)
                ).sum()
            ) if not conf_subset.empty else 0
            safety_sig_count = int(
                early_subset["beneficial_and_holm_significant"].astype(bool).sum()
            ) if not early_subset.empty else 0
            safety_harmful_count = int(
                (
                    early_subset["holm_reject"].astype(bool)
                    & ~early_subset["beneficial"].astype(bool)
                ).sum()
            ) if not early_subset.empty else 0
            rows.append(
                {
                    "split_family": split_family,
                    "split_tier": split_tier,
                    "tau": tau,
                    "confidence_primary_metric_count": int(conf_subset.shape[0]),
                    "confidence_primary_sig_beneficial_count": int(len(confidence_sig_metrics)),
                    "confidence_all_primary_metrics_sig_beneficial": (
                        set(conf_subset["metric"].astype(str)) == set(PRIMARY_METRICS)
                        and confidence_sig_metrics == set(PRIMARY_METRICS)
                    ),
                    "confidence_harmful_sig_metric_count": confidence_harmful_count,
                    "safety_metric_count": int(early_subset.shape[0]),
                    "safety_sig_beneficial_count": safety_sig_count,
                    "safety_harmful_sig_metric_count": safety_harmful_count,
                    "confidence_mean_nearest_curve_budget_gap": _mean_or_nan(
                        conf_subset["mean_nearest_curve_budget_gap"].astype(float).tolist()
                    ) if not conf_subset.empty else float("nan"),
                    "confidence_extrapolated_fraction": _max_or_nan(
                        conf_subset["extrapolated_fraction"].astype(float).tolist()
                    ) if not conf_subset.empty else float("nan"),
                    "earliest_mean_nearest_curve_budget_gap": _mean_or_nan(
                        early_subset["mean_nearest_curve_budget_gap"].astype(float).tolist()
                    ) if not early_subset.empty else float("nan"),
                    "earliest_extrapolated_fraction": _max_or_nan(
                        early_subset["extrapolated_fraction"].astype(float).tolist()
                    ) if not early_subset.empty else float("nan"),
                }
            )
    return pd.DataFrame(rows).sort_values(["split_family", "tau"]).reset_index(drop=True)


def _build_exact_budget_claim_summary(
    *,
    confidence_pairwise: pd.DataFrame,
    earliest_pairwise: pd.DataFrame,
    primary_families: list[str],
    boundary_families: list[str],
) -> pd.DataFrame:
    columns = [
        "split_family",
        "split_tier",
        "tau",
        "confidence_primary_metric_count",
        "confidence_primary_sig_beneficial_count",
        "confidence_all_primary_metrics_sig_beneficial",
        "confidence_harmful_sig_metric_count",
        "safety_metric_count",
        "safety_sig_beneficial_count",
        "safety_harmful_sig_metric_count",
        "confidence_mean_abs_intervention_rate_gap",
        "confidence_exact_match_fraction",
        "earliest_mean_abs_intervention_rate_gap",
        "earliest_exact_match_fraction",
    ]
    normalized_confidence = _normalize_exact_budget_pairwise_frame(confidence_pairwise)
    normalized_earliest = _normalize_exact_budget_pairwise_frame(earliest_pairwise)
    if normalized_confidence.empty and normalized_earliest.empty:
        return pd.DataFrame(columns=columns)

    split_values = sorted(
        set(normalized_confidence["split_family"].astype(str)).union(
            normalized_earliest["split_family"].astype(str)
        )
    )
    tau_values = sorted(
        {
            str(tau)
            for tau in pd.concat(
                [
                    normalized_confidence["tau"].astype(str),
                    normalized_earliest["tau"].astype(str),
                ],
                ignore_index=True,
            )
            if str(tau).strip() and str(tau) != "nan"
        },
        key=_sort_tau_key,
    )

    rows: list[dict[str, Any]] = []
    for split_family in split_values:
        split_tier = _split_tier(split_family, primary_families, boundary_families)
        for tau in tau_values:
            conf_subset = normalized_confidence.loc[
                (normalized_confidence["split_family"].astype(str) == split_family)
                & (normalized_confidence["tau"].astype(str) == tau)
                & normalized_confidence["metric"].astype(str).isin(PRIMARY_METRICS)
            ].copy()
            early_subset = normalized_earliest.loc[
                (normalized_earliest["split_family"].astype(str) == split_family)
                & (normalized_earliest["tau"].astype(str) == tau)
                & normalized_earliest["metric"].astype(str).isin(SAFETY_COMPARATOR_METRICS)
            ].copy()
            if conf_subset.empty and early_subset.empty:
                continue
            confidence_sig_metrics = set(
                conf_subset.loc[
                    conf_subset["beneficial_and_holm_significant"].astype(bool),
                    "metric",
                ].astype(str)
            )
            confidence_harmful_count = int(
                (
                    conf_subset["holm_reject"].astype(bool)
                    & ~conf_subset["beneficial"].astype(bool)
                ).sum()
            ) if not conf_subset.empty else 0
            safety_sig_count = int(
                early_subset["beneficial_and_holm_significant"].astype(bool).sum()
            ) if not early_subset.empty else 0
            safety_harmful_count = int(
                (
                    early_subset["holm_reject"].astype(bool)
                    & ~early_subset["beneficial"].astype(bool)
                ).sum()
            ) if not early_subset.empty else 0
            rows.append(
                {
                    "split_family": split_family,
                    "split_tier": split_tier,
                    "tau": tau,
                    "confidence_primary_metric_count": int(conf_subset.shape[0]),
                    "confidence_primary_sig_beneficial_count": int(len(confidence_sig_metrics)),
                    "confidence_all_primary_metrics_sig_beneficial": (
                        set(conf_subset["metric"].astype(str)) == set(PRIMARY_METRICS)
                        and confidence_sig_metrics == set(PRIMARY_METRICS)
                    ),
                    "confidence_harmful_sig_metric_count": confidence_harmful_count,
                    "safety_metric_count": int(early_subset.shape[0]),
                    "safety_sig_beneficial_count": safety_sig_count,
                    "safety_harmful_sig_metric_count": safety_harmful_count,
                    "confidence_mean_abs_intervention_rate_gap": _mean_or_nan(
                        conf_subset["mean_abs_intervention_rate_gap"].astype(float).tolist()
                    ) if not conf_subset.empty else float("nan"),
                    "confidence_exact_match_fraction": _mean_or_nan(
                        conf_subset["exact_match_fraction"].astype(float).tolist()
                    ) if not conf_subset.empty else float("nan"),
                    "earliest_mean_abs_intervention_rate_gap": _mean_or_nan(
                        early_subset["mean_abs_intervention_rate_gap"].astype(float).tolist()
                    ) if not early_subset.empty else float("nan"),
                    "earliest_exact_match_fraction": _mean_or_nan(
                        early_subset["exact_match_fraction"].astype(float).tolist()
                    ) if not early_subset.empty else float("nan"),
                }
            )
    return pd.DataFrame(rows).sort_values(["split_family", "tau"]).reset_index(drop=True)


def _build_universal_tau_regret(
    *,
    points: pd.DataFrame,
    pairwise_rows: list[dict[str, Any]],
    primary_families: list[str],
    iso_budget_claim_summary: pd.DataFrame,
) -> pd.DataFrame:
    agency = points.loc[
        points["split_family"].astype(str).isin(primary_families)
        & (points["policy_family"].astype(str) == "agency_margin")
        & points["tau"].notna()
    ].copy()
    if agency.empty:
        return pd.DataFrame(
            columns=[
                "tau",
                "n_primary_families_present",
                "n_primary_families_non_dominated",
                "n_primary_families_sig_both_primary",
                "n_primary_families_iso_budget_sig_both_primary",
                "n_primary_families_iso_budget_harmful_safety",
                "n_primary_families_dominates_plain_conf",
                "n_primary_families_dominated_by_plain_conf",
                "mean_regret_active_macro_f1",
                "mean_regret_active_risk_coverage_auc",
                "mean_regret_stable_safe_episode_rate",
                "mean_regret_mean_ali",
            ]
        )

    pairwise_df = pd.DataFrame(pairwise_rows)
    iso_budget_df = iso_budget_claim_summary.copy()
    rows: list[dict[str, Any]] = []
    tau_values = sorted(agency["tau"].astype(str).unique().tolist(), key=_sort_tau_key)
    for tau in tau_values:
        tau_df = agency.loc[agency["tau"].astype(str) == tau].copy()
        metric_regrets = {metric: [] for metric in REGRET_METRICS}
        n_non_dominated = int(tau_df["is_non_dominated"].astype(bool).sum())
        n_dominates_plain_conf = 0
        n_dominated_by_plain_conf = 0
        n_sig_both_primary = 0
        n_iso_budget_sig_both_primary = 0
        n_iso_budget_harmful_safety = 0
        for split_family in primary_families:
            family_df = agency.loc[agency["split_family"].astype(str) == split_family].copy()
            current = tau_df.loc[tau_df["split_family"].astype(str) == split_family]
            if current.empty or family_df.empty:
                continue
            current_row = current.iloc[0]
            for metric in REGRET_METRICS:
                metric_regrets[metric].append(
                    _metric_regret(
                        family_df=family_df,
                        current_value=_safe_float(current_row.get(metric)),
                        metric=metric,
                    )
                )
            plain_conf_row = points.loc[
                (points["split_family"].astype(str) == split_family)
                & (points["policy_family"].astype(str) == "plain_conf_threshold_matched")
                & (points["tau"].astype(str) == tau)
            ]
            if not plain_conf_row.empty:
                other = plain_conf_row.iloc[0]
                if _row_dominates(current_row, other, COMPARATOR_OBJECTIVES):
                    n_dominates_plain_conf += 1
                if _row_dominates(other, current_row, COMPARATOR_OBJECTIVES):
                    n_dominated_by_plain_conf += 1
            if not pairwise_df.empty:
                pair_subset = pairwise_df.loc[
                    (pairwise_df["split_family"].astype(str) == split_family)
                    & (pairwise_df["tau"].astype(str) == tau)
                    & pairwise_df["metric"].astype(str).isin(PRIMARY_METRICS)
                    & pairwise_df["significant_beneficial"].astype(bool)
                ]
                if set(pair_subset["metric"].astype(str)) == set(PRIMARY_METRICS):
                    n_sig_both_primary += 1
            if not iso_budget_df.empty:
                iso_subset = iso_budget_df.loc[
                    (iso_budget_df["split_family"].astype(str) == split_family)
                    & (iso_budget_df["tau"].astype(str) == tau)
                ]
                if not iso_subset.empty:
                    row = iso_subset.iloc[0]
                    if bool(row.get("confidence_all_primary_metrics_sig_beneficial")):
                        n_iso_budget_sig_both_primary += 1
                    if int(row.get("safety_harmful_sig_metric_count", 0)) > 0:
                        n_iso_budget_harmful_safety += 1

        rows.append(
            {
                "tau": tau,
                "n_primary_families_present": int(
                    tau_df["split_family"].astype(str).nunique()
                ),
                "n_primary_families_non_dominated": n_non_dominated,
                "n_primary_families_sig_both_primary": int(n_sig_both_primary),
                "n_primary_families_iso_budget_sig_both_primary": int(
                    n_iso_budget_sig_both_primary
                ),
                "n_primary_families_iso_budget_harmful_safety": int(
                    n_iso_budget_harmful_safety
                ),
                "n_primary_families_dominates_plain_conf": int(n_dominates_plain_conf),
                "n_primary_families_dominated_by_plain_conf": int(n_dominated_by_plain_conf),
                "mean_regret_active_macro_f1": _mean_or_nan(metric_regrets["active_macro_f1"]),
                "max_regret_active_macro_f1": _max_or_nan(metric_regrets["active_macro_f1"]),
                "mean_regret_active_risk_coverage_auc": _mean_or_nan(
                    metric_regrets["active_risk_coverage_auc"]
                ),
                "max_regret_active_risk_coverage_auc": _max_or_nan(
                    metric_regrets["active_risk_coverage_auc"]
                ),
                "mean_regret_stable_safe_episode_rate": _mean_or_nan(
                    metric_regrets["stable_safe_episode_rate"]
                ),
                "max_regret_stable_safe_episode_rate": _max_or_nan(
                    metric_regrets["stable_safe_episode_rate"]
                ),
                "mean_regret_mean_ali": _mean_or_nan(metric_regrets["mean_ali"]),
                "max_regret_mean_ali": _max_or_nan(metric_regrets["mean_ali"]),
            }
        )
    return pd.DataFrame(rows).sort_values("tau", key=lambda col: col.map(_sort_tau_key)).reset_index(drop=True)


def _build_operating_point_instability(
    *,
    points: pd.DataFrame,
    primary_families: list[str],
    universal_taus: list[str],
) -> dict[str, Any]:
    metrics = ("active_macro_f1", "active_risk_coverage_auc", "stable_safe_episode_rate", "mean_ali")
    agency = points.loc[
        points["split_family"].astype(str).isin(primary_families)
        & (points["policy_family"].astype(str) == "agency_margin")
        & points["tau"].notna()
    ].copy()
    best_tau_by_family_metric: dict[str, dict[str, str | None]] = {}
    unique_best_taus: set[str] = set()
    disagreement_families: list[str] = []
    for split_family, family_df in agency.groupby("split_family", sort=True):
        family_best: dict[str, str | None] = {}
        family_tau_set: set[str] = set()
        for metric in metrics:
            best_tau = _best_agency_tau_for_metric(family_df, metric=metric)
            family_best[metric] = best_tau
            if best_tau is not None:
                family_tau_set.add(best_tau)
                unique_best_taus.add(best_tau)
        if len(family_tau_set) > 1:
            disagreement_families.append(str(split_family))
        best_tau_by_family_metric[str(split_family)] = family_best

    tau_values = sorted(agency["tau"].astype(str).unique().tolist(), key=_sort_tau_key)
    non_dominated_counts = (
        agency.loc[agency["is_non_dominated"].astype(bool)]
        .groupby("tau")["split_family"]
        .nunique()
        .to_dict()
    )
    return {
        "primary_split_families": list(primary_families),
        "best_tau_by_family_metric": best_tau_by_family_metric,
        "unique_best_taus": sorted(unique_best_taus, key=_sort_tau_key),
        "primary_family_metric_disagreement_families": sorted(disagreement_families),
        "non_dominated_primary_family_counts_by_tau": {
            str(tau): int(non_dominated_counts.get(tau, 0)) for tau in tau_values
        },
        "primary_universal_tau_candidates": list(universal_taus),
    }


def _normalize_iso_budget_pairwise_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "split_family",
        "comparison_family",
        "metric",
        "tau",
        "beneficial",
        "mean_nearest_curve_budget_gap",
        "extrapolated_fraction",
    }
    if frame.empty:
        return pd.DataFrame(columns=list(required) + ["holm_reject", "beneficial_and_holm_significant"])
    missing = sorted(required.difference(frame.columns))
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"iso-budget pairwise artifact is missing required columns: {names}")
    normalized = frame.copy()
    if "holm_reject" not in normalized.columns:
        normalized["holm_reject"] = False
    if "beneficial_and_holm_significant" not in normalized.columns:
        normalized["beneficial_and_holm_significant"] = False
    normalized = normalized.loc[
        normalized["comparison_family"].astype(str) == ISO_BUDGET_COMPARISON_FAMILY
    ].copy()
    if normalized.empty:
        return normalized
    duplicates = normalized.duplicated(
        subset=["split_family", "comparison_family", "metric", "tau"],
        keep=False,
    )
    if bool(duplicates.any()):
        raise ValueError(
            "iso-budget pairwise artifact must contain at most one row per split_family / metric / tau"
        )
    normalized["tau"] = normalized["tau"].map(
        lambda value: _normalize_tau_value(value) if not pd.isna(value) else None
    )
    return normalized


def _normalize_exact_budget_pairwise_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "split_family",
        "comparison_family",
        "metric",
        "tau",
        "beneficial",
        "mean_abs_intervention_rate_gap",
        "exact_match_fraction",
    }
    if frame.empty:
        return pd.DataFrame(columns=list(required) + ["holm_reject", "beneficial_and_holm_significant"])
    missing = sorted(required.difference(frame.columns))
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"exact-budget pairwise artifact is missing required columns: {names}")
    normalized = frame.copy()
    if "holm_reject" not in normalized.columns:
        normalized["holm_reject"] = False
    if "beneficial_and_holm_significant" not in normalized.columns:
        normalized["beneficial_and_holm_significant"] = False
    normalized = normalized.loc[
        normalized["comparison_family"].astype(str) == EXACT_BUDGET_COMPARISON_FAMILY
    ].copy()
    if normalized.empty:
        return normalized
    duplicates = normalized.duplicated(
        subset=["split_family", "comparison_family", "metric", "tau"],
        keep=False,
    )
    if bool(duplicates.any()):
        raise ValueError(
            "exact-budget pairwise artifact must contain at most one row per split_family / metric / tau"
        )
    normalized["tau"] = normalized["tau"].map(
        lambda value: _normalize_tau_value(value) if not pd.isna(value) else None
    )
    return normalized


def _resolve_primary_iso_budget_supported_taus(
    summary: pd.DataFrame,
    *,
    primary_families: list[str],
) -> list[str]:
    if summary.empty or not primary_families:
        return []
    supported: list[str] = []
    required = set(primary_families)
    for tau, tau_df in summary.groupby("tau", sort=True):
        present = set(tau_df["split_family"].astype(str))
        if present != required:
            continue
        if not tau_df["confidence_all_primary_metrics_sig_beneficial"].astype(bool).all():
            continue
        if bool((tau_df["confidence_extrapolated_fraction"].astype(float) > 0.0).any()):
            continue
        if bool((tau_df["earliest_extrapolated_fraction"].astype(float) > 0.0).any()):
            continue
        if bool((tau_df["safety_harmful_sig_metric_count"].astype(int) > 0).any()):
            continue
        supported.append(str(tau))
    return sorted(supported, key=_sort_tau_key)


def _resolve_primary_exact_budget_audit_taus(
    summary: pd.DataFrame,
    *,
    primary_families: list[str],
) -> list[str]:
    if summary.empty or not primary_families:
        return []
    audited: list[str] = []
    required = set(primary_families)
    for tau, tau_df in summary.groupby("tau", sort=True):
        present = set(tau_df["split_family"].astype(str))
        if present != required:
            continue
        if not tau_df["confidence_all_primary_metrics_sig_beneficial"].astype(bool).all():
            continue
        if bool((tau_df["safety_harmful_sig_metric_count"].astype(int) > 0).any()):
            continue
        audited.append(str(tau))
    return sorted(audited, key=_sort_tau_key)


def _summarize_comparator_integrity(
    *,
    budget_match_summary: pd.DataFrame,
    iso_budget_claim_summary: pd.DataFrame,
    exact_budget_claim_summary: pd.DataFrame,
    primary_families: list[str],
) -> dict[str, Any]:
    primary_df = budget_match_summary.loc[
        budget_match_summary["split_family"].astype(str).isin(primary_families)
    ].copy()
    primary_iso = iso_budget_claim_summary.loc[
        iso_budget_claim_summary["split_family"].astype(str).isin(primary_families)
    ].copy()
    primary_exact = exact_budget_claim_summary.loc[
        exact_budget_claim_summary["split_family"].astype(str).isin(primary_families)
    ].copy()
    if primary_df.empty and primary_iso.empty and primary_exact.empty:
        return {
            "worst_primary_mean_abs_raw_gap": None,
            "worst_primary_max_abs_raw_gap": None,
            "mean_primary_nearest_curve_gap": None,
            "primary_iso_budget_supported_taus": [],
            "primary_exact_budget_audit_taus": [],
            "worst_primary_confidence_extrapolated_fraction": None,
            "worst_primary_earliest_extrapolated_fraction": None,
            "worst_primary_exact_budget_confidence_gap": None,
            "worst_primary_exact_budget_earliest_gap": None,
            "lowest_primary_exact_budget_confidence_match_fraction": None,
            "lowest_primary_exact_budget_earliest_match_fraction": None,
        }
    return {
        "worst_primary_mean_abs_raw_gap": float(
            primary_df["mean_abs_raw_intervention_rate_gap"].max()
        ) if not primary_df.empty else None,
        "worst_primary_max_abs_raw_gap": float(
            primary_df["max_abs_raw_intervention_rate_gap"].max()
        ) if not primary_df.empty else None,
        "mean_primary_nearest_curve_gap": float(
            primary_df["mean_nearest_curve_budget_gap"].mean()
        ) if not primary_df.empty else None,
        "primary_iso_budget_supported_taus": _resolve_primary_iso_budget_supported_taus(
            primary_iso,
            primary_families=primary_families,
        ),
        "primary_exact_budget_audit_taus": _resolve_primary_exact_budget_audit_taus(
            primary_exact,
            primary_families=primary_families,
        ),
        "worst_primary_confidence_extrapolated_fraction": float(
            primary_iso["confidence_extrapolated_fraction"].max()
        ) if not primary_iso.empty else None,
        "worst_primary_earliest_extrapolated_fraction": float(
            primary_iso["earliest_extrapolated_fraction"].max()
        ) if not primary_iso.empty else None,
        "worst_primary_exact_budget_confidence_gap": float(
            primary_exact["confidence_mean_abs_intervention_rate_gap"].max()
        ) if not primary_exact.empty else None,
        "worst_primary_exact_budget_earliest_gap": float(
            primary_exact["earliest_mean_abs_intervention_rate_gap"].max()
        ) if not primary_exact.empty else None,
        "lowest_primary_exact_budget_confidence_match_fraction": float(
            primary_exact["confidence_exact_match_fraction"].min()
        ) if not primary_exact.empty else None,
        "lowest_primary_exact_budget_earliest_match_fraction": float(
            primary_exact["earliest_exact_match_fraction"].min()
        ) if not primary_exact.empty else None,
    }


def _extract_scorecard_statuses(scorecard: dict[str, Any]) -> tuple[bool, bool, str, str]:
    gates = scorecard.get("gates") or {}
    if not isinstance(gates, dict):
        gates = {}
    boundary_gate = scorecard.get("boundary_conditions") or gates.get("boundary_conditions") or {}
    readiness_gate = (
        scorecard.get("readiness")
        or scorecard.get("readiness_reproducibility")
        or gates.get("readiness")
        or gates.get("readiness_reproducibility")
        or {}
    )
    boundary_passed = bool(boundary_gate.get("passed"))
    readiness_passed = bool(readiness_gate.get("passed"))
    boundary_status = str(boundary_gate.get("status") or boundary_gate.get("report_status") or "UNKNOWN")
    readiness_status = str(readiness_gate.get("status") or ("PASS" if readiness_passed else "UNKNOWN"))
    return boundary_passed, readiness_passed, boundary_status, readiness_status


def _load_primary_pairwise(
    path: str | Path,
    *,
    required_families: list[str],
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required_columns = {
        "dataset_id",
        "split_family",
        "comparison_family",
        "metric",
        "policy_a",
        "estimate",
        "holm_reject_0_05",
    }
    missing = required_columns.difference(frame.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"primary_pairwise.csv is missing required columns: {names}")

    filtered = frame.copy()
    filtered = filtered.loc[filtered["dataset_id"].astype(str).str.lower() == "j1"].copy()
    filtered = filtered.loc[
        filtered["comparison_family"].astype(str).isin(PRIMARY_COMPARISON_FAMILIES)
    ].copy()
    filtered = filtered.loc[filtered["metric"].astype(str).isin(PRIMARY_METRICS)].copy()
    filtered = filtered.loc[filtered["split_family"].astype(str).isin(required_families)].copy()
    filtered["tau"] = filtered.apply(_extract_tau_from_pairwise_row, axis=1)
    filtered = filtered.loc[filtered["tau"].notna()].copy()
    if filtered.empty:
        raise ValueError("no J1 primary-comparison rows were found for the required frontier readout")
    return filtered


def _filter_pairwise_families(frame: pd.DataFrame, *, families: list[str]) -> pd.DataFrame:
    if not families:
        return frame.iloc[0:0].copy()
    return frame.loc[frame["split_family"].astype(str).isin(families)].copy()


def _summarize_pairwise_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(["split_family", "metric", "tau"], sort=True, dropna=False)
    for (split_family, metric, tau), tau_df in grouped:
        if len(tau_df.index) != 1:
            raise ValueError(
                "expected one primary pairwise row per split_family / metric / tau; "
                f"got {len(tau_df.index)} for {split_family} / {metric} / {tau}"
            )
        record = tau_df.iloc[0]
        estimate = float(record["estimate"])
        beneficial = _estimate_is_beneficial(str(metric), estimate)
        significant_beneficial = bool(record["holm_reject_0_05"]) and beneficial
        rows.append(
            {
                "split_family": str(split_family),
                "metric": str(metric),
                "tau": str(tau),
                "estimate": estimate,
                "directional_estimate": _directional_estimate(str(metric), estimate),
                "beneficial": beneficial,
                "significant_beneficial": significant_beneficial,
            }
        )
    return rows


def _resolve_universal_taus(
    rows: list[dict[str, Any]],
    *,
    required_families: list[str],
) -> list[str]:
    candidates = sorted({row["tau"] for row in rows}, key=_sort_tau_key)
    universal_taus: list[str] = []
    required_family_set = set(required_families)
    for tau in candidates:
        is_universal = True
        for metric in PRIMARY_METRICS:
            observed = {
                row["split_family"]
                for row in rows
                if row["tau"] == tau
                and row["metric"] == metric
                and row["significant_beneficial"]
            }
            if not required_family_set.issubset(observed):
                is_universal = False
                break
        if is_universal:
            universal_taus.append(str(tau))
    return universal_taus


def _build_family_summary(
    rows: list[dict[str, Any]],
    *,
    required_families: list[str],
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["split_family"], row["metric"]), []).append(row)

    for split_family in required_families:
        for metric in PRIMARY_METRICS:
            metric_rows = grouped.get((split_family, metric), [])
            if not metric_rows:
                summary.append(
                    {
                        "split_family": split_family,
                        "metric": metric,
                        "best_tau": None,
                        "best_estimate": None,
                        "best_is_beneficial": False,
                        "best_is_significant_beneficial": False,
                        "beneficial_taus": [],
                        "significant_beneficial_taus": [],
                    }
                )
                continue

            ordered = sorted(metric_rows, key=lambda row: _row_sort_key(metric, row), reverse=True)
            best = ordered[0]
            beneficial_taus = sorted(
                [row["tau"] for row in metric_rows if row["beneficial"]],
                key=_sort_tau_key,
            )
            significant_beneficial_taus = sorted(
                [row["tau"] for row in metric_rows if row["significant_beneficial"]],
                key=_sort_tau_key,
            )
            summary.append(
                {
                    "split_family": split_family,
                    "metric": metric,
                    "best_tau": best["tau"],
                    "best_estimate": best["estimate"],
                    "best_is_beneficial": best["beneficial"],
                    "best_is_significant_beneficial": best["significant_beneficial"],
                    "beneficial_taus": beneficial_taus,
                    "significant_beneficial_taus": significant_beneficial_taus,
                }
            )

    return summary


def _build_interpretation(
    *,
    scorecard_decision: str,
    universal_taus: list[str],
    best_tau_set: set[str],
    metric_disagreement_families: list[str],
    boundary_passed: bool,
    readiness_passed: bool,
) -> list[str]:
    lines: list[str] = []
    if universal_taus:
        lines.append(
            "At least one fixed tau clears the matched-budget primary gate across every required family, "
            "so a universal-winner claim remains live."
        )
    else:
        lines.append(
            "No fixed tau clears the matched-budget primary gate across every required family and both "
            "primary metrics."
        )
    if len(best_tau_set) > 1:
        lines.append(
            "Best taus vary across families or metrics, which is direct operating-point instability rather "
            "than simple tuning noise."
        )
    elif best_tau_set:
        lines.append(
            "Best-tau agreement is higher than expected, but the scorecard still needs a matched-budget "
            "dominance story before claiming a universal winner."
        )
    if metric_disagreement_families:
        joined = ", ".join(metric_disagreement_families)
        lines.append(
            f"Within-family tension remains visible because primary metrics disagree on the best tau for {joined}."
        )
    if boundary_passed and readiness_passed:
        lines.append(
            "Boundary and reproducibility gates pass, so the failed universal-winner story is a scientific "
            "result, not a pipeline failure."
        )
    lines.append(
        f"The current release should be framed as J1-Frontier: a causal-prefix intervention frontier benchmark "
        f"under shift, with the universal-winner scorecard recorded as {scorecard_decision}."
    )
    return lines


def _infer_policy_family(policy: str) -> str:
    value = str(policy)
    if value.startswith("agency_margin_tau_"):
        return "agency_margin"
    if value.startswith("set_acsa_tau_"):
        return "set_acsa"
    if value.startswith("plain_conf_threshold_matched_tau_"):
        return "plain_conf_threshold_matched"
    if value.startswith("confidence_blend_alpha_"):
        return "confidence_blend"
    if value.startswith("selective_prediction"):
        return "selective_prediction"
    if value == "user_only":
        return "user_only"
    if value == "assist_only":
        return "assist_only"
    return value


def _split_tier(split_family: str, primary_families: list[str], boundary_families: list[str]) -> str:
    if split_family in set(primary_families):
        return "primary"
    if split_family in set(boundary_families):
        return "boundary"
    return "descriptive"


def _best_agency_tau_for_metric(family_df: pd.DataFrame, *, metric: str) -> str | None:
    agency = family_df.loc[
        (family_df["policy_family"].astype(str) == "agency_margin")
        & family_df["tau"].notna()
        & family_df[metric].notna()
    ].copy()
    if agency.empty:
        return None
    agency["directional_metric"] = agency[metric].astype(float).map(
        lambda value: _directional_estimate(metric, float(value))
    )
    ordered = agency.sort_values(
        ["directional_metric", "tau"],
        ascending=[False, True],
    )
    return str(ordered.iloc[0]["tau"])


def _frontier_label(policy_family: str, tau: str | None) -> str:
    if tau is None or (isinstance(tau, float) and np.isnan(tau)) or str(tau).lower() == "nan":
        return policy_family
    return f"{policy_family}@{tau}"


def _pareto_mask(frame: pd.DataFrame, objectives: dict[str, str]) -> list[bool]:
    values = frame.loc[:, list(objectives)].to_numpy(dtype=float)
    transformed = values.copy()
    directions = [objectives[column] for column in objectives]
    for index, direction in enumerate(directions):
        if direction == "min":
            transformed[:, index] = -transformed[:, index]
    n_rows = transformed.shape[0]
    mask = np.ones(n_rows, dtype=bool)
    for row_index in range(n_rows):
        for other_index in range(n_rows):
            if row_index == other_index:
                continue
            other = transformed[other_index]
            current = transformed[row_index]
            if np.all(other >= current - 1e-12) and np.any(other > current + 1e-12):
                mask[row_index] = False
                break
    return mask.tolist()


def _metric_regret(
    *,
    family_df: pd.DataFrame,
    current_value: float,
    metric: str,
) -> float:
    if np.isnan(current_value):
        return float("nan")
    best_directional = family_df[metric].astype(float).map(
        lambda value: _directional_estimate(metric, float(value))
    ).max()
    current_directional = _directional_estimate(metric, current_value)
    return float(best_directional - current_directional)


def _row_dominates(row_a: pd.Series, row_b: pd.Series, objectives: dict[str, str]) -> bool:
    better_or_equal = True
    strictly_better = False
    for metric, direction in objectives.items():
        value_a = _safe_float(row_a.get(metric))
        value_b = _safe_float(row_b.get(metric))
        if np.isnan(value_a) or np.isnan(value_b):
            return False
        if direction == "max":
            better_or_equal = better_or_equal and value_a >= value_b - 1e-12
            strictly_better = strictly_better or value_a > value_b + 1e-12
        else:
            better_or_equal = better_or_equal and value_a <= value_b + 1e-12
            strictly_better = strictly_better or value_a < value_b - 1e-12
    return bool(better_or_equal and strictly_better)


def _interpolate_policy_curve(
    *,
    curve_df: pd.DataFrame,
    target_budget: float,
    metrics: tuple[str, ...],
) -> dict[str, Any]:
    budgets = curve_df["intervention_rate"].to_numpy(dtype=float)
    nearest_gap = float(np.min(np.abs(budgets - float(target_budget)))) if budgets.size else float("nan")
    extrapolated = bool(target_budget < float(np.nanmin(budgets)) or target_budget > float(np.nanmax(budgets)))
    values: dict[str, float] = {}
    for metric in metrics:
        metric_df = curve_df.loc[curve_df[metric].notna(), ["intervention_rate", metric]].copy()
        if metric_df.empty:
            values[metric] = float("nan")
            continue
        metric_df = (
            metric_df.groupby("intervention_rate", as_index=False)[metric]
            .mean()
            .sort_values("intervention_rate")
            .reset_index(drop=True)
        )
        xs = metric_df["intervention_rate"].to_numpy(dtype=float)
        ys = metric_df[metric].to_numpy(dtype=float)
        if xs.size == 1:
            values[metric] = float(ys[0])
        else:
            values[metric] = float(np.interp(float(target_budget), xs, ys))
    return {
        "nearest_gap": nearest_gap,
        "extrapolated": extrapolated,
        "values": values,
    }


def _extract_tau_from_pairwise_row(row: pd.Series) -> str | None:
    for column in ("policy_a", "policy_b"):
        if column in row and isinstance(row[column], str):
            tau = _extract_tau(row[column])
            if tau is not None:
                return tau
    return None


def _extract_tau(value: str) -> str | None:
    match = TAU_PATTERN.search(str(value))
    if match is None:
        return None
    return _normalize_tau_value(match.group(1))


def _normalize_tau_value(value: object) -> str:
    numeric = float(str(value).strip())
    return f"{numeric:.2f}"


def _estimate_is_beneficial(metric: str, estimate: float) -> bool:
    if metric in PACKAGE_LOWER_IS_BETTER_METRICS:
        return estimate < 0.0
    return estimate > 0.0


def _directional_estimate(metric: str, estimate: float) -> float:
    if metric in PACKAGE_LOWER_IS_BETTER_METRICS:
        return -estimate
    return estimate


def _directional_delta_or_nan(metric: str, left: float, right: float) -> float:
    if np.isnan(left) or np.isnan(right):
        return float("nan")
    if metric in PACKAGE_LOWER_IS_BETTER_METRICS:
        return float(right - left)
    return float(left - right)


def _delta_or_nan(left: float, right: float) -> float:
    if np.isnan(left) or np.isnan(right):
        return float("nan")
    return float(left - right)


def _row_sort_key(metric: str, row: dict[str, Any]) -> tuple[int, int, float, float]:
    return (
        int(bool(row["significant_beneficial"])),
        int(bool(row["beneficial"])),
        float(row["directional_estimate"]),
        -_sort_tau_key(str(row["tau"])),
    )


def _sort_tau_key(value: str) -> float:
    return float(str(value))


def _normalize_string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if str(value).strip()]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _mean_or_nan(values: list[float]) -> float:
    cleaned = np.asarray([value for value in values if not np.isnan(value)], dtype=float)
    if cleaned.size == 0:
        return float("nan")
    return float(cleaned.mean())


def _max_or_nan(values: list[float]) -> float:
    cleaned = np.asarray([value for value in values if not np.isnan(value)], dtype=float)
    if cleaned.size == 0:
        return float("nan")
    return float(cleaned.max())


def _frame_to_markdown(
    frame: pd.DataFrame,
    *,
    columns: list[str],
    max_rows: int,
) -> list[str]:
    available = [column for column in columns if column in frame.columns]
    if not available or frame.empty:
        return ["- <no rows>"]
    table = frame.loc[:, available].head(max_rows).copy()
    for column in table.columns:
        if pd.api.types.is_float_dtype(table[column]):
            table[column] = table[column].map(_format_float)
    header = "| " + " | ".join(table.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    lines = [header, divider]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in table.columns) + " |")
    if frame.shape[0] > max_rows:
        lines.append(f"- Showing first {max_rows} of {frame.shape[0]} rows.")
    return lines


def _format_float(value: object) -> str:
    if value is None:
        return "<none>"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(numeric):
        return "<none>"
    return f"{numeric:.4f}"


__all__ = [
    "COMPARATOR_OBJECTIVES",
    "FRONTIER_OBJECTIVES",
    "J1_FRONTIER_PACKAGE_ARTIFACTS",
    "PRIMARY_COMPARISON_FAMILIES",
    "PRIMARY_METRICS",
    "REGRET_METRICS",
    "build_j1_frontier_package",
    "build_j1_frontier_readout_payload",
    "render_j1_frontier_package",
    "render_j1_frontier_readout",
]

