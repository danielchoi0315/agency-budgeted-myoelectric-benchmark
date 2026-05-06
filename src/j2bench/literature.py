from __future__ import annotations

from typing import Any


COMPARABILITY_LEVELS = ("direct", "partial", "not_comparable", "contextual", "boundary")
COMPARABILITY_CRITERIA_FIELDS = (
    "dataset",
    "task_family",
    "split_shift",
    "population",
    "adaptation",
    "decision_object",
    "metric",
)


def anchors_for_target(anchors: list[dict[str, Any]], target_id: str) -> list[dict[str, Any]]:
    return [anchor for anchor in anchors if target_id in set(anchor.get("targets", []))]


def split_anchor_groups(anchors: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {
        "direct": [],
        "partial": [],
        "contextual": [],
        "boundary": [],
        "other": [],
    }
    for anchor in anchors:
        comparability = str(anchor.get("comparability", "other")).lower()
        groups.setdefault(comparability, []).append(anchor)
    return groups


def comparison_rows_for_target(rows: list[dict[str, Any]], target_id: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("target_id", "")) == str(target_id)]


def split_comparison_groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {level: [] for level in COMPARABILITY_LEVELS}
    for row in rows:
        level = str(row.get("derived_comparability", "not_comparable")).lower()
        groups.setdefault(level, []).append(row)
    return groups


def derive_comparison_classification(comparison: dict[str, Any]) -> str:
    role = str(comparison.get("role", "comparator")).strip().lower()
    if role == "boundary":
        return "boundary"
    if role == "context":
        return "contextual"

    criteria = _normalize_criteria(comparison.get("criteria", {}))
    direct_exact_fields = ("dataset", "task_family", "split_shift", "population", "adaptation", "decision_object")
    if all(criteria[field] == "exact" for field in direct_exact_fields) and criteria["metric"] in {"exact", "convertible"}:
        return "direct"
    if criteria["dataset"] == "exact" and criteria["task_family"] in {"exact", "related"} and criteria["population"] in {"exact", "related"}:
        return "partial"
    return "not_comparable"


def build_comparability_rows(
    comparability_cfg: dict[str, Any],
    *,
    targets: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    target_lookup = {str(target.get("target_id", "")): target for target in targets or []}
    paper_lookup = {str(paper.get("paper_id", "")): paper for paper in comparability_cfg.get("papers", [])}
    rows: list[dict[str, Any]] = []
    for comparison in comparability_cfg.get("comparisons", []):
        paper_id = str(comparison.get("paper_id", ""))
        target_id = str(comparison.get("target_id", ""))
        paper = dict(paper_lookup.get(paper_id, {}))
        target = dict(target_lookup.get(target_id, {}))
        criteria = _normalize_criteria(comparison.get("criteria", {}))
        derived = derive_comparison_classification(comparison)
        blocking_fields = [
            field
            for field, value in criteria.items()
            if field in COMPARABILITY_CRITERIA_FIELDS and value not in {"exact", "convertible"}
        ]
        row = {
            "comparison_id": str(comparison.get("comparison_id", f"{paper_id}__{target_id}")),
            "paper_id": paper_id,
            "citation": str(paper.get("citation", comparison.get("citation", ""))),
            "url": str(paper.get("url", comparison.get("url", ""))),
            "artifact_type": str(paper.get("artifact_type", "")),
            "target_id": target_id,
            "target_dataset_id": str(target.get("dataset_id", comparison.get("dataset_id", ""))),
            "target_split_family": str(target.get("split_family", "")),
            "target_priority": str(target.get("priority", "")),
            "target_claim_surface": str(target.get("claim_surface", "benchmark_only")),
            "target_direct_sota_eligible": bool(target.get("direct_sota_eligible", False)),
            "target_metric_name": str(target.get("local_metric_name", target.get("local_metric_field", ""))),
            "role": str(comparison.get("role", "comparator")),
            "anchor_id": str(comparison.get("anchor_id", "")),
            "derived_comparability": derived,
            "claim_blocked": derived != "direct",
            "blocking_fields": "|".join(blocking_fields),
            "note": str(comparison.get("note", "")),
        }
        for field in COMPARABILITY_CRITERIA_FIELDS:
            row[field] = criteria[field]
        reported_metric = comparison.get("reported_metric", {})
        row["reported_metric_name"] = str(reported_metric.get("name", ""))
        row["reported_metric_value"] = _coerce_optional_float(reported_metric.get("value"))
        row["reported_metric_unit"] = str(reported_metric.get("unit", ""))
        rows.append(row)
    return rows


def summarize_comparability_matrix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for target_id in sorted({str(row.get("target_id", "")) for row in rows}):
        target_rows = comparison_rows_for_target(rows, target_id)
        groups = split_comparison_groups(target_rows)
        exemplar = target_rows[0] if target_rows else {}
        summaries.append(
            {
                "target_id": target_id,
                "target_dataset_id": str(exemplar.get("target_dataset_id", "")),
                "target_split_family": str(exemplar.get("target_split_family", "")),
                "target_priority": str(exemplar.get("target_priority", "")),
                "target_claim_surface": str(exemplar.get("target_claim_surface", "benchmark_only")),
                "target_direct_sota_eligible": bool(exemplar.get("target_direct_sota_eligible", False)),
                "n_direct": len(groups.get("direct", [])),
                "n_partial": len(groups.get("partial", [])),
                "n_not_comparable": len(groups.get("not_comparable", [])),
                "n_contextual": len(groups.get("contextual", [])),
                "n_boundary": len(groups.get("boundary", [])),
            }
        )
    return summaries


def select_best_numeric_anchor(
    anchors: list[dict[str, Any]],
    *,
    higher_is_better: bool,
) -> dict[str, Any] | None:
    numeric = [anchor for anchor in anchors if _coerce_optional_float(anchor.get("metric_value")) is not None]
    if not numeric:
        return None
    key = lambda item: float(item["metric_value"])
    return max(numeric, key=key) if higher_is_better else min(numeric, key=key)


def evaluate_target(
    target: dict[str, Any],
    observation_row: dict[str, Any] | None,
    anchors: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    target_id = str(target["target_id"])
    higher_is_better = bool(target.get("higher_is_better", True))
    target_anchors = anchors_for_target(anchors, target_id)
    groups = split_anchor_groups(target_anchors)
    target_comparisons = comparison_rows_for_target(comparison_rows or [], target_id)
    comparison_groups = split_comparison_groups(target_comparisons)
    if target_comparisons:
        direct_anchor_ids = {str(row.get("anchor_id", "")) for row in comparison_groups.get("direct", []) if row.get("anchor_id")}
        partial_anchor_ids = {str(row.get("anchor_id", "")) for row in comparison_groups.get("partial", []) if row.get("anchor_id")}
        direct_anchor_pool = [anchor for anchor in target_anchors if str(anchor.get("anchor_id", "")) in direct_anchor_ids]
        partial_anchor_pool = [anchor for anchor in target_anchors if str(anchor.get("anchor_id", "")) in partial_anchor_ids]
    else:
        direct_anchor_pool = groups.get("direct", [])
        partial_anchor_pool = groups.get("partial", [])
    result: dict[str, Any] = {
        "target_id": target_id,
        "dataset_id": str(target["dataset_id"]),
        "split_family": str(target["split_family"]),
        "priority": str(target.get("priority", "supporting")),
        "description": str(target.get("description", "")),
        "claim_surface": str(target.get("claim_surface", "benchmark_only")),
        "direct_sota_eligible": bool(target.get("direct_sota_eligible", False)),
        "local_metric_field": str(target["local_metric_field"]),
        "local_metric_name": str(target.get("local_metric_name", target["local_metric_field"])),
        "direct_anchor_count": len(comparison_groups.get("direct", [])) if target_comparisons else len(groups.get("direct", [])),
        "partial_anchor_count": len(comparison_groups.get("partial", [])) if target_comparisons else len(groups.get("partial", [])),
        "contextual_anchor_count": len(comparison_groups.get("contextual", [])) if target_comparisons else len(groups.get("contextual", [])),
        "boundary_anchor_count": len(comparison_groups.get("boundary", [])) if target_comparisons else len(groups.get("boundary", [])),
        "not_comparable_anchor_count": len(comparison_groups.get("not_comparable", [])) if target_comparisons else 0,
        "local_observation_available": observation_row is not None,
    }
    if observation_row is None:
        result.update(
            {
                "status": "missing_local_result",
                "claim_ready": False,
                "detail": "No local benchmark observation matched the configured dataset and split family.",
            }
        )
        return result

    local_metric_value = _coerce_optional_float(observation_row.get(target["local_metric_field"]))
    baseline_metric_value = _coerce_optional_float(observation_row.get(target.get("baseline_metric_field", "")))
    local_policy = observation_row.get(target.get("policy_field", "best_agency_policy"))
    direct_best = select_best_numeric_anchor(direct_anchor_pool, higher_is_better=higher_is_better)
    partial_numeric = [
        anchor
        for anchor in partial_anchor_pool
        if bool(anchor.get("surface_numeric_reference", False)) or _coerce_optional_float(anchor.get("metric_value")) is not None
    ]
    partial_best = select_best_numeric_anchor(partial_numeric, higher_is_better=higher_is_better)

    result.update(
        {
            "local_metric_value": local_metric_value,
            "baseline_metric_value": baseline_metric_value,
            "local_policy": str(local_policy) if local_policy is not None else "",
            "baseline_delta": (local_metric_value - baseline_metric_value)
            if local_metric_value is not None and baseline_metric_value is not None
            else None,
            "partial_numeric_anchor_count": len(partial_numeric),
            "best_direct_anchor_id": direct_best.get("anchor_id") if direct_best else None,
            "best_direct_anchor_value": _coerce_optional_float(direct_best.get("metric_value")) if direct_best else None,
            "best_partial_anchor_id": partial_best.get("anchor_id") if partial_best else None,
            "best_partial_anchor_value": _coerce_optional_float(partial_best.get("metric_value")) if partial_best else None,
        }
    )

    if local_metric_value is None:
        result.update(
            {
                "status": "missing_local_metric",
                "claim_ready": False,
                "detail": "Configured local metric field is missing or non-numeric in the observation row.",
            }
        )
        return result

    if not direct_anchor_pool and not comparison_groups.get("direct", []):
        result.update(
            {
                "status": "not_established_no_direct_anchor",
                "claim_ready": False,
                "detail": "No directly comparable published anchor is registered for this dataset/protocol/metric target.",
            }
        )
        return result

    if direct_best is None:
        result.update(
            {
                "status": "not_established_non_numeric_direct_anchor",
                "claim_ready": False,
                "detail": "Direct anchors exist, but none carry a numeric metric value that can be compared safely.",
            }
        )
        return result

    direct_value = float(direct_best["metric_value"])
    better_than_direct = local_metric_value > direct_value if higher_is_better else local_metric_value < direct_value
    statistical_support_ready = bool(target.get("statistical_support_ready", False))
    if better_than_direct:
        if not bool(target.get("direct_sota_eligible", False)):
            result.update(
                {
                    "status": "candidate_direct_numeric_win_benchmark_only",
                    "claim_ready": False,
                    "detail": (
                        "Local result exceeds the best directly comparable published anchor, "
                        "but this target is registered as benchmark-only rather than a direct-SOTA release target."
                    ),
                }
            )
            return result
        if not statistical_support_ready:
            result.update(
                {
                    "status": "candidate_direct_numeric_win_pending_stats",
                    "claim_ready": False,
                    "detail": (
                        "Local result exceeds the best directly comparable published anchor, "
                        "but the registered statistical support gate is not yet satisfied."
                    ),
                }
            )
            return result
        result.update(
            {
                "status": "candidate_direct_sota",
                "claim_ready": True,
                "detail": f"Local result exceeds the best directly comparable published anchor ({direct_best['anchor_id']}).",
            }
        )
        return result

    result.update(
        {
            "status": "not_established_direct_anchor_not_beaten",
            "claim_ready": False,
            "detail": f"Local result does not exceed the best directly comparable published anchor ({direct_best['anchor_id']}).",
        }
    )
    return result


def summarize_overall_status(
    target_results: list[dict[str, Any]],
    *,
    anchor_status: str,
    guardrails: dict[str, Any] | None = None,
) -> dict[str, Any]:
    guardrails = guardrails or {}
    primary_targets = [result for result in target_results if result.get("priority") == "primary"]
    eligible_primary_targets = [result for result in primary_targets if bool(result.get("direct_sota_eligible", False))]
    if anchor_status != "provisionally_ready":
        status = "blocked_benchmark_not_ready"
        recommendation = "Do not make a SOTA claim while the benchmark readiness gates are not satisfied."
    elif any(
        result["status"] in {"missing_local_result", "missing_local_metric"} for result in primary_targets
    ):
        status = "blocked_primary_evidence_incomplete"
        recommendation = "Do not make a benchmark or SOTA claim until every primary target has valid local evidence."
    elif not eligible_primary_targets:
        status = "claim_safe_benchmark_only"
        recommendation = (
            "No primary target is registered as a paper-matched direct-SOTA release target. "
            "Use benchmark-only wording even when comparator papers are discussed."
        )
    elif any(result["status"] == "candidate_direct_numeric_win_pending_stats" for result in eligible_primary_targets):
        status = "direct_numeric_win_pending_stats"
        recommendation = (
            "A direct comparable numeric win exists, but the statistical support gate is not yet satisfied. "
            "Do not claim SOTA."
        )
    elif any(result["status"] == "candidate_direct_sota" for result in eligible_primary_targets) and all(
        result["claim_ready"] for result in eligible_primary_targets
    ):
        if any(int(result.get("boundary_anchor_count", 0)) > 0 for result in eligible_primary_targets):
            status = "direct_numeric_win_boundary_limited"
            recommendation = (
                "A direct comparable numeric win exists, but boundary anchors still block broader SOTA wording. "
                "Keep the claim benchmark-scoped and non-clinical."
            )
        else:
            status = "direct_comparable_sota_supported"
            recommendation = "A protocol-specific SOTA claim is supportable for the primary targets registered here."
    elif any(result["status"] == "not_established_direct_anchor_not_beaten" for result in eligible_primary_targets):
        status = "claim_safe_non_sota"
        recommendation = "Do not claim SOTA; at least one directly comparable published anchor remains stronger."
    elif any(result["status"] == "not_established_non_numeric_direct_anchor" for result in eligible_primary_targets):
        status = "claim_safe_benchmark_only"
        recommendation = "Do not claim SOTA; the registered direct literature anchors are not numerically comparable yet."
    else:
        status = "claim_safe_benchmark_only"
        recommendation = (
            "Do not claim true SOTA. The current evidence supports a reproducible benchmark/policy-layer contribution, "
            "but not a blanket best-published claim."
        )

    return {
        "status": status,
        "recommendation": recommendation,
        "disallowed_terms": list(guardrails.get("disallowed_terms_without_direct_support", [])),
        "preferred_terms": list(guardrails.get("preferred_terms", [])),
    }


def _normalize_criteria(criteria: dict[str, Any] | None) -> dict[str, str]:
    normalized: dict[str, str] = {}
    payload = criteria or {}
    for field in COMPARABILITY_CRITERIA_FIELDS:
        token = str(payload.get(field, "unknown")).strip().lower()
        normalized[field] = token if token else "unknown"
    return normalized


def _coerce_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

