from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PRIMARY_METRICS = ("active_macro_f1", "active_risk_coverage_auc")
SAFETY_METRICS = (
    "stable_safe_episode_rate",
    "final_correct_rate",
    "median_earliest_stable_safe_s",
)
LOWER_IS_BETTER_METRICS = {
    "active_risk_coverage_auc",
    "median_earliest_safe_s",
    "median_earliest_stable_safe_s",
}
EXACT_BUDGET_COMPARISON_FAMILY = "agency_vs_plain_conf_exact_budget"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a cross-dataset exact-budget instability report from publication artifacts."
    )
    parser.add_argument(
        "--publication-root",
        type=Path,
        default=Path("results/reports/publication_clean"),
    )
    parser.add_argument(
        "--j1-root",
        type=Path,
        default=Path("results/reports/j1_open"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("results/reports/publication_clean"),
    )
    args = parser.parse_args()

    payload = build_payload(
        publication_root=args.publication_root,
        j1_root=args.j1_root,
    )
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    by_tau = pd.DataFrame(payload["by_tau"])
    overview = pd.DataFrame(payload["dataset_overview"])
    by_tau.to_csv(out_root / "exact_budget_instability_by_tau.csv", index=False)
    overview.to_csv(out_root / "exact_budget_instability_summary.csv", index=False)
    (out_root / "exact_budget_instability_status.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (out_root / "exact_budget_instability_report.md").write_text(
        build_markdown(payload),
        encoding="utf-8",
    )
    print(f"Wrote exact-budget instability report to {out_root}")


def build_payload(
    *,
    publication_root: Path,
    j1_root: Path,
) -> dict[str, Any]:
    dataset_specs = [
        {
            "dataset_id": "db10",
            "confidence_pairwise": publication_root / "db10_confidence_gate_exact_budget_pairwise.csv",
            "earliest_pairwise": publication_root / "db10_earliest_safe_exact_budget_pairwise.csv",
            "required_families_path": None,
        },
        {
            "dataset_id": "hyser",
            "confidence_pairwise": publication_root / "hyser_confidence_gate_exact_budget_pairwise.csv",
            "earliest_pairwise": publication_root / "hyser_earliest_safe_exact_budget_pairwise.csv",
            "required_families_path": None,
        },
        {
            "dataset_id": "cemhsey",
            "confidence_pairwise": publication_root / "cemhsey_confidence_gate_exact_budget_pairwise.csv",
            "earliest_pairwise": publication_root / "cemhsey_earliest_safe_exact_budget_pairwise.csv",
            "required_families_path": None,
        },
        {
            "dataset_id": "j1",
            "confidence_pairwise": j1_root / "publication_clean" / "j1_confidence_gate_exact_budget_pairwise.csv",
            "earliest_pairwise": j1_root / "publication_clean" / "j1_earliest_safe_exact_budget_pairwise.csv",
            "required_families_path": j1_root / "j1_scorecard.json",
        },
    ]

    by_tau_rows: list[dict[str, Any]] = []
    overview_rows: list[dict[str, Any]] = []
    for spec in dataset_specs:
        confidence = _load_exact_budget_pairwise(spec["confidence_pairwise"])
        earliest = _load_exact_budget_pairwise(spec["earliest_pairwise"])
        observed_families = sorted(
            set(confidence["split_family"].astype(str)).union(earliest["split_family"].astype(str))
        )
        required_families = _resolve_required_families(
            observed_families=observed_families,
            scorecard_path=spec["required_families_path"],
        )
        by_tau = _build_claim_summary(
            dataset_id=spec["dataset_id"],
            confidence_pairwise=confidence,
            earliest_pairwise=earliest,
            required_families=required_families,
        )
        by_tau_rows.extend(by_tau.to_dict(orient="records"))
        overview_rows.append(
            _build_dataset_overview(
                dataset_id=spec["dataset_id"],
                by_tau=by_tau,
                confidence_pairwise=confidence,
                required_families=required_families,
                observed_families=observed_families,
            )
        )
    return {
        "status": _overall_status(overview_rows),
        "dataset_overview": overview_rows,
        "by_tau": by_tau_rows,
    }


def _load_exact_budget_pairwise(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "split_family",
                "comparison_family",
                "metric",
                "tau",
                "estimate",
                "beneficial",
                "holm_reject",
                "beneficial_and_holm_significant",
                "mean_abs_intervention_rate_gap",
                "exact_match_fraction",
            ]
        )
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
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
    normalized["tau"] = normalized["tau"].map(_normalize_tau_value)
    return normalized


def _resolve_required_families(
    *,
    observed_families: list[str],
    scorecard_path: Path | None,
) -> list[str]:
    if scorecard_path is None or not scorecard_path.exists():
        return list(observed_families)
    payload = json.loads(scorecard_path.read_text(encoding="utf-8"))
    required = [
        str(family)
        for family in payload.get("required_split_families", [])
        if str(family).strip()
    ]
    return required or list(observed_families)


def _build_claim_summary(
    *,
    dataset_id: str,
    confidence_pairwise: pd.DataFrame,
    earliest_pairwise: pd.DataFrame,
    required_families: list[str],
) -> pd.DataFrame:
    columns = [
        "dataset_id",
        "split_family",
        "scope",
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
    if confidence_pairwise.empty and earliest_pairwise.empty:
        return pd.DataFrame(columns=columns)

    tau_values = sorted(
        {
            str(tau)
            for tau in pd.concat(
                [
                    confidence_pairwise.get("tau", pd.Series(dtype=str)).astype(str),
                    earliest_pairwise.get("tau", pd.Series(dtype=str)).astype(str),
                ],
                ignore_index=True,
            )
            if str(tau).strip() and str(tau) != "nan"
        },
        key=_sort_tau_key,
    )
    split_values = sorted(
        set(confidence_pairwise["split_family"].astype(str)).union(
            earliest_pairwise["split_family"].astype(str)
        )
    )

    rows: list[dict[str, Any]] = []
    required_set = set(required_families)
    for split_family in split_values:
        scope = "required" if split_family in required_set else "descriptive"
        for tau in tau_values:
            conf_subset = confidence_pairwise.loc[
                (confidence_pairwise["split_family"].astype(str) == split_family)
                & (confidence_pairwise["tau"].astype(str) == tau)
                & confidence_pairwise["metric"].astype(str).isin(PRIMARY_METRICS)
            ].copy()
            early_subset = earliest_pairwise.loc[
                (earliest_pairwise["split_family"].astype(str) == split_family)
                & (earliest_pairwise["tau"].astype(str) == tau)
                & earliest_pairwise["metric"].astype(str).isin(SAFETY_METRICS)
            ].copy()
            if conf_subset.empty and early_subset.empty:
                continue
            conf_sig_metrics = set(
                conf_subset.loc[
                    conf_subset["beneficial_and_holm_significant"].astype(bool),
                    "metric",
                ].astype(str)
            )
            conf_harmful_sig = int(
                (
                    conf_subset["holm_reject"].astype(bool)
                    & ~conf_subset["beneficial"].astype(bool)
                ).sum()
            ) if not conf_subset.empty else 0
            safety_sig = int(
                early_subset["beneficial_and_holm_significant"].astype(bool).sum()
            ) if not early_subset.empty else 0
            safety_harmful_sig = int(
                (
                    early_subset["holm_reject"].astype(bool)
                    & ~early_subset["beneficial"].astype(bool)
                ).sum()
            ) if not early_subset.empty else 0
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "split_family": split_family,
                    "scope": scope,
                    "tau": tau,
                    "confidence_primary_metric_count": int(conf_subset.shape[0]),
                    "confidence_primary_sig_beneficial_count": int(len(conf_sig_metrics)),
                    "confidence_all_primary_metrics_sig_beneficial": (
                        set(conf_subset["metric"].astype(str)) == set(PRIMARY_METRICS)
                        and conf_sig_metrics == set(PRIMARY_METRICS)
                    ),
                    "confidence_harmful_sig_metric_count": conf_harmful_sig,
                    "safety_metric_count": int(early_subset.shape[0]),
                    "safety_sig_beneficial_count": safety_sig,
                    "safety_harmful_sig_metric_count": safety_harmful_sig,
                    "confidence_mean_abs_intervention_rate_gap": _mean_or_none(
                        conf_subset.get("mean_abs_intervention_rate_gap", pd.Series(dtype=float))
                    ),
                    "confidence_exact_match_fraction": _mean_or_none(
                        conf_subset.get("exact_match_fraction", pd.Series(dtype=float))
                    ),
                    "earliest_mean_abs_intervention_rate_gap": _mean_or_none(
                        early_subset.get("mean_abs_intervention_rate_gap", pd.Series(dtype=float))
                    ),
                    "earliest_exact_match_fraction": _mean_or_none(
                        early_subset.get("exact_match_fraction", pd.Series(dtype=float))
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["dataset_id", "split_family", "tau"]).reset_index(drop=True)


def _build_dataset_overview(
    *,
    dataset_id: str,
    by_tau: pd.DataFrame,
    confidence_pairwise: pd.DataFrame,
    required_families: list[str],
    observed_families: list[str],
) -> dict[str, Any]:
    required_set = set(required_families)
    required_rows = by_tau.loc[by_tau["scope"].astype(str) == "required"].copy()
    universal_taus = _resolve_universal_taus(required_rows, required_families=required_families)
    best_tau_payload = _summarize_best_taus(
        confidence_pairwise=confidence_pairwise,
        required_families=required_families,
    )
    return {
        "dataset_id": dataset_id,
        "observed_split_family_count": len(observed_families),
        "required_split_family_count": len(required_families),
        "required_split_families": "|".join(required_families),
        "universal_exact_budget_taus": "|".join(universal_taus),
        "n_universal_exact_budget_taus": len(universal_taus),
        "fixed_tau_supported": bool(universal_taus),
        "n_unique_best_taus": len(best_tau_payload["unique_best_taus"]),
        "unique_best_taus": "|".join(best_tau_payload["unique_best_taus"]),
        "metric_disagreement_families": "|".join(best_tau_payload["metric_disagreement_families"]),
        "worst_confidence_gap": _max_or_none(required_rows["confidence_mean_abs_intervention_rate_gap"]),
        "worst_earliest_gap": _max_or_none(required_rows["earliest_mean_abs_intervention_rate_gap"]),
        "lowest_confidence_exact_match_fraction": _min_or_none(required_rows["confidence_exact_match_fraction"]),
        "lowest_earliest_exact_match_fraction": _min_or_none(required_rows["earliest_exact_match_fraction"]),
        "worst_safety_harmful_sig_metric_count": _max_or_none(required_rows["safety_harmful_sig_metric_count"]),
        "instability_story": "fixed_tau_supported" if universal_taus else "fixed_tau_unstable",
        "required_scope": "scorecard_required" if dataset_id == "j1" else "all_split_families",
        "required_scope_missing_families": "|".join(
            sorted(required_set.difference(set(observed_families)))
        ),
    }


def _resolve_universal_taus(
    frame: pd.DataFrame,
    *,
    required_families: list[str],
) -> list[str]:
    if frame.empty or not required_families:
        return []
    required_set = set(required_families)
    supported: list[str] = []
    for tau, tau_df in frame.groupby("tau", sort=True):
        present = set(tau_df["split_family"].astype(str))
        if present != required_set:
            continue
        if not tau_df["confidence_all_primary_metrics_sig_beneficial"].astype(bool).all():
            continue
        if bool((tau_df["confidence_harmful_sig_metric_count"].astype(int) > 0).any()):
            continue
        if bool((tau_df["safety_harmful_sig_metric_count"].astype(int) > 0).any()):
            continue
        if bool((tau_df["confidence_primary_metric_count"].astype(int) < len(PRIMARY_METRICS)).any()):
            continue
        if bool((tau_df["safety_metric_count"].astype(int) < len(SAFETY_METRICS)).any()):
            continue
        supported.append(str(tau))
    return sorted(supported, key=_sort_tau_key)


def _summarize_best_taus(
    *,
    confidence_pairwise: pd.DataFrame,
    required_families: list[str],
) -> dict[str, Any]:
    required_set = set(required_families)
    required_rows = confidence_pairwise.loc[
        confidence_pairwise["split_family"].astype(str).isin(required_set)
        & confidence_pairwise["metric"].astype(str).isin(PRIMARY_METRICS)
    ].copy()
    if required_rows.empty:
        return {
            "unique_best_taus": [],
            "metric_disagreement_families": [],
        }
    required_rows["directional_estimate"] = required_rows.apply(
        lambda row: _directional_estimate(
            str(row["metric"]),
            float(row["estimate"]),
        ),
        axis=1,
    )
    best_tau_set: set[str] = set()
    disagreement_families: list[str] = []
    for split_family, family_df in required_rows.groupby("split_family", sort=True):
        family_best: set[str] = set()
        for metric in PRIMARY_METRICS:
            metric_df = family_df.loc[family_df["metric"].astype(str) == metric].copy()
            if metric_df.empty:
                continue
            ordered = metric_df.sort_values(
                ["directional_estimate", "tau"],
                ascending=[False, True],
            )
            best_tau = str(ordered.iloc[0]["tau"])
            family_best.add(best_tau)
            best_tau_set.add(best_tau)
        if len(family_best) > 1:
            disagreement_families.append(str(split_family))
    return {
        "unique_best_taus": sorted(best_tau_set, key=_sort_tau_key),
        "metric_disagreement_families": sorted(disagreement_families),
    }


def build_markdown(payload: dict[str, Any]) -> str:
    overview = pd.DataFrame(payload.get("dataset_overview", []))
    by_tau = pd.DataFrame(payload.get("by_tau", []))
    lines = [
        "# Exact-Budget Instability Report",
        "",
        "## Bottom Line",
        f"- Status: {payload.get('status', 'unknown')}",
        "- The exact-budget audit holds plain-confidence to a dense global threshold bank and reports residual budget gaps explicitly.",
        "- If fixed taus still fail here, the instability story is scientific rather than a comparator-support artifact.",
        "",
        "## Dataset Overview",
    ]
    if not overview.empty:
        lines.extend(
            overview[
                [
                    "dataset_id",
                    "required_split_family_count",
                    "n_universal_exact_budget_taus",
                    "fixed_tau_supported",
                    "n_unique_best_taus",
                    "unique_best_taus",
                    "worst_confidence_gap",
                    "lowest_confidence_exact_match_fraction",
                    "worst_safety_harmful_sig_metric_count",
                    "instability_story",
                ]
            ]
            .fillna("")
            .to_markdown(index=False)
            .splitlines()
        )
    else:
        lines.append("_No data_")
    lines.append("")
    lines.append("## Required-Scope Tau Summary")
    required = by_tau.loc[by_tau.get("scope", pd.Series(dtype=str)).astype(str) == "required"].copy()
    if not required.empty:
        lines.extend(
            required[
                [
                    "dataset_id",
                    "split_family",
                    "tau",
                    "confidence_all_primary_metrics_sig_beneficial",
                    "safety_harmful_sig_metric_count",
                    "confidence_mean_abs_intervention_rate_gap",
                    "confidence_exact_match_fraction",
                ]
            ]
            .fillna("")
            .to_markdown(index=False)
            .splitlines()
        )
    else:
        lines.append("_No data_")
    lines.extend(
        [
            "",
            "## Interpretation",
            "- `fixed_tau_supported = True` means at least one tau clears every required split family on both primary metrics with no harmful significant safety row under the exact-budget audit.",
            "- `n_unique_best_taus > 1` or non-empty `metric_disagreement_families` indicates operating-point instability even when a universal tau exists somewhere else.",
            "- Small residual budget gaps with no universal tau mean the fixed-policy story is failing on behavior, not on comparator mismatch.",
            "",
        ]
    )
    return "\n".join(lines)


def _overall_status(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "no_data"
    if any(bool(row.get("fixed_tau_supported")) for row in rows):
        return "mixed_fixed_tau_support"
    return "fixed_tau_unstable_under_exact_budget"


def _normalize_tau_value(value: object) -> str:
    return f"{float(str(value).strip()):.2f}"


def _sort_tau_key(value: object) -> tuple[int, float]:
    try:
        return (0, float(str(value)))
    except (TypeError, ValueError):
        return (1, float("inf"))


def _directional_estimate(metric: str, estimate: float) -> float:
    if metric in LOWER_IS_BETTER_METRICS:
        return -estimate
    return estimate


def _mean_or_none(series: pd.Series) -> float | None:
    if series.empty:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _max_or_none(series: pd.Series) -> float | int | None:
    if series.empty:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.max())


def _min_or_none(series: pd.Series) -> float | int | None:
    if series.empty:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.min())


if __name__ == "__main__":
    main()

