from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myoagency.provenance import build_output_manifest, write_manifest  # noqa: E402
from myoagency.stats import (  # noqa: E402
    hierarchical_bootstrap_ci,
    holm_bonferroni,
    paired_policy_difference_test,
)


PRIMARY_METRICS = (
    "macro_f1",
    "active_macro_f1",
    "balanced_accuracy",
    "risk_coverage_auc",
    "active_risk_coverage_auc",
    "ece",
    "mean_ali",
)
TAU_GRID = ("0.02", "0.05", "0.10", "0.15", "0.20")
POLICY_CI_REPEATS = 1000
PAIRWISE_REPEATS = 2000


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate split-level inferential statistics for real benchmarks.")
    parser.add_argument("--real-root", type=Path, default=Path("results/real"))
    parser.add_argument("--metrics", action="append", default=[])
    parser.add_argument("--out-root", type=Path, default=Path("results/reports/stats"))
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    policy_ci_rows: list[dict[str, object]] = []
    primary_rows: list[dict[str, object]] = []
    secondary_rows: list[dict[str, object]] = []

    metrics_paths = resolve_metrics_paths(args.real_root, explicit_metrics=args.metrics)
    for metrics_path in metrics_paths:
        if not metrics_path.exists():
            continue
        dataset_dir = metrics_path.parent
        data = pd.read_csv(metrics_path)
        if data.empty:
            continue
        data = normalize_unit_columns(data)
        dataset_id = str(data["dataset_id"].iloc[0]) if "dataset_id" in data.columns else dataset_dir.name
        if not should_include_discovered_metrics_path(dataset_dir=dataset_dir, dataset_id=dataset_id, explicit_metrics=bool(args.metrics)):
            continue
        for split_family, split_df in data.groupby("split_family", sort=True):
            unit_cols = infer_unit_cols(split_df)
            policy_ci_rows.extend(
                compute_policy_ci_rows(dataset_id=dataset_id, split_family=str(split_family), split_df=split_df, unit_cols=unit_cols)
            )
            primary_rows.extend(
                compute_pairwise_rows(
                    dataset_id=dataset_id,
                    split_family=str(split_family),
                    split_df=split_df,
                    unit_cols=unit_cols,
                    comparison_family=primary_comparison_family(dataset_id),
                    comparisons=primary_pairs_for_dataset(dataset_id, split_df["policy"]),
                )
            )
            secondary_rows.extend(
                compute_pairwise_rows(
                    dataset_id=dataset_id,
                    split_family=str(split_family),
                    split_df=split_df,
                    unit_cols=unit_cols,
                    comparison_family="vs_user_only",
                    comparisons=vs_user_only_pairs(split_df["policy"]),
                )
            )

    policy_ci = pd.DataFrame(policy_ci_rows)
    if not policy_ci.empty:
        policy_ci = policy_ci.sort_values(["dataset_id", "split_family", "metric", "policy"]).reset_index(drop=True)
    primary = apply_holm(pd.DataFrame(primary_rows))
    secondary = apply_holm(pd.DataFrame(secondary_rows))

    out_root.mkdir(parents=True, exist_ok=True)
    policy_ci.to_csv(out_root / "policy_ci.csv", index=False)
    primary.to_csv(out_root / "primary_pairwise.csv", index=False)
    secondary.to_csv(out_root / "secondary_pairwise.csv", index=False)
    stats_provenance = build_output_manifest(
        out_root,
        output_kind="stats",
        config_paths=(Path("config/config.yaml"),),
        input_paths=tuple(metrics_paths),
        extra_metadata={
            "n_metrics_inputs": int(len(metrics_paths)),
        },
        exclude_paths=(out_root / "stats_provenance.json",),
    )
    write_manifest(stats_provenance, out_root / "stats_provenance.json")
    print(f"Wrote stats tables to {out_root}")


def resolve_metrics_paths(real_root: Path, *, explicit_metrics: list[str]) -> list[Path]:
    if explicit_metrics:
        return [Path(path) for path in explicit_metrics]
    return sorted(real_root.glob("*/metrics_by_policy_unit.csv"))


def should_include_discovered_metrics_path(*, dataset_dir: Path, dataset_id: str, explicit_metrics: bool) -> bool:
    if explicit_metrics:
        return True
    return str(dataset_dir.name) == str(dataset_id)


def normalize_unit_columns(data: pd.DataFrame) -> pd.DataFrame:
    normalized = data.copy()
    for col in ["dataset_id", "split_id", "split_family", "policy", "subject_id", "session", "day"]:
        if col not in normalized.columns:
            normalized[col] = "<missing>"
        normalized[col] = normalized[col].astype("object").where(normalized[col].notna(), "<missing>").astype(str)
    return normalized


def infer_unit_cols(data: pd.DataFrame) -> list[str]:
    candidate_cols = []
    for col in ["split_id", "subject_id", "session", "day"]:
        if col not in data.columns:
            continue
        series = data[col].astype(str)
        if set(series.unique()) == {"<missing>"}:
            continue
        candidate_cols.append(col)
    if not candidate_cols:
        raise ValueError("no valid inferential unit columns found")
    return candidate_cols


def compute_policy_ci_rows(
    dataset_id: str,
    split_family: str,
    split_df: pd.DataFrame,
    unit_cols: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for policy, policy_df in split_df.groupby("policy", sort=True):
        unit_level = policy_df.groupby(unit_cols, as_index=False)[list(PRIMARY_METRICS)].mean()
        n_units = int(unit_level.shape[0])
        for metric in PRIMARY_METRICS:
            observed, ci_low, ci_high = hierarchical_bootstrap_ci(
                unit_level,
                value_col=metric,
                cluster_cols=unit_cols,
                repeats=POLICY_CI_REPEATS,
            )
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "split_family": split_family,
                    "policy": str(policy),
                    "metric": metric,
                    "unit_cols": "|".join(unit_cols),
                    "n_units": n_units,
                    "observed": observed,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return rows


def compute_pairwise_rows(
    dataset_id: str,
    split_family: str,
    split_df: pd.DataFrame,
    unit_cols: list[str],
    comparison_family: str,
    comparisons: list[tuple[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric in PRIMARY_METRICS:
        for policy_a, policy_b in comparisons:
            estimate, ci_low, ci_high, p_value, n_units = paired_policy_difference_test(
                split_df,
                policy_a=policy_a,
                policy_b=policy_b,
                value_col=metric,
                unit_cols=unit_cols,
                repeats=PAIRWISE_REPEATS,
            )
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "split_family": split_family,
                    "comparison_family": comparison_family,
                    "metric": metric,
                    "policy_a": policy_a,
                    "policy_b": policy_b,
                    "unit_cols": "|".join(unit_cols),
                    "n_units": n_units,
                    "estimate": estimate,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": p_value,
                }
            )
    return rows


def primary_comparison_family(dataset_id: str) -> str:
    return "matched_tau"


def primary_pairs_for_dataset(dataset_id: str, policies: pd.Series) -> list[tuple[str, str]]:
    return matched_tau_pairs(policies)


def matched_tau_pairs(policies: pd.Series) -> list[tuple[str, str]]:
    policy_set = {str(policy) for policy in policies.astype(str)}
    pairs: list[tuple[str, str]] = []
    for tau in TAU_GRID:
        agency = f"agency_margin_tau_{tau}"
        set_acsa = f"set_acsa_tau_{tau}"
        if agency in policy_set and set_acsa in policy_set:
            pairs.append((agency, set_acsa))
    return pairs


def vs_user_only_pairs(policies: pd.Series) -> list[tuple[str, str]]:
    policy_set = {str(policy) for policy in policies.astype(str)}
    if "user_only" not in policy_set:
        return []
    pairs: list[tuple[str, str]] = []
    for tau in TAU_GRID:
        agency = f"agency_margin_tau_{tau}"
        if agency in policy_set:
            pairs.append((agency, "user_only"))
    return pairs


def apply_holm(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data
    annotated = data.copy()
    annotated["holm_reject_0_05"] = False
    annotated["p_value_rank"] = 0
    for (_, _, metric), group in annotated.groupby(["dataset_id", "split_family", "metric"], sort=False):
        p_values = group["p_value"].astype(float).tolist()
        rejected = holm_bonferroni(p_values)
        for rank, row_index in enumerate(group.sort_values("p_value").index, start=1):
            annotated.loc[row_index, "p_value_rank"] = rank
        for row_index, reject in zip(group.index, rejected):
            annotated.loc[row_index, "holm_reject_0_05"] = bool(reject)
    return annotated.sort_values(
        ["dataset_id", "split_family", "comparison_family", "metric", "policy_a", "policy_b"]
    ).reset_index(drop=True)


if __name__ == "__main__":
    main()

