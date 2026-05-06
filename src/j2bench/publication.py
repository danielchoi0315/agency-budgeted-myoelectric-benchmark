from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from .metrics import summarize_decisions
from .policies import (
    plain_confidence_threshold_policy,
    select_plain_confidence_threshold_for_target_intervention_rate,
    trace_decision_time_ms,
)
from .schemas import PolicyDecision, PredictionTrace
from .splits import Split
from .stats import paired_delta_test, paired_policy_difference_test


DEFAULT_PUBLICATION_TAU_GRID = (0.02, 0.05, 0.10, 0.15, 0.20)
DEFAULT_ABLATION_METRICS = (
    "active_macro_f1",
    "active_risk_coverage_auc",
    "ece",
    "mean_ali",
    "intervention_rate",
)
DEFAULT_EARLIEST_METRICS = (
    "safe_episode_rate",
    "stable_safe_episode_rate",
    "final_correct_rate",
    "median_earliest_safe_s",
    "median_earliest_stable_safe_s",
)
LOWER_IS_BETTER_PUBLICATION_METRICS = {
    "active_risk_coverage_auc",
    "ece",
    "mean_ali",
    "median_earliest_safe_s",
    "median_earliest_stable_safe_s",
}
DEFAULT_ISO_BUDGET_ABLATION_METRICS = tuple(
    metric for metric in DEFAULT_ABLATION_METRICS if metric != "intervention_rate"
)
ISO_BUDGET_PLAIN_CONF_CURVE_LABEL = "plain_conf_threshold_iso_budget_curve"
DEFAULT_EXACT_BUDGET_THRESHOLD_BANK_SIZE = 201
EXACT_BUDGET_PLAIN_CONF_POLICY_PREFIX = "plain_conf_threshold_exact_budget_tau_"
EXACT_BUDGET_COMPARISON_FAMILY = "agency_vs_plain_conf_exact_budget"


def traces_to_frame(traces: Iterable[PredictionTrace]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trace in traces:
        rows.append(
            {
                "dataset_id": trace.dataset_id,
                "split_id": trace.split_id,
                "split_family": str(trace.split_id).split("|", 1)[0],
                "subject_id": trace.subject_id,
                "session": trace.session,
                "day": trace.day,
                "label": int(trace.label),
                "timestamp_s": float(trace.timestamp_s),
                "prefix_time_s": float(trace.metadata.get("prefix_time_s", trace.timestamp_s)),
                "episode_id": str(trace.metadata.get("episode_id", "")),
                "group": str(trace.metadata.get("group", "")),
                "label_raw": trace.metadata.get("label_raw"),
                "dynamic_flag": trace.metadata.get("dynamic_flag"),
                "position_id": trace.metadata.get("position_id"),
                "object_id": trace.metadata.get("object_id"),
                "grasp_repetition": trace.metadata.get("grasp_repetition"),
                "object_repetition": trace.metadata.get("object_repetition"),
                "user_action": int(np.argmax(trace.p_user)),
                "assist_action": int(np.argmax(trace.p_assist)),
                "user_confidence": float(np.max(trace.p_user)),
                "assist_confidence": float(np.max(trace.p_assist)),
                "user_correct": bool(int(np.argmax(trace.p_user)) == int(trace.label)),
                "assist_correct": bool(int(np.argmax(trace.p_assist)) == int(trace.label)),
                "user_temperature": float(trace.user_temperature),
                "assist_temperature": float(trace.assist_temperature),
                "p_user": np.asarray(trace.p_user, dtype=float),
                "p_assist": np.asarray(trace.p_assist, dtype=float),
            }
        )
    return pd.DataFrame(rows)


def decisions_to_frame(decisions: Iterable[PolicyDecision]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for decision in decisions:
        row = {
            "dataset_id": decision.dataset_id,
            "split_id": decision.split_id,
            "split_family": str(decision.split_id).split("|", 1)[0],
            "policy": decision.policy,
            "subject_id": decision.subject_id,
            "session": decision.session,
            "day": decision.day,
            "label": int(decision.label),
            "selected_action": int(decision.selected_action),
            "user_action": int(decision.user_action),
            "assist_action": int(decision.assist_action),
            "timestamp_s": float(decision.timestamp_s),
            "selected_confidence": float(decision.selected_confidence),
            "coverage_confidence": float(decision.coverage_confidence),
            "control_source": decision.control_source,
            "action_changed": bool(decision.action_changed),
            "autonomy_used": bool(decision.autonomy_used),
            "intervene": bool(decision.intervene),
            "defer": bool(decision.defer),
            "abstain": bool(decision.abstain),
            "expected_utility": float(decision.expected_utility),
            "expected_ali": float(decision.expected_ali),
            "tau": float(decision.tau),
            "agency_margin": float(decision.agency_margin),
            "decision_time_ms": float(decision.decision_time_ms),
            "correct": bool(decision.correct),
        }
        for key, value in (decision.metadata or {}).items():
            if key in row:
                row[f"meta_{key}"] = value
            else:
                row[key] = value
        if "prefix_time_s" not in row:
            row["prefix_time_s"] = float(decision.timestamp_s)
        if "episode_id" not in row:
            row["episode_id"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_earliest_safe_by_episode(decision_frame: pd.DataFrame) -> pd.DataFrame:
    if decision_frame.empty:
        return pd.DataFrame()
    required = {"dataset_id", "split_id", "policy", "episode_id", "prefix_time_s", "selected_action", "label", "expected_ali", "tau", "autonomy_used", "correct"}
    missing = required.difference(decision_frame.columns)
    if missing:
        raise ValueError(f"decision_frame is missing required columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    group_cols = ["dataset_id", "split_id", "split_family", "policy", "subject_id", "session", "day", "episode_id"]
    for keys, episode_df in decision_frame.groupby(group_cols, dropna=False, sort=True):
        ordered = episode_df.sort_values(["prefix_time_s", "timestamp_s"]).reset_index(drop=True)
        prefix = ordered["prefix_time_s"].to_numpy(dtype=float)
        correct = ordered["correct"].astype(bool).to_numpy()
        intervene = ordered["autonomy_used"].astype(bool).to_numpy()
        within_budget = ordered["expected_ali"].to_numpy(dtype=float) <= ordered["tau"].to_numpy(dtype=float) + 1e-12
        safe = intervene & correct & within_budget
        final_action = int(ordered["selected_action"].iloc[-1])
        stable_future = _stable_future_mask(ordered["selected_action"].to_numpy(dtype=int), final_action)
        stable_safe = safe & stable_future & bool(ordered["correct"].iloc[-1])
        rows.append(
            {
                "dataset_id": keys[0],
                "split_id": keys[1],
                "split_family": keys[2],
                "policy": keys[3],
                "subject_id": keys[4],
                "session": keys[5],
                "day": keys[6],
                "episode_id": keys[7],
                "n_prefixes": int(len(ordered)),
                "earliest_intervene_s": _first_or_nan(prefix, intervene),
                "earliest_correct_s": _first_or_nan(prefix, correct),
                "earliest_safe_s": _first_or_nan(prefix, safe),
                "earliest_stable_safe_s": _first_or_nan(prefix, stable_safe),
                "safe_episode": bool(np.any(safe)),
                "stable_safe_episode": bool(np.any(stable_safe)),
                "final_correct": bool(ordered["correct"].iloc[-1]),
                "final_action": final_action,
                "final_prefix_s": float(prefix[-1]),
            }
        )
    return pd.DataFrame(rows)


def summarize_earliest_safe_by_unit(episode_frame: pd.DataFrame) -> pd.DataFrame:
    if episode_frame.empty:
        return pd.DataFrame()
    group_cols = ["dataset_id", "split_family", "split_id", "policy", "subject_id", "session", "day"]
    rows: list[dict[str, Any]] = []
    for keys, unit_df in episode_frame.groupby(group_cols, dropna=False, sort=True):
        rows.append(
            {
                "dataset_id": keys[0],
                "split_family": keys[1],
                "split_id": keys[2],
                "policy": keys[3],
                "subject_id": keys[4],
                "session": keys[5],
                "day": keys[6],
                "n_episodes": int(len(unit_df)),
                "safe_episode_rate": float(unit_df["safe_episode"].mean()),
                "stable_safe_episode_rate": float(unit_df["stable_safe_episode"].mean()),
                "final_correct_rate": float(unit_df["final_correct"].mean()),
                "median_earliest_intervene_s": _nanmedian(unit_df["earliest_intervene_s"]),
                "median_earliest_correct_s": _nanmedian(unit_df["earliest_correct_s"]),
                "median_earliest_safe_s": _nanmedian(unit_df["earliest_safe_s"]),
                "median_earliest_stable_safe_s": _nanmedian(unit_df["earliest_stable_safe_s"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_earliest_safe_aggregate(unit_frame: pd.DataFrame) -> pd.DataFrame:
    if unit_frame.empty:
        return pd.DataFrame()
    numeric_cols = [
        "safe_episode_rate",
        "stable_safe_episode_rate",
        "final_correct_rate",
        "median_earliest_intervene_s",
        "median_earliest_correct_s",
        "median_earliest_safe_s",
        "median_earliest_stable_safe_s",
    ]
    return (
        unit_frame.groupby(["dataset_id", "split_family", "policy"], as_index=False)[numeric_cols]
        .mean()
        .sort_values(["dataset_id", "split_family", "policy"])
        .reset_index(drop=True)
    )


def matched_policy_pairs(
    policies: Iterable[str],
    *,
    left_prefix: str,
    right_prefix: str,
    tau_grid: Iterable[float],
) -> list[tuple[str, str]]:
    policy_set = {str(policy) for policy in policies}
    pairs: list[tuple[str, str]] = []
    for tau in tau_grid:
        tau_token = f"{float(tau):.2f}"
        left = f"{left_prefix}{tau_token}"
        right = f"{right_prefix}{tau_token}"
        if left in policy_set and right in policy_set:
            pairs.append((left, right))
    return pairs


def build_dense_plain_confidence_threshold_bank(
    traces: Iterable[PredictionTrace],
    *,
    max_thresholds: int = DEFAULT_EXACT_BUDGET_THRESHOLD_BANK_SIZE,
) -> np.ndarray:
    trace_rows = list(traces)
    if not trace_rows:
        return np.asarray([0.0, 1.01], dtype=float)
    assist_conf = np.asarray([float(np.max(trace.p_assist)) for trace in trace_rows], dtype=float)
    unique_conf = np.unique(assist_conf)
    if unique_conf.size <= int(max_thresholds):
        bank = unique_conf
    else:
        quantiles = np.linspace(0.0, 1.0, int(max_thresholds), dtype=float)
        bank = np.quantile(assist_conf, quantiles)
    return np.unique(np.concatenate([np.asarray(bank, dtype=float), np.array([0.0, 1.01], dtype=float)]))


def infer_unit_cols(data: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in ["split_id", "subject_id", "session", "day"]:
        if col not in data.columns:
            continue
        values = data[col].astype("object").where(data[col].notna(), "<missing>").astype(str)
        if set(values.unique()) == {"<missing>"}:
            continue
        cols.append(col)
    if not cols:
        raise ValueError("no valid inferential unit columns found")
    return cols


def normalize_unit_columns(data: pd.DataFrame) -> pd.DataFrame:
    normalized = data.copy()
    for col in ["split_id", "subject_id", "session", "day"]:
        if col not in normalized.columns:
            normalized[col] = "<missing>"
        normalized[col] = normalized[col].astype("object").where(normalized[col].notna(), "<missing>").astype(str)
    return normalized


def merge_budget_into_unit_frame(
    unit_frame: pd.DataFrame,
    *,
    budget_frame: pd.DataFrame,
) -> pd.DataFrame:
    if unit_frame.empty:
        return unit_frame.copy()
    normalized_unit = normalize_unit_columns(unit_frame)
    normalized_budget = normalize_unit_columns(budget_frame)
    join_cols = ["dataset_id", "split_family", "split_id", "policy", "subject_id", "session", "day"]
    missing_unit = sorted(set(join_cols).difference(normalized_unit.columns))
    if missing_unit:
        raise ValueError(f"unit_frame is missing required columns: {missing_unit}")
    required_budget = set(join_cols).union({"intervention_rate"})
    missing_budget = sorted(required_budget.difference(normalized_budget.columns))
    if missing_budget:
        raise ValueError(f"budget_frame is missing required columns: {missing_budget}")
    merged = normalized_unit.merge(
        normalized_budget.loc[:, join_cols + ["intervention_rate"]].copy(),
        on=join_cols,
        how="left",
        validate="one_to_one",
    )
    missing_rows = int(merged["intervention_rate"].isna().sum())
    if missing_rows:
        raise ValueError(
            "budget_frame is missing intervention_rate values for "
            f"{missing_rows} unit/policy rows"
        )
    return merged


def interpolate_metric_curve(
    curve_df: pd.DataFrame,
    *,
    target_budget: float,
    metric: str,
    budget_col: str = "intervention_rate",
    allow_extrapolation: bool = True,
    min_points: int = 2,
) -> dict[str, Any]:
    if budget_col not in curve_df.columns:
        raise ValueError(f"curve_df is missing required budget column: {budget_col}")
    if metric not in curve_df.columns:
        raise ValueError(f"curve_df is missing required metric column: {metric}")
    metric_df = curve_df.loc[curve_df[metric].notna(), [budget_col, metric]].copy()
    if metric_df.empty:
        return {
            "value": float("nan"),
            "nearest_gap": float("nan"),
            "extrapolated": False,
            "supported": False,
            "n_points": 0,
        }
    metric_df = (
        metric_df.groupby(budget_col, as_index=False)[metric]
        .mean()
        .sort_values(budget_col)
        .reset_index(drop=True)
    )
    xs = metric_df[budget_col].to_numpy(dtype=float)
    ys = metric_df[metric].to_numpy(dtype=float)
    budget = float(target_budget)
    nearest_gap = float(np.min(np.abs(xs - budget)))
    extrapolated = bool(budget < float(xs.min()) or budget > float(xs.max()))
    exact_match = bool(nearest_gap <= 1e-12)
    supported = bool(exact_match or xs.size >= int(min_points)) and (
        allow_extrapolation or not extrapolated
    )
    if not supported:
        return {
            "value": float("nan"),
            "nearest_gap": nearest_gap,
            "extrapolated": extrapolated,
            "supported": False,
            "n_points": int(xs.size),
        }
    if xs.size == 1:
        value = float(ys[0])
    else:
        value = float(np.interp(budget, xs, ys))
    return {
        "value": value,
        "nearest_gap": nearest_gap,
        "extrapolated": extrapolated,
        "supported": True,
        "n_points": int(xs.size),
    }


def compute_iso_budget_pairwise_stats(
    frame: pd.DataFrame,
    *,
    policy_pairs: list[tuple[str, str]],
    curve_policy_prefix: str,
    metrics: tuple[str, ...],
    repeats: int,
    comparison_family: str = "agency_vs_plain_conf_iso_budget",
    curve_policy_label: str = ISO_BUDGET_PLAIN_CONF_CURVE_LABEL,
    lower_is_better: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty or not policy_pairs:
        return _empty_iso_budget_unit_delta_frame(), _empty_iso_budget_pairwise_frame()

    required = {
        "dataset_id",
        "split_family",
        "policy",
        "intervention_rate",
        *metrics,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"frame is missing required columns: {missing}")

    lower_is_better = set(lower_is_better or set())
    normalized = normalize_unit_columns(frame)
    unit_cols = infer_unit_cols(normalized)
    numeric_cols = ["intervention_rate", *metrics]
    unit_rows: list[dict[str, object]] = []

    for split_family, split_df in normalized.groupby("split_family", sort=True):
        for unit_keys, unit_df in split_df.groupby(unit_cols, dropna=False, sort=True):
            keys = unit_keys if isinstance(unit_keys, tuple) else (unit_keys,)
            dataset_id = str(unit_df["dataset_id"].iloc[0])
            policy_view = (
                unit_df.groupby("policy", as_index=False)[numeric_cols]
                .mean()
                .sort_values("policy")
                .reset_index(drop=True)
            )
            curve_df = policy_view.loc[
                policy_view["policy"].astype(str).str.startswith(curve_policy_prefix)
            ].copy()
            if curve_df.empty:
                continue
            key_payload = dict(zip(unit_cols, (str(value) for value in keys)))
            for policy_a, policy_b_same_tau in policy_pairs:
                agency_df = policy_view.loc[policy_view["policy"].astype(str) == policy_a].copy()
                if agency_df.empty:
                    continue
                same_tau_df = policy_view.loc[
                    policy_view["policy"].astype(str) == policy_b_same_tau
                ].copy()
                agency_row = agency_df.iloc[0]
                same_tau_row = same_tau_df.iloc[0] if not same_tau_df.empty else None
                agency_budget = float(agency_row["intervention_rate"])
                same_tau_budget = (
                    float(same_tau_row["intervention_rate"]) if same_tau_row is not None else float("nan")
                )
                raw_gap = (
                    float(agency_budget - same_tau_budget)
                    if not np.isnan(same_tau_budget)
                    else float("nan")
                )
                tau = _extract_tau_token(policy_a)
                for metric in metrics:
                    interp = interpolate_metric_curve(
                        curve_df,
                        target_budget=agency_budget,
                        metric=metric,
                        allow_extrapolation=False,
                        min_points=2,
                    )
                    agency_value = _safe_float(agency_row.get(metric))
                    same_tau_value = _safe_float(
                        same_tau_row.get(metric) if same_tau_row is not None else float("nan")
                    )
                    iso_budget_value = _safe_float(interp["value"])
                    unit_rows.append(
                        {
                            "dataset_id": dataset_id,
                            "split_family": str(split_family),
                            "comparison_family": comparison_family,
                            "tau": tau,
                            "policy_a": str(policy_a),
                            "policy_b_same_tau": str(policy_b_same_tau),
                            "policy_b": curve_policy_label,
                            "metric": metric,
                            **key_payload,
                            "agency_intervention_rate": agency_budget,
                            "plain_conf_same_tau_intervention_rate": same_tau_budget,
                            "raw_intervention_rate_gap": raw_gap,
                            "nearest_curve_budget_gap": float(interp["nearest_gap"]),
                            "curve_budget_extrapolated": bool(interp["extrapolated"]),
                            "curve_supported": bool(interp["supported"]),
                            "curve_metric_point_count": int(interp["n_points"]),
                            "agency_value": agency_value,
                            "plain_conf_same_tau_value": same_tau_value,
                            "plain_conf_iso_budget_value": iso_budget_value,
                            "delta_same_tau": _delta_or_nan(agency_value, same_tau_value),
                            "delta_iso_budget": _delta_or_nan(agency_value, iso_budget_value),
                            "directional_delta_same_tau": _directional_delta(
                                metric,
                                agency_value,
                                same_tau_value,
                            ),
                            "directional_delta_iso_budget": _directional_delta(
                                metric,
                                agency_value,
                                iso_budget_value,
                            ),
                        }
                    )

    unit_delta_frame = pd.DataFrame(unit_rows)
    if unit_delta_frame.empty:
        return _empty_iso_budget_unit_delta_frame(), _empty_iso_budget_pairwise_frame()

    pairwise_rows: list[dict[str, object]] = []
    group_cols = [
        "split_family",
        "comparison_family",
        "metric",
        "tau",
        "policy_a",
        "policy_b",
        "policy_b_same_tau",
    ]
    for keys, group_df in unit_delta_frame.groupby(group_cols, dropna=False, sort=True):
        available = group_df.loc[group_df["delta_iso_budget"].notna()].copy()
        if available.empty:
            continue
        estimate, ci_low, ci_high, p_value, n_units = paired_delta_test(
            available["delta_iso_budget"].to_numpy(dtype=float),
            repeats=repeats,
        )
        metric = str(keys[2])
        beneficial = estimate < 0.0 if metric in lower_is_better else estimate > 0.0
        pairwise_rows.append(
            {
                "split_family": str(keys[0]),
                "comparison_family": str(keys[1]),
                "metric": metric,
                "tau": None if pd.isna(keys[3]) else str(keys[3]),
                "policy_a": str(keys[4]),
                "policy_b": str(keys[5]),
                "policy_b_same_tau": str(keys[6]),
                "unit_cols": "|".join(unit_cols),
                "n_units": int(n_units),
                "n_units_total": int(group_df.shape[0]),
                "n_units_missing_metric": int(group_df.shape[0] - available.shape[0]),
                "n_extrapolated_units": int(
                    group_df["curve_budget_extrapolated"].astype(bool).sum()
                ),
                "n_supported_units": int(available["curve_supported"].astype(bool).sum()),
                "extrapolated_fraction": float(
                    group_df["curve_budget_extrapolated"].astype(bool).mean()
                ),
                "mean_abs_raw_intervention_rate_gap": float(
                    group_df["raw_intervention_rate_gap"].abs().mean()
                ),
                "max_abs_raw_intervention_rate_gap": float(
                    group_df["raw_intervention_rate_gap"].abs().max()
                ),
                "mean_nearest_curve_budget_gap": float(
                    group_df["nearest_curve_budget_gap"].mean()
                ),
                "max_nearest_curve_budget_gap": float(
                    group_df["nearest_curve_budget_gap"].max()
                ),
                "mean_curve_metric_point_count": float(
                    group_df["curve_metric_point_count"].mean()
                ),
                "n_singleton_curve_units": int(
                    (group_df["curve_metric_point_count"].astype(int) == 1).sum()
                ),
                "estimate": estimate,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": p_value,
                "beneficial": bool(beneficial),
            }
        )

    pairwise_frame = pd.DataFrame(pairwise_rows)
    unit_sort_cols = [
        column
        for column in ["split_family", "metric", "tau", "split_id", "subject_id", "session", "day"]
        if column in unit_delta_frame.columns
    ]
    if pairwise_frame.empty:
        return (
            unit_delta_frame.sort_values(unit_sort_cols).reset_index(drop=True),
            _empty_iso_budget_pairwise_frame(),
        )
    return (
        unit_delta_frame.sort_values(unit_sort_cols).reset_index(drop=True),
        pairwise_frame.sort_values(
            ["split_family", "metric", "tau", "policy_a"]
        ).reset_index(drop=True),
    )


def materialize_exact_budget_plain_confidence_units(
    *,
    traces: Iterable[PredictionTrace],
    budget_frame: pd.DataFrame,
    policy_pairs: list[tuple[str, str]],
    threshold_bank: np.ndarray | None = None,
    threshold_bank_size: int = DEFAULT_EXACT_BUDGET_THRESHOLD_BANK_SIZE,
    comparator_policy_prefix: str = EXACT_BUDGET_PLAIN_CONF_POLICY_PREFIX,
    comparison_family: str = EXACT_BUDGET_COMPARISON_FAMILY,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {
        "dataset_id",
        "split_family",
        "split_id",
        "policy",
        "subject_id",
        "session",
        "day",
        "intervention_rate",
    }
    missing = sorted(required.difference(budget_frame.columns))
    if missing:
        raise ValueError(f"budget_frame is missing required columns: {missing}")

    trace_rows = list(traces)
    if not trace_rows or budget_frame.empty or not policy_pairs:
        return (
            _empty_exact_budget_selection_frame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    threshold_values = (
        np.asarray(threshold_bank, dtype=float)
        if threshold_bank is not None
        else build_dense_plain_confidence_threshold_bank(
            trace_rows,
            max_thresholds=threshold_bank_size,
        )
    )
    normalized_budget = normalize_unit_columns(budget_frame)
    unit_cols = infer_unit_cols(normalized_budget)
    policy_specs: list[dict[str, str]] = []
    policy_names = set()
    for policy_a, policy_b_same_tau in policy_pairs:
        tau = _extract_tau_token(policy_a)
        if tau is None:
            continue
        policy_b = f"{comparator_policy_prefix}{tau}"
        policy_specs.append(
            {
                "policy_a": str(policy_a),
                "policy_b": policy_b,
                "policy_b_same_tau": str(policy_b_same_tau),
                "tau": tau,
            }
        )
        policy_names.add(str(policy_a))
    if not policy_specs:
        return (
            _empty_exact_budget_selection_frame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    trace_groups: dict[tuple[str, ...], list[PredictionTrace]] = {}
    for trace in trace_rows:
        split_family = str(trace.split_id).split("|", 1)[0]
        key = (split_family, *(_trace_unit_key(trace, unit_cols)))
        trace_groups.setdefault(key, []).append(trace)

    budget_subset = normalized_budget.loc[
        normalized_budget["policy"].astype(str).isin(policy_names)
    ].copy()
    selection_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    earliest_rows: list[dict[str, object]] = []
    group_cols = ["split_family", *unit_cols]
    for keys, unit_budget_df in budget_subset.groupby(group_cols, dropna=False, sort=True):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        split_family = str(key_tuple[0])
        unit_key = tuple(str(value) for value in key_tuple[1:])
        unit_traces = trace_groups.get((split_family, *unit_key))
        if not unit_traces:
            continue
        unit_meta = dict(zip(unit_cols, unit_key))
        dataset_id = str(unit_budget_df["dataset_id"].iloc[0])
        for spec in policy_specs:
            agency_rows = unit_budget_df.loc[
                unit_budget_df["policy"].astype(str) == spec["policy_a"]
            ]
            if agency_rows.empty:
                continue
            agency_row = agency_rows.iloc[0]
            target_rate = _safe_float(agency_row.get("intervention_rate"))
            selection = select_plain_confidence_threshold_for_target_intervention_rate(
                unit_traces,
                target_rate=target_rate,
                candidate_thresholds=threshold_values,
            )
            selected_threshold = float(selection["threshold"])
            tau_value = float(spec["tau"])
            unit_decisions = [
                plain_confidence_threshold_policy(
                    trace,
                    threshold=selected_threshold,
                    tau=tau_value,
                    decision_time_ms=trace_decision_time_ms(trace),
                    policy_name=spec["policy_b"],
                )
                for trace in unit_traces
            ]
            metric_record = asdict(
                summarize_decisions(
                    unit_decisions,
                    dataset_id=dataset_id,
                    split_id=str(unit_meta.get("split_id", agency_row["split_id"])),
                    policy=spec["policy_b"],
                    subject_id=str(unit_meta.get("subject_id", agency_row["subject_id"])),
                    session=str(unit_meta.get("session", agency_row["session"])),
                    day=str(unit_meta.get("day", agency_row["day"])),
                )
            )
            metric_record["split_family"] = split_family
            metric_rows.append(metric_record)

            unit_decision_frame = decisions_to_frame(unit_decisions)
            episode_frame = summarize_earliest_safe_by_episode(unit_decision_frame)
            unit_earliest = summarize_earliest_safe_by_unit(episode_frame)
            if not unit_earliest.empty:
                earliest_rows.append(unit_earliest.iloc[0].to_dict())

            selection_rows.append(
                {
                    "dataset_id": dataset_id,
                    "split_family": split_family,
                    "comparison_family": comparison_family,
                    "tau": spec["tau"],
                    "policy_a": spec["policy_a"],
                    "policy_b_same_tau": spec["policy_b_same_tau"],
                    "policy_b": spec["policy_b"],
                    **unit_meta,
                    "selected_threshold": selected_threshold,
                    "target_intervention_rate": target_rate,
                    "selected_intervention_rate": float(selection["matched_intervention_rate"]),
                    "intervention_rate_gap": float(selection["intervention_rate_gap"]),
                    "exact_budget_match": bool(float(selection["intervention_rate_gap"]) <= 1e-12),
                    "candidate_threshold_count": int(selection["candidate_threshold_count"]),
                    "trace_count": int(selection["trace_count"]),
                    "disagreement_trace_count": int(selection["disagreement_trace_count"]),
                    "max_attainable_intervention_rate": float(
                        selection["max_attainable_intervention_rate"]
                    ),
                }
            )

    selection_frame = pd.DataFrame(selection_rows)
    metrics_frame = pd.DataFrame(metric_rows)
    earliest_frame = pd.DataFrame(earliest_rows)
    if not selection_frame.empty:
        selection_frame = selection_frame.sort_values(
            [
                column
                for column in [
                    "split_family",
                    "tau",
                    "split_id",
                    "subject_id",
                    "session",
                    "day",
                ]
                if column in selection_frame.columns
            ]
        ).reset_index(drop=True)
    if not metrics_frame.empty:
        metrics_frame = metrics_frame.sort_values(
            [
                column
                for column in [
                    "split_family",
                    "policy",
                    "split_id",
                    "subject_id",
                    "session",
                    "day",
                ]
                if column in metrics_frame.columns
            ]
        ).reset_index(drop=True)
    if not earliest_frame.empty:
        earliest_frame = earliest_frame.sort_values(
            [
                column
                for column in [
                    "split_family",
                    "policy",
                    "split_id",
                    "subject_id",
                    "session",
                    "day",
                ]
                if column in earliest_frame.columns
            ]
        ).reset_index(drop=True)
    return selection_frame, metrics_frame, earliest_frame


def annotate_exact_budget_pairwise_diagnostics(
    pairwise_frame: pd.DataFrame,
    *,
    selection_frame: pd.DataFrame,
) -> pd.DataFrame:
    annotated = pairwise_frame.copy()
    annotated["tau"] = annotated["policy_a"].map(_extract_tau_token)
    if annotated.empty:
        for column in [
            "policy_b_same_tau",
            "n_audit_units",
            "mean_abs_intervention_rate_gap",
            "max_abs_intervention_rate_gap",
            "exact_match_fraction",
            "n_exact_match_units",
            "mean_selected_threshold",
            "min_selected_threshold",
            "max_selected_threshold",
            "mean_candidate_threshold_count",
            "mean_max_attainable_intervention_rate",
        ]:
            annotated[column] = pd.Series(dtype=float if column != "policy_b_same_tau" else object)
        return annotated
    if selection_frame.empty:
        annotated["policy_b_same_tau"] = ""
        annotated["n_audit_units"] = 0
        for column in [
            "mean_abs_intervention_rate_gap",
            "max_abs_intervention_rate_gap",
            "exact_match_fraction",
            "n_exact_match_units",
            "mean_selected_threshold",
            "min_selected_threshold",
            "max_selected_threshold",
            "mean_candidate_threshold_count",
            "mean_max_attainable_intervention_rate",
        ]:
            annotated[column] = float("nan")
        return annotated

    diagnostics = (
        selection_frame.groupby(
            ["split_family", "tau", "policy_a", "policy_b"],
            as_index=False,
        )
        .agg(
            policy_b_same_tau=("policy_b_same_tau", "first"),
            n_audit_units=("policy_b", "count"),
            mean_abs_intervention_rate_gap=("intervention_rate_gap", lambda values: float(np.mean(np.abs(values)))),
            max_abs_intervention_rate_gap=("intervention_rate_gap", lambda values: float(np.max(np.abs(values)))),
            exact_match_fraction=("exact_budget_match", "mean"),
            n_exact_match_units=("exact_budget_match", "sum"),
            mean_selected_threshold=("selected_threshold", "mean"),
            min_selected_threshold=("selected_threshold", "min"),
            max_selected_threshold=("selected_threshold", "max"),
            mean_candidate_threshold_count=("candidate_threshold_count", "mean"),
            mean_max_attainable_intervention_rate=("max_attainable_intervention_rate", "mean"),
        )
    )
    return annotated.merge(
        diagnostics,
        on=["split_family", "tau", "policy_a", "policy_b"],
        how="left",
        validate="many_to_one",
    )


def compute_pairwise_stats(
    frame: pd.DataFrame,
    *,
    pair_builders: list[tuple[str, list[tuple[str, str]]]],
    metrics: tuple[str, ...],
    repeats: int,
    lower_is_better: set[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if frame.empty:
        return pd.DataFrame()
    lower_is_better = set(lower_is_better or set())
    for split_family, split_df in frame.groupby("split_family", sort=True):
        normalized = normalize_unit_columns(split_df)
        unit_cols = infer_unit_cols(normalized)
        for comparison_family, pairs in pair_builders:
            for metric in metrics:
                for policy_a, policy_b in pairs:
                    try:
                        estimate, ci_low, ci_high, p_value, n_units = paired_policy_difference_test(
                            normalized,
                            policy_a=policy_a,
                            policy_b=policy_b,
                            value_col=metric,
                            unit_cols=unit_cols,
                            repeats=repeats,
                        )
                    except ValueError:
                        continue
                    beneficial = estimate < 0 if metric in lower_is_better else estimate > 0
                    rows.append(
                        {
                            "split_family": str(split_family),
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
                            "beneficial": bool(beneficial),
                        }
                    )
    return pd.DataFrame(rows)


def build_wang_db10_splits(metadata: pd.DataFrame) -> list[Split]:
    subset = metadata.loc[
        (metadata["group"].astype(str) == "able_bodied")
        & (metadata["label_raw"].astype(int) > 0)
        & (metadata["dynamic_flag"].astype(int) == 0)
        & (metadata["position_id"].astype(int) == 1)
        & (metadata["object_repetition"].astype(int).isin([5, 6, 7, 8]))
    ].copy()
    repetitions = sorted(subset["object_repetition"].astype(int).unique().tolist())
    splits: list[Split] = []
    for repetition in repetitions:
        train_idx = subset.index[subset["object_repetition"].astype(int) != repetition].to_list()
        test_idx = subset.index[subset["object_repetition"].astype(int) == repetition].to_list()
        if train_idx and test_idx:
            splits.append(
                Split(
                    split_id=f"db10_anchor_wang_loro|rep{int(repetition)}",
                    train_index=train_idx,
                    test_index=test_idx,
                    train_groups={"object_repetition": set(subset.loc[train_idx, "object_repetition"].astype(str))},
                    test_groups={"object_repetition": {str(repetition)}},
                )
            )
    return splits


def build_cognolato_db10_splits(metadata: pd.DataFrame) -> list[Split]:
    episode_meta = metadata.loc[metadata["label_raw"].astype(int) > 0].copy()
    episode_meta = episode_meta.drop_duplicates(subset=["episode_id"]).copy()
    episode_meta["fold_id"] = -1
    for subject_id in sorted(episode_meta["subject_id"].astype(str).unique()):
        subject_mask = episode_meta["subject_id"].astype(str) == subject_id
        subject_df = episode_meta.loc[subject_mask].copy()
        fold_map = _assign_cognolato_subject_folds(subject_df)
        if not fold_map:
            continue
        episode_meta.loc[subject_mask, "fold_id"] = episode_meta.loc[subject_mask, "episode_id"].map(fold_map).fillna(-1).astype(int)
    assigned = episode_meta.loc[episode_meta["fold_id"] >= 0].copy()
    if assigned.empty:
        return []
    fold_lookup = dict(zip(assigned["episode_id"].astype(str), assigned["fold_id"].astype(int)))
    row_folds = metadata["episode_id"].astype(str).map(fold_lookup).fillna(-1).astype(int)
    splits: list[Split] = []
    for subject_id in sorted(assigned["subject_id"].astype(str).unique()):
        subject_assigned = assigned.loc[assigned["subject_id"].astype(str) == subject_id]
        group_name = str(subject_assigned["group"].iloc[0])
        for fold_id in range(4):
            subject_rows = metadata["subject_id"].astype(str) == subject_id
            train_idx = metadata.index[subject_rows & (row_folds != fold_id) & (row_folds >= 0)].to_list()
            test_idx = metadata.index[subject_rows & (row_folds == fold_id)].to_list()
            if train_idx and test_idx:
                splits.append(
                    Split(
                        split_id=f"db10_anchor_cognolato|{group_name}|{subject_id}|fold{fold_id + 1}",
                        train_index=train_idx,
                        test_index=test_idx,
                        train_groups={"subject_id": {subject_id}, "fold_id": set(str(value) for value in range(4) if value != fold_id)},
                        test_groups={"subject_id": {subject_id}, "fold_id": {str(fold_id)}},
                    )
                )
    return splits


def compute_wang_anchor_metrics(trace_frame: pd.DataFrame) -> dict[str, Any]:
    subset = trace_frame.loc[trace_frame["split_family"] == "db10_anchor_wang_loro"].copy()
    if subset.empty:
        return {"n_trials": 0}
    episode_rows: list[dict[str, Any]] = []
    for (_, episode_id), episode_df in subset.groupby(["split_id", "episode_id"], sort=True):
        ordered = episode_df.sort_values("prefix_time_s").reset_index(drop=True)
        early_idx = int(np.argmin(np.abs(ordered["prefix_time_s"].to_numpy(dtype=float) - 0.2)))
        early = ordered.iloc[early_idx]
        final = ordered.iloc[-1]
        fused = np.asarray(early["p_assist"], dtype=float) * np.asarray(final["p_user"], dtype=float)
        fused_sum = float(fused.sum())
        if fused_sum <= 0:
            fused = np.asarray(final["p_user"], dtype=float)
            fused_sum = float(fused.sum())
        fused /= fused_sum
        fused_action = int(np.argmax(fused))
        episode_rows.append(
            {
                "split_id": str(final["split_id"]),
                "episode_id": str(episode_id),
                "label": int(final["label"]),
                "emg_final_correct": bool(final["user_action"] == final["label"]),
                "integrated_correct": bool(fused_action == int(final["label"])),
                "assist_early_correct": bool(early["assist_action"] == early["label"]),
            }
        )
    episode_df = pd.DataFrame(episode_rows)
    if episode_df.empty:
        return {"n_trials": 0}
    split_df = (
        episode_df.groupby("split_id", as_index=False)
        .agg(
            emg_accuracy=("emg_final_correct", "mean"),
            integrated_accuracy=("integrated_correct", "mean"),
            assist_early_accuracy=("assist_early_correct", "mean"),
            n_trials=("episode_id", "count"),
        )
    )
    return {
        "n_trials": int(episode_df.shape[0]),
        "n_splits": int(split_df.shape[0]),
        "emg_accuracy_percent": float(split_df["emg_accuracy"].mean() * 100.0),
        "integrated_accuracy_percent": float(split_df["integrated_accuracy"].mean() * 100.0),
        "assist_early_accuracy_percent": float(split_df["assist_early_accuracy"].mean() * 100.0),
        "gain_percentage_points": float((split_df["integrated_accuracy"] - split_df["emg_accuracy"]).mean() * 100.0),
    }


def compute_cognolato_anchor_metrics(trace_frame: pd.DataFrame) -> dict[str, Any]:
    subset = trace_frame.loc[trace_frame["split_family"] == "db10_anchor_cognolato"].copy()
    if subset.empty:
        return {"groups": {}}
    final_rows = (
        subset.sort_values(["split_id", "episode_id", "prefix_time_s"])
        .groupby(["split_id", "episode_id"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    per_split = (
        final_rows.groupby(["split_id", "group", "subject_id"], as_index=False)
        .agg(
            user_accuracy=("user_correct", "mean"),
            assist_accuracy=("assist_correct", "mean"),
        )
    )
    per_split["gain_pp"] = (per_split["assist_accuracy"] - per_split["user_accuracy"]) * 100.0
    summary: dict[str, Any] = {"groups": {}, "n_splits": int(per_split["split_id"].nunique())}
    for group_name, group_df in per_split.groupby("group", sort=True):
        subject_df = (
            group_df.groupby("subject_id", as_index=False)
            .agg(
                user_accuracy=("user_accuracy", "mean"),
                assist_accuracy=("assist_accuracy", "mean"),
                gain_pp=("gain_pp", "mean"),
            )
        )
        summary["groups"][str(group_name)] = {
            "n_subjects": int(subject_df.shape[0]),
            "user_accuracy_percent": float(subject_df["user_accuracy"].mean() * 100.0),
            "assist_accuracy_percent": float(subject_df["assist_accuracy"].mean() * 100.0),
            "gain_percentage_points": float(subject_df["gain_pp"].mean()),
            "gain_std_percentage_points": float(subject_df["gain_pp"].std(ddof=0)),
        }
    return summary


def _empty_iso_budget_unit_delta_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "dataset_id",
            "split_family",
            "comparison_family",
            "tau",
            "policy_a",
            "policy_b_same_tau",
            "policy_b",
            "metric",
            "split_id",
            "subject_id",
            "session",
            "day",
            "agency_intervention_rate",
            "plain_conf_same_tau_intervention_rate",
            "raw_intervention_rate_gap",
            "nearest_curve_budget_gap",
            "curve_budget_extrapolated",
            "curve_supported",
            "curve_metric_point_count",
            "agency_value",
            "plain_conf_same_tau_value",
            "plain_conf_iso_budget_value",
            "delta_same_tau",
            "delta_iso_budget",
            "directional_delta_same_tau",
            "directional_delta_iso_budget",
        ]
    )


def _empty_exact_budget_selection_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "dataset_id",
            "split_family",
            "comparison_family",
            "tau",
            "policy_a",
            "policy_b_same_tau",
            "policy_b",
            "split_id",
            "subject_id",
            "session",
            "day",
            "selected_threshold",
            "target_intervention_rate",
            "selected_intervention_rate",
            "intervention_rate_gap",
            "exact_budget_match",
            "candidate_threshold_count",
            "trace_count",
            "disagreement_trace_count",
            "max_attainable_intervention_rate",
        ]
    )


def _empty_iso_budget_pairwise_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "split_family",
            "comparison_family",
            "metric",
            "tau",
            "policy_a",
            "policy_b",
            "policy_b_same_tau",
            "unit_cols",
            "n_units",
            "n_units_total",
            "n_units_missing_metric",
            "n_extrapolated_units",
            "n_supported_units",
            "extrapolated_fraction",
            "mean_abs_raw_intervention_rate_gap",
            "max_abs_raw_intervention_rate_gap",
            "mean_nearest_curve_budget_gap",
            "max_nearest_curve_budget_gap",
            "mean_curve_metric_point_count",
            "n_singleton_curve_units",
            "estimate",
            "ci_low",
            "ci_high",
            "p_value",
            "beneficial",
        ]
    )


def _extract_tau_token(policy_name: object) -> str | None:
    token = str(policy_name)
    marker = "tau_"
    if marker not in token:
        return None
    try:
        return f"{float(token.rsplit(marker, 1)[1]):.2f}"
    except ValueError:
        return None


def _delta_or_nan(left: float, right: float) -> float:
    if np.isnan(left) or np.isnan(right):
        return float("nan")
    return float(left - right)


def _directional_delta(metric: str, left: float, right: float) -> float:
    if np.isnan(left) or np.isnan(right):
        return float("nan")
    if metric in LOWER_IS_BETTER_PUBLICATION_METRICS:
        return float(right - left)
    return float(left - right)


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _trace_unit_key(trace: PredictionTrace, unit_cols: list[str]) -> tuple[str, ...]:
    payload = {
        "split_id": trace.split_id,
        "subject_id": trace.subject_id,
        "session": trace.session,
        "day": trace.day,
    }
    return tuple(_normalize_key_component(payload.get(column)) for column in unit_cols)


def _normalize_key_component(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "<missing>"
    return str(value)


def _assign_cognolato_subject_folds(subject_df: pd.DataFrame) -> dict[str, int]:
    quotas = {
        (0, 0): 3,
        (0, 1): 3,
    }
    fold_map: dict[str, int] = {}
    for label in sorted(subject_df["label_raw"].astype(int).unique().tolist()):
        label_df = subject_df.loc[subject_df["label_raw"].astype(int) == label]
        for block_key, block_df in label_df.groupby(["dynamic_flag", "position_id"], sort=True):
            block = block_df.sort_values(["object_repetition", "grasp_repetition", "object_id", "episode_id"]).reset_index(drop=True)
            dynamic_flag = int(block_key[0])
            quota = 2 if dynamic_flag == 1 else quotas.get((int(block_key[0]), int(block_key[1])), 0)
            if quota <= 0:
                continue
            usable = min(int(block.shape[0]), quota * 4)
            for idx, episode_id in enumerate(block["episode_id"].astype(str).tolist()[:usable]):
                fold_map[episode_id] = idx % 4
    return fold_map


def _first_or_nan(values: np.ndarray, mask: np.ndarray) -> float:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return float("nan")
    return float(values[int(idx[0])])


def _nanmedian(series: pd.Series) -> float:
    arr = series.to_numpy(dtype=float)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmedian(arr))


def _stable_future_mask(actions: np.ndarray, final_action: int) -> np.ndarray:
    stable = np.zeros(actions.shape[0], dtype=bool)
    all_match = True
    for index in range(actions.shape[0] - 1, -1, -1):
        all_match = all_match and int(actions[index]) == int(final_action)
        stable[index] = all_match
    return stable

