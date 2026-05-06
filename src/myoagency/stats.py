from __future__ import annotations

import numpy as np
import pandas as pd


def hierarchical_bootstrap_ci(
    data: pd.DataFrame,
    value_col: str,
    cluster_cols: list[str],
    repeats: int = 2000,
    ci: float = 0.95,
    seed: int = 20260415,
) -> tuple[float, float, float]:
    """Cluster bootstrap mean with nested resampling over provided cluster columns."""
    if data.empty:
        raise ValueError("cannot bootstrap an empty dataframe")
    data = data.copy()
    usable_cluster_cols = [col for col in cluster_cols if col in data.columns]
    for col in usable_cluster_cols:
        data[col] = data[col].astype("object").where(data[col].notna(), "<missing>")
    if not cluster_cols:
        values = data[value_col].to_numpy(dtype=float)
        return _simple_bootstrap_ci(values, repeats, ci, seed)
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    observed = float(data[value_col].mean())
    top_col = usable_cluster_cols[0] if usable_cluster_cols else None
    if top_col is None:
        values = data[value_col].to_numpy(dtype=float)
        return _simple_bootstrap_ci(values, repeats, ci, seed)
    top_groups = data[top_col].unique()
    for _ in range(repeats):
        sampled_groups = rng.choice(top_groups, size=len(top_groups), replace=True)
        sampled = _nested_resample(data, value_col, usable_cluster_cols, sampled_groups, rng)
        estimates.append(float(sampled[value_col].mean()))
    alpha = (1.0 - ci) / 2.0
    return (
        observed,
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1.0 - alpha)),
    )


def _nested_resample(
    data: pd.DataFrame,
    value_col: str,
    cluster_cols: list[str],
    sampled_groups: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    top_col = cluster_cols[0]
    pieces = []
    for group in sampled_groups:
        group_frame = data[data[top_col] == group]
        if len(cluster_cols) == 1:
            pieces.append(group_frame)
        else:
            next_col = cluster_cols[1]
            next_groups = group_frame[next_col].unique()
            nested_groups = rng.choice(next_groups, size=len(next_groups), replace=True)
            pieces.append(_nested_resample(group_frame, value_col, cluster_cols[1:], nested_groups, rng))
    return pd.concat(pieces, ignore_index=True)


def paired_policy_difference_ci(
    data: pd.DataFrame,
    policy_a: str,
    policy_b: str,
    value_col: str,
    unit_cols: list[str],
    repeats: int = 2000,
    seed: int = 20260415,
) -> tuple[float, float, float]:
    paired = _paired_policy_matrix(data, policy_a=policy_a, policy_b=policy_b, value_col=value_col, unit_cols=unit_cols)
    diff = (paired[policy_a] - paired[policy_b]).to_numpy(dtype=float)
    return _simple_bootstrap_ci(diff, repeats=repeats, ci=0.95, seed=seed)


def paired_policy_difference_test(
    data: pd.DataFrame,
    policy_a: str,
    policy_b: str,
    value_col: str,
    unit_cols: list[str],
    repeats: int = 4000,
    seed: int = 20260415,
) -> tuple[float, float, float, float, int]:
    paired = _paired_policy_matrix(data, policy_a=policy_a, policy_b=policy_b, value_col=value_col, unit_cols=unit_cols)
    diff = (paired[policy_a] - paired[policy_b]).to_numpy(dtype=float)
    return paired_delta_test(diff, repeats=repeats, seed=seed)


def paired_delta_test(
    deltas: np.ndarray | pd.Series | list[float],
    repeats: int = 4000,
    seed: int = 20260415,
) -> tuple[float, float, float, float, int]:
    arr = np.asarray(deltas, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        raise ValueError("cannot compute a paired delta test on zero values")
    observed, ci_low, ci_high = _simple_bootstrap_ci(arr, repeats=repeats, ci=0.95, seed=seed)
    p_value = _paired_signflip_p_value(arr, repeats=repeats, seed=seed + 1)
    return observed, ci_low, ci_high, p_value, int(arr.shape[0])


def _paired_policy_matrix(
    data: pd.DataFrame,
    policy_a: str,
    policy_b: str,
    value_col: str,
    unit_cols: list[str],
) -> pd.DataFrame:
    wide = data.pivot_table(index=unit_cols, columns="policy", values=value_col, aggfunc="mean")
    if policy_a not in wide or policy_b not in wide:
        raise ValueError("both policies must exist in data")
    paired = wide[[policy_a, policy_b]]
    if paired.isna().any().any():
        missing_units = int(paired.isna().any(axis=1).sum())
        raise ValueError(f"missing paired policy values for {missing_units} inferential units")
    return paired


def _simple_bootstrap_ci(values: np.ndarray, repeats: int, ci: float, seed: int) -> tuple[float, float, float]:
    if values.size == 0:
        raise ValueError("cannot bootstrap zero values")
    rng = np.random.default_rng(seed)
    estimates = [float(np.mean(rng.choice(values, size=values.shape[0], replace=True))) for _ in range(repeats)]
    alpha = (1.0 - ci) / 2.0
    return float(np.mean(values)), float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1.0 - alpha))


def _paired_signflip_p_value(values: np.ndarray, repeats: int, seed: int) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("cannot compute a sign-flip test on zero values")
    observed = abs(float(np.mean(arr)))
    if arr.size == 1:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(repeats, arr.shape[0]), replace=True)
    simulated = np.abs((signs * arr[None, :]).mean(axis=1))
    return float((np.count_nonzero(simulated >= observed) + 1) / (repeats + 1))


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    order = np.argsort(p_values)
    rejected = [False] * len(p_values)
    for rank, idx in enumerate(order):
        threshold = alpha / (len(p_values) - rank)
        if p_values[idx] <= threshold:
            rejected[idx] = True
        else:
            break
    return rejected


def annotate_holm_bonferroni(
    data: pd.DataFrame,
    *,
    group_cols: list[str],
    p_value_col: str = "p_value",
    alpha: float = 0.05,
) -> pd.DataFrame:
    if data.empty:
        annotated = data.copy()
        annotated["holm_rank"] = pd.Series(dtype=int)
        annotated["holm_n_tests"] = pd.Series(dtype=int)
        annotated["holm_threshold"] = pd.Series(dtype=float)
        annotated["holm_reject"] = pd.Series(dtype=bool)
        return annotated
    annotated = data.copy()
    annotated["holm_rank"] = 0
    annotated["holm_n_tests"] = 0
    annotated["holm_threshold"] = np.nan
    annotated["holm_reject"] = False
    if not group_cols:
        group_iterator = [((), annotated.index.to_list())]
    else:
        group_iterator = annotated.groupby(group_cols, dropna=False, sort=True).groups.items()
    for _, indices in group_iterator:
        ordered = annotated.loc[list(indices), p_value_col].astype(float).sort_values(kind="mergesort")
        p_values = ordered.to_list()
        rejected = holm_bonferroni(p_values, alpha=alpha)
        n_tests = len(p_values)
        for rank, (row_idx, p_value) in enumerate(ordered.items(), start=1):
            threshold = alpha / (n_tests - rank + 1)
            annotated.at[row_idx, "holm_rank"] = int(rank)
            annotated.at[row_idx, "holm_n_tests"] = int(n_tests)
            annotated.at[row_idx, "holm_threshold"] = float(threshold)
            annotated.at[row_idx, "holm_reject"] = bool(rejected[rank - 1])
    return annotated

