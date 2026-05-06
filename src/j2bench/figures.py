from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns
import numpy as np
import math

matplotlib.use("Agg", force=True)

from matplotlib import pyplot as plt


def save_policy_tradeoff(metrics: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    style = "split_family" if "split_family" in metrics.columns else None
    sns.scatterplot(
        data=metrics,
        x="mean_ali",
        y="macro_f1",
        hue="policy",
        style=style,
        size="intervention_rate",
        ax=ax,
    )
    ax.set_xlabel("Mean expected ALI")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Performance-agency tradeoff")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def save_matched_budget_ablation(
    aggregate: pd.DataFrame,
    out: Path,
    *,
    title_prefix: str = "",
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    subset = aggregate.loc[
        aggregate["policy"].astype(str).str.startswith("agency_margin_tau_")
        | aggregate["policy"].astype(str).str.startswith("plain_conf_threshold_matched_tau_")
    ].copy()
    if subset.empty:
        return
    subset["tau"] = subset["policy"].astype(str).str.extract(r"(\d+\.\d+)$").astype(float)
    subset["family"] = np.where(
        subset["policy"].astype(str).str.startswith("agency_margin_tau_"),
        "agency_margin",
        "plain_conf_threshold",
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for family, family_df in subset.groupby("family", sort=True):
        mean_df = family_df.groupby("tau", as_index=False).agg(
            active_macro_f1=("active_macro_f1", "mean"),
            active_risk_coverage_auc=("active_risk_coverage_auc", "mean"),
        )
        axes[0].plot(mean_df["tau"], mean_df["active_macro_f1"], marker="o", label=family)
        axes[1].plot(mean_df["tau"], mean_df["active_risk_coverage_auc"], marker="o", label=family)
    axes[0].set_title(_prefixed_title(title_prefix, "Matched-Budget Active Macro-F1"))
    axes[0].set_xlabel("Tau")
    axes[0].set_ylabel("Active Macro-F1")
    axes[1].set_title(_prefixed_title(title_prefix, "Matched-Budget Active Risk-Coverage AUC"))
    axes[1].set_xlabel("Tau")
    axes[1].set_ylabel("AUC")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def save_earliest_safe_summary(
    summary: pd.DataFrame,
    out: Path,
    *,
    title_prefix: str = "",
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    subset = summary.loc[
        summary["policy"].astype(str).str.startswith("agency_margin_tau_")
        | summary["policy"].astype(str).str.startswith("set_acsa_tau_")
        | summary["policy"].astype(str).str.startswith("plain_conf_threshold_matched_tau_")
    ].copy()
    if subset.empty:
        return
    subset["tau"] = subset["policy"].astype(str).str.extract(r"(\d+\.\d+)$").astype(float)
    subset["family"] = np.where(
        subset["policy"].astype(str).str.startswith("agency_margin_tau_"),
        "agency_margin",
        np.where(
            subset["policy"].astype(str).str.startswith("set_acsa_tau_"),
            "set_acsa",
            "plain_conf_threshold",
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for family, family_df in subset.groupby("family", sort=True):
        mean_df = family_df.groupby("tau", as_index=False).agg(
            stable_safe_episode_rate=("stable_safe_episode_rate", "mean"),
            median_earliest_stable_safe_s=("median_earliest_stable_safe_s", "mean"),
        )
        axes[0].plot(mean_df["tau"], mean_df["stable_safe_episode_rate"], marker="o", label=family)
        axes[1].plot(mean_df["tau"], mean_df["median_earliest_stable_safe_s"], marker="o", label=family)
    axes[0].set_title(_prefixed_title(title_prefix, "Stable Safe Episode Rate"))
    axes[0].set_xlabel("Tau")
    axes[0].set_ylabel("Rate")
    axes[1].set_title(_prefixed_title(title_prefix, "Median Earliest Stable Safe Time"))
    axes[1].set_xlabel("Tau")
    axes[1].set_ylabel("Seconds")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def save_j1_frontier_atlas(
    points: pd.DataFrame,
    out: Path,
    *,
    split_families: list[str] | None = None,
    title_prefix: str = "",
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if points.empty:
        return
    subset = points.copy()
    if split_families:
        subset = subset.loc[subset["split_family"].astype(str).isin(split_families)].copy()
    if subset.empty:
        return

    families = split_families or sorted(subset["split_family"].astype(str).unique().tolist())
    ncols = 2
    nrows = max(1, math.ceil(len(families) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.6 * nrows), squeeze=False)
    palette = _policy_palette(subset["policy_family"].astype(str).unique().tolist())
    size_min = 60.0
    size_max = 260.0

    stable_vals = subset["stable_safe_episode_rate"].astype(float)
    stable_min = float(stable_vals.min()) if stable_vals.notna().any() else 0.0
    stable_max = float(stable_vals.max()) if stable_vals.notna().any() else 1.0

    for axis, family in zip(axes.flat, families):
        family_df = subset.loc[subset["split_family"].astype(str) == str(family)].copy()
        if family_df.empty:
            axis.axis("off")
            continue

        for policy_family, family_points in family_df.groupby("policy_family", sort=True):
            sizes = family_points["stable_safe_episode_rate"].astype(float).map(
                lambda value: _scale_size(value, stable_min, stable_max, size_min, size_max)
            )
            axis.scatter(
                family_points["active_risk_coverage_auc"],
                family_points["active_macro_f1"],
                s=sizes,
                color=palette[str(policy_family)],
                alpha=0.72,
                edgecolors="white",
                linewidths=0.6,
                label=str(policy_family),
            )
        frontier = family_df.loc[family_df["is_non_dominated"].astype(bool)].copy()
        if not frontier.empty:
            sizes = frontier["stable_safe_episode_rate"].astype(float).map(
                lambda value: _scale_size(value, stable_min, stable_max, size_min, size_max) + 40.0
            )
            axis.scatter(
                frontier["active_risk_coverage_auc"],
                frontier["active_macro_f1"],
                s=sizes,
                facecolors="none",
                edgecolors="black",
                linewidths=1.4,
            )
            for _, row in frontier.iterrows():
                label = str(row["frontier_label"])
                axis.annotate(
                    label,
                    (float(row["active_risk_coverage_auc"]), float(row["active_macro_f1"])),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                )
        axis.set_title(str(family))
        axis.set_xlabel("Active Risk-Coverage AUC")
        axis.set_ylabel("Active Macro-F1")
        axis.grid(alpha=0.2, linestyle=":")

    for axis in axes.flat[len(families):]:
        axis.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        fig.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper center",
            ncol=min(4, len(by_label)),
            frameon=False,
        )
    title = _prefixed_title(title_prefix, "J1 Frontier Atlas")
    fig.suptitle(title, y=0.995, fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out, dpi=300)
    plt.close(fig)


def save_j1_safety_timing_frontier(
    points: pd.DataFrame,
    out: Path,
    *,
    split_families: list[str] | None = None,
    title_prefix: str = "",
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if points.empty:
        return
    subset = points.copy()
    if split_families:
        subset = subset.loc[subset["split_family"].astype(str).isin(split_families)].copy()
    subset = subset.loc[
        subset["policy_family"].astype(str).isin(
            ["agency_margin", "set_acsa", "plain_conf_threshold_matched"]
        )
    ].copy()
    subset = subset.loc[subset["tau"].notna()].copy()
    if subset.empty:
        return

    families = split_families or sorted(subset["split_family"].astype(str).unique().tolist())
    ncols = 2
    nrows = max(1, math.ceil(len(families) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.6 * nrows), squeeze=False)
    palette = _policy_palette(subset["policy_family"].astype(str).unique().tolist())

    for axis, family in zip(axes.flat, families):
        family_df = subset.loc[subset["split_family"].astype(str) == str(family)].copy()
        if family_df.empty:
            axis.axis("off")
            continue
        for policy_family, family_points in family_df.groupby("policy_family", sort=True):
            ordered = family_points.sort_values("tau", key=lambda col: col.astype(float))
            axis.plot(
                ordered["median_earliest_stable_safe_s"],
                ordered["stable_safe_episode_rate"],
                marker="o",
                linewidth=1.7,
                label=str(policy_family),
                color=palette[str(policy_family)],
            )
            for _, row in ordered.iterrows():
                axis.annotate(
                    str(row["tau"]),
                    (
                        float(row["median_earliest_stable_safe_s"]),
                        float(row["stable_safe_episode_rate"]),
                    ),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                )
        axis.set_title(str(family))
        axis.set_xlabel("Median Earliest Stable Safe Time (s)")
        axis.set_ylabel("Stable Safe Episode Rate")
        axis.grid(alpha=0.2, linestyle=":")

    for axis in axes.flat[len(families):]:
        axis.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        fig.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper center",
            ncol=min(3, len(by_label)),
            frameon=False,
        )
    title = _prefixed_title(title_prefix, "J1 Safety-Timing Frontier")
    fig.suptitle(title, y=0.995, fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out, dpi=300)
    plt.close(fig)


def _prefixed_title(prefix: str, title: str) -> str:
    cleaned = str(prefix).strip()
    if not cleaned:
        return title
    return f"{cleaned} {title}"


def _policy_palette(policy_families: list[str]) -> dict[str, tuple[float, float, float]]:
    colors = sns.color_palette("tab10", n_colors=max(3, len(policy_families)))
    return {
        str(policy_family): colors[index]
        for index, policy_family in enumerate(sorted(set(policy_families)))
    }


def _scale_size(
    value: float,
    lower: float,
    upper: float,
    size_min: float,
    size_max: float,
) -> float:
    if np.isnan(value):
        return size_min
    if upper <= lower + 1e-12:
        return (size_min + size_max) / 2.0
    normalized = (float(value) - lower) / (upper - lower)
    normalized = min(max(normalized, 0.0), 1.0)
    return size_min + normalized * (size_max - size_min)

