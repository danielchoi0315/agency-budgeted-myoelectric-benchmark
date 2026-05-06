from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns
import numpy as np

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


def _prefixed_title(prefix: str, title: str) -> str:
    cleaned = str(prefix).strip()
    if not cleaned:
        return title
    return f"{cleaned} {title}"


