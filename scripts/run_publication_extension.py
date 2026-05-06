from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myoagency.artifact_schemas import validate_publication_manifest  # noqa: E402
from myoagency.figures import save_earliest_safe_summary, save_matched_budget_ablation  # noqa: E402
from myoagency.metrics import summarize_by_policy  # noqa: E402
from myoagency.policies import apply_policy_grid, apply_rate_matched_confidence_gate  # noqa: E402
from myoagency.publication import (  # noqa: E402
    DEFAULT_ABLATION_METRICS,
    DEFAULT_EARLIEST_METRICS,
    DEFAULT_ISO_BUDGET_ABLATION_METRICS,
    DEFAULT_PUBLICATION_TAU_GRID,
    EXACT_BUDGET_COMPARISON_FAMILY,
    EXACT_BUDGET_PLAIN_CONF_POLICY_PREFIX,
    ISO_BUDGET_PLAIN_CONF_CURVE_LABEL,
    LOWER_IS_BETTER_PUBLICATION_METRICS,
    annotate_exact_budget_pairwise_diagnostics,
    build_cognolato_db10_splits,
    build_dense_plain_confidence_threshold_bank,
    build_wang_db10_splits,
    compute_cognolato_anchor_metrics,
    compute_iso_budget_pairwise_stats,
    compute_pairwise_stats,
    compute_wang_anchor_metrics,
    decisions_to_frame,
    materialize_exact_budget_plain_confidence_units,
    matched_policy_pairs,
    merge_budget_into_unit_frame,
    summarize_earliest_safe_aggregate,
    summarize_earliest_safe_by_episode,
    summarize_earliest_safe_by_unit,
    traces_to_frame,
)
from myoagency.real_benchmark import load_optional_sequence_payloads, score_dataset_traces  # noqa: E402
from myoagency.realdata import load_prepared_dataset  # noqa: E402
from myoagency.stats import annotate_holm_bonferroni  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run publication-clean matched-budget and earliest-safe analyses for a prepared dataset. "
            "DB10 can optionally emit local anchor reproductions."
        )
    )
    parser.add_argument("--dataset", required=True, choices=["db10", "hyser", "cemhsey", "grabmyo"])
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=Path("results/reports/publication_clean"))
    parser.add_argument("--user-model", required=True)
    parser.add_argument("--assist-model", required=True)
    parser.add_argument("--pairwise-repeats", type=int, default=2000)
    parser.add_argument("--include-db10-anchors", action="store_true")
    args = parser.parse_args()

    success_file = args.prepared_root / "_SUCCESS"
    if not success_file.exists():
        raise FileNotFoundError(f"prepared dataset is incomplete: missing {success_file}")
    if args.include_db10_anchors and args.dataset != "db10":
        raise ValueError("--include-db10-anchors is only valid for --dataset db10")

    args.out_root.mkdir(parents=True, exist_ok=True)
    bundle = load_prepared_dataset(args.prepared_root)
    sequence_payloads, _ = load_optional_sequence_payloads(args.prepared_root, expected_rows=len(bundle.metadata))

    traces = score_dataset_traces(
        bundle,
        dataset_id=args.dataset,
        user_model_name=args.user_model,
        assist_model_name=args.assist_model,
        extra_inputs=sequence_payloads,
    )
    decisions = apply_policy_grid(traces, tau_grid=DEFAULT_PUBLICATION_TAU_GRID)
    matched_decisions, match_rows = apply_rate_matched_confidence_gate(traces, tau_grid=DEFAULT_PUBLICATION_TAU_GRID)
    all_decisions = decisions + matched_decisions

    metrics_unit = pd.DataFrame([asdict(metric) for metric in summarize_by_policy(all_decisions)])
    metrics_unit["split_family"] = metrics_unit["split_id"].astype(str).str.split("|").str[0]
    aggregate = (
        metrics_unit.groupby(["split_family", "policy"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            macro_f1=("macro_f1", "mean"),
            active_macro_f1=("active_macro_f1", "mean"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            mean_ali=("mean_ali", "mean"),
            intervention_rate=("intervention_rate", "mean"),
            risk_coverage_auc=("risk_coverage_auc", "mean"),
            active_risk_coverage_auc=("active_risk_coverage_auc", "mean"),
            ece=("ece", "mean"),
            median_decision_time_ms=("median_decision_time_ms", "mean"),
            n=("n", "sum"),
            n_active=("n_active", "sum"),
        )
        .sort_values(["split_family", "mean_ali", "active_macro_f1"], ascending=[True, True, False])
        .reset_index(drop=True)
    )

    prefix = f"{args.dataset}_"
    metrics_unit.to_csv(args.out_root / f"{prefix}publication_metrics_by_policy_unit.csv", index=False)
    aggregate.to_csv(args.out_root / f"{prefix}publication_aggregate_policy_metrics.csv", index=False)
    pd.DataFrame(match_rows).to_csv(args.out_root / f"{prefix}confidence_gate_match_summary.csv", index=False)

    decision_frame = decisions_to_frame(all_decisions)
    episode_frame = summarize_earliest_safe_by_episode(decision_frame)
    unit_frame = summarize_earliest_safe_by_unit(episode_frame)
    earliest_summary = summarize_earliest_safe_aggregate(unit_frame)
    episode_frame.to_csv(args.out_root / f"{prefix}earliest_safe_by_episode.csv", index=False)
    unit_frame.to_csv(args.out_root / f"{prefix}earliest_safe_by_unit.csv", index=False)
    earliest_summary.to_csv(args.out_root / f"{prefix}earliest_safe_summary.csv", index=False)

    agency_vs_set_pairs = matched_policy_pairs(
        metrics_unit["policy"],
        left_prefix="agency_margin_tau_",
        right_prefix="set_acsa_tau_",
        tau_grid=DEFAULT_PUBLICATION_TAU_GRID,
    )
    agency_vs_plain_pairs = matched_policy_pairs(
        metrics_unit["policy"],
        left_prefix="agency_margin_tau_",
        right_prefix="plain_conf_threshold_matched_tau_",
        tau_grid=DEFAULT_PUBLICATION_TAU_GRID,
    )
    pair_builders = [
        ("agency_vs_set_acsa", agency_vs_set_pairs),
        ("agency_vs_plain_conf", agency_vs_plain_pairs),
    ]
    confidence_pairwise = compute_pairwise_stats(
        metrics_unit,
        pair_builders=pair_builders,
        metrics=DEFAULT_ABLATION_METRICS,
        repeats=args.pairwise_repeats,
        lower_is_better=LOWER_IS_BETTER_PUBLICATION_METRICS,
    )
    confidence_pairwise = annotate_holm_bonferroni(
        confidence_pairwise,
        group_cols=["split_family", "comparison_family", "metric"],
    )
    confidence_pairwise["beneficial_and_holm_significant"] = (
        confidence_pairwise["beneficial"].astype(bool) & confidence_pairwise["holm_reject"].astype(bool)
    )
    confidence_pairwise.to_csv(args.out_root / f"{prefix}confidence_gate_pairwise.csv", index=False)
    confidence_iso_unit, confidence_iso_pairwise = compute_iso_budget_pairwise_stats(
        metrics_unit,
        policy_pairs=agency_vs_plain_pairs,
        curve_policy_prefix="plain_conf_threshold_matched_tau_",
        metrics=DEFAULT_ISO_BUDGET_ABLATION_METRICS,
        repeats=args.pairwise_repeats,
        lower_is_better=LOWER_IS_BETTER_PUBLICATION_METRICS,
        curve_policy_label=ISO_BUDGET_PLAIN_CONF_CURVE_LABEL,
    )
    if confidence_iso_pairwise.empty:
        confidence_iso_pairwise = confidence_iso_pairwise.copy()
        confidence_iso_pairwise["beneficial_and_holm_significant"] = pd.Series(dtype=bool)
    else:
        confidence_iso_pairwise = annotate_holm_bonferroni(
            confidence_iso_pairwise,
            group_cols=["split_family", "comparison_family", "metric"],
        )
        confidence_iso_pairwise["beneficial_and_holm_significant"] = (
            confidence_iso_pairwise["beneficial"].astype(bool)
            & confidence_iso_pairwise["holm_reject"].astype(bool)
        )
    confidence_iso_unit.to_csv(
        args.out_root / f"{prefix}confidence_gate_iso_budget_unit_deltas.csv",
        index=False,
    )
    confidence_iso_pairwise.to_csv(
        args.out_root / f"{prefix}confidence_gate_iso_budget_pairwise.csv",
        index=False,
    )
    exact_budget_threshold_bank = build_dense_plain_confidence_threshold_bank(traces)
    exact_budget_selection, exact_budget_metrics_unit, exact_budget_earliest_unit = (
        materialize_exact_budget_plain_confidence_units(
            traces=traces,
            budget_frame=metrics_unit,
            policy_pairs=agency_vs_plain_pairs,
            threshold_bank=exact_budget_threshold_bank,
        )
    )
    exact_metric_frame = pd.concat([metrics_unit, exact_budget_metrics_unit], ignore_index=True)
    agency_vs_exact_pairs = matched_policy_pairs(
        exact_metric_frame["policy"],
        left_prefix="agency_margin_tau_",
        right_prefix=EXACT_BUDGET_PLAIN_CONF_POLICY_PREFIX,
        tau_grid=DEFAULT_PUBLICATION_TAU_GRID,
    )
    confidence_exact_pairwise = compute_pairwise_stats(
        exact_metric_frame,
        pair_builders=[(EXACT_BUDGET_COMPARISON_FAMILY, agency_vs_exact_pairs)],
        metrics=DEFAULT_ISO_BUDGET_ABLATION_METRICS,
        repeats=args.pairwise_repeats,
        lower_is_better=LOWER_IS_BETTER_PUBLICATION_METRICS,
    )
    confidence_exact_pairwise = annotate_exact_budget_pairwise_diagnostics(
        confidence_exact_pairwise,
        selection_frame=exact_budget_selection,
    )
    if confidence_exact_pairwise.empty:
        confidence_exact_pairwise = confidence_exact_pairwise.copy()
        confidence_exact_pairwise["beneficial_and_holm_significant"] = pd.Series(dtype=bool)
    else:
        confidence_exact_pairwise = annotate_holm_bonferroni(
            confidence_exact_pairwise,
            group_cols=["split_family", "comparison_family", "metric"],
        )
        confidence_exact_pairwise["beneficial_and_holm_significant"] = (
            confidence_exact_pairwise["beneficial"].astype(bool)
            & confidence_exact_pairwise["holm_reject"].astype(bool)
        )
    exact_budget_selection.to_csv(
        args.out_root / f"{prefix}plain_conf_exact_budget_selection_by_unit.csv",
        index=False,
    )
    confidence_exact_pairwise.to_csv(
        args.out_root / f"{prefix}confidence_gate_exact_budget_pairwise.csv",
        index=False,
    )

    earliest_pairwise = compute_pairwise_stats(
        unit_frame,
        pair_builders=pair_builders,
        metrics=DEFAULT_EARLIEST_METRICS,
        repeats=args.pairwise_repeats,
        lower_is_better=LOWER_IS_BETTER_PUBLICATION_METRICS,
    )
    earliest_pairwise = annotate_holm_bonferroni(
        earliest_pairwise,
        group_cols=["split_family", "comparison_family", "metric"],
    )
    earliest_pairwise["beneficial_and_holm_significant"] = (
        earliest_pairwise["beneficial"].astype(bool) & earliest_pairwise["holm_reject"].astype(bool)
    )
    earliest_pairwise.to_csv(args.out_root / f"{prefix}earliest_safe_pairwise.csv", index=False)
    earliest_budgeted_unit = merge_budget_into_unit_frame(
        unit_frame,
        budget_frame=metrics_unit,
    )
    earliest_iso_unit, earliest_iso_pairwise = compute_iso_budget_pairwise_stats(
        earliest_budgeted_unit,
        policy_pairs=agency_vs_plain_pairs,
        curve_policy_prefix="plain_conf_threshold_matched_tau_",
        metrics=DEFAULT_EARLIEST_METRICS,
        repeats=args.pairwise_repeats,
        lower_is_better=LOWER_IS_BETTER_PUBLICATION_METRICS,
        curve_policy_label=ISO_BUDGET_PLAIN_CONF_CURVE_LABEL,
    )
    if earliest_iso_pairwise.empty:
        earliest_iso_pairwise = earliest_iso_pairwise.copy()
        earliest_iso_pairwise["beneficial_and_holm_significant"] = pd.Series(dtype=bool)
    else:
        earliest_iso_pairwise = annotate_holm_bonferroni(
            earliest_iso_pairwise,
            group_cols=["split_family", "comparison_family", "metric"],
        )
        earliest_iso_pairwise["beneficial_and_holm_significant"] = (
            earliest_iso_pairwise["beneficial"].astype(bool)
            & earliest_iso_pairwise["holm_reject"].astype(bool)
        )
    earliest_iso_unit.to_csv(
        args.out_root / f"{prefix}earliest_safe_iso_budget_unit_deltas.csv",
        index=False,
    )
    earliest_iso_pairwise.to_csv(
        args.out_root / f"{prefix}earliest_safe_iso_budget_pairwise.csv",
        index=False,
    )
    exact_earliest_frame = pd.concat([unit_frame, exact_budget_earliest_unit], ignore_index=True)
    earliest_exact_pairwise = compute_pairwise_stats(
        exact_earliest_frame,
        pair_builders=[(EXACT_BUDGET_COMPARISON_FAMILY, agency_vs_exact_pairs)],
        metrics=DEFAULT_EARLIEST_METRICS,
        repeats=args.pairwise_repeats,
        lower_is_better=LOWER_IS_BETTER_PUBLICATION_METRICS,
    )
    earliest_exact_pairwise = annotate_exact_budget_pairwise_diagnostics(
        earliest_exact_pairwise,
        selection_frame=exact_budget_selection,
    )
    if earliest_exact_pairwise.empty:
        earliest_exact_pairwise = earliest_exact_pairwise.copy()
        earliest_exact_pairwise["beneficial_and_holm_significant"] = pd.Series(dtype=bool)
    else:
        earliest_exact_pairwise = annotate_holm_bonferroni(
            earliest_exact_pairwise,
            group_cols=["split_family", "comparison_family", "metric"],
        )
        earliest_exact_pairwise["beneficial_and_holm_significant"] = (
            earliest_exact_pairwise["beneficial"].astype(bool)
            & earliest_exact_pairwise["holm_reject"].astype(bool)
        )
    earliest_exact_pairwise.to_csv(
        args.out_root / f"{prefix}earliest_safe_exact_budget_pairwise.csv",
        index=False,
    )

    save_earliest_safe_summary(
        earliest_summary,
        args.out_root / f"{prefix}earliest_safe.png",
        title_prefix=args.dataset.upper(),
    )
    save_matched_budget_ablation(
        aggregate,
        args.out_root / f"{prefix}matched_budget_ablation.png",
        title_prefix=args.dataset.upper(),
    )

    manifest = {
        "dataset": args.dataset,
        "prepared_root": str(args.prepared_root),
        "user_model": args.user_model,
        "assist_model": args.assist_model,
        "pairwise_repeats": int(args.pairwise_repeats),
        "trace_count": int(len(traces)),
        "decision_count": int(len(all_decisions)),
        "split_families": sorted(metrics_unit["split_family"].astype(str).unique().tolist()),
        "artifact_inventory": {
            "publication_metrics_by_policy_unit": f"{prefix}publication_metrics_by_policy_unit.csv",
            "publication_aggregate_policy_metrics": f"{prefix}publication_aggregate_policy_metrics.csv",
            "confidence_gate_match_summary": f"{prefix}confidence_gate_match_summary.csv",
            "confidence_gate_pairwise": f"{prefix}confidence_gate_pairwise.csv",
            "confidence_gate_iso_budget_unit_deltas": f"{prefix}confidence_gate_iso_budget_unit_deltas.csv",
            "confidence_gate_iso_budget_pairwise": f"{prefix}confidence_gate_iso_budget_pairwise.csv",
            "plain_conf_exact_budget_selection_by_unit": f"{prefix}plain_conf_exact_budget_selection_by_unit.csv",
            "confidence_gate_exact_budget_pairwise": f"{prefix}confidence_gate_exact_budget_pairwise.csv",
            "earliest_safe_by_episode": f"{prefix}earliest_safe_by_episode.csv",
            "earliest_safe_by_unit": f"{prefix}earliest_safe_by_unit.csv",
            "earliest_safe_summary": f"{prefix}earliest_safe_summary.csv",
            "earliest_safe_pairwise": f"{prefix}earliest_safe_pairwise.csv",
            "earliest_safe_iso_budget_unit_deltas": f"{prefix}earliest_safe_iso_budget_unit_deltas.csv",
            "earliest_safe_iso_budget_pairwise": f"{prefix}earliest_safe_iso_budget_pairwise.csv",
            "earliest_safe_exact_budget_pairwise": f"{prefix}earliest_safe_exact_budget_pairwise.csv",
            "matched_budget_ablation_figure": f"{prefix}matched_budget_ablation.png",
            "earliest_safe_figure": f"{prefix}earliest_safe.png",
        },
    }
    if args.include_db10_anchors:
        anchor_payload = build_db10_anchor_payload(
            bundle=bundle,
            sequence_payloads=sequence_payloads,
            user_model=args.user_model,
            assist_model=args.assist_model,
        )
        (args.out_root / "db10_anchor_reproductions.json").write_text(json.dumps(anchor_payload, indent=2), encoding="utf-8")
        (args.out_root / "db10_anchor_reproductions.md").write_text(build_anchor_markdown(anchor_payload), encoding="utf-8")
        manifest["db10_anchor_payload_keys"] = sorted(anchor_payload)
    validate_publication_manifest(manifest, expected_dataset=args.dataset)
    (args.out_root / f"{prefix}publication_extension_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {args.dataset} publication extension outputs to {args.out_root}")


def build_db10_anchor_payload(
    *,
    bundle,
    sequence_payloads: dict[str, object],
    user_model: str,
    assist_model: str,
) -> dict[str, object]:
    wang_traces = score_dataset_traces(
        bundle,
        dataset_id="db10",
        user_model_name=user_model,
        assist_model_name=assist_model,
        extra_inputs=sequence_payloads,
        splits=build_wang_db10_splits(bundle.metadata),
    )
    cognolato_traces = score_dataset_traces(
        bundle,
        dataset_id="db10",
        user_model_name=user_model,
        assist_model_name=assist_model,
        extra_inputs=sequence_payloads,
        splits=build_cognolato_db10_splits(bundle.metadata),
    )
    wang_metrics = compute_wang_anchor_metrics(traces_to_frame(wang_traces))
    cognolato_metrics = compute_cognolato_anchor_metrics(traces_to_frame(cognolato_traces))
    return {
        "wang_frobt_2022": {
            "published": {
                "emg_accuracy_percent": 85.5,
                "integrated_accuracy_percent": 90.06,
                "gain_percentage_points": 4.56,
            },
            "local": wang_metrics,
            "match_level": "split_matched_phase_gated_local_reproduction",
            "note": "Local reproduction uses able-bodied DB10 leave-one-repetition-out splits with early assistive evidence fused into the final user decoder decision.",
        },
        "cognolato_frontiers_ai_2022": {
            "published": {
                "amputee_gain_percentage_points": 15.61,
                "able_bodied_gain_percentage_points": 7.37,
            },
            "local": cognolato_metrics,
            "match_level": "within_subject_four_fold_local_reproduction",
            "note": "Local reproduction builds subject-wise four-fold DB10 splits from repetition metadata and compares final-prefix EMG-only vs assistive multimodal decoders.",
        },
    }


def build_anchor_markdown(payload: dict[str, object]) -> str:
    lines = ["# DB10 Anchor Reproductions", ""]
    for anchor_id, entry in payload.items():
        lines.append(f"## {anchor_id}")
        lines.append("")
        lines.append(f"- Match level: {entry['match_level']}")
        lines.append(f"- Note: {entry['note']}")
        lines.append(f"- Published: `{json.dumps(entry['published'])}`")
        lines.append(f"- Local: `{json.dumps(entry['local'])}`")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

