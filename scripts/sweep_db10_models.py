from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myoagency.figures import save_policy_tradeoff  # noqa: E402
from myoagency.model_selection import default_db10_candidate_pairs, evaluate_db10_candidate, infer_db10_candidate_lane  # noqa: E402
from myoagency.real_benchmark import load_optional_sequence_payloads, run_dataset_benchmark  # noqa: E402
from myoagency.realdata import load_prepared_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DB10 model sweeps and rank candidates against the matched-budget gate.")
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=Path("results/experiments/db10_model_sweep"))
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate pair in user_model:assist_model form. Repeat for multiple pairs.",
    )
    parser.add_argument(
        "--lane",
        action="append",
        choices=["safe", "sota"],
        help="Default sweep lane to include when --candidate is omitted. Repeat to include multiple lanes. Defaults to both.",
    )
    parser.add_argument("--pairwise-repeats", type=int, default=2000)
    parser.add_argument("--promote-best-to", type=Path, help="Optional output directory to replace with the best DB10 result.")
    parser.add_argument("--force", action="store_true", help="Recompute candidates even if prior outputs already exist.")
    args = parser.parse_args()

    success_file = args.prepared_root / "_SUCCESS"
    if not success_file.exists():
        raise FileNotFoundError(f"prepared dataset is incomplete: missing {success_file}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    bundle = load_prepared_dataset(args.prepared_root)
    sequence_payloads, sequence_sources = load_optional_sequence_payloads(
        args.prepared_root,
        expected_rows=len(bundle.metadata),
    )
    candidates = parse_candidates(args.candidate, lanes=args.lane)
    if not candidates:
        raise ValueError("no DB10 sweep candidates resolved for the requested lane selection")
    summary_rows: list[dict[str, object]] = []
    pairwise_frames: list[pd.DataFrame] = []
    best_candidate_dir: Path | None = None
    best_rank_key: tuple[object, ...] | None = None

    for user_model, assist_model in candidates:
        candidate_id = f"{user_model}__{assist_model}"
        candidate_dir = args.out_root / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = candidate_dir / "metrics_by_policy_unit.csv"
        aggregate_path = candidate_dir / "aggregate_policy_metrics.csv"
        pairwise_path = candidate_dir / "matched_tau_pairwise.csv"
        manifest_path = candidate_dir / "benchmark_manifest.json"
        if not args.force and all(path.exists() for path in [metrics_path, aggregate_path, pairwise_path, manifest_path]):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            row = dict(manifest["candidate_summary"])
            row.setdefault("candidate_lane", infer_db10_candidate_lane(user_model, assist_model))
            row["elapsed_seconds"] = float(manifest.get("elapsed_seconds", 0.0))
            summary_rows.append(row)
            pairwise_df = pd.read_csv(pairwise_path)
            if not pairwise_df.empty:
                pairwise_df["candidate_id"] = candidate_id
                pairwise_df["candidate_lane"] = row["candidate_lane"]
                pairwise_frames.append(pairwise_df)
            rank_key = candidate_rank_key(row)
            if best_rank_key is None or rank_key > best_rank_key:
                best_rank_key = rank_key
                best_candidate_dir = candidate_dir
            print(f"{candidate_id}: reusing existing sweep output")
            continue
        start = time.perf_counter()
        metrics_df, aggregate_df = run_dataset_benchmark(
            bundle,
            dataset_id="db10",
            user_model_name=user_model,
            assist_model_name=assist_model,
            extra_inputs=sequence_payloads,
        )
        elapsed_s = time.perf_counter() - start
        metrics_df.to_csv(metrics_path, index=False)
        aggregate_df.to_csv(aggregate_path, index=False)
        save_policy_tradeoff(aggregate_df, candidate_dir / "policy_tradeoff.png")
        summary, pairwise_df = evaluate_db10_candidate(
            metrics_df,
            aggregate_df,
            user_model=user_model,
            assist_model=assist_model,
            repeats=args.pairwise_repeats,
        )
        pairwise_df.to_csv(candidate_dir / "matched_tau_pairwise.csv", index=False)
        manifest = {
            "dataset": "db10",
            "prepared_root": str(args.prepared_root),
            "prepared_success_file": str(success_file),
            "user_model": user_model,
            "assist_model": assist_model,
            "pairwise_repeats": int(args.pairwise_repeats),
            "elapsed_seconds": elapsed_s,
            "candidate_lane": infer_db10_candidate_lane(user_model, assist_model),
            "available_sequence_payload_roles": sorted(sequence_payloads),
            "sequence_payload_sources": sequence_sources,
            "n_rows": int(len(bundle.metadata)),
            "candidate_summary": summary.to_dict(),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        row = summary.to_dict()
        row["elapsed_seconds"] = elapsed_s
        summary_rows.append(row)
        if not pairwise_df.empty:
            tagged_pairwise = pairwise_df.copy()
            tagged_pairwise["candidate_id"] = candidate_id
            tagged_pairwise["candidate_lane"] = row["candidate_lane"]
            pairwise_frames.append(tagged_pairwise)
        rank_key = candidate_rank_key(row)
        if best_rank_key is None or rank_key > best_rank_key:
            best_rank_key = rank_key
            best_candidate_dir = candidate_dir
        print(
            f"{candidate_id} [{row['candidate_lane']}]: qualifying_taus={row['qualifying_taus'] or '<none>'} "
            f"active_f1_cells={row['beneficial_active_f1_cells']} "
            f"active_risk_cells={row['beneficial_active_risk_cells']} "
            f"elapsed_s={elapsed_s:.1f}"
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        [
            "all_families_passed",
            "qualifying_tau_count",
            "beneficial_active_f1_cells",
            "beneficial_active_risk_cells",
            "best_agency_active_macro_f1",
            "best_agency_active_risk_coverage_auc",
        ],
        ascending=[False, False, False, False, False, True],
    )
    summary_df.to_csv(args.out_root / "candidate_summary.csv", index=False)
    if pairwise_frames:
        pd.concat(pairwise_frames, ignore_index=True).to_csv(args.out_root / "candidate_pairwise.csv", index=False)
    if args.promote_best_to is not None and best_candidate_dir is not None:
        if args.promote_best_to.exists():
            shutil.rmtree(args.promote_best_to)
        shutil.copytree(best_candidate_dir, args.promote_best_to)
    print(f"Wrote DB10 sweep summary to {args.out_root}")


def parse_candidates(values: list[str], lanes: list[str] | None = None) -> list[tuple[str, str]]:
    if not values:
        resolved_lanes = tuple(lanes) if lanes else None
        return default_db10_candidate_pairs(lanes=resolved_lanes)
    pairs: list[tuple[str, str]] = []
    for value in values:
        parts = value.split(":")
        if len(parts) != 2 or not all(part.strip() for part in parts):
            raise ValueError(f"invalid candidate '{value}', expected user_model:assist_model")
        pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs


def candidate_rank_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        bool(row["all_families_passed"]),
        int(row["qualifying_tau_count"]),
        int(row["beneficial_active_f1_cells"]),
        int(row["beneficial_active_risk_cells"]),
        float(row["best_agency_active_macro_f1"]),
        -float(row["best_agency_active_risk_coverage_auc"]),
    )


if __name__ == "__main__":
    main()

