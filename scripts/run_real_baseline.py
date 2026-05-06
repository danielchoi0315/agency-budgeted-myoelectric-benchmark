from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myoagency.artifact_schemas import validate_benchmark_manifest  # noqa: E402
from myoagency.figures import save_policy_tradeoff  # noqa: E402
from myoagency.provenance import build_output_manifest, write_manifest  # noqa: E402
from myoagency.real_benchmark import load_optional_sequence_payloads, run_dataset_benchmark  # noqa: E402
from myoagency.realdata import load_prepared_dataset  # noqa: E402


def infer_available_sequence_keys(payloads: dict[str, object]) -> list[str]:
    keys: set[str] = set()

    def visit(prefix: str, value: object) -> None:
        if isinstance(value, Mapping):
            for name, item in value.items():
                key = f"{prefix}.{name}" if prefix else str(name)
                keys.add(key)
                visit(key, item)

    for role, payload in payloads.items():
        keys.add(str(role))
        visit(str(role), payload)
    return sorted(keys)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline policy benchmarks on prepared real data.")
    parser.add_argument("--dataset", choices=["db10", "hyser", "cemhsey", "grabmyo"], required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--user-model", default="lda")
    parser.add_argument("--assist-model", default="logistic")
    parser.add_argument("--split-family", action="append", default=[], help="Optional split-family allowlist.")
    parser.add_argument("--split-id", action="append", default=[], help="Optional exact split-id allowlist.")
    parser.add_argument("--max-splits-per-family", type=int, help="Optional cap on split count per split family.")
    args = parser.parse_args()

    success_file = args.prepared_root / "_SUCCESS"
    if not success_file.exists():
        raise FileNotFoundError(f"prepared dataset is incomplete: missing {success_file}")

    bundle = load_prepared_dataset(args.prepared_root)
    sequence_payloads, sequence_sources = load_optional_sequence_payloads(
        args.prepared_root,
        expected_rows=len(bundle.metadata),
    )
    metrics_df, aggregate_df = run_dataset_benchmark(
        bundle,
        dataset_id=args.dataset,
        user_model_name=args.user_model,
        assist_model_name=args.assist_model,
        extra_inputs=sequence_payloads,
        split_family_allowlist=tuple(args.split_family) or None,
        split_id_allowlist=tuple(args.split_id) or None,
        max_splits_per_family=args.max_splits_per_family,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(args.out / "metrics_by_policy_unit.csv", index=False)
    aggregate_df.to_csv(args.out / "aggregate_policy_metrics.csv", index=False)
    context_modes = []
    if "assist_context_mode" in bundle.metadata.columns:
        context_modes = sorted(bundle.metadata["assist_context_mode"].dropna().astype(str).unique().tolist())
    benchmark_manifest = {
        "dataset": args.dataset,
        "prepared_root": str(args.prepared_root),
        "prepared_success_file": str(success_file),
        "user_model": args.user_model,
        "assist_model": args.assist_model,
        "assist_context_modes": context_modes,
        "available_sequence_payload_roles": sorted(sequence_payloads),
        "available_sequence_payload_keys": infer_available_sequence_keys(sequence_payloads),
        "sequence_payload_sources": sequence_sources,
        "split_family_allowlist": list(args.split_family),
        "split_id_allowlist": list(args.split_id),
        "max_splits_per_family": args.max_splits_per_family,
        "n_rows": int(len(bundle.metadata)),
    }
    validate_benchmark_manifest(benchmark_manifest, expected_dataset=args.dataset)
    (args.out / "benchmark_manifest.json").write_text(
        json.dumps(benchmark_manifest, indent=2),
        encoding="utf-8",
    )
    save_policy_tradeoff(aggregate_df, args.out / "policy_tradeoff.png")
    benchmark_provenance = build_output_manifest(
        args.out,
        output_kind="benchmark",
        dataset_id=args.dataset,
        config_paths=(Path("config/config.yaml"),),
        input_paths=(args.prepared_root,),
        extra_metadata={
            "benchmark_manifest_path": str(args.out / "benchmark_manifest.json"),
            "available_sequence_payload_keys": benchmark_manifest["available_sequence_payload_keys"],
        },
        exclude_paths=(args.out / "benchmark_provenance.json",),
    )
    write_manifest(benchmark_provenance, args.out / "benchmark_provenance.json")
    print(f"Baseline benchmark complete for {args.dataset} -> {args.out}")


if __name__ == "__main__":
    main()

