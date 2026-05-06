from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from j1_fixtures import (  # noqa: E402
    J1PreparedBundleFixture,
    j1_prepared_bundle,
    j1_prepared_bundle_missing_success,
)
from j2bench.artifact_schemas import (  # noqa: E402
    ArtifactSchemaError,
    summarize_prepared_bundle,
    validate_j1_benchmark_manifest,
    validate_j1_prepared_bundle_summary,
    validate_j1_publication_manifest,
)
from j2bench.j1_contract import validate_j1_prepared_bundle  # noqa: E402


def _load_script_module(script_name: str):
    module_path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_j1_end_to_end_regression_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    j1_prepared_bundle: J1PreparedBundleFixture,
) -> None:
    baseline_module = _load_script_module("run_real_baseline.py")
    split_report_module = _load_script_module("build_split_report.py")
    publication_module = _load_script_module("run_publication_extension.py")

    contract_manifest = validate_j1_prepared_bundle(j1_prepared_bundle.root)
    assert contract_manifest["status"] == "PASS"
    assert contract_manifest["n_records"] == j1_prepared_bundle.n_rows
    assert contract_manifest["n_episodes"] == j1_prepared_bundle.n_episodes
    assert contract_manifest["n_subjects"] == j1_prepared_bundle.n_subjects
    assert contract_manifest["source_dataset_counts"] == j1_prepared_bundle.expected_source_dataset_counts
    assert contract_manifest["group_counts"] == j1_prepared_bundle.expected_group_counts
    assert contract_manifest["label_counts"] == j1_prepared_bundle.expected_label_counts
    assert contract_manifest["feature_shapes"] == {"user_features": [32, 4], "assist_features": [32, 5]}
    assert contract_manifest["prefixes_per_episode"] == {"min": 2, "max": 2, "mean": 2.0}
    assert set(contract_manifest["sequence_payloads"]["names"]) == set(j1_prepared_bundle.expected_sequence_payload_keys)

    prepared_summary = validate_j1_prepared_bundle_summary(summarize_prepared_bundle(j1_prepared_bundle.root))
    assert prepared_summary["n_rows"] == j1_prepared_bundle.n_rows
    assert prepared_summary["has_sequence_payloads"] is True
    assert prepared_summary["sequence_payload_keys"] == list(j1_prepared_bundle.expected_sequence_payload_keys)

    benchmark_out = tmp_path / "benchmark"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_real_baseline.py",
            "--dataset",
            "j1",
            "--prepared-root",
            str(j1_prepared_bundle.root),
            "--out",
            str(benchmark_out),
            "--user-model",
            "logistic",
            "--assist-model",
            "logistic",
            "--split-family",
            "j1_subject_logo",
            "--max-splits-per-family",
            "1",
        ],
    )
    baseline_module.main()

    benchmark_manifest_path = benchmark_out / "benchmark_manifest.json"
    metrics_path = benchmark_out / "metrics_by_policy_unit.csv"
    aggregate_path = benchmark_out / "aggregate_policy_metrics.csv"
    tradeoff_path = benchmark_out / "policy_tradeoff.png"
    assert benchmark_manifest_path.is_file()
    assert metrics_path.is_file()
    assert aggregate_path.is_file()
    assert tradeoff_path.is_file()

    benchmark_manifest = validate_j1_benchmark_manifest(benchmark_manifest_path)
    assert benchmark_manifest["dataset"] == "j1"
    assert benchmark_manifest["user_model"] == "logistic"
    assert benchmark_manifest["assist_model"] == "logistic"
    assert benchmark_manifest["n_rows"] == j1_prepared_bundle.n_rows
    assert benchmark_manifest["available_sequence_payload_roles"] == list(j1_prepared_bundle.expected_sequence_roles)
    assert set(benchmark_manifest["sequence_payload_sources"]) == set(j1_prepared_bundle.expected_sequence_roles)
    assert benchmark_manifest["split_family_allowlist"] == ["j1_subject_logo"]
    assert benchmark_manifest["max_splits_per_family"] == 1

    metrics_df = pd.read_csv(metrics_path)
    aggregate_df = pd.read_csv(aggregate_path)
    assert not metrics_df.empty
    assert not aggregate_df.empty
    assert set(metrics_df["split_family"].astype(str)) == {"j1_subject_logo"}
    assert set(aggregate_df["split_family"].astype(str)) == {"j1_subject_logo"}
    assert metrics_df["policy"].astype(str).str.startswith("plain_conf_threshold_matched_tau_").any()

    report_out = tmp_path / "split_report"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_split_report.py",
            "--prepared-root",
            str(j1_prepared_bundle.root),
            "--out-root",
            str(report_out),
        ],
    )
    split_report_module.main()

    split_report_json = report_out / "split_report.json"
    split_report_md = report_out / "split_report.md"
    assert split_report_json.is_file()
    assert split_report_md.is_file()

    split_report_payload = json.loads(split_report_json.read_text(encoding="utf-8"))
    split_report_markdown = split_report_md.read_text(encoding="utf-8")
    family_ids = {row["split_family"] for row in split_report_payload["families"]}
    assert split_report_payload["status"] == "ok"
    assert split_report_payload["inputs"]["prepared_success_file_present"] is True
    assert split_report_payload["metadata"]["rows"] == j1_prepared_bundle.n_rows
    assert split_report_payload["summary"]["critical_overlap_checks_passed"] is True
    assert split_report_payload["summary"]["forward_order_checks_passed"] is True
    assert set(j1_prepared_bundle.expected_split_families).issubset(family_ids)
    assert "# J1 Split Leakage Report" in split_report_markdown
    assert "Forward Order Checks" in split_report_markdown
    assert "j1_source_dataset_holdout\\|db10" in split_report_markdown

    publication_out = tmp_path / "publication"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_publication_extension.py",
            "--dataset",
            "j1",
            "--prepared-root",
            str(j1_prepared_bundle.root),
            "--out-root",
            str(publication_out),
            "--user-model",
            "logistic",
            "--assist-model",
            "logistic",
            "--pairwise-repeats",
            "16",
        ],
    )
    publication_module.main()

    publication_manifest_path = publication_out / "j1_publication_extension_manifest.json"
    expected_publication_files = [
        publication_manifest_path,
        publication_out / "j1_publication_metrics_by_policy_unit.csv",
        publication_out / "j1_publication_aggregate_policy_metrics.csv",
        publication_out / "j1_confidence_gate_match_summary.csv",
        publication_out / "j1_earliest_safe_by_episode.csv",
        publication_out / "j1_earliest_safe_by_unit.csv",
        publication_out / "j1_earliest_safe_summary.csv",
        publication_out / "j1_confidence_gate_pairwise.csv",
        publication_out / "j1_confidence_gate_iso_budget_unit_deltas.csv",
        publication_out / "j1_confidence_gate_iso_budget_pairwise.csv",
        publication_out / "j1_plain_conf_exact_budget_selection_by_unit.csv",
        publication_out / "j1_confidence_gate_exact_budget_pairwise.csv",
        publication_out / "j1_earliest_safe_pairwise.csv",
        publication_out / "j1_earliest_safe_iso_budget_unit_deltas.csv",
        publication_out / "j1_earliest_safe_iso_budget_pairwise.csv",
        publication_out / "j1_earliest_safe_exact_budget_pairwise.csv",
        publication_out / "j1_earliest_safe.png",
        publication_out / "j1_matched_budget_ablation.png",
    ]
    for path in expected_publication_files:
        assert path.is_file()

    publication_manifest = validate_j1_publication_manifest(publication_manifest_path)
    assert publication_manifest["dataset"] == "j1"
    assert publication_manifest["user_model"] == "logistic"
    assert publication_manifest["assist_model"] == "logistic"
    assert publication_manifest["pairwise_repeats"] == 16
    assert publication_manifest["trace_count"] > 0
    assert publication_manifest["decision_count"] >= publication_manifest["trace_count"]
    assert set(publication_manifest["split_families"]) == set(j1_prepared_bundle.expected_split_families)


def test_j1_end_to_end_fail_closed_on_incomplete_or_tampered_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    j1_prepared_bundle: J1PreparedBundleFixture,
    j1_prepared_bundle_missing_success: J1PreparedBundleFixture,
) -> None:
    baseline_module = _load_script_module("run_real_baseline.py")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_real_baseline.py",
            "--dataset",
            "j1",
            "--prepared-root",
            str(j1_prepared_bundle_missing_success.root),
            "--out",
            str(tmp_path / "broken_benchmark"),
            "--user-model",
            "logistic",
            "--assist-model",
            "logistic",
        ],
    )
    with pytest.raises(FileNotFoundError, match="_SUCCESS"):
        baseline_module.main()

    tampered_summary = summarize_prepared_bundle(j1_prepared_bundle.root)
    tampered_summary["sequence_payload_keys"] = []
    with pytest.raises(ArtifactSchemaError, match="sequence_payload_keys must be non-empty"):
        validate_j1_prepared_bundle_summary(tampered_summary)

    benchmark_out = tmp_path / "benchmark"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_real_baseline.py",
            "--dataset",
            "j1",
            "--prepared-root",
            str(j1_prepared_bundle.root),
            "--out",
            str(benchmark_out),
            "--user-model",
            "logistic",
            "--assist-model",
            "logistic",
            "--split-family",
            "j1_subject_logo",
            "--max-splits-per-family",
            "1",
        ],
    )
    baseline_module.main()

    tampered_benchmark_manifest = json.loads((benchmark_out / "benchmark_manifest.json").read_text(encoding="utf-8"))
    tampered_benchmark_manifest["sequence_payload_sources"] = {
        "assist": tampered_benchmark_manifest["sequence_payload_sources"]["assist"]
    }
    with pytest.raises(ArtifactSchemaError, match="available_sequence_payload_roles"):
        validate_j1_benchmark_manifest(tampered_benchmark_manifest)

