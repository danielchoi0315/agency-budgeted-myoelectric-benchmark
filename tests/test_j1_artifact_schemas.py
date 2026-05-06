from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from j2bench.artifact_schemas import (
    ArtifactSchemaError,
    summarize_prepared_bundle,
    validate_benchmark_manifest,
    validate_prepared_bundle_summary,
    validate_publication_manifest,
    validate_scorecard,
)
from j2bench.realdata import PreparedDataset, save_prepared_dataset


def _prepared_bundle_summary_payload() -> dict[str, object]:
    return {
        "dataset": "j1",
        "prepared_root": "/tmp/j1_prepared",
        "success_file": "/tmp/j1_prepared/_SUCCESS",
        "required_files": [
            "metadata.csv",
            "user_features.npy",
            "assist_features.npy",
            "labels.npy",
            "label_vocab.npy",
            "_SUCCESS",
        ],
        "metadata_columns": ["dataset_id", "subject_id", "label_raw", "label"],
        "n_rows": 2,
        "label_vocab_size": 2,
        "user_feature_shape": [2, 3],
        "assist_feature_shape": [2, 5],
        "has_sequence_payloads": True,
        "sequence_payload_keys": ["user_emg"],
    }


def _benchmark_manifest_payload() -> dict[str, object]:
    return {
        "dataset": "j1",
        "prepared_root": "/tmp/j1_prepared",
        "prepared_success_file": "/tmp/j1_prepared/_SUCCESS",
        "user_model": "logistic",
        "assist_model": "logistic",
        "assist_context_modes": [],
        "available_sequence_payload_roles": ["assist", "user"],
        "sequence_payload_sources": {
            "assist": "/tmp/j1_prepared/sequence_payloads/assist",
            "user": "/tmp/j1_prepared/sequence_payloads/user",
        },
        "split_family_allowlist": [],
        "split_id_allowlist": [],
        "max_splits_per_family": None,
        "n_rows": 16,
    }


def _publication_manifest_payload() -> dict[str, object]:
    return {
        "dataset": "j1",
        "prepared_root": "/tmp/j1_prepared",
        "user_model": "logistic",
        "assist_model": "logistic",
        "pairwise_repeats": 2000,
        "trace_count": 40,
        "decision_count": 920,
        "split_families": [
            "j1_able_to_amputee",
            "j1_amputee_loso",
            "j1_mixed_to_amputee",
            "j1_subject_logo",
        ],
    }


def _scorecard_text(*, include_decision_log: bool = True) -> str:
    lines = [
        "# J1-open Scorecard",
        "",
        "## Decision Banner",
        "GO for the frozen benchmark-only claim set.",
        "",
        "## Frozen Candidate and Operating Point",
        "Candidate: logistic/logistic at tau 0.10.",
        "",
        "## Primary DB10 Gate",
        "Worst-family lower bound remains above zero.",
        "",
        "## Safety/Timing Panel",
        "Stable-safe timing is unchanged at the promoted operating point.",
        "",
        "## Robustness Panel",
        "Supporting datasets preserve the matched-budget direction.",
        "",
        "## Boundary Conditions",
        "Claim scope is benchmark-only outside the audited families.",
        "",
        "## Readiness and Reproducibility Status",
        "All figures trace to audited inputs and a clean rerun reproduces them.",
        "",
        "## Candidate Provenance",
        "The candidate was frozen before publication-clean analysis.",
        "",
    ]
    if include_decision_log:
        lines.extend(
            [
                "## Decision Log",
                "2026-04-21: promoted after passing the primary and reproducibility gates.",
                "",
            ]
        )
    return "\n".join(lines)


def test_summarize_prepared_bundle_returns_valid_summary(tmp_path) -> None:
    bundle = PreparedDataset(
        metadata=pd.DataFrame(
            [
                {"dataset_id": "j1", "subject_id": "S101", "label_raw": 1, "label": 0},
                {"dataset_id": "j1", "subject_id": "S102", "label_raw": 2, "label": 1},
            ]
        ),
        user_features=np.ones((2, 3), dtype=np.float32),
        assist_features=np.ones((2, 5), dtype=np.float32),
        labels=np.array([0, 1], dtype=int),
        label_vocab=[1, 2],
        sequence_payloads={"user_emg": np.ones((2, 4, 2), dtype=np.float32)},
    )
    out_dir = tmp_path / "prepared"
    save_prepared_dataset(bundle, out_dir)

    summary = summarize_prepared_bundle(out_dir)

    assert summary["dataset"] == "j1"
    assert summary["n_rows"] == 2
    assert summary["has_sequence_payloads"] is True
    assert summary["sequence_payload_keys"] == ["user_emg"]


def test_validate_prepared_bundle_summary_requires_required_fields() -> None:
    payload = _prepared_bundle_summary_payload()
    payload.pop("assist_feature_shape")

    with pytest.raises(ArtifactSchemaError, match="assist_feature_shape"):
        validate_prepared_bundle_summary(payload)


def test_validate_benchmark_manifest_accepts_valid_payload() -> None:
    manifest = validate_benchmark_manifest(json.dumps(_benchmark_manifest_payload()))

    assert manifest["dataset"] == "j1"
    assert manifest["n_rows"] == 16


def test_validate_benchmark_manifest_fails_closed_on_role_mismatch() -> None:
    payload = _benchmark_manifest_payload()
    payload["sequence_payload_sources"] = {"assist": "/tmp/j1_prepared/sequence_payloads/assist"}

    with pytest.raises(ArtifactSchemaError, match="available_sequence_payload_roles"):
        validate_benchmark_manifest(payload)


def test_validate_publication_manifest_requires_required_fields() -> None:
    payload = _publication_manifest_payload()
    payload.pop("split_families")

    with pytest.raises(ArtifactSchemaError, match="split_families"):
        validate_publication_manifest(payload)


def test_validate_publication_manifest_fails_closed_on_invalid_counts() -> None:
    payload = _publication_manifest_payload()
    payload["decision_count"] = 10

    with pytest.raises(ArtifactSchemaError, match="decision_count"):
        validate_publication_manifest(payload)


def test_validate_scorecard_accepts_required_sections() -> None:
    scorecard = validate_scorecard(_scorecard_text())

    assert scorecard["title"] == "J1-open Scorecard"
    assert "decision log" in scorecard["sections"]


def test_validate_scorecard_requires_all_sections() -> None:
    with pytest.raises(ArtifactSchemaError, match="decision log"):
        validate_scorecard(_scorecard_text(include_decision_log=False))

