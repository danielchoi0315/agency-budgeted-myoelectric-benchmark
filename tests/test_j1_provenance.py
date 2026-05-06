from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from j2bench import provenance


def _write_prepared_bundle(root) -> None:
    root.mkdir(parents=True, exist_ok=True)
    metadata = pd.DataFrame(
        [
            {
                "dataset_id": "j1",
                "subject_id": "S101",
                "group": "amputee",
                "label_raw": 1,
                "label": 0,
            },
            {
                "dataset_id": "j1",
                "subject_id": "S010",
                "group": "able_bodied",
                "label_raw": 2,
                "label": 1,
            },
        ]
    )
    metadata.to_csv(root / "metadata.csv", index=False)
    np.save(root / "user_features.npy", np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    np.save(
        root / "assist_features.npy",
        np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], dtype=np.float32),
    )
    np.save(root / "labels.npy", np.array([0, 1], dtype=np.int64))
    np.save(root / "label_vocab.npy", np.array([1, 2], dtype=np.int64))
    sequence_dir = root / "sequence_payloads"
    sequence_dir.mkdir()
    np.save(sequence_dir / "user_emg.npy", np.arange(12, dtype=np.float32).reshape(2, 2, 3))
    (sequence_dir / "manifest.json").write_text(
        json.dumps({"user_emg": {"shape": [2, 2, 3], "dtype": "float32"}}, indent=2),
        encoding="utf-8",
    )
    (root / "_SUCCESS").write_text("ok\n", encoding="utf-8")


def test_build_prepared_bundle_manifest_captures_hashes_rows_and_runtime(
    monkeypatch,
    tmp_path,
) -> None:
    prepared_root = tmp_path / "prepared"
    _write_prepared_bundle(prepared_root)
    config_path = tmp_path / "config" / "config.yaml"
    config_path.parent.mkdir()
    config_path.write_text("seed: 20260415\n", encoding="utf-8")

    def fake_version(name: str) -> str:
        if name == "numpy":
            return "9.9.9"
        raise provenance.importlib_metadata.PackageNotFoundError

    monkeypatch.setattr(provenance.importlib_metadata, "version", fake_version)

    manifest = provenance.build_prepared_bundle_manifest(
        prepared_root,
        config_paths=[config_path],
        seed_metadata={"bootstrap_seed": 11, "split_seed": 20260415},
        package_names=["missing-pkg", "numpy"],
        path_root=tmp_path,
    )

    assert manifest["artifact_type"] == "prepared_bundle"
    assert manifest["dataset_id"] == "j1"
    assert manifest["artifact_root"] == "prepared"
    assert manifest["row_counts"] == {
        "assist_features.npy": 2,
        "label_vocab.npy": 2,
        "labels.npy": 2,
        "metadata.csv": 2,
        "sequence_payloads/user_emg.npy": 2,
        "user_features.npy": 2,
    }
    assert manifest["seed_metadata"] == {"bootstrap_seed": 11, "split_seed": 20260415}
    assert manifest["runtime"]["package_versions"] == {
        "found": {"numpy": "9.9.9"},
        "missing": ["missing-pkg"],
    }

    files_by_path = {record["path"]: record for record in manifest["files"]}
    assert sorted(files_by_path) == [
        "_SUCCESS",
        "assist_features.npy",
        "label_vocab.npy",
        "labels.npy",
        "metadata.csv",
        "sequence_payloads/manifest.json",
        "sequence_payloads/user_emg.npy",
        "user_features.npy",
    ]
    assert files_by_path["_SUCCESS"]["sha256"] == hashlib.sha256(
        (prepared_root / "_SUCCESS").read_bytes()
    ).hexdigest()
    assert files_by_path["user_features.npy"]["shape"] == [2, 2]
    assert files_by_path["sequence_payloads/user_emg.npy"]["shape"] == [2, 2, 3]
    assert manifest["config_files"] == [
        {
            "path": "config/config.yaml",
            "kind": "yaml",
            "exists": True,
            "size_bytes": config_path.stat().st_size,
            "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "row_count": None,
            "shape": None,
        }
    ]
    assert manifest["runtime"]["python"]["version"]
    assert manifest["runtime"]["platform"]["system"]


def test_build_output_manifest_is_deterministic_and_excludes_written_manifest(tmp_path) -> None:
    prepared_root = tmp_path / "prepared"
    _write_prepared_bundle(prepared_root)
    output_root = tmp_path / "results" / "j1"
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"dataset_id": "j1", "split_id": "split_a", "policy": "user_only"},
            {"dataset_id": "j1", "split_id": "split_b", "policy": "assist_only"},
        ]
    ).to_csv(output_root / "metrics_by_policy_unit.csv", index=False)
    pd.DataFrame([{"dataset_id": "j1", "policy": "user_only", "macro_f1": 0.5}]).to_csv(
        output_root / "aggregate_policy_metrics.csv",
        index=False,
    )
    (output_root / "benchmark_manifest.json").write_text(
        json.dumps({"dataset": "j1", "n_rows": 2}, indent=2),
        encoding="utf-8",
    )
    (output_root / "anchor_report.md").write_text("# Report\n", encoding="utf-8")
    config_path = tmp_path / "config" / "benchmark.yaml"
    config_path.parent.mkdir(exist_ok=True)
    config_path.write_text("user_model: logistic\n", encoding="utf-8")

    manifest = provenance.build_output_manifest(
        output_root,
        output_kind="benchmark",
        dataset_id="j1",
        config_paths=[config_path],
        input_paths=[prepared_root],
        seed_metadata={"model_seed": 7},
        package_names=(),
        path_root=tmp_path,
        extra_metadata={"stage": "baseline"},
    )

    assert manifest["artifact_type"] == "benchmark_output"
    assert manifest["artifact_root"] == "results/j1"
    assert manifest["row_counts"] == {
        "aggregate_policy_metrics.csv": 1,
        "metrics_by_policy_unit.csv": 2,
    }
    assert manifest["extra_metadata"] == {"stage": "baseline"}
    assert manifest["runtime"]["package_versions"] == {"found": {}, "missing": []}
    assert [record["path"] for record in manifest["files"]] == sorted(
        record["path"] for record in manifest["files"]
    )
    input_paths = {record["path"] for record in manifest["input_files"]}
    assert "prepared/_SUCCESS" in input_paths
    assert "prepared/metadata.csv" in input_paths

    provenance_path = output_root / "provenance.json"
    provenance.write_manifest(manifest, provenance_path)
    assert provenance_path.read_text(encoding="utf-8").endswith("\n")
    assert json.loads(provenance_path.read_text(encoding="utf-8")) == manifest

    rebuilt = provenance.build_output_manifest(
        output_root,
        output_kind="benchmark",
        dataset_id="j1",
        config_paths=[config_path],
        input_paths=[prepared_root],
        seed_metadata={"model_seed": 7},
        package_names=(),
        path_root=tmp_path,
        extra_metadata={"stage": "baseline"},
        exclude_paths=[provenance_path],
    )

    assert rebuilt == manifest
    assert "provenance.json" not in {record["path"] for record in rebuilt["files"]}

