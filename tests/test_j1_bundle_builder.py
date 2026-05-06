from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from j2bench.realdata import PreparedDataset, load_prepared_dataset, save_prepared_dataset


def _load_build_j1_bundle_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "build_j1_bundle.py"
    spec = importlib.util.spec_from_file_location("build_j1_bundle_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_prepared_root(
    root: Path,
    *,
    metadata_rows: list[dict[str, object]],
    user_features: np.ndarray,
    assist_features: np.ndarray,
    sequence_payloads: dict[str, np.ndarray] | None = None,
) -> Path:
    metadata = pd.DataFrame(metadata_rows)
    label_vocab = sorted(int(label) for label in metadata["label_raw"].astype(int).unique())
    label_map = {label: idx for idx, label in enumerate(label_vocab)}
    metadata = metadata.copy()
    metadata["label"] = metadata["label_raw"].astype(int).map(label_map).astype(int)
    bundle = PreparedDataset(
        metadata=metadata,
        user_features=np.asarray(user_features, dtype=np.float32),
        assist_features=np.asarray(assist_features, dtype=np.float32),
        labels=metadata["label"].to_numpy(dtype=int),
        label_vocab=label_vocab,
        sequence_payloads=(
            None
            if not sequence_payloads
            else {name: np.asarray(payload, dtype=np.float32) for name, payload in sequence_payloads.items()}
        ),
    )
    save_prepared_dataset(bundle, root)
    return root


def _write_spec(path: Path, spec: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return path


def _minimal_source_row(
    dataset_id: str,
    label_raw: int,
    *,
    subject_id: str = "subject01",
    group: str = "able_bodied",
    session: str = "session1",
    day: str = "D01",
    task: str = "grasp",
    episode_id: str = "episode_a",
    record_id: str = "record_a",
    prefix_time_s: float = 0.2,
    timestamp_s: float = 0.2,
) -> dict[str, object]:
    return {
        "dataset_id": dataset_id,
        "subject_id": subject_id,
        "group": group,
        "session": session,
        "day": day,
        "task": task,
        "episode_id": episode_id,
        "record_id": record_id,
        "label_raw": label_raw,
        "prefix_time_s": prefix_time_s,
        "timestamp_s": timestamp_s,
    }


def test_build_j1_bundle_single_source_cli(tmp_path) -> None:
    module = _load_build_j1_bundle_module()
    source_root = _write_prepared_root(
        tmp_path / "db10_source",
        metadata_rows=[
            {
                "dataset_id": "handoff_bundle",
                "source_dataset_id": "db10",
                "subject_id": "S101",
                "group": "amputee",
                "session": "session1",
                "day": "D01",
                "task": "grasp",
                "episode_id": "ep_a",
                "record_id": "rec_a",
                "label_raw": 10,
                "prefix_time_s": 0.2,
                "timestamp_s": 0.2,
            },
            {
                "dataset_id": "handoff_bundle",
                "source_dataset_id": "db10",
                "subject_id": "S010",
                "group": "able_bodied",
                "session": "session1",
                "day": "D01",
                "task": "grasp",
                "episode_id": "ep_b",
                "record_id": "rec_b",
                "label_raw": 20,
                "prefix_time_s": 0.4,
                "timestamp_s": 0.4,
            },
        ],
        user_features=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        assist_features=np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], dtype=np.float32),
    )
    spec_path = _write_spec(
        tmp_path / "j1_single.yaml",
        {
            "version": 1,
            "bundle": {"dataset_id": "j1"},
            "sources": [
                {
                    "name": "db10_open",
                    "prepared_root": str(source_root),
                    "feature_schema": {
                        "user_features": "db10_user_v1",
                        "assist_features": "db10_assist_v1",
                    },
                    "label_map": {"10": 1, "20": 2},
                }
            ],
        },
    )
    out_root = tmp_path / "j1_out"

    assert module.main(["--spec", str(spec_path), "--out", str(out_root)]) == 0

    built = load_prepared_dataset(out_root)
    assert set(built.metadata["dataset_id"].astype(str)) == {"j1"}
    assert set(built.metadata["source_dataset_id"].astype(str)) == {"db10"}
    assert built.metadata["source_label_raw"].astype(int).tolist() == [10, 20]
    assert built.metadata["label_raw"].astype(int).tolist() == [1, 2]
    assert set(built.metadata["source_prepared_root"].astype(str)) == {str(source_root)}
    np.testing.assert_array_equal(
        built.user_features,
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        built.assist_features,
        np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], dtype=np.float32),
    )
    assert built.label_vocab == [1, 2]


def test_build_j1_bundle_multi_source_with_explicit_projection(tmp_path) -> None:
    module = _load_build_j1_bundle_module()
    root_a = _write_prepared_root(
        tmp_path / "source_a",
        metadata_rows=[
            {
                "dataset_id": "db10",
                "subject_id": "S101",
                "group": "amputee",
                "session": "session1",
                "day": "D01",
                "task": "grasp",
                "episode_id": "ep_a",
                "record_id": "rec_a",
                "label_raw": 1,
                "prefix_time_s": 0.2,
                "timestamp_s": 0.2,
            }
        ],
        user_features=np.array([[1.0, 2.0]], dtype=np.float32),
        assist_features=np.array([[10.0, 11.0, 12.0]], dtype=np.float32),
    )
    root_b = _write_prepared_root(
        tmp_path / "source_b",
        metadata_rows=[
            {
                "dataset_id": "hyser",
                "subject_id": "subject01",
                "group": "able_bodied",
                "session": "session1",
                "day": "D01",
                "task": "dynamic",
                "episode_id": "ep_b",
                "record_id": "rec_b",
                "label_raw": 10,
                "prefix_time_s": 0.1,
                "timestamp_s": 0.1,
            },
            {
                "dataset_id": "hyser",
                "subject_id": "subject02",
                "group": "able_bodied",
                "session": "session1",
                "day": "D01",
                "task": "dynamic",
                "episode_id": "ep_c",
                "record_id": "rec_c",
                "label_raw": 20,
                "prefix_time_s": 0.3,
                "timestamp_s": 0.3,
            },
        ],
        user_features=np.array([[100.0, 101.0, 102.0, 103.0], [200.0, 201.0, 202.0, 203.0]], dtype=np.float32),
        assist_features=np.array(
            [[300.0, 301.0, 302.0, 303.0, 304.0], [400.0, 401.0, 402.0, 403.0, 404.0]],
            dtype=np.float32,
        ),
    )
    spec_path = _write_spec(
        tmp_path / "j1_multi.yaml",
        {
            "version": 1,
            "bundle": {"dataset_id": "j1"},
            "composition": {
                "strategy": "concatenate",
                "output_feature_schema": {
                    "user_features": "canonical_user_v1",
                    "assist_features": "canonical_assist_v1",
                },
            },
            "sources": [
                {
                    "name": "db10_source",
                    "prepared_root": str(root_a),
                    "feature_schema": {
                        "user_features": "canonical_user_v1",
                        "assist_features": "canonical_assist_v1",
                    },
                    "label_map": {"1": 1},
                },
                {
                    "name": "hyser_source",
                    "prepared_root": str(root_b),
                    "feature_schema": {
                        "user_features": "hyser_user_v2",
                        "assist_features": "hyser_assist_v2",
                    },
                    "label_map": {"10": 1, "20": 2},
                    "projection": {
                        "user_features": {
                            "from_schema": "hyser_user_v2",
                            "to_schema": "canonical_user_v1",
                            "columns": [1, 3],
                        },
                        "assist_features": {
                            "from_schema": "hyser_assist_v2",
                            "to_schema": "canonical_assist_v1",
                            "columns": [0, 2, 4],
                        },
                    },
                },
            ],
        },
    )

    built = module.build_j1_bundle(spec_path)

    assert len(built.metadata) == 3
    assert set(built.metadata["dataset_id"].astype(str)) == {"j1"}
    assert set(built.metadata["source_dataset_id"].astype(str)) == {"db10", "hyser"}
    np.testing.assert_array_equal(
        built.user_features,
        np.array([[1.0, 2.0], [101.0, 103.0], [201.0, 203.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        built.assist_features,
        np.array([[10.0, 11.0, 12.0], [300.0, 302.0, 304.0], [400.0, 402.0, 404.0]], dtype=np.float32),
    )
    assert built.label_vocab == [1, 2]


def test_build_j1_bundle_rejects_incomplete_label_map(tmp_path) -> None:
    module = _load_build_j1_bundle_module()
    source_root = _write_prepared_root(
        tmp_path / "source",
        metadata_rows=[
            _minimal_source_row("db10", 1, episode_id="ep_a", record_id="rec_a", prefix_time_s=0.2, timestamp_s=0.2),
            _minimal_source_row("db10", 2, episode_id="ep_b", record_id="rec_b", prefix_time_s=0.4, timestamp_s=0.4),
        ],
        user_features=np.array([[1.0], [2.0]], dtype=np.float32),
        assist_features=np.array([[3.0], [4.0]], dtype=np.float32),
    )
    spec_path = _write_spec(
        tmp_path / "bad_label_map.yaml",
        {
            "version": 1,
            "bundle": {"dataset_id": "j1"},
            "sources": [
                {
                    "name": "db10_source",
                    "prepared_root": str(source_root),
                    "feature_schema": {
                        "user_features": "db10_user_v1",
                        "assist_features": "db10_assist_v1",
                    },
                    "label_map": {"1": 1},
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="label_map must cover exactly"):
        module.build_j1_bundle(spec_path)


def test_build_j1_bundle_rejects_schema_mismatch_without_projection(tmp_path) -> None:
    module = _load_build_j1_bundle_module()
    root_a = _write_prepared_root(
        tmp_path / "source_a",
        metadata_rows=[_minimal_source_row("db10", 1)],
        user_features=np.array([[1.0, 2.0]], dtype=np.float32),
        assist_features=np.array([[3.0, 4.0, 5.0]], dtype=np.float32),
    )
    root_b = _write_prepared_root(
        tmp_path / "source_b",
        metadata_rows=[_minimal_source_row("hyser", 2, subject_id="subject02", episode_id="ep_b", record_id="rec_b")],
        user_features=np.array([[6.0, 7.0]], dtype=np.float32),
        assist_features=np.array([[8.0, 9.0, 10.0]], dtype=np.float32),
    )
    spec_path = _write_spec(
        tmp_path / "schema_mismatch.yaml",
        {
            "version": 1,
            "bundle": {"dataset_id": "j1"},
            "composition": {
                "strategy": "concatenate",
                "output_feature_schema": {
                    "user_features": "canonical_user_v1",
                    "assist_features": "canonical_assist_v1",
                },
            },
            "sources": [
                {
                    "name": "db10_source",
                    "prepared_root": str(root_a),
                    "feature_schema": {
                        "user_features": "canonical_user_v1",
                        "assist_features": "canonical_assist_v1",
                    },
                    "label_map": {"1": 1},
                },
                {
                    "name": "hyser_source",
                    "prepared_root": str(root_b),
                    "feature_schema": {
                        "user_features": "hyser_user_v1",
                        "assist_features": "hyser_assist_v1",
                    },
                    "label_map": {"2": 2},
                },
            ],
        },
    )

    with pytest.raises(ValueError, match="schema mismatch"):
        module.build_j1_bundle(spec_path)


def test_build_j1_bundle_requires_explicit_multi_source_composition(tmp_path) -> None:
    module = _load_build_j1_bundle_module()
    root_a = _write_prepared_root(
        tmp_path / "source_a",
        metadata_rows=[_minimal_source_row("db10", 1)],
        user_features=np.array([[1.0]], dtype=np.float32),
        assist_features=np.array([[2.0]], dtype=np.float32),
    )
    root_b = _write_prepared_root(
        tmp_path / "source_b",
        metadata_rows=[_minimal_source_row("hyser", 2, subject_id="subject02", episode_id="ep_b", record_id="rec_b")],
        user_features=np.array([[3.0]], dtype=np.float32),
        assist_features=np.array([[4.0]], dtype=np.float32),
    )
    spec_path = _write_spec(
        tmp_path / "missing_composition.yaml",
        {
            "version": 1,
            "bundle": {"dataset_id": "j1"},
            "sources": [
                {
                    "name": "db10_source",
                    "prepared_root": str(root_a),
                    "feature_schema": {
                        "user_features": "shared_user_v1",
                        "assist_features": "shared_assist_v1",
                    },
                    "label_map": {"1": 1},
                },
                {
                    "name": "hyser_source",
                    "prepared_root": str(root_b),
                    "feature_schema": {
                        "user_features": "shared_user_v1",
                        "assist_features": "shared_assist_v1",
                    },
                    "label_map": {"2": 2},
                },
            ],
        },
    )

    with pytest.raises(ValueError, match="multi-source J1 composition requires an explicit spec.composition block"):
        module.build_j1_bundle(spec_path)


def test_build_j1_bundle_rejects_mixed_sequence_and_tabular_sources(tmp_path) -> None:
    module = _load_build_j1_bundle_module()
    root_a = _write_prepared_root(
        tmp_path / "sequence_source",
        metadata_rows=[_minimal_source_row("db10", 1)],
        user_features=np.array([[1.0, 2.0]], dtype=np.float32),
        assist_features=np.array([[3.0, 4.0]], dtype=np.float32),
        sequence_payloads={"user_emg": np.ones((1, 4, 2), dtype=np.float32)},
    )
    root_b = _write_prepared_root(
        tmp_path / "tabular_source",
        metadata_rows=[_minimal_source_row("hyser", 2, subject_id="subject02", episode_id="ep_b", record_id="rec_b")],
        user_features=np.array([[5.0, 6.0]], dtype=np.float32),
        assist_features=np.array([[7.0, 8.0]], dtype=np.float32),
    )
    spec_path = _write_spec(
        tmp_path / "mixed_sequences.yaml",
        {
            "version": 1,
            "bundle": {"dataset_id": "j1"},
            "composition": {
                "strategy": "concatenate",
                "output_feature_schema": {
                    "user_features": "shared_user_v1",
                    "assist_features": "shared_assist_v1",
                },
            },
            "sources": [
                {
                    "name": "sequence_source",
                    "prepared_root": str(root_a),
                    "feature_schema": {
                        "user_features": "shared_user_v1",
                        "assist_features": "shared_assist_v1",
                    },
                    "label_map": {"1": 1},
                },
                {
                    "name": "tabular_source",
                    "prepared_root": str(root_b),
                    "feature_schema": {
                        "user_features": "shared_user_v1",
                        "assist_features": "shared_assist_v1",
                    },
                    "label_map": {"2": 2},
                },
            ],
        },
    )

    with pytest.raises(ValueError, match="cannot merge sequence-enabled and tabular-only prepared roots"):
        module.build_j1_bundle(spec_path)

