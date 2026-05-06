from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from j2bench.j1_contract import validate_j1_metadata, validate_j1_prepared_bundle


def _base_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_id": "j1",
                "source_dataset_id": "db10",
                "subject_id": "S101",
                "group": "amputee",
                "session": "session1",
                "day": "D01",
                "task": "grasp",
                "record_id": "rec_a",
                "episode_id": "ep_a",
                "label": 0,
                "label_raw": 1,
                "prefix_time_s": 0.2,
                "timestamp_s": 0.2,
            },
            {
                "dataset_id": "j1",
                "source_dataset_id": "db10",
                "subject_id": "S101",
                "group": "amputee",
                "session": "session1",
                "day": "D01",
                "task": "grasp",
                "record_id": "rec_a",
                "episode_id": "ep_a",
                "label": 0,
                "label_raw": 1,
                "prefix_time_s": 0.4,
                "timestamp_s": 0.4,
            },
            {
                "dataset_id": "j1",
                "source_dataset_id": "hyser",
                "subject_id": "subject01",
                "group": "able_bodied",
                "session": "session2",
                "day": "D02",
                "task": "gesture",
                "record_id": "rec_b",
                "episode_id": "ep_b",
                "label": 1,
                "label_raw": 2,
                "prefix_time_s": 0.2,
                "timestamp_s": 5.2,
            },
            {
                "dataset_id": "j1",
                "source_dataset_id": "hyser",
                "subject_id": "subject01",
                "group": "able_bodied",
                "session": "session2",
                "day": "D02",
                "task": "gesture",
                "record_id": "rec_b",
                "episode_id": "ep_b",
                "label": 1,
                "label_raw": 2,
                "prefix_time_s": 0.4,
                "timestamp_s": 5.4,
            },
        ]
    )


def _write_bundle(
    root,
    *,
    metadata: pd.DataFrame,
    user_features: np.ndarray | None = None,
    assist_features: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    label_vocab: np.ndarray | None = None,
    sequence_payloads: dict[str, np.ndarray] | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    n_rows = len(metadata)
    metadata.to_csv(root / "metadata.csv", index=False)
    np.save(
        root / "user_features.npy",
        np.asarray(
            user_features
            if user_features is not None
            else np.arange(n_rows * 2, dtype=np.float32).reshape(n_rows, 2)
        ),
    )
    np.save(
        root / "assist_features.npy",
        np.asarray(
            assist_features
            if assist_features is not None
            else np.arange(n_rows * 3, dtype=np.float32).reshape(n_rows, 3)
        ),
    )
    np.save(
        root / "labels.npy",
        np.asarray(labels if labels is not None else metadata["label"].to_numpy(dtype=np.int64)),
    )
    np.save(
        root / "label_vocab.npy",
        np.asarray(label_vocab if label_vocab is not None else np.array([1, 2], dtype=np.int64)),
    )
    (root / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    if sequence_payloads:
        sequence_dir = root / "sequence_payloads"
        sequence_dir.mkdir(exist_ok=True)
        manifest: dict[str, dict[str, object]] = {}
        for name, payload in sequence_payloads.items():
            arr = np.asarray(payload)
            np.save(sequence_dir / f"{name}.npy", arr)
            manifest[name] = {
                "shape": [int(dim) for dim in arr.shape],
                "dtype": str(arr.dtype),
            }
        (sequence_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )


def test_validate_j1_prepared_bundle_returns_manifest(tmp_path) -> None:
    root = tmp_path / "bundle"
    metadata = _base_metadata()
    _write_bundle(
        root,
        metadata=metadata,
        sequence_payloads={"user_emg": np.ones((len(metadata), 2, 3), dtype=np.float32)},
    )

    manifest = validate_j1_prepared_bundle(root)

    assert manifest["status"] == "PASS"
    assert manifest["n_records"] == 4
    assert manifest["n_episodes"] == 2
    assert manifest["n_subjects"] == 2
    assert manifest["source_dataset_counts"] == {"db10": 2, "hyser": 2}
    assert manifest["label_coverage"] == {
        "n_vocab": 2,
        "n_observed": 2,
        "unused_label_raw": [],
    }
    assert manifest["prefixes_per_episode"] == {"min": 2, "max": 2, "mean": 2.0}
    assert manifest["feature_shapes"] == {"user_features": [4, 2], "assist_features": [4, 3]}
    assert manifest["sequence_payloads"] == {"count": 1, "names": ["user_emg"]}


def test_validate_j1_prepared_bundle_requires_source_dataset_id(tmp_path) -> None:
    root = tmp_path / "bundle"
    metadata = _base_metadata().drop(columns=["source_dataset_id"])
    _write_bundle(root, metadata=metadata)

    with pytest.raises(ValueError, match="source_dataset_id"):
        validate_j1_prepared_bundle(root)


def test_validate_j1_prepared_bundle_rejects_nonfinite_features(tmp_path) -> None:
    root = tmp_path / "bundle"
    metadata = _base_metadata()
    user_features = np.arange(len(metadata) * 2, dtype=np.float32).reshape(len(metadata), 2)
    user_features[0, 0] = np.nan
    _write_bundle(root, metadata=metadata, user_features=user_features)

    with pytest.raises(ValueError, match="user_features must contain only finite values"):
        validate_j1_prepared_bundle(root)


def test_validate_j1_metadata_rejects_duplicate_records() -> None:
    metadata = _base_metadata()
    metadata = pd.concat([metadata, metadata.iloc[[1]]], ignore_index=True)
    labels = np.array([0, 0, 1, 1, 0], dtype=np.int64)

    with pytest.raises(ValueError, match="duplicate J1 records detected"):
        validate_j1_metadata(metadata, labels=labels, label_vocab=np.array([1, 2], dtype=np.int64))


def test_validate_j1_metadata_rejects_non_monotone_prefixes() -> None:
    metadata = _base_metadata()
    metadata.loc[1, "prefix_time_s"] = 0.1
    metadata.loc[1, "timestamp_s"] = 0.1

    with pytest.raises(ValueError, match="metadata.prefix_time_s must be strictly increasing"):
        validate_j1_metadata(
            metadata,
            labels=metadata["label"].to_numpy(dtype=np.int64),
            label_vocab=np.array([1, 2], dtype=np.int64),
        )


def test_validate_j1_metadata_rejects_label_mismatch() -> None:
    metadata = _base_metadata()

    with pytest.raises(ValueError, match="metadata.label does not match labels"):
        validate_j1_metadata(
            metadata,
            labels=np.array([0, 0, 1, 0], dtype=np.int64),
            label_vocab=np.array([1, 2], dtype=np.int64),
        )


def test_validate_j1_metadata_rejects_non_default_index() -> None:
    metadata = _base_metadata()
    metadata.index = pd.Index([10, 11, 12, 13], dtype=int)

    with pytest.raises(ValueError, match="RangeIndex"):
        validate_j1_metadata(
            metadata,
            labels=metadata["label"].to_numpy(dtype=np.int64),
            label_vocab=np.array([1, 2], dtype=np.int64),
        )

