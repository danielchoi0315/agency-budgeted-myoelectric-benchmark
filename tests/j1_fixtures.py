from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from j2bench.realdata import PreparedDataset, save_prepared_dataset


_LABEL_VOCAB = (1, 2)
_PREFIX_GRID_S = (0.2, 0.5)
_DAY_LAYOUT = (("D01", "session1"), ("D02", "session2"))
_SUBJECT_LAYOUT = (
    ("db10", "S010", "able_bodied", -2.0),
    ("db10", "S011", "able_bodied", -1.0),
    ("hyser", "S101", "amputee", 1.0),
    ("hyser", "S102", "amputee", 2.0),
)
_EXPECTED_SPLIT_FAMILIES = (
    "j1_able_to_amputee",
    "j1_amputee_loso",
    "j1_forward_day_logo",
    "j1_mixed_to_amputee",
    "j1_source_dataset_holdout",
    "j1_subject_logo",
)


@dataclass(frozen=True)
class J1PreparedBundleFixture:
    root: Path
    bundle: PreparedDataset
    expected_split_families: tuple[str, ...]
    expected_sequence_roles: tuple[str, ...]
    expected_sequence_payload_keys: tuple[str, ...]
    expected_source_dataset_counts: dict[str, int]
    expected_group_counts: dict[str, int]
    expected_label_counts: dict[str, int]
    n_rows: int
    n_episodes: int
    n_subjects: int

    def copy_to(self, root: Path, *, keep_success: bool = True) -> J1PreparedBundleFixture:
        shutil.copytree(self.root, root)
        success_path = root / "_SUCCESS"
        if not keep_success and success_path.exists():
            success_path.unlink()
        return replace(self, root=root)


def create_synthetic_j1_prepared_bundle(
    root: Path,
    *,
    include_sequences: bool = True,
) -> J1PreparedBundleFixture:
    rows: list[dict[str, object]] = []
    user_rows: list[np.ndarray] = []
    assist_rows: list[np.ndarray] = []
    sequence_rows: dict[str, list[np.ndarray]] = {
        "user_emg": [],
        "user_time_mask": [],
        "assist_emg": [],
        "assist_time_mask": [],
        "assist_context": [],
        "assist_context_mask": [],
    }

    for source_index, (source_dataset_id, subject_id, group, base) in enumerate(_SUBJECT_LAYOUT):
        for day_index, (day, session) in enumerate(_DAY_LAYOUT, start=1):
            for label_index, label_raw in enumerate(_LABEL_VOCAB):
                episode_id = f"{source_dataset_id}_{subject_id}_{day}_class{label_raw}"
                record_id = episode_id
                for prefix_index, prefix_time_s in enumerate(_PREFIX_GRID_S, start=1):
                    timestamp_s = float(day_index * 10.0 + prefix_time_s)
                    signal = float(base + label_index * 4.0 + day_index * 0.25 + prefix_index * 0.05)
                    rows.append(
                        {
                            "dataset_id": "j1",
                            "source_dataset_id": source_dataset_id,
                            "subject_id": subject_id,
                            "group": group,
                            "session": session,
                            "day": day,
                            "task": "grasp",
                            "record_id": record_id,
                            "episode_id": episode_id,
                            "label": label_index,
                            "label_raw": label_raw,
                            "prefix_time_s": prefix_time_s,
                            "timestamp_s": timestamp_s,
                        }
                    )
                    user_rows.append(
                        np.array(
                            [
                                signal,
                                float(label_index),
                                float(day_index),
                                prefix_time_s,
                            ],
                            dtype=np.float32,
                        )
                    )
                    assist_rows.append(
                        np.array(
                            [
                                signal * 1.5,
                                float(source_index),
                                float(day_index),
                                prefix_time_s,
                                signal * prefix_time_s,
                            ],
                            dtype=np.float32,
                        )
                    )
                    if include_sequences:
                        user_emg = np.column_stack(
                            [
                                np.linspace(signal, signal + 0.3, 4, dtype=np.float32),
                                np.linspace(prefix_time_s, prefix_time_s + 0.3, 4, dtype=np.float32),
                            ]
                        )
                        assist_emg = np.column_stack(
                            [
                                np.linspace(signal * 1.2, signal * 1.2 + 0.4, 5, dtype=np.float32),
                                np.linspace(float(label_index), float(label_index) + 0.4, 5, dtype=np.float32),
                                np.linspace(float(day_index), float(day_index) + 0.4, 5, dtype=np.float32),
                            ]
                        )
                        sequence_rows["user_emg"].append(user_emg.astype(np.float32))
                        sequence_rows["user_time_mask"].append(np.ones((4, 1), dtype=np.float32))
                        sequence_rows["assist_emg"].append(assist_emg.astype(np.float32))
                        sequence_rows["assist_time_mask"].append(np.ones((5, 1), dtype=np.float32))
                        sequence_rows["assist_context"].append(
                            np.array(
                                [[[float(source_index), float(day_index), float(label_index)]]],
                                dtype=np.float32,
                            ).reshape(1, 3)
                        )
                        sequence_rows["assist_context_mask"].append(np.ones((1, 1), dtype=np.float32))

    metadata = pd.DataFrame(rows).reset_index(drop=True)
    labels = metadata["label"].to_numpy(dtype=np.int64)
    sequence_payloads = None
    if include_sequences:
        sequence_payloads = {
            name: np.stack(payload_rows).astype(np.float32)
            for name, payload_rows in sequence_rows.items()
        }

    bundle = PreparedDataset(
        metadata=metadata,
        user_features=np.vstack(user_rows).astype(np.float32),
        assist_features=np.vstack(assist_rows).astype(np.float32),
        labels=labels,
        label_vocab=list(_LABEL_VOCAB),
        sequence_payloads=sequence_payloads,
    )
    save_prepared_dataset(bundle, root)

    return J1PreparedBundleFixture(
        root=root,
        bundle=bundle,
        expected_split_families=_EXPECTED_SPLIT_FAMILIES,
        expected_sequence_roles=("assist", "user") if include_sequences else (),
        expected_sequence_payload_keys=tuple(sorted(sequence_payloads or {})),
        expected_source_dataset_counts={"db10": 16, "hyser": 16},
        expected_group_counts={"able_bodied": 16, "amputee": 16},
        expected_label_counts={"1": 16, "2": 16},
        n_rows=int(len(metadata)),
        n_episodes=16,
        n_subjects=4,
    )


@pytest.fixture
def j1_prepared_bundle(tmp_path: Path) -> J1PreparedBundleFixture:
    return create_synthetic_j1_prepared_bundle(tmp_path / "j1_prepared")


@pytest.fixture
def j1_prepared_bundle_missing_success(
    tmp_path: Path,
    j1_prepared_bundle: J1PreparedBundleFixture,
) -> J1PreparedBundleFixture:
    return j1_prepared_bundle.copy_to(tmp_path / "j1_prepared_missing_success", keep_success=False)


__all__ = [
    "J1PreparedBundleFixture",
    "create_synthetic_j1_prepared_bundle",
    "j1_prepared_bundle",
    "j1_prepared_bundle_missing_success",
]

