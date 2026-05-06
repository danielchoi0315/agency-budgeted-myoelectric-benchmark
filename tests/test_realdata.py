from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.io as sio
import wfdb

try:
    from j2bench.real_benchmark import load_optional_sequence_payloads, run_dataset_benchmark
except ModuleNotFoundError as exc:
    load_optional_sequence_payloads = None
    run_dataset_benchmark = None
    REAL_BENCHMARK_IMPORT_ERROR = exc
else:
    REAL_BENCHMARK_IMPORT_ERROR = None

from j2bench.realdata import (
    DB10_CONTEXT_MODES,
    _db10_context_features,
    PreparedDataset,
    apply_cemhsey_trial_exclusions,
    cemhsey_exclusion_keys,
    db10_subject_group,
    load_prepared_dataset,
    load_cemhsey_failed_trials,
    merge_prepared_datasets,
    parse_grabmyo_record_path,
    parse_cemhsey_grasp_member,
    parse_hyser_label_text,
    prepare_j1_dataset,
    prepare_db10_dataset,
    prepare_grabmyo_dataset,
    save_prepared_dataset,
)


def _require_run_dataset_benchmark():
    if run_dataset_benchmark is None:
        pytest.skip(f"real benchmark import unavailable in this workspace: {REAL_BENCHMARK_IMPORT_ERROR}")
    return run_dataset_benchmark


def _require_load_optional_sequence_payloads():
    if load_optional_sequence_payloads is None:
        pytest.skip(f"real benchmark import unavailable in this workspace: {REAL_BENCHMARK_IMPORT_ERROR}")
    return load_optional_sequence_payloads


def test_db10_subject_group() -> None:
    assert db10_subject_group("S010") == "able_bodied"
    assert db10_subject_group("S115") == "amputee"


def test_db10_context_feature_modes() -> None:
    assert "annotated_object_proxy" in DB10_CONTEXT_MODES
    sensor = _db10_context_features(
        context_mode="sensor_summary",
        prefix_s=0.2,
        prefix_grid_s=(0.2, 0.4),
        dynamic_flag=1,
        position_id=0,
        object_id=5,
        object_part=3,
        gaze_available=True,
        pupil_left_available=True,
        pupil_right_available=False,
        acc_available=True,
        gyr_available=False,
    )
    protocol = _db10_context_features(
        context_mode="protocol_context",
        prefix_s=0.2,
        prefix_grid_s=(0.2, 0.4),
        dynamic_flag=1,
        position_id=0,
        object_id=5,
        object_part=3,
        gaze_available=True,
        pupil_left_available=True,
        pupil_right_available=False,
        acc_available=True,
        gyr_available=False,
    )
    annotated = _db10_context_features(
        context_mode="annotated_object_proxy",
        prefix_s=0.2,
        prefix_grid_s=(0.2, 0.4),
        dynamic_flag=1,
        position_id=0,
        object_id=5,
        object_part=3,
        gaze_available=True,
        pupil_left_available=True,
        pupil_right_available=False,
        acc_available=True,
        gyr_available=False,
    )
    assert sensor.shape == (0,)
    assert protocol.shape[0] > 0
    assert annotated.shape[0] > protocol.shape[0]


def test_parse_hyser_label_text() -> None:
    assert parse_hyser_label_text("1,1,2,3\n4,5") == [1, 1, 2, 3, 4, 5]


def test_parse_grabmyo_record_path() -> None:
    parsed = parse_grabmyo_record_path("Session3/session3_participant12/session3_participant12_gesture14_trial7")
    assert parsed["subject_id"] == "participant12"
    assert parsed["session"] == "session3"
    assert parsed["day"] == "D03"
    assert parsed["gesture_label_raw"] == 14
    assert parsed["trial"] == 7


def test_parse_cemhsey_grasp_member() -> None:
    parsed = parse_cemhsey_grasp_member("S1/D11/S1_Day11_Session7_Task3_Trial2.mat")
    assert parsed["subject_id"] == "S1"
    assert parsed["day"] == "D11"
    assert parsed["grasp_block"] == "Session7"
    assert parsed["grasp_label_raw"] == 7
    assert parsed["force_level"] == 3


def test_cemhsey_exclusion_keys() -> None:
    keys = cemhsey_exclusion_keys(r"S4\D1\S4_Day1_Session1_Task1_Trial1.mat")
    assert "S4/D1/S4_Day1_Session1_Task1_Trial1.mat" in keys
    assert "S4_Day1_Session1_Task1_Trial1.mat" in keys


def test_apply_cemhsey_trial_exclusions(tmp_path) -> None:
    exclusion_path = tmp_path / "exclusions.yaml"
    exclusion_path.write_text(
        "cemhsey:\n  grasp_failed_trials:\n    - S4_Day1_Session1_Task1_Trial1.mat\n",
        encoding="utf-8",
    )
    manifest = pd.DataFrame(
        [
            {"member_path": "S4/D1/S4_Day1_Session1_Task1_Trial1.mat", "archive_path": "GRASP_S4.zip"},
            {"member_path": "S4/D1/S4_Day1_Session1_Task1_Trial2.mat", "archive_path": "GRASP_S4.zip"},
        ]
    )
    filtered, report = apply_cemhsey_trial_exclusions(manifest, subset="grasp", exclusion_config=exclusion_path)
    assert len(filtered) == 1
    assert filtered.iloc[0]["member_path"].endswith("Trial2.mat")
    assert report["n_excluded_trials"] == 1
    assert report["matched_failed_trials"] == ["S4_Day1_Session1_Task1_Trial1.mat"]


def test_load_cemhsey_failed_trials(tmp_path) -> None:
    exclusion_path = tmp_path / "exclusions.yaml"
    exclusion_path.write_text(
        "cemhsey:\n  grasp_failed_trials:\n    - trial_a.mat\n  gesture_failed_trials:\n    - trial_b.mat\n",
        encoding="utf-8",
    )
    assert load_cemhsey_failed_trials(exclusion_path, subset="grasp") == {"trial_a.mat"}
    assert load_cemhsey_failed_trials(exclusion_path, subset="gesture") == {"trial_b.mat"}


def test_apply_cemhsey_trial_exclusions_requires_matching_config(tmp_path) -> None:
    manifest = pd.DataFrame([{"member_path": "S4/D1/S4_Day1_Session1_Task1_Trial1.mat"}])
    missing_path = tmp_path / "missing.yaml"
    try:
        apply_cemhsey_trial_exclusions(manifest, subset="grasp", exclusion_config=missing_path)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected missing exclusion config to fail closed")


def test_prepared_dataset_validate() -> None:
    bundle = PreparedDataset(
        metadata=pd.DataFrame({"subject_id": ["a", "b"]}),
        user_features=np.ones((2, 3)),
        assist_features=np.ones((2, 5)),
        labels=np.array([0, 1]),
        label_vocab=[1, 2],
        sequence_payloads={"user_emg": np.ones((2, 4, 2), dtype=np.float32)},
    )
    bundle.validate()


def test_prepared_dataset_validate_rejects_sequence_row_mismatch() -> None:
    bundle = PreparedDataset(
        metadata=pd.DataFrame({"subject_id": ["a", "b"]}),
        user_features=np.ones((2, 3)),
        assist_features=np.ones((2, 5)),
        labels=np.array([0, 1]),
        label_vocab=[1, 2],
        sequence_payloads={"user_emg": np.ones((1, 4, 2), dtype=np.float32)},
    )
    try:
        bundle.validate()
    except ValueError as exc:
        assert "sequence payload 'user_emg' row count does not match metadata" in str(exc)
    else:
        raise AssertionError("expected sequence payload row mismatch to fail validation")


def test_prepared_dataset_sequence_payloads_save_load(tmp_path) -> None:
    metadata = pd.DataFrame(
        [
            {"dataset_id": "db10", "subject_id": "S010", "label_raw": 1, "label": 0},
            {"dataset_id": "db10", "subject_id": "S010", "label_raw": 2, "label": 1},
        ]
    )
    bundle = PreparedDataset(
        metadata=metadata,
        user_features=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        assist_features=np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], dtype=np.float32),
        labels=np.array([0, 1], dtype=int),
        label_vocab=[1, 2],
        sequence_payloads={
            "user_emg": np.arange(24, dtype=np.float32).reshape(2, 4, 3),
            "assist_time_mask": np.array(
                [
                    [[0.0], [1.0], [1.0]],
                    [[1.0], [1.0], [1.0]],
                ],
                dtype=np.float32,
            ),
        },
    )
    out_dir = tmp_path / "prepared"
    save_prepared_dataset(bundle, out_dir)
    assert (out_dir / "sequence_payloads").exists()
    loaded = load_prepared_dataset(out_dir)
    pd.testing.assert_frame_equal(loaded.metadata, bundle.metadata)
    np.testing.assert_array_equal(loaded.user_features, bundle.user_features)
    np.testing.assert_array_equal(loaded.assist_features, bundle.assist_features)
    np.testing.assert_array_equal(loaded.labels, bundle.labels)
    assert loaded.label_vocab == bundle.label_vocab
    assert loaded.sequence_payloads is not None
    assert set(loaded.sequence_payloads) == set(bundle.sequence_payloads or {})
    for name, payload in (bundle.sequence_payloads or {}).items():
        np.testing.assert_array_equal(loaded.sequence_payloads[name], payload)


def test_load_optional_sequence_payloads_from_saved_directory(tmp_path) -> None:
    loader = _require_load_optional_sequence_payloads()
    metadata = pd.DataFrame(
        [
            {"dataset_id": "db10", "subject_id": "S010", "label_raw": 1, "label": 0},
            {"dataset_id": "db10", "subject_id": "S010", "label_raw": 2, "label": 1},
        ]
    )
    bundle = PreparedDataset(
        metadata=metadata,
        user_features=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        assist_features=np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], dtype=np.float32),
        labels=np.array([0, 1], dtype=int),
        label_vocab=[1, 2],
        sequence_payloads={
            "user_emg": np.arange(24, dtype=np.float32).reshape(2, 4, 3),
            "user_time_mask": np.ones((2, 4, 1), dtype=np.float32),
            "assist_emg": np.arange(48, dtype=np.float32).reshape(2, 8, 3),
            "assist_time_mask": np.ones((2, 8, 1), dtype=np.float32),
            "assist_gaze": np.ones((2, 8, 2), dtype=np.float32),
            "assist_gaze_mask": np.ones((2, 8, 1), dtype=np.float32),
            "assist_pupil": np.ones((2, 8, 2), dtype=np.float32),
            "assist_pupil_mask": np.ones((2, 8, 2), dtype=np.float32),
            "assist_acc": np.ones((2, 8, 3), dtype=np.float32),
            "assist_acc_mask": np.ones((2, 8, 1), dtype=np.float32),
            "assist_gyr": np.ones((2, 8, 3), dtype=np.float32),
            "assist_gyr_mask": np.ones((2, 8, 1), dtype=np.float32),
            "assist_context": np.ones((2, 1, 4), dtype=np.float32),
            "assist_context_mask": np.ones((2, 1, 1), dtype=np.float32),
        },
    )
    out_dir = tmp_path / "prepared"
    save_prepared_dataset(bundle, out_dir)
    payloads, sources = loader(out_dir, expected_rows=2)
    assert sorted(payloads) == ["assist", "user"]
    assert sorted(sources) == ["assist", "user"]
    np.testing.assert_array_equal(payloads["user"]["user_emg"], bundle.sequence_payloads["user_emg"])
    np.testing.assert_array_equal(payloads["assist"]["assist_context"], bundle.sequence_payloads["assist_context"])


def test_prepare_db10_dataset_emit_sequences(tmp_path) -> None:
    root = tmp_path / "db10"
    root.mkdir()
    n_samples = 40
    fs = 20.0
    time = np.arange(n_samples, dtype=np.float32) / fs
    emg = np.column_stack(
        [
            np.sin(time),
            np.cos(time),
            time,
            time**2,
        ]
    ).astype(np.float32)
    gaze = np.column_stack([time, 1.0 - time]).astype(np.float32)
    gaze_invalid = np.zeros((n_samples, 1), dtype=np.uint8)
    gaze_invalid[:2] = 1
    pupil_left = (2.5 + time).reshape(-1, 1).astype(np.float32)
    pupil_right = (2.0 + 0.5 * time).reshape(-1, 1).astype(np.float32)
    pupil_left_invalid = np.zeros((n_samples, 1), dtype=np.uint8)
    pupil_right_invalid = np.zeros((n_samples, 1), dtype=np.uint8)
    pupil_left_invalid[5:7] = 1
    pupil_right_invalid[:1] = 1
    acc = np.column_stack([time, time + 1.0, time + 2.0]).astype(np.float32)
    acc_invalid = np.zeros((n_samples, 1), dtype=np.uint8)
    acc_invalid[:3] = 1
    gyr = np.column_stack([time**2, time**2 + 1.0, time**2 + 2.0]).astype(np.float32)
    gyr_invalid = np.zeros((n_samples, 1), dtype=np.uint8)
    gyr_invalid[10:12] = 1
    sio.savemat(
        root / "S010_ex1.mat",
        {
            "ts": time.reshape(-1, 1),
            "emg": emg,
            "grasp": np.ones((n_samples, 1), dtype=np.int16),
            "dynamic": np.ones((n_samples, 1), dtype=np.int16),
            "position": np.zeros((n_samples, 1), dtype=np.int16),
            "object": np.full((n_samples, 1), 5, dtype=np.int16),
            "grasprepetition": np.ones((n_samples, 1), dtype=np.int16),
            "objectrepetition": np.full((n_samples, 1), 2, dtype=np.int16),
            "objectpart": np.full((n_samples, 1), 3, dtype=np.int16),
            "gazepoint": gaze,
            "gazepoint_invalid": gaze_invalid,
            "pupildiameterleft": pupil_left,
            "pupildiameterleft_invalid": pupil_left_invalid,
            "pupildiameterright": pupil_right,
            "pupildiameterright_invalid": pupil_right_invalid,
            "tobiiacc": acc,
            "tobiiacc_invalid": acc_invalid,
            "tobiigyr": gyr,
            "tobiigyr_invalid": gyr_invalid,
        },
    )
    bundle = prepare_db10_dataset(
        root,
        prefix_grid_s=(0.5, 1.0),
        user_window_s=0.25,
        min_segment_s=0.25,
        limit_files=1,
        context_mode="annotated_object_proxy",
        emit_sequences=True,
    )
    assert len(bundle.metadata) == 2
    assert bundle.sequence_payloads is not None
    seq = bundle.sequence_payloads
    assert seq["user_emg"].shape == (2, 5, 12)
    assert seq["user_time_mask"].shape == (2, 5, 1)
    assert seq["assist_emg"].shape == (2, 20, 12)
    assert seq["assist_time_mask"].shape == (2, 20, 1)
    assert seq["assist_gaze"].shape == (2, 20, 2)
    assert seq["assist_gaze_mask"].shape == (2, 20, 1)
    assert seq["assist_pupil"].shape == (2, 20, 2)
    assert seq["assist_pupil_mask"].shape == (2, 20, 2)
    assert seq["assist_acc"].shape == (2, 20, 3)
    assert seq["assist_acc_mask"].shape == (2, 20, 1)
    assert seq["assist_gyr"].shape == (2, 20, 3)
    assert seq["assist_gyr_mask"].shape == (2, 20, 1)
    assert seq["assist_context"].shape[:2] == (2, 1)
    assert seq["assist_context"].shape[2] > 0
    assert seq["assist_context_mask"].shape == (2, 1, 1)
    np.testing.assert_array_equal(seq["user_time_mask"].sum(axis=(1, 2)), np.array([5.0, 5.0], dtype=np.float32))
    np.testing.assert_array_equal(seq["assist_time_mask"].sum(axis=(1, 2)), np.array([10.0, 20.0], dtype=np.float32))
    np.testing.assert_array_equal(seq["assist_gaze_mask"].sum(axis=(1, 2)), np.array([8.0, 18.0], dtype=np.float32))
    np.testing.assert_array_equal(seq["assist_pupil_mask"].sum(axis=(1, 2)), np.array([17.0, 37.0], dtype=np.float32))
    np.testing.assert_array_equal(seq["assist_acc_mask"].sum(axis=(1, 2)), np.array([7.0, 17.0], dtype=np.float32))
    np.testing.assert_array_equal(seq["assist_gyr_mask"].sum(axis=(1, 2)), np.array([10.0, 18.0], dtype=np.float32))


def test_prepare_grabmyo_dataset_smoke(tmp_path) -> None:
    root = tmp_path / "grabmyo"
    record_dir = root / "Session1" / "session1_participant1"
    record_dir.mkdir(parents=True)
    signal = np.column_stack(
        [
            np.sin(np.linspace(0.0, 4.0 * np.pi, 500, dtype=np.float32)),
            np.cos(np.linspace(0.0, 4.0 * np.pi, 500, dtype=np.float32)),
        ]
    ).astype(np.float32)
    wfdb.wrsamp(
        record_name="session1_participant1_gesture2_trial1",
        fs=100.0,
        fmt=["16", "16"],
        adc_gain=[1000.0, 1000.0],
        baseline=[0, 0],
        units=["mV", "mV"],
        sig_name=["c1", "c2"],
        p_signal=signal,
        write_dir=str(record_dir),
    )
    (root / "RECORDS").write_text(
        "Session1/session1_participant1/session1_participant1_gesture2_trial1\n",
        encoding="utf-8",
    )
    bundle = prepare_grabmyo_dataset(
        root,
        prefix_grid_s=(0.2, 0.4),
        user_window_s=0.1,
    )
    assert len(bundle.metadata) == 2
    assert set(bundle.metadata["session"].astype(str)) == {"session1"}
    assert set(bundle.metadata["day"].astype(str)) == {"D01"}
    assert set(bundle.metadata["label_raw"].astype(int)) == {2}
    assert bundle.user_features.shape[0] == 2
    assert bundle.assist_features.shape[0] == 2


def test_merge_prepared_datasets() -> None:
    bundle_a = PreparedDataset(
        metadata=pd.DataFrame([{"dataset_id": "cemhsey", "subject_id": "S1", "label_raw": 1, "label": 0}]),
        user_features=np.array([[1.0, 2.0]], dtype=float),
        assist_features=np.array([[3.0, 4.0, 5.0]], dtype=float),
        labels=np.array([0]),
        label_vocab=[1],
    )
    bundle_b = PreparedDataset(
        metadata=pd.DataFrame([{"dataset_id": "cemhsey", "subject_id": "S2", "label_raw": 2, "label": 0}]),
        user_features=np.array([[6.0, 7.0]], dtype=float),
        assist_features=np.array([[8.0, 9.0, 10.0]], dtype=float),
        labels=np.array([0]),
        label_vocab=[2],
    )
    merged = merge_prepared_datasets([bundle_a, bundle_b])
    assert len(merged.metadata) == 2
    assert merged.label_vocab == [1, 2]
    assert merged.user_features.shape == (2, 2)
    assert merged.assist_features.shape == (2, 3)


def test_prepare_j1_dataset_from_prepared_bundle(tmp_path) -> None:
    source_root = tmp_path / "handoff_bundle"
    source_bundle = PreparedDataset(
        metadata=pd.DataFrame(
            [
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
                },
                {
                    "dataset_id": "db10",
                    "subject_id": "S010",
                    "group": "able_bodied",
                    "session": "session1",
                    "day": "D01",
                    "task": "grasp",
                    "episode_id": "ep_b",
                    "record_id": "rec_b",
                    "label_raw": 2,
                    "prefix_time_s": 0.4,
                    "timestamp_s": 0.4,
                },
            ]
        ),
        user_features=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        assist_features=np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], dtype=np.float32),
        labels=np.array([0, 1], dtype=int),
        label_vocab=[1, 2],
        sequence_payloads={"user_emg": np.arange(12, dtype=np.float32).reshape(2, 2, 3)},
    )
    save_prepared_dataset(source_bundle, source_root)

    j1_bundle = prepare_j1_dataset(source_root)
    assert set(j1_bundle.metadata["dataset_id"].astype(str)) == {"j1"}
    assert set(j1_bundle.metadata["source_dataset_id"].astype(str)) == {"db10"}
    assert "source_prepared_root" in j1_bundle.metadata.columns
    assert j1_bundle.sequence_payloads is not None
    np.testing.assert_array_equal(j1_bundle.user_features, source_bundle.user_features)
    np.testing.assert_array_equal(j1_bundle.assist_features, source_bundle.assist_features)
    assert j1_bundle.label_vocab == [1, 2]


def test_prepare_j1_dataset_rejects_missing_required_metadata(tmp_path) -> None:
    source_root = tmp_path / "handoff_bundle_missing_subject"
    source_bundle = PreparedDataset(
        metadata=pd.DataFrame(
            [
                {
                    "dataset_id": "db10",
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
            ]
        ),
        user_features=np.array([[1.0, 2.0]], dtype=np.float32),
        assist_features=np.array([[5.0, 6.0, 7.0]], dtype=np.float32),
        labels=np.array([0], dtype=int),
        label_vocab=[1],
    )
    save_prepared_dataset(source_bundle, source_root)

    with pytest.raises(ValueError, match="missing required columns"):
        prepare_j1_dataset(source_root)


def test_prepare_j1_dataset_rejects_nonmonotone_episode_times(tmp_path) -> None:
    source_root = tmp_path / "handoff_bundle_bad_prefix"
    source_bundle = PreparedDataset(
        metadata=pd.DataFrame(
            [
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
                    "prefix_time_s": 0.4,
                    "timestamp_s": 0.4,
                },
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
                },
            ]
        ),
        user_features=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        assist_features=np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], dtype=np.float32),
        labels=np.array([0, 0], dtype=int),
        label_vocab=[1],
    )
    save_prepared_dataset(source_bundle, source_root)

    with pytest.raises(ValueError, match="duplicated rows|nondecreasing|strictly increasing"):
        prepare_j1_dataset(source_root)
def test_run_dataset_benchmark_hyser_smoke() -> None:
    benchmark_fn = _require_run_dataset_benchmark()
    rows = []
    user_features = []
    assist_features = []
    labels = []
    for subject_id, base in [("subject01", -2.0), ("subject02", 2.0)]:
        for session in ["session1", "session2"]:
            for label in [0, 1]:
                for prefix_time_s in [0.2, 0.4]:
                    rows.append(
                        {
                            "dataset_id": "hyser",
                            "subject_id": subject_id,
                            "group": "able_bodied",
                            "session": session,
                            "day": session,
                            "task": "dynamic",
                            "episode_id": f"{subject_id}_{session}_{label}_{prefix_time_s}",
                            "record_id": f"{subject_id}_{session}_{label}",
                            "label_raw": label + 1,
                            "prefix_time_s": prefix_time_s,
                            "timestamp_s": prefix_time_s,
                        }
                    )
                    feat = np.array([base + label * 4.0, prefix_time_s], dtype=float)
                    user_features.append(feat)
                    assist_features.append(np.array([feat[0], feat[1], feat[0] * feat[1]], dtype=float))
                    labels.append(label)
    bundle = PreparedDataset(
        metadata=pd.DataFrame(rows),
        user_features=np.vstack(user_features),
        assist_features=np.vstack(assist_features),
        labels=np.asarray(labels, dtype=int),
        label_vocab=[1, 2],
    )
    metrics_df, aggregate_df = benchmark_fn(
        bundle,
        dataset_id="hyser",
        user_model_name="logistic",
        assist_model_name="logistic",
    )
    assert not metrics_df.empty
    assert not aggregate_df.empty
    assert "split_family" in metrics_df.columns
    assert "split_family" in aggregate_df.columns


def test_run_dataset_benchmark_db10_assist_hybrid_fusion_smoke() -> None:
    benchmark_fn = _require_run_dataset_benchmark()
    rows = []
    user_features = []
    assist_features = []
    labels = []
    sequence_rows = {
        "assist_emg": [],
        "assist_time_mask": [],
        "assist_gaze": [],
        "assist_gaze_mask": [],
        "assist_pupil": [],
        "assist_pupil_mask": [],
        "assist_acc": [],
        "assist_acc_mask": [],
        "assist_gyr": [],
        "assist_gyr_mask": [],
        "assist_context": [],
        "assist_context_mask": [],
    }
    subjects = [
        ("S010", "able_bodied", -1.5),
        ("S011", "able_bodied", -0.5),
        ("S101", "amputee", 0.5),
        ("S102", "amputee", 1.5),
    ]
    max_len = 6
    time = np.linspace(0.0, 1.0, max_len, dtype=np.float32)
    for subject_id, group, base in subjects:
        for label in [0, 1]:
            for rep in [0, 1]:
                prefix_time_s = 0.2 + 0.2 * rep
                rows.append(
                    {
                        "dataset_id": "db10",
                        "subject_id": subject_id,
                        "group": group,
                        "session": "ex1",
                        "day": None,
                        "task": "grasp_stream",
                        "episode_id": f"{subject_id}_{label}_{rep}",
                        "record_id": f"{subject_id}_{label}",
                        "label_raw": label + 1,
                        "prefix_time_s": prefix_time_s,
                        "timestamp_s": prefix_time_s,
                    }
                )
                feat = np.array([base, float(label), prefix_time_s], dtype=np.float32)
                user_features.append(np.array([feat[0], feat[1] * 2.0], dtype=np.float32))
                assist_features.append(np.array([feat[0], feat[1] * 3.0, feat[2], feat[0] * (label + 1)], dtype=np.float32))
                labels.append(label)
                assist_emg = np.zeros((max_len, 3), dtype=np.float32)
                assist_emg[:, 0] = base + 0.2 * time
                assist_emg[:, 1] = float(label) + time
                assist_emg[:, 2] = prefix_time_s
                assist_gaze = np.column_stack([time + label, 1.0 - time]).astype(np.float32)
                assist_pupil = np.column_stack([0.5 + label + 0.1 * time, 0.2 + 0.05 * time]).astype(np.float32)
                assist_acc = np.column_stack([base + time, time, np.full(max_len, label, dtype=np.float32)]).astype(np.float32)
                assist_gyr = np.column_stack([np.cos(time + label), np.sin(time + label), np.full(max_len, base, dtype=np.float32)]).astype(np.float32)
                context = np.array([[base, float(label), prefix_time_s, 1.0]], dtype=np.float32)
                sequence_rows["assist_emg"].append(assist_emg)
                sequence_rows["assist_time_mask"].append(np.ones((max_len, 1), dtype=np.float32))
                sequence_rows["assist_gaze"].append(assist_gaze)
                sequence_rows["assist_gaze_mask"].append(np.ones((max_len, 1), dtype=np.float32))
                sequence_rows["assist_pupil"].append(assist_pupil)
                sequence_rows["assist_pupil_mask"].append(np.ones((max_len, 2), dtype=np.float32))
                sequence_rows["assist_acc"].append(assist_acc)
                sequence_rows["assist_acc_mask"].append(np.ones((max_len, 1), dtype=np.float32))
                sequence_rows["assist_gyr"].append(assist_gyr)
                sequence_rows["assist_gyr_mask"].append(np.ones((max_len, 1), dtype=np.float32))
                sequence_rows["assist_context"].append(context)
                sequence_rows["assist_context_mask"].append(np.ones((1, 1), dtype=np.float32))
    bundle = PreparedDataset(
        metadata=pd.DataFrame(rows),
        user_features=np.vstack(user_features),
        assist_features=np.vstack(assist_features),
        labels=np.asarray(labels, dtype=int),
        label_vocab=[1, 2],
        sequence_payloads={name: np.stack(values).astype(np.float32) for name, values in sequence_rows.items()},
    )
    metrics_df, aggregate_df = benchmark_fn(
        bundle,
        dataset_id="db10",
        user_model_name="logistic",
        assist_model_name="assist_hybrid_fusion",
        extra_inputs={"assist": bundle.sequence_payloads},
    )
    assert not metrics_df.empty
    assert not aggregate_df.empty
    assert "db10_amputee_loso" in set(metrics_df["split_family"])
    assert "agency_margin_tau_0.20" in set(aggregate_df["policy"])


def test_run_dataset_benchmark_cemhsey_forward_day_smoke() -> None:
    benchmark_fn = _require_run_dataset_benchmark()
    rows = []
    user_features = []
    assist_features = []
    labels = []
    for day_index, day in enumerate(["D01", "D02", "D03"], start=1):
        for label in [0, 1]:
            for rep in [0, 1]:
                rows.append(
                    {
                        "dataset_id": "cemhsey",
                        "subject_id": "S1",
                        "group": "able_bodied",
                        "session": "grasp_protocol",
                        "day": day,
                        "task": "grasp",
                        "episode_id": f"{day}_{label}_{rep}",
                        "record_id": f"{day}_{label}_{rep}",
                        "label_raw": label + 1,
                        "prefix_time_s": 5.0 + rep,
                        "timestamp_s": 10.0 + rep,
                    }
                )
                feat = np.array([day_index, label * 3.0 + rep], dtype=float)
                user_features.append(feat)
                assist_features.append(np.array([feat[0], feat[1], feat[0] * feat[1]], dtype=float))
                labels.append(label)
    bundle = PreparedDataset(
        metadata=pd.DataFrame(rows),
        user_features=np.vstack(user_features),
        assist_features=np.vstack(assist_features),
        labels=np.asarray(labels, dtype=int),
        label_vocab=[1, 2],
    )
    metrics_df, aggregate_df = benchmark_fn(
        bundle,
        dataset_id="cemhsey",
        user_model_name="logistic",
        assist_model_name="logistic",
    )
    assert not metrics_df.empty
    assert "cemhsey_forward_day_logo" in set(metrics_df["split_family"])
    assert "cemhsey_forward_day_logo" in set(aggregate_df["split_family"])


def test_run_dataset_benchmark_grabmyo_smoke() -> None:
    benchmark_fn = _require_run_dataset_benchmark()
    rows = []
    user_features = []
    assist_features = []
    labels = []
    for subject_id, base in [("participant01", -2.0), ("participant02", 2.0)]:
        for session_index, session in enumerate(["session1", "session2", "session3"], start=1):
            for label in [0, 1]:
                rows.append(
                    {
                        "dataset_id": "grabmyo",
                        "subject_id": subject_id,
                        "group": "able_bodied",
                        "session": session,
                        "day": f"D{session_index:02d}",
                        "task": "gesture",
                        "episode_id": f"{subject_id}_{session}_{label}",
                        "record_id": f"{subject_id}_{session}_{label}",
                        "label_raw": label + 1,
                        "prefix_time_s": 0.2 * session_index,
                        "timestamp_s": 0.2 * session_index,
                    }
                )
                feat = np.array([base + label * 4.0, float(session_index)], dtype=float)
                user_features.append(feat)
                assist_features.append(np.array([feat[0], feat[1], feat[0] * feat[1]], dtype=float))
                labels.append(label)
    bundle = PreparedDataset(
        metadata=pd.DataFrame(rows),
        user_features=np.vstack(user_features),
        assist_features=np.vstack(assist_features),
        labels=np.asarray(labels, dtype=int),
        label_vocab=[1, 2],
    )
    metrics_df, aggregate_df = benchmark_fn(
        bundle,
        dataset_id="grabmyo",
        user_model_name="logistic",
        assist_model_name="logistic",
    )
    assert not metrics_df.empty
    assert "grabmyo_within_subject_dayshift" in set(metrics_df["split_family"])
    assert "grabmyo_subject_logo" in set(metrics_df["split_family"])
    assert "grabmyo_within_subject_dayshift" in set(aggregate_df["split_family"])


def test_run_dataset_benchmark_respects_split_family_filters() -> None:
    benchmark_fn = _require_run_dataset_benchmark()
    metadata = pd.DataFrame(
        [
            {
                "dataset_id": "grabmyo",
                "subject_id": "participant01",
                "group": "able_bodied",
                "session": "session1",
                "day": "D01",
                "task": "gesture",
                "episode_id": "p1_s1_a",
                "timestamp_s": 0.2,
                "prefix_time_s": 0.2,
                "label_raw": 1,
            },
            {
                "dataset_id": "grabmyo",
                "subject_id": "participant01",
                "group": "able_bodied",
                "session": "session1",
                "day": "D01",
                "task": "gesture",
                "episode_id": "p1_s1_b",
                "timestamp_s": 0.4,
                "prefix_time_s": 0.4,
                "label_raw": 2,
            },
            {
                "dataset_id": "grabmyo",
                "subject_id": "participant01",
                "group": "able_bodied",
                "session": "session2",
                "day": "D02",
                "task": "gesture",
                "episode_id": "p1_s2_a",
                "timestamp_s": 0.2,
                "prefix_time_s": 0.2,
                "label_raw": 1,
            },
            {
                "dataset_id": "grabmyo",
                "subject_id": "participant01",
                "group": "able_bodied",
                "session": "session2",
                "day": "D02",
                "task": "gesture",
                "episode_id": "p1_s2_b",
                "timestamp_s": 0.4,
                "prefix_time_s": 0.4,
                "label_raw": 2,
            },
            {
                "dataset_id": "grabmyo",
                "subject_id": "participant02",
                "group": "able_bodied",
                "session": "session1",
                "day": "D01",
                "task": "gesture",
                "episode_id": "p2_s1_b",
                "timestamp_s": 0.2,
                "prefix_time_s": 0.2,
                "label_raw": 2,
            },
            {
                "dataset_id": "grabmyo",
                "subject_id": "participant02",
                "group": "able_bodied",
                "session": "session1",
                "day": "D01",
                "task": "gesture",
                "episode_id": "p2_s1_a",
                "timestamp_s": 0.4,
                "prefix_time_s": 0.4,
                "label_raw": 1,
            },
            {
                "dataset_id": "grabmyo",
                "subject_id": "participant02",
                "group": "able_bodied",
                "session": "session2",
                "day": "D02",
                "task": "gesture",
                "episode_id": "p2_s2_b",
                "timestamp_s": 0.2,
                "prefix_time_s": 0.2,
                "label_raw": 2,
            },
            {
                "dataset_id": "grabmyo",
                "subject_id": "participant02",
                "group": "able_bodied",
                "session": "session2",
                "day": "D02",
                "task": "gesture",
                "episode_id": "p2_s2_a",
                "timestamp_s": 0.4,
                "prefix_time_s": 0.4,
                "label_raw": 1,
            },
        ]
    )
    bundle = PreparedDataset(
        metadata=metadata,
        user_features=np.array(
            [
                [0.1, 0.2, 0.0],
                [1.0, 1.0, 1.1],
                [0.2, 0.1, 0.0],
                [1.1, 1.0, 1.1],
                [1.0, 1.1, 1.2],
                [0.1, 0.1, 0.2],
                [1.1, 1.0, 1.2],
                [0.2, 0.2, 0.1],
            ],
            dtype=np.float32,
        ),
        assist_features=np.array(
            [
                [0.1, 0.2],
                [1.0, 1.1],
                [0.2, 0.1],
                [1.1, 1.0],
                [1.0, 1.1],
                [0.1, 0.2],
                [1.1, 1.0],
                [0.2, 0.1],
            ],
            dtype=np.float32,
        ),
        labels=np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=int),
        label_vocab=[1, 2],
    )

    metrics_df, aggregate_df = benchmark_fn(
        bundle,
        dataset_id="grabmyo",
        user_model_name="logistic",
        assist_model_name="logistic",
        split_family_allowlist=("grabmyo_subject_logo",),
        max_splits_per_family=1,
    )

    assert set(metrics_df["split_family"]) == {"grabmyo_subject_logo"}
    assert set(aggregate_df["split_family"]) == {"grabmyo_subject_logo"}
    assert metrics_df["split_id"].astype(str).nunique() == 1


def test_run_dataset_benchmark_j1_smoke() -> None:
    benchmark_fn = _require_run_dataset_benchmark()
    rows = []
    user_features = []
    assist_features = []
    labels = []
    subjects = [
        ("S010", "able_bodied", -1.5),
        ("S011", "able_bodied", -0.5),
        ("S101", "amputee", 0.5),
        ("S102", "amputee", 1.5),
    ]
    for subject_id, group, base in subjects:
        for label in [0, 1]:
            for rep in [0, 1]:
                prefix_time_s = 0.2 + 0.2 * rep
                rows.append(
                    {
                        "dataset_id": "j1",
                        "source_dataset_id": "db10",
                        "subject_id": subject_id,
                        "group": group,
                        "session": "session1",
                        "day": "D01",
                        "task": "grasp",
                        "episode_id": f"{subject_id}_{label}_{rep}",
                        "record_id": f"{subject_id}_{label}_{rep}",
                        "label_raw": label + 1,
                        "prefix_time_s": prefix_time_s,
                        "timestamp_s": prefix_time_s,
                    }
                )
                feat = np.array([base, float(label), prefix_time_s], dtype=np.float32)
                user_features.append(np.array([feat[0], feat[1] * 2.0], dtype=np.float32))
                assist_features.append(np.array([feat[0], feat[1] * 3.0, feat[2]], dtype=np.float32))
                labels.append(label)
    bundle = PreparedDataset(
        metadata=pd.DataFrame(rows),
        user_features=np.vstack(user_features),
        assist_features=np.vstack(assist_features),
        labels=np.asarray(labels, dtype=int),
        label_vocab=[1, 2],
    )

    metrics_df, aggregate_df = benchmark_fn(
        bundle,
        dataset_id="j1",
        user_model_name="logistic",
        assist_model_name="logistic",
    )

    assert not metrics_df.empty
    assert not aggregate_df.empty
    assert "j1_amputee_loso" in set(metrics_df["split_family"])
    assert "j1_subject_logo" in set(metrics_df["split_family"])
    assert metrics_df["policy"].astype(str).str.startswith("plain_conf_threshold_matched_tau_").any()

