import pandas as pd
import pytest

from j2bench.real_benchmark import _hyser_splits
from j2bench.splits import (
    assert_no_leakage,
    group_k_fold_splits,
    held_out_group_with_background_splits,
    leave_one_group_out_splits,
)
from j2bench.synthetic import make_synthetic_metadata


def test_leave_one_day_out_has_no_day_leakage():
    metadata = make_synthetic_metadata(n_subjects=4, n_days=2, n_trials_per_cell=2)
    splits = leave_one_group_out_splits(metadata, group_key="day", strict_disjoint_keys=["day"])
    assert len(splits) == 2
    for split in splits:
        assert set(split.train_index).isdisjoint(split.test_index)


def test_group_k_fold_has_no_subject_leakage():
    metadata = make_synthetic_metadata(n_subjects=6, n_trials_per_cell=2)
    splits = group_k_fold_splits(metadata, group_key="subject_id", n_splits=3, strict_disjoint_keys=["subject_id"])
    assert len(splits) == 3
    for split in splits:
        assert split.train_groups["subject_id"].isdisjoint(split.test_groups["subject_id"])


def test_assert_no_leakage_raises_on_overlap():
    metadata = pd.DataFrame({"subject_id": ["S1", "S1", "S2"]})
    with pytest.raises(AssertionError):
        assert_no_leakage(metadata, [0], [1, 2], ["subject_id"])


def test_held_out_amputee_loso_keeps_subjects_disjoint():
    metadata = make_synthetic_metadata(n_subjects=8, n_trials_per_cell=2)
    splits = held_out_group_with_background_splits(metadata, "group", "transradial_amputee")
    assert len(splits) == metadata.query("group == 'transradial_amputee'")["subject_id"].nunique()
    for split in splits:
        assert split.train_groups["subject_id"].isdisjoint(split.test_groups["subject_id"])


def test_missing_strict_key_fails():
    metadata = make_synthetic_metadata(n_subjects=4)
    with pytest.raises(KeyError):
        leave_one_group_out_splits(metadata, group_key="day", strict_disjoint_keys=["participant_uid"])


def test_hyser_within_subject_dayshift_is_forward_only():
    metadata = pd.DataFrame(
        {
            "dataset_id": ["hyser"] * 4,
            "subject_id": ["S1", "S1", "S2", "S2"],
            "session": ["session1", "session2", "session1", "session2"],
        }
    )
    splits = _hyser_splits(metadata)
    shift_splits = [split for split in splits if split.split_id.startswith("hyser_within_subject_dayshift|")]
    assert len(shift_splits) == 1
    split = shift_splits[0]
    assert split.split_id.endswith("session2")
    assert set(metadata.loc[split.train_index, "session"]) == {"session1"}
    assert set(metadata.loc[split.test_index, "session"]) == {"session2"}

