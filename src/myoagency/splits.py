from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut


@dataclass(frozen=True)
class Split:
    split_id: str
    train_index: list[int]
    test_index: list[int]
    train_groups: dict[str, set[str]]
    test_groups: dict[str, set[str]]


def _require_columns(metadata: pd.DataFrame, keys: list[str]) -> None:
    missing = [key for key in keys if key not in metadata.columns]
    if missing:
        raise KeyError(f"metadata missing required split columns: {missing}")


def assert_no_leakage(metadata: pd.DataFrame, train_idx: list[int], test_idx: list[int], keys: list[str]) -> None:
    _require_columns(metadata, keys)
    if set(train_idx) & set(test_idx):
        raise AssertionError("train/test index overlap detected")
    train = metadata.iloc[train_idx]
    test = metadata.iloc[test_idx]
    for key in keys:
        overlap = set(train[key].dropna().astype(str)) & set(test[key].dropna().astype(str))
        if overlap:
            raise AssertionError(f"leakage detected for {key}: {sorted(overlap)[:10]}")


def leave_one_group_out_splits(
    metadata: pd.DataFrame,
    group_key: str,
    strict_disjoint_keys: list[str] | None = None,
    report_keys: list[str] | None = None,
    prefix: str = "logo",
) -> list[Split]:
    strict_disjoint_keys = strict_disjoint_keys or [group_key]
    report_keys = report_keys or strict_disjoint_keys
    _require_columns(metadata, [group_key, *strict_disjoint_keys])
    groups = metadata[group_key].astype(str).to_numpy()
    splits: list[Split] = []
    for fold, (train_idx, test_idx) in enumerate(LeaveOneGroupOut().split(metadata, groups=groups)):
        train_list = train_idx.tolist()
        test_list = test_idx.tolist()
        assert_no_leakage(metadata, train_list, test_list, strict_disjoint_keys)
        splits.append(
            Split(
                split_id=f"{prefix}_{fold:03d}_{group_key}_{metadata.iloc[test_idx][group_key].iloc[0]}",
                train_index=train_list,
                test_index=test_list,
                train_groups={key: set(metadata.iloc[train_idx][key].astype(str)) for key in report_keys if key in metadata},
                test_groups={key: set(metadata.iloc[test_idx][key].astype(str)) for key in report_keys if key in metadata},
            )
        )
    return splits


def group_k_fold_splits(
    metadata: pd.DataFrame,
    group_key: str,
    n_splits: int,
    strict_disjoint_keys: list[str] | None = None,
    report_keys: list[str] | None = None,
    prefix: str = "gkf",
) -> list[Split]:
    strict_disjoint_keys = strict_disjoint_keys or [group_key]
    report_keys = report_keys or strict_disjoint_keys
    _require_columns(metadata, [group_key, *strict_disjoint_keys])
    groups = metadata[group_key].astype(str).to_numpy()
    splits: list[Split] = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(metadata, groups=groups)):
        train_list = train_idx.tolist()
        test_list = test_idx.tolist()
        assert_no_leakage(metadata, train_list, test_list, strict_disjoint_keys)
        splits.append(
            Split(
                split_id=f"{prefix}_{fold:03d}",
                train_index=train_list,
                test_index=test_list,
                train_groups={key: set(metadata.iloc[train_idx][key].astype(str)) for key in report_keys if key in metadata},
                test_groups={key: set(metadata.iloc[test_idx][key].astype(str)) for key in report_keys if key in metadata},
            )
        )
    return splits


def held_out_group_with_background_splits(
    metadata: pd.DataFrame,
    heldout_group_key: str,
    heldout_group_value: str,
    subject_key: str = "subject_id",
    background_group_values: list[str] | None = None,
    group_col: str = "group",
    prefix: str = "heldout",
) -> list[Split]:
    """Train on all background rows plus non-held-out target-group subjects."""
    background_group_values = background_group_values or ["able_bodied"]
    _require_columns(metadata, [heldout_group_key, subject_key, group_col])
    target = metadata[metadata[heldout_group_key].astype(str) == heldout_group_value]
    splits: list[Split] = []
    for fold, subject in enumerate(sorted(target[subject_key].astype(str).unique())):
        test_mask = (metadata[heldout_group_key].astype(str) == heldout_group_value) & (
            metadata[subject_key].astype(str) == subject
        )
        train_mask = ~test_mask
        train_idx = metadata.index[train_mask].to_list()
        test_idx = metadata.index[test_mask].to_list()
        assert_no_leakage(metadata, train_idx, test_idx, [subject_key])
        splits.append(
            Split(
                split_id=f"{prefix}_{fold:03d}_{heldout_group_value}_{subject}",
                train_index=train_idx,
                test_index=test_idx,
                train_groups={
                    subject_key: set(metadata.loc[train_idx, subject_key].astype(str)),
                    group_col: set(metadata.loc[train_idx, group_col].astype(str)),
                },
                test_groups={
                    subject_key: set(metadata.loc[test_idx, subject_key].astype(str)),
                    group_col: set(metadata.loc[test_idx, group_col].astype(str)),
                },
            )
        )
    return splits

