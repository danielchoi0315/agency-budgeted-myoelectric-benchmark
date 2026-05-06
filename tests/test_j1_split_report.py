from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


def _load_build_split_report_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "build_split_report.py"
    spec = importlib.util.spec_from_file_location("build_split_report_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(
    *,
    source_dataset_id: str,
    subject_id: str,
    group: str,
    session: str,
    day: str,
    episode_id: str,
    label_raw: int = 1,
) -> dict[str, object]:
    return {
        "dataset_id": "j1",
        "source_dataset_id": source_dataset_id,
        "subject_id": subject_id,
        "group": group,
        "session": session,
        "day": day,
        "task": "grasp",
        "episode_id": episode_id,
        "record_id": episode_id,
        "label_raw": label_raw,
        "prefix_time_s": 0.2,
        "timestamp_s": 0.2,
    }


def test_build_split_report_uses_composite_subject_ids_for_cross_source_holdouts() -> None:
    module = _load_build_split_report_module()
    metadata = pd.DataFrame(
        [
            _row(
                source_dataset_id="db10",
                subject_id="shared01",
                group="able_bodied",
                session="session1",
                day="D01",
                episode_id="db10_ep1",
            ),
            _row(
                source_dataset_id="hyser",
                subject_id="shared01",
                group="able_bodied",
                session="session1",
                day="D01",
                episode_id="hyser_ep1",
            ),
        ]
    )

    payload = module.build_split_report_payload(metadata)
    split = next(row for row in payload["splits"] if row["split_id"] == "j1_source_dataset_holdout|db10")

    assert split["overlap_checks"]["subject_id"]["count"] == 0
    assert split["overlap_checks"]["source_dataset_id"]["count"] == 0
    assert split["critical_overlap_passed"] is True


def test_build_split_report_summarizes_forward_day_coverage() -> None:
    module = _load_build_split_report_module()
    metadata = pd.DataFrame(
        [
            _row(
                source_dataset_id="db10",
                subject_id=subject_id,
                group="able_bodied",
                session="session1",
                day=day,
                episode_id=f"{subject_id}_{day}",
            )
            for subject_id in ("S010", "S011")
            for day in ("D01", "D02", "D03")
        ]
    )

    payload = module.build_split_report_payload(
        metadata,
        split_family_allowlist=("j1_forward_day_logo",),
    )

    family = payload["families"][0]
    split = next(row for row in payload["splits"] if row["split_id"] == "j1_forward_day_logo|D02")

    assert family["split_family"] == "j1_forward_day_logo"
    assert family["n_splits"] == 2
    assert family["covered_test_days"] == 2
    assert family["covered_test_days_pct"] == pytest.approx(2 / 3)
    assert family["forward_ordering_passed"] is True
    assert split["boundary_train_values"] == ["D01"]
    assert split["boundary_test_values"] == ["D02"]
    assert split["forward_ordering_passed"] is True


def test_main_writes_filtered_split_report_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_build_split_report_module()
    prepared_root = tmp_path / "prepared"
    out_root = tmp_path / "report"
    prepared_root.mkdir()

    metadata = pd.DataFrame(
        [
            _row(
                source_dataset_id="db10",
                subject_id=subject_id,
                group="able_bodied",
                session=session,
                day="D01",
                episode_id=f"{subject_id}_{session}",
            )
            for subject_id in ("S010", "S011")
            for session in ("session1", "session2")
        ]
    )
    metadata.to_csv(prepared_root / "metadata.csv", index=False)
    (prepared_root / "_SUCCESS").write_text("ok\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_split_report.py",
            "--prepared-root",
            str(prepared_root),
            "--out-root",
            str(out_root),
            "--split-family",
            "j1_forward_session_logo",
        ],
    )
    module.main()

    summary_path = out_root / "split_report.json"
    report_path = out_root / "split_report.md"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    markdown = report_path.read_text(encoding="utf-8")

    assert payload["summary"]["n_families"] == 1
    assert payload["summary"]["forward_order_checks_applicable"] == 1
    assert payload["inputs"]["prepared_success_file_present"] is True
    assert payload["families"][0]["split_family"] == "j1_forward_session_logo"
    assert "j1_forward_session_logo\\|session2" in markdown
    assert "Forward Order Checks" in markdown

