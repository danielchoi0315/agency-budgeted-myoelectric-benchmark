from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_build_sota_report_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "build_sota_report.py"
    spec = importlib.util.spec_from_file_location("build_sota_report_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_repo_path_uses_repo_root_for_relative_inputs() -> None:
    module = _load_build_sota_report_module()
    repo_root = Path("/repo")

    resolved = module.resolve_repo_path(repo_root, Path("config/config.yaml"))

    assert resolved == repo_root / "config" / "config.yaml"


def test_build_markdown_blocks_optimistic_text_for_incomplete_status() -> None:
    module = _load_build_sota_report_module()
    markdown = module.build_markdown(
        {
            "status": "blocked_primary_evidence_incomplete",
            "benchmark_prerequisite_status": "provisionally_ready",
            "recommendation": "Do not make a benchmark or SOTA claim until every primary target has valid local evidence.",
            "disallowed_terms": [],
            "preferred_terms": [],
            "targets": [],
            "literature_anchors": [],
            "comparability_summary": [],
            "comparability_matrix": [],
            "executed_datasets": ["db10"],
            "expansion_candidates": [],
        }
    )

    assert "incomplete for manuscript-ready benchmarking claims" in markdown
    assert "supports a bounded benchmark contribution" not in markdown
    assert "## Direct Comparability Matrix" in markdown
    assert "## Release Gate" in markdown

