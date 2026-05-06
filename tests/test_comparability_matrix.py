from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_build_comparability_matrix_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "build_comparability_matrix.py"
    spec = importlib.util.spec_from_file_location("build_comparability_matrix_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_markdown_mentions_direct_gate() -> None:
    module = _load_build_comparability_matrix_module()
    markdown = module.build_markdown(
        summary_rows=[
            {
                "target_id": "db10_cross_population_policy",
                "target_dataset_id": "db10",
                "target_split_family": "db10_mixed_to_amputee",
                "target_priority": "primary",
                "target_claim_surface": "benchmark_primary",
                "target_direct_sota_eligible": False,
                "n_direct": 0,
                "n_partial": 1,
                "n_not_comparable": 0,
                "n_contextual": 0,
                "n_boundary": 1,
            }
        ],
        rows=[
            {
                "target_id": "db10_cross_population_policy",
                "target_claim_surface": "benchmark_primary",
                "target_direct_sota_eligible": False,
                "paper_id": "paper_a",
                "citation": "Paper A",
                "role": "comparator",
                "derived_comparability": "partial",
                "dataset": "exact",
                "task_family": "related",
                "split_shift": "different",
                "population": "related",
                "adaptation": "exact",
                "decision_object": "different",
                "metric": "different",
                "anchor_id": "",
                "blocking_fields": "split_shift|decision_object|metric",
                "note": "Not directly comparable.",
            }
        ],
    )

    assert "Direct rows are the only rows that can support literal best-published or SOTA wording." in markdown
    assert "A target must also be explicitly marked as direct-SOTA-eligible" in markdown

