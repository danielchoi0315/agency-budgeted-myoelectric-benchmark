from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from j2bench.io import load_yaml  # noqa: E402
from j2bench.literature import build_comparability_rows, summarize_comparability_matrix  # noqa: E402


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build a target-by-paper direct-comparability matrix for J2.")
    parser.add_argument("--targets", type=Path, default=Path("config/literature_anchors.yaml"))
    parser.add_argument("--comparability", type=Path, default=Path("config/literature_comparability.yaml"))
    parser.add_argument("--out-root", type=Path, default=Path("results/reports/publication_clean"))
    args = parser.parse_args()

    targets_cfg = load_yaml(resolve_repo_path(repo_root, args.targets))
    comparability_cfg = load_yaml(resolve_repo_path(repo_root, args.comparability))
    out_root = resolve_repo_path(repo_root, args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = build_comparability_rows(comparability_cfg, targets=list(targets_cfg.get("targets", [])))
    summary_rows = summarize_comparability_matrix(rows)
    payload = {
        "metadata": comparability_cfg.get("metadata", {}),
        "rows": rows,
        "target_summary": summary_rows,
    }
    (out_root / "direct_comparability_matrix.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_root / "direct_comparability_matrix.csv", index=False)
    (out_root / "direct_comparability_matrix.md").write_text(build_markdown(summary_rows, rows), encoding="utf-8")
    print(f"Wrote direct comparability matrix to {out_root}")


def build_markdown(summary_rows: list[dict[str, object]], rows: list[dict[str, object]]) -> str:
    lines = [
        "# Direct Comparability Matrix",
        "",
        "## Target Summary",
    ]
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary[
            [
                "target_id",
                "target_dataset_id",
                "target_split_family",
                "target_priority",
                "target_claim_surface",
                "target_direct_sota_eligible",
                "n_direct",
                "n_partial",
                "n_not_comparable",
                "n_contextual",
                "n_boundary",
            ]
        ].copy()
    lines.extend(render_table(summary))
    lines.append("")
    lines.append("## Matrix Rows")
    matrix = pd.DataFrame(rows)
    if not matrix.empty:
        matrix = matrix[
            [
                "target_id",
                "target_claim_surface",
                "target_direct_sota_eligible",
                "paper_id",
                "citation",
                "role",
                "derived_comparability",
                "dataset",
                "task_family",
                "split_shift",
                "population",
                "adaptation",
                "decision_object",
                "metric",
                "anchor_id",
                "blocking_fields",
                "note",
            ]
        ].copy()
    lines.extend(render_table(matrix))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- Direct rows are the only rows that can support literal best-published or SOTA wording.")
    lines.append("- A target must also be explicitly marked as direct-SOTA-eligible before any direct row can release SOTA language.")
    lines.append("- Partial rows are useful comparator context but still block a literal direct-comparison claim.")
    lines.append("- Contextual, boundary, and not-comparable rows are reviewer guardrails, not numeric comparators.")
    lines.append("")
    return "\n".join(lines)


def render_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["_No data_"]
    return frame.fillna("").to_markdown(index=False).splitlines()


def resolve_repo_path(repo_root: Path, candidate: Path) -> Path:
    return candidate if candidate.is_absolute() else repo_root / candidate


if __name__ == "__main__":
    main()

