from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


DEFAULT_ALLOWED_FAMILIES = {"ArialMT", "Arial-BoldMT"}


def font_family(name: str) -> str:
    return re.sub(r"^[A-Z]{6}\+", "", name)


def parse_pdffonts(output: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("name ") or stripped.startswith("---"):
            continue
        parts = stripped.split()
        if len(parts) < 7:
            continue
        rows.append(
            {
                "name": parts[0],
                "family": font_family(parts[0]),
                "type": parts[1],
                "encoding": parts[2],
                "embedded": parts[3],
                "subset": parts[4],
                "unicode": parts[5],
            }
        )
    return rows


def check_pdf(path: Path, allowed_families: set[str]) -> list[str]:
    result = subprocess.run(
        ["pdffonts", str(path)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return [f"{path}: pdffonts failed: {result.stderr.strip()}"]

    issues: list[str] = []
    rows = parse_pdffonts(result.stdout)
    if not rows:
        issues.append(f"{path}: no fonts reported by pdffonts")
        return issues

    for row in rows:
        if row["type"].lower() == "type3":
            issues.append(f"{path}: Type 3 font found: {row['name']}")
        if row["embedded"] != "yes":
            issues.append(f"{path}: font not embedded: {row['name']}")
        if row["subset"] != "yes":
            issues.append(f"{path}: font not subset: {row['name']}")
        if row["unicode"] != "yes":
            issues.append(f"{path}: font has no Unicode map: {row['name']}")
        if row["family"] not in allowed_families:
            issues.append(f"{path}: unexpected font family {row['family']} from {row['name']}")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Check publication figure PDF font embedding.")
    parser.add_argument("--pdf-dir", type=Path, default=Path("results/reports/tnsre_figures/pdf"))
    parser.add_argument("--allow-family", action="append", default=sorted(DEFAULT_ALLOWED_FAMILIES))
    args = parser.parse_args()

    pdfs = sorted(args.pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDF files found in {args.pdf_dir}", file=sys.stderr)
        return 2

    allowed_families = set(args.allow_family)
    issues: list[str] = []
    for pdf in pdfs:
        issues.extend(check_pdf(pdf, allowed_families))

    if issues:
        print("Figure font check failed:", file=sys.stderr)
        for issue in issues:
            print(f"- {issue}", file=sys.stderr)
        return 1

    print(f"Figure font check passed: {len(pdfs)} PDFs, allowed families: {', '.join(sorted(allowed_families))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
