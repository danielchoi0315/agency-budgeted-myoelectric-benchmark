from __future__ import annotations

import argparse
from pathlib import Path


BLOCKED_SUFFIXES = {
    ".mat",
    ".h5",
    ".hdf5",
    ".edf",
    ".bdf",
    ".npy",
    ".npz",
    ".parquet",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".ckpt",
    ".joblib",
    ".dat",
    ".hea",
    ".avi",
    ".mp4",
    ".mov",
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
    ".tgz",
}

SENSITIVE_PATTERNS = [
    "C:\\Users",
    "C:/Users",
    "D:\\prosthetic_data",
    "E:\\prosthetic_data",
    "D:/prosthetic_data",
    "E:/prosthetic_data",
    "j2_agency_benchmark",
    "Author~One",
    "example.edu",
    "editorial_worker",
]

TEXT_SUFFIXES = {
    ".bib",
    ".cff",
    ".csv",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".r",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

REQUIRED_PATHS = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "pyproject.toml",
    "environment.yml",
    ".github/workflows/ci.yml",
    "docs/REPRODUCIBILITY.md",
    "docs/DATA.md",
    "docs/SUBMISSION_CHECKLIST.md",
    "results/README.md",
    "manuscript/README.md",
    "manuscript/tnsre_overleaf/main.tex",
    "manuscript/tnsre_overleaf/supplement.tex",
    "manuscript/tnsre_overleaf/references.bib",
    "scripts/build_tnsre_figures.R",
    "scripts/check_no_raw_data.py",
    "scripts/check_tnsre_figures.py",
]

BLOCKED_COMMITTED_PATHS = [
    "manuscript/tnsre_overleaf/main.pdf",
    "manuscript/tnsre_overleaf/supplement.pdf",
]


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name in {".gitignore", "LICENSE"}


def iter_repo_files(root: Path):
    for path in root.rglob("*"):
        if ".git" in path.parts or ".venv" in path.parts:
            continue
        if path.is_file():
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(description="Publication-repository integrity gate.")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()

    root = args.root.resolve()
    self_path = Path(__file__).resolve()
    issues: list[str] = []

    for rel in REQUIRED_PATHS:
        if not (root / rel).is_file():
            issues.append(f"missing required file: {rel}")

    for rel in BLOCKED_COMMITTED_PATHS:
        if (root / rel).exists():
            issues.append(f"local build product should not be committed: {rel}")

    for path in iter_repo_files(root):
        if path.resolve() == self_path:
            continue
        rel = path.relative_to(root).as_posix()
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            issues.append(f"blocked raw/large-data suffix: {rel}")
            continue
        if not is_text_file(path):
            continue
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
        for pattern in SENSITIVE_PATTERNS:
            if pattern in text:
                issues.append(f"sensitive or placeholder string {pattern!r} in {rel}")

    if issues:
        print("Release integrity check failed:")
        for issue in issues[:100]:
            print(f"- {issue}")
        if len(issues) > 100:
            print(f"- ... {len(issues) - 100} more")
        return 1

    print("Release integrity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
