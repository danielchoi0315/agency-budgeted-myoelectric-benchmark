from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


BLOCKED_SUFFIXES = {
    ".mat",
    ".h5",
    ".hdf5",
    ".edf",
    ".bdf",
    ".npy",
    ".npz",
    ".avi",
    ".mp4",
    ".mov",
    ".zip",
    ".rar",
    ".7z",
    ".dat",
    ".hea",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail if likely raw data files are present in the repo.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--ok-file", type=Path)
    args = parser.parse_args()
    offenders = []
    for path in args.root.rglob("*"):
        if ".venv" in path.parts or ".git" in path.parts:
            continue
        if path.is_file() and path.suffix.lower() in BLOCKED_SUFFIXES:
            offenders.append(path)
    if offenders:
        joined = "\n".join(str(path) for path in offenders[:50])
        raise SystemExit(f"Likely raw data files found in repository:\n{joined}")
    if args.ok_file is not None:
        args.ok_file.parent.mkdir(parents=True, exist_ok=True)
        args.ok_file.write_text("ok\n", encoding="utf-8")
    print("No likely raw data files found in repository.")


if __name__ == "__main__":
    main()

