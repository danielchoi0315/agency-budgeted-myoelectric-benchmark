from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from stream_unzip import stream_unzip


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CEMHSEY zip archives to a fast local workspace.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--subset", choices=["all", "grasp", "gesture"], default="all")
    parser.add_argument("--limit-archives", type=int)
    args = parser.parse_args()

    archives = sorted(args.root.rglob("*.zip"))
    if args.subset == "grasp":
        archives = [path for path in archives if path.name.startswith("GRASP_")]
    elif args.subset == "gesture":
        archives = [path for path in archives if path.name.startswith("GESTURE_")]
    if args.limit_archives is not None:
        archives = archives[: args.limit_archives]

    args.out.mkdir(parents=True, exist_ok=True)
    for archive_path in archives:
        print(f"extracting {archive_path.name}")
        extract_archive(archive_path, args.out / archive_path.stem)


def extract_archive(archive_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as handle:
        if any(info.compress_type == 9 for info in handle.infolist()):
            _extract_with_stream_unzip(archive_path, out_dir)
        else:
            handle.extractall(out_dir)


def _extract_with_stream_unzip(archive_path: Path, out_dir: Path) -> None:
    with archive_path.open("rb") as handle:
        for file_name, _, chunks in stream_unzip(handle):
            decoded_name = file_name.decode("utf-8")
            relative_path = Path(decoded_name)
            target = out_dir / relative_path
            if decoded_name.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as out_handle:
                for chunk in chunks:
                    out_handle.write(chunk)


if __name__ == "__main__":
    main()

