from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen


BASE_URL = "https://physionet-open.s3.amazonaws.com/hd-semg/2.0.0"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch missing Hyser PR label txt files from PhysioNet S3.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--ok-file", type=Path)
    args = parser.parse_args()

    sha_index = load_sha_index(args.root / "SHA256SUMS.txt")
    expected_label_paths = sorted(
        rel_path
        for rel_path in sha_index
        if rel_path.startswith("pr_dataset/") and rel_path.endswith(".txt") and "label_" in rel_path
    )
    if len(expected_label_paths) != 80:
        raise ValueError(f"expected 80 Hyser PR label files, found {len(expected_label_paths)} in SHA256SUMS.txt")
    downloaded = 0
    for rel_path in expected_label_paths:
        target = args.root / Path(rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            continue
        url = f"{args.base_url}/{rel_path}"
        data = urlopen(url).read()
        expected_sha = sha_index.get(rel_path)
        if expected_sha and hashlib.sha256(data).hexdigest() != expected_sha:
            raise ValueError(f"sha256 mismatch for {rel_path}")
        target.write_bytes(data)
        downloaded += 1
        print(f"downloaded {rel_path}")
    missing = [rel_path for rel_path in expected_label_paths if not (args.root / Path(rel_path)).exists()]
    if missing:
        raise FileNotFoundError(f"missing Hyser label files after fetch: {missing[:5]}")
    if args.ok_file is not None:
        args.ok_file.parent.mkdir(parents=True, exist_ok=True)
        args.ok_file.write_text("ok\n", encoding="utf-8")
    print(f"Fetched {downloaded} label files.")


def load_sha_index(path: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        sha256, relative_path = parts
        index[relative_path.strip()] = sha256.strip()
    return index


if __name__ == "__main__":
    main()

