from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import requests


DEFAULT_RAW_ROOT = Path(os.environ.get("J2_DATA_ROOT", "data")) / "raw"

PHYSIONET_DATASETS = {
    "hyser": {
        "base_url": "https://physionet.org/files/hd-semg/2.0.0/",
        "default_root": DEFAULT_RAW_ROOT / "hyser",
        "metadata": ["LICENSE.txt", "RECORDS", "SHA256SUMS.txt", "readme.txt", "equipment_info.pdf"],
    },
    "grabmyo": {
        "base_url": "https://physionet.org/files/grabmyo/1.1.0/",
        "default_root": DEFAULT_RAW_ROOT / "grabmyo",
        "metadata": ["LICENSE.txt", "RECORDS", "SHA256SUMS.txt"],
    },
}


def download_file(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    existing = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with requests.get(url, stream=True, timeout=60, headers=headers) as response:
        if response.status_code not in {200, 206}:
            response.raise_for_status()
        mode = "ab" if response.status_code == 206 and existing else "wb"
        with tmp.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)
    tmp.replace(destination)


def download_metadata(dataset: str, root: Path) -> None:
    spec = PHYSIONET_DATASETS[dataset]
    for name in spec["metadata"]:
        url = spec["base_url"] + name
        destination = root / name
        print(f"Downloading {url} -> {destination}")
        download_file(url, destination)


def download_records(dataset: str, root: Path, limit: int | None = None) -> None:
    spec = PHYSIONET_DATASETS[dataset]
    records_file = root / "RECORDS"
    if not records_file.exists():
        download_metadata(dataset, root)
    records = [line.strip() for line in records_file.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    if limit is not None:
        records = records[:limit]
    for record in records:
        for suffix in [".hea", ".dat"]:
            url = spec["base_url"] + record + suffix
            destination = root / f"{record}{suffix}"
            print(f"Downloading {url} -> {destination}")
            download_file(url, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download open PhysioNet metadata or WFDB records for J2 datasets.")
    parser.add_argument("--dataset", choices=sorted(PHYSIONET_DATASETS), required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--limit-records", type=int, help="Download only the first N WFDB records for smoke testing.")
    args = parser.parse_args()

    root = args.root or Path(PHYSIONET_DATASETS[args.dataset]["default_root"])
    download_metadata(args.dataset, root)
    if not args.metadata_only:
        download_records(args.dataset, root, limit=args.limit_records)


if __name__ == "__main__":
    main()


