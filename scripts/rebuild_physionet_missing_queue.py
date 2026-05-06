from __future__ import annotations

import argparse
from pathlib import Path


PHYSIONET_DATASETS = {
    "hyser": {
        "base_url": "https://physionet-open.s3.amazonaws.com/hd-semg/2.0.0/",
        "metadata": ["LICENSE.txt", "RECORDS", "SHA256SUMS.txt", "readme.txt", "equipment_info.pdf"],
    },
    "grabmyo": {
        "base_url": "https://physionet-open.s3.amazonaws.com/grabmyo/1.1.0/",
        "metadata": ["LICENSE.txt", "RECORDS", "SHA256SUMS.txt"],
    },
}


def add_block(lines: list[str], url: str, destination: Path, root: Path) -> None:
    rel_parent = destination.parent.relative_to(root)
    lines.append(url)
    lines.append(f"  dir={(root / rel_parent).as_posix()}")
    lines.append(f"  out={destination.name}")
    lines.append("  continue=true")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild a PhysioNet aria2 queue containing only missing files.")
    parser.add_argument("--dataset", choices=sorted(PHYSIONET_DATASETS), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    spec = PHYSIONET_DATASETS[args.dataset]
    root = args.root
    records_path = root / "RECORDS"
    if not records_path.exists():
        raise SystemExit(f"Missing RECORDS file: {records_path}")

    lines: list[str] = []
    missing = 0

    for name in spec["metadata"]:
        destination = root / name
        if not destination.exists():
            add_block(lines, spec["base_url"] + name, destination, root)
            missing += 1

    seen_records: set[str] = set()
    records: list[str] = []
    for line in records_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        record = line.strip()
        if not record or record in seen_records:
            continue
        seen_records.add(record)
        records.append(record)
    for record in records:
        for suffix in (".hea", ".dat"):
            destination = root / f"{record}{suffix}"
            if not destination.exists():
                add_block(lines, spec["base_url"] + record + suffix, destination, root)
                missing += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"Wrote {missing} missing-file URLs to {args.out}")


if __name__ == "__main__":
    main()

