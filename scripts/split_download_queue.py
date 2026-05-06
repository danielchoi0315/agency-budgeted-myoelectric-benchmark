from __future__ import annotations

import argparse
from pathlib import Path


GROUPS = {
    "dataverse": ("dataverse.harvard.edu",),
    "physionet": ("physionet.org", "physionet-open.s3.amazonaws.com"),
    "zenodo": ("zenodo.org",),
    "figshare": ("figshare.com",),
    "ninapro": ("ninapro.hevs.ch",),
}


def iter_aria2_blocks(path: Path) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("http"):
            if current:
                blocks.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append(current)
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser(description="Split aria2 input queue by host.")
    parser.add_argument("--queue", type=Path, default=Path("results/downloads/aria2_remaining.session"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/downloads/split"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {name: [] for name in GROUPS}
    outputs["other"] = []
    for block in iter_aria2_blocks(args.queue):
        url = block[0]
        target = "other"
        for name, hosts in GROUPS.items():
            if any(host in url for host in hosts):
                target = name
                break
        outputs[target].extend(block)
    for name, lines in outputs.items():
        out = args.out_dir / f"{name}.aria2.txt"
        out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        url_count = sum(1 for line in lines if line.startswith("http"))
        print(f"{name}: {url_count} URLs -> {out}")


if __name__ == "__main__":
    main()

