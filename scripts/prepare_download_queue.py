from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from urllib.parse import urljoin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import requests


DATA_ROOT = Path(os.environ.get("MYOAGENCY_DATA_ROOT", "data"))
RAW_ROOT = DATA_ROOT / "raw"
ARCHIVE_ROOT = DATA_ROOT / "archives"


def add_aria2(lines: list[str], url: str, directory: Path, out: str | None = None) -> None:
    lines.append(url)
    lines.append(f"  dir={directory.as_posix()}")
    if out:
        lines.append(f"  out={out}")
    lines.append("  continue=true")


def prepare_physionet(lines: list[str], dataset: str, base_url: str, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    metadata = ["LICENSE.txt", "RECORDS", "SHA256SUMS.txt"]
    if dataset == "hyser":
        metadata.extend(["readme.txt", "equipment_info.pdf"])
    for name in metadata:
        add_aria2(lines, base_url + name, root, name)
    records_path = root / "RECORDS"
    if records_path.exists():
        records = [line.strip() for line in records_path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
        for record in records:
            for suffix in (".hea", ".dat"):
                add_aria2(lines, base_url + record + suffix, root / Path(record).parent, Path(record).name + suffix)


def prepare_dataverse(lines: list[str], persistent_id: str, root: Path, blockers: list[str]) -> None:
    api = "https://dataverse.harvard.edu/api/datasets/:persistentId"
    try:
        response = requests.get(api, params={"persistentId": persistent_id}, timeout=60)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        blockers.append(f"DB10 Dataverse API failed for {persistent_id}: {exc}")
        return
    data = response.json()["data"]["latestVersion"]["files"]
    for item in data:
        data_file = item["dataFile"]
        file_id = data_file["id"]
        filename = data_file["filename"]
        directory = root / item.get("directoryLabel", "")
        url = f"https://dataverse.harvard.edu/api/access/datafile/{file_id}"
        add_aria2(lines, url, directory, filename)


def prepare_zenodo(lines: list[str], record_id: str, root: Path, blockers: list[str]) -> None:
    try:
        response = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=60)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        blockers.append(f"Zenodo API failed for {record_id}: {exc}")
        return
    record = response.json()
    for item in record.get("files", []):
        link = item["links"]["self"]
        key = item["key"]
        add_aria2(lines, link, root / record_id, key)


def prepare_figshare(lines: list[str], article_id: str, version: str, root: Path, blockers: list[str]) -> None:
    url = f"https://api.figshare.com/v2/articles/{article_id}/versions/{version}"
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        blockers.append(f"Figshare API failed for article {article_id} version {version}: {exc}")
        return
    for item in response.json().get("files", []):
        add_aria2(lines, item["download_url"], root, item["name"])


def prepare_ninapro_page(lines: list[str], page_url: str, root: Path, blockers: list[str]) -> None:
    try:
        text = requests.get(page_url, timeout=60).text
    except Exception as exc:  # noqa: BLE001
        blockers.append(f"NinaPro page failed for {page_url}: {exc}")
        return
    hrefs = sorted(set(re.findall(r'href="([^"]+\.zip)"', text, flags=re.IGNORECASE)))
    if not hrefs:
        blockers.append(f"No zip links found on {page_url}")
        return
    for href in hrefs:
        url = urljoin(page_url, href)
        add_aria2(lines, url, root, Path(href).name)


def write_blockers(blockers: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = ["# Download blockers / manual steps", ""]
    content.extend(f"- {item}" for item in blockers)
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare one aria2 download queue for the benchmark open datasets.")
    parser.add_argument("--out", type=Path, default=Path("results/downloads/aria2_all_open_datasets.txt"))
    parser.add_argument("--blockers", type=Path, default=Path("results/downloads/manual_blockers.md"))
    args = parser.parse_args()

    lines: list[str] = []
    blockers: list[str] = []

    prepare_dataverse(lines, "doi:10.7910/DVN/1Z3IOM", RAW_ROOT / "db10", blockers)
    prepare_physionet(lines, "hyser", "https://physionet-open.s3.amazonaws.com/hd-semg/2.0.0/", RAW_ROOT / "hyser")
    prepare_physionet(lines, "grabmyo", "https://physionet-open.s3.amazonaws.com/grabmyo/1.1.0/", RAW_ROOT / "grabmyo")
    prepare_zenodo(lines, "14224328", RAW_ROOT / "cemhsey", blockers)
    prepare_zenodo(lines, "14272463", RAW_ROOT / "cemhsey", blockers)
    prepare_figshare(lines, "7210397", "1", RAW_ROOT / "capgmyo_dba", blockers)
    prepare_figshare(lines, "13625828", "1", RAW_ROOT / "seeds_65", blockers)
    prepare_ninapro_page(lines, "https://ninapro.hevs.ch/instructions/DB2.html", RAW_ROOT / "ninapro_db2", blockers)
    prepare_ninapro_page(lines, "https://ninapro.hevs.ch/instructions/DB3.html", RAW_ROOT / "ninapro_db3", blockers)
    prepare_ninapro_page(lines, "https://ninapro.hevs.ch/instructions/DB6.html", RAW_ROOT / "ninapro_db6", blockers)

    blockers.append("putEMG requires its project downloader or manual chmura.put.poznan.pl flow; not added to aria2 queue.")
    blockers.append("HD-FW KIN is PhysioNet restricted-access and requires signed DUA/login; not added to open aria2 queue.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_blockers(blockers, args.blockers)
    print(f"Wrote {len([line for line in lines if not line.startswith('  ')])} URLs to {args.out}")
    print(f"Wrote blockers/manual steps to {args.blockers}")


if __name__ == "__main__":
    main()

