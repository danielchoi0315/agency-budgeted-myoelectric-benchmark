from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


RAW_ROOT = Path(os.environ.get("J2_DATA_ROOT", "data")) / "raw"


@dataclass(frozen=True)
class DownloadItem:
    dataset: str
    url: str
    target: Path
    size: int | None = None
    checksum: str | None = None
    checksum_type: str | None = None
    source: str = ""


def build_session() -> requests.Session:
    retry = Retry(
        total=8,
        connect=8,
        read=8,
        status=8,
        backoff_factor=2.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "j2-agency-benchmark-downloader/0.1"})
    return session


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_complete(item: DownloadItem) -> bool:
    if not item.target.exists():
        return False
    if item.size is not None and item.target.stat().st_size != item.size:
        return False
    if item.checksum and item.checksum_type == "sha256":
        return sha256_file(item.target).lower() == item.checksum.lower()
    return True


def stream_download(item: DownloadItem, retries: int = 5, timeout: int = 120) -> dict[str, Any]:
    item.target.parent.mkdir(parents=True, exist_ok=True)
    if file_complete(item):
        return {"status": "skip", "target": str(item.target), "bytes": item.target.stat().st_size}

    part = item.target.with_suffix(item.target.suffix + ".part")
    last_error = ""
    for attempt in range(1, retries + 1):
        session = build_session()
        resume_from = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
        mode = "ab" if resume_from else "wb"
        try:
            with session.get(item.url, headers=headers, stream=True, timeout=timeout, allow_redirects=True) as response:
                if response.status_code == 416 and file_complete(item):
                    part.unlink(missing_ok=True)
                    return {"status": "skip", "target": str(item.target), "bytes": item.target.stat().st_size}
                if response.status_code != 206 and resume_from:
                    resume_from = 0
                    mode = "wb"
                response.raise_for_status()
                with part.open(mode + "b" if mode in {"a", "w"} else mode) as handle:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if item.size is not None and part.stat().st_size != item.size:
                last_error = f"size mismatch: got {part.stat().st_size}, expected {item.size}"
                time.sleep(min(60, 2**attempt))
                continue
            if item.checksum and item.checksum_type == "sha256":
                digest = sha256_file(part)
                if digest.lower() != item.checksum.lower():
                    last_error = f"sha256 mismatch: got {digest}, expected {item.checksum}"
                    time.sleep(min(60, 2**attempt))
                    continue
            part.replace(item.target)
            return {"status": "downloaded", "target": str(item.target), "bytes": item.target.stat().st_size}
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            time.sleep(min(90, 2**attempt))
        finally:
            session.close()
    return {"status": "error", "target": str(item.target), "error": last_error}


def figshare_items(article_id: str, version: str, dataset: str, root: Path) -> list[DownloadItem]:
    session = build_session()
    url = f"https://api.figshare.com/v2/articles/{article_id}/versions/{version}"
    response = session.get(url, timeout=60)
    response.raise_for_status()
    items = []
    for file_item in response.json().get("files", []):
        items.append(
            DownloadItem(
                dataset=dataset,
                url=file_item["download_url"],
                target=root / dataset / file_item["name"],
                size=file_item.get("size"),
                source=f"figshare:{article_id}/{version}",
            )
        )
    session.close()
    return items


def zenodo_items(record_id: str, dataset: str, root: Path) -> list[DownloadItem]:
    session = build_session()
    response = session.get(f"https://zenodo.org/api/records/{record_id}", timeout=60)
    response.raise_for_status()
    items = []
    for file_item in response.json().get("files", []):
        checksum = file_item.get("checksum", "")
        checksum_type = None
        checksum_value = None
        if checksum.startswith("sha256:"):
            checksum_type = "sha256"
            checksum_value = checksum.split(":", 1)[1]
        items.append(
            DownloadItem(
                dataset=dataset,
                url=file_item["links"]["self"],
                target=root / dataset / record_id / file_item["key"],
                size=file_item.get("size"),
                checksum=checksum_value,
                checksum_type=checksum_type,
                source=f"zenodo:{record_id}",
            )
        )
    session.close()
    return items


def write_manifest(results: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download fresh-link HTTP datasets used by the J2 benchmark.")
    parser.add_argument("--root", type=Path, default=RAW_ROOT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--manifest", type=Path, default=Path("results/downloads/http_fresh_manifest.json"))
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument(
        "--figshare",
        action="append",
        default=[],
        help="Figshare spec as article_id:version:dataset, e.g. 7210397:1:capgmyo_dba",
    )
    parser.add_argument(
        "--zenodo",
        action="append",
        default=[],
        help="Zenodo spec as record_id:dataset, e.g. 14224328:cemhsey",
    )
    args = parser.parse_args()

    items: list[DownloadItem] = []
    for spec in args.figshare:
        article_id, version, dataset = spec.split(":", 2)
        items.extend(figshare_items(article_id, version, dataset, args.root))
    for spec in args.zenodo:
        record_id, dataset = spec.split(":", 1)
        items.extend(zenodo_items(record_id, dataset, args.root))
    if args.limit_files is not None:
        items = items[: args.limit_files]

    print(f"Prepared {len(items)} files across {len(set(item.dataset for item in items))} datasets")
    if not items:
        write_manifest([], args.manifest)
        return

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(stream_download, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = {"status": "error", "target": str(item.target), "error": str(exc)}
            result["dataset"] = item.dataset
            result["source"] = item.source
            results.append(result)
            print(json.dumps(result, sort_keys=True))
            write_manifest(results, args.manifest)

    errors = [result for result in results if result["status"] == "error"]
    if errors:
        raise SystemExit(f"{len(errors)} downloads failed; see {args.manifest}")


if __name__ == "__main__":
    main()

