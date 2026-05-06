from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .io import load_yaml


@dataclass(frozen=True)
class DatasetConfig:
    dataset_id: str
    name: str
    role: str
    official_page: str
    data_sources: dict[str, str]
    expected_modalities: list[str]
    expected_groups: list[str]


def load_dataset_configs(config_path: Path) -> dict[str, DatasetConfig]:
    raw = load_yaml(config_path)["datasets"]
    return {
        dataset_id: DatasetConfig(
            dataset_id=dataset_id,
            name=item["name"],
            role=item["role"],
            official_page=item["official_page"],
            data_sources=item["data_sources"],
            expected_modalities=item["expected_modalities"],
            expected_groups=item["expected_groups"],
        )
        for dataset_id, item in raw.items()
    }


def dataset_download_instructions(config: DatasetConfig) -> str:
    lines = [
        f"# {config.dataset_id}: {config.name}",
        "",
        f"Official page: {config.official_page}",
        "",
        "Data sources:",
    ]
    for name, url in config.data_sources.items():
        lines.append(f"- {name}: {url}")
    lines.extend(
        [
            "",
            "Download policy:",
            "- Store raw archives and extracted files outside git.",
            f"- Recommended root: <DATA_ROOT>/raw/{config.dataset_id}",
            "- Record checksums with `python scripts/audit_dataset.py --hash-files` after download.",
        ]
    )
    return "\n".join(lines)


