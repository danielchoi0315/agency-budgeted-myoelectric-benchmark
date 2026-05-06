from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from j2bench.datasets import dataset_download_instructions, load_dataset_configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Render official download instructions from dataset config.")
    parser.add_argument("--dataset", choices=["db10", "hyser", "cemhsey", "grabmyo", "capgmyo_dba", "putemg", "ninapro_db2", "ninapro_db3", "ninapro_db6", "seeds_65", "hd_fw_kin"], required=True)
    parser.add_argument("--config", type=Path, default=Path("config/datasets.yaml"))
    args = parser.parse_args()
    configs = load_dataset_configs(args.config)
    print(dataset_download_instructions(configs[args.dataset]))


if __name__ == "__main__":
    main()

