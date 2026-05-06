from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_publication_extension import main as publication_main

    sys.argv = [
        sys.argv[0],
        "--dataset",
        "db10",
        "--include-db10-anchors",
        *sys.argv[1:],
    ]
    publication_main()


if __name__ == "__main__":
    main()

