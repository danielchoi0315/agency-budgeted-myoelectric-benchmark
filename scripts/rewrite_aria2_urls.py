from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite URLs in an aria2 input/session file while preserving options.")
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--from-prefix", dest="from_prefix", default="")
    parser.add_argument("--to-prefix", dest="to_prefix", default="")
    parser.add_argument(
        "--strip-session-state",
        action="store_true",
        help="Drop aria2 saved runtime options that can override the lane profile.",
    )
    args = parser.parse_args()

    keep_option_prefixes = ("dir=", "out=")
    volatile_options = (
        " gid=",
        " timeout=",
        " connect-timeout=",
        " max-tries=",
        " split=",
        " lowest-speed-limit=",
        " file-allocation=",
        " allow-overwrite=",
        " continue=",
        " auto-file-renaming=",
        " max-connection-per-server=",
        " min-split-size=",
        " retry-wait=",
    )
    lines = []
    changed = 0
    for line in args.queue.read_text(encoding="utf-8").splitlines():
        if args.strip_session_state and line.startswith(volatile_options):
            continue
        if args.from_prefix and line.startswith(args.from_prefix):
            line = args.to_prefix + line[len(args.from_prefix) :]
            changed += 1
        if args.strip_session_state and line.startswith(" ") and not line.lstrip().startswith(keep_option_prefixes):
            continue
        lines.append(line)
        if args.strip_session_state and line.startswith("http"):
            lines.append("  continue=true")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Rewrote {changed} URL lines -> {args.out}")


if __name__ == "__main__":
    main()

