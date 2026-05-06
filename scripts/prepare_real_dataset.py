from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myoagency.realdata import (  # noqa: E402
    prepare_cemhsey_dataset,
    prepare_db10_dataset,
    prepare_grabmyo_dataset,
    prepare_hyser_dataset,
    save_prepared_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a real prosthetics dataset into benchmark features.")
    parser.add_argument("--dataset", choices=["db10", "hyser", "cemhsey", "grabmyo"], required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--exclude-rest", action="store_true")
    parser.add_argument(
        "--db10-context-mode",
        choices=["sensor_summary", "protocol_context", "annotated_object_proxy"],
        default="annotated_object_proxy",
    )
    parser.add_argument(
        "--db10-sequences",
        action="store_true",
        help="Emit optional fixed-size DB10 sequence payloads alongside the existing tabular features.",
    )
    parser.add_argument(
        "--db10-sequence-user-max-steps",
        type=int,
        default=128,
        help="Maximum token length for DB10 user-window sequences when --db10-sequences is enabled.",
    )
    parser.add_argument(
        "--db10-sequence-assist-max-steps",
        type=int,
        default=256,
        help="Maximum token length for DB10 assist-window sequences when --db10-sequences is enabled.",
    )
    args = parser.parse_args()

    if args.dataset == "db10":
        bundle = prepare_db10_dataset(
            args.root,
            include_rest=not args.exclude_rest,
            limit_files=args.limit,
            context_mode=args.db10_context_mode,
            emit_sequences=args.db10_sequences,
            sequence_user_max_steps=args.db10_sequence_user_max_steps,
            sequence_assist_max_steps=args.db10_sequence_assist_max_steps,
        )
    elif args.dataset == "hyser":
        bundle = prepare_hyser_dataset(args.root, limit_records=args.limit)
    elif args.dataset == "grabmyo":
        bundle = prepare_grabmyo_dataset(args.root, limit_records=args.limit)
    else:
        bundle = prepare_cemhsey_dataset(args.root, limit_records=args.limit)
    save_prepared_dataset(bundle, args.out)
    print(f"Prepared {args.dataset}: {len(bundle.metadata)} rows -> {args.out}")


if __name__ == "__main__":
    main()

