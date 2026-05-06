from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from j2bench.audit import audit_dataset_root, write_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a downloaded open prosthetics dataset.")
    parser.add_argument("--dataset", required=True, choices=["db10", "hyser", "cemhsey", "grabmyo", "capgmyo_dba", "putemg", "j1"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--hash-files", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    audit = audit_dataset_root(args.dataset, args.root, hash_files=args.hash_files)
    write_audit(audit, args.out)
    print(f"Wrote audit for {args.dataset}: {audit.file_count} files, {audit.total_bytes} bytes -> {args.out}")
    if audit.warnings:
        print("Warnings:")
        for warning in audit.warnings:
            print(f"- {warning}")
    if args.require_pass and audit.status != "PASS":
        raise SystemExit(f"Audit status is {audit.status}; refusing to continue.")


if __name__ == "__main__":
    main()

