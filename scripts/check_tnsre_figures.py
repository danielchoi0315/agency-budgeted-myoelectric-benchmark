from __future__ import annotations

import argparse
import csv
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PDF_SIGNATURE = b"%PDF-"

DEFAULT_REQUIRED_FIGURES = [
    "db10_matched_budget_ablation.png",
    "db10_earliest_safe.png",
    "hyser_matched_budget_ablation.png",
    "hyser_earliest_safe.png",
    "cemhsey_matched_budget_ablation.png",
    "cemhsey_earliest_safe.png",
]


@dataclass(frozen=True)
class PngMetadata:
    width: int
    height: int
    x_dpi: float | None
    y_dpi: float | None


def normalized_path(path: Path) -> str:
    resolved = str(path.resolve(strict=False))
    if os.name == "nt":
        return resolved.lower()
    return resolved


def display_path(path: Path, base: Path) -> str:
    try:
        return str(path.resolve(strict=False).relative_to(base.resolve(strict=False)))
    except ValueError:
        return str(path)


def resolve_target(out_dir: Path, spec: str) -> Path:
    path = Path(spec)
    if path.is_absolute():
        return path
    return out_dir / path


def load_required_list(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, list):
            return [str(item) for item in payload]
        if isinstance(payload, dict):
            for key in ("required", "required_files", "required_figures"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [str(item) for item in value]
        raise ValueError(
            "JSON required list must be a list, or an object with required, "
            "required_files, or required_figures."
        )

    required = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            required.append(stripped)
    return required


def load_manifest_required(path: Path, out_dir: Path) -> list[str]:
    required: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "file" not in reader.fieldnames:
            raise ValueError("Manifest must contain a 'file' column.")
        for row in reader:
            value = (row.get("file") or "").strip()
            if not value:
                continue
            file_path = Path(value)
            if file_path.is_absolute():
                try:
                    required.append(str(file_path.resolve(strict=False).relative_to(out_dir.resolve(strict=False))))
                except ValueError:
                    required.append(str(file_path))
            else:
                required.append(str(file_path))
    if not required:
        raise ValueError("Manifest did not list any required figure files.")
    return required


def read_png_metadata(path: Path) -> PngMetadata:
    data = path.read_bytes()
    if len(data) < 33:
        raise ValueError("PNG is too short to contain a valid IHDR chunk.")
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("File does not start with the PNG signature.")

    offset = len(PNG_SIGNATURE)
    width: int | None = None
    height: int | None = None
    x_dpi: float | None = None
    y_dpi: float | None = None

    while offset + 8 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_start = offset + 8
        chunk_end = chunk_start + length
        next_offset = chunk_end + 4
        if next_offset > len(data):
            raise ValueError(f"PNG chunk {chunk_type!r} is truncated.")

        chunk = data[chunk_start:chunk_end]
        if chunk_type == b"IHDR":
            if length != 13:
                raise ValueError("PNG IHDR chunk has an invalid length.")
            width, height = struct.unpack(">II", chunk[:8])
        elif chunk_type == b"pHYs":
            if length == 9:
                x_pixels_per_unit, y_pixels_per_unit, unit = struct.unpack(">IIB", chunk)
                if unit == 1:
                    x_dpi = x_pixels_per_unit * 0.0254
                    y_dpi = y_pixels_per_unit * 0.0254
        elif chunk_type == b"IEND":
            break

        offset = next_offset

    if width is None or height is None:
        raise ValueError("PNG IHDR chunk was not found.")
    return PngMetadata(width=width, height=height, x_dpi=x_dpi, y_dpi=y_dpi)


def check_png(path: Path, args: argparse.Namespace) -> tuple[dict[str, Any], list[str], list[str]]:
    issues: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {}

    size_bytes = path.stat().st_size
    details["size_bytes"] = size_bytes
    if size_bytes < args.min_png_bytes:
        issues.append(f"PNG is smaller than {args.min_png_bytes} bytes.")

    try:
        metadata = read_png_metadata(path)
    except ValueError as exc:
        issues.append(str(exc))
        return details, issues, warnings

    details["width_px"] = metadata.width
    details["height_px"] = metadata.height
    if metadata.width < args.min_width:
        issues.append(f"PNG width {metadata.width}px is below {args.min_width}px.")
    if metadata.height < args.min_height:
        issues.append(f"PNG height {metadata.height}px is below {args.min_height}px.")

    if metadata.x_dpi is None or metadata.y_dpi is None:
        message = "PNG does not include pHYs DPI metadata."
        if args.min_dpi > 0:
            if args.allow_missing_dpi:
                warnings.append(message)
            else:
                issues.append(message)
    else:
        details["x_dpi"] = round(metadata.x_dpi, 3)
        details["y_dpi"] = round(metadata.y_dpi, 3)
        dpi_floor = min(metadata.x_dpi, metadata.y_dpi)
        if args.min_dpi > 0 and dpi_floor + args.dpi_tolerance < args.min_dpi:
            issues.append(
                f"PNG DPI {dpi_floor:.1f} is below {args.min_dpi:g} "
                f"(tolerance {args.dpi_tolerance:g})."
            )

    return details, issues, warnings


def check_pdf(path: Path, args: argparse.Namespace) -> tuple[dict[str, Any], list[str], list[str]]:
    issues: list[str] = []
    details: dict[str, Any] = {}

    size_bytes = path.stat().st_size
    details["size_bytes"] = size_bytes
    if size_bytes < args.min_pdf_bytes:
        issues.append(f"PDF is smaller than {args.min_pdf_bytes} bytes.")

    with path.open("rb") as handle:
        header = handle.read(len(PDF_SIGNATURE))
    if header != PDF_SIGNATURE:
        issues.append("File does not start with the PDF signature.")

    return details, issues, []


def check_asset(path: Path, out_dir: Path, required: bool, args: argparse.Namespace) -> dict[str, Any]:
    suffix = path.suffix.lower()
    kind = suffix.lstrip(".") or "unknown"
    result: dict[str, Any] = {
        "path": display_path(path, out_dir),
        "kind": kind,
        "required": required,
        "present": path.is_file(),
        "status": "ok",
        "issues": [],
        "warnings": [],
        "details": {},
    }

    if not path.is_file():
        result["status"] = "fail" if required else "missing"
        result["issues"] = ["Required file is missing."] if required else ["File is missing."]
        return result

    if suffix == ".png":
        details, issues, warnings = check_png(path, args)
    elif suffix == ".pdf":
        details, issues, warnings = check_pdf(path, args)
    else:
        details = {"size_bytes": path.stat().st_size}
        issues = [f"Unsupported required figure extension: {suffix or '<none>'}."]
        warnings = []

    result["details"] = details
    result["issues"] = issues
    result["warnings"] = warnings
    if issues:
        result["status"] = "fail"
    elif warnings:
        result["status"] = "warn"
    return result


def discover_assets(out_dir: Path, recursive: bool) -> list[Path]:
    if not out_dir.is_dir():
        return []
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in out_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in {".png", ".pdf"}
    )


def build_required_specs(args: argparse.Namespace) -> list[str]:
    user_required: list[str] = []
    if args.required_list is not None:
        user_required.extend(load_required_list(args.required_list))
    if args.required:
        user_required.extend(args.required)

    if user_required:
        required = user_required
    else:
        manifest = args.manifest
        if manifest is None:
            default_manifest = args.out_dir / "tnsre_figure_manifest.csv"
            manifest = default_manifest if default_manifest.is_file() else None
        required = load_manifest_required(manifest, args.out_dir) if manifest is not None else list(DEFAULT_REQUIRED_FIGURES)

    if args.require_pdf_sidecars:
        sidecars = [
            str(Path(spec).with_suffix(".pdf"))
            for spec in required
            if Path(spec).suffix.lower() == ".png"
        ]
        required.extend(sidecars)

    deduped: list[str] = []
    seen: set[str] = set()
    for spec in required:
        key = str(Path(spec)).lower() if os.name == "nt" else str(Path(spec))
        if key not in seen:
            deduped.append(spec)
            seen.add(key)
    return deduped


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir
    required_paths = [resolve_target(out_dir, spec) for spec in build_required_specs(args)]
    required_keys = {normalized_path(path) for path in required_paths}

    ordered_paths = list(required_paths)
    for path in discover_assets(out_dir, recursive=args.recursive):
        if normalized_path(path) not in required_keys:
            ordered_paths.append(path)

    seen: set[str] = set()
    assets = []
    for path in ordered_paths:
        key = normalized_path(path)
        if key in seen:
            continue
        seen.add(key)
        assets.append(check_asset(path, out_dir, required=key in required_keys, args=args))

    root_issues: list[str] = []
    if not out_dir.exists():
        root_issues.append("Output directory does not exist.")
    elif not out_dir.is_dir():
        root_issues.append("Output path exists but is not a directory.")

    failures = sum(1 for item in assets if item["status"] == "fail")
    warnings = sum(1 for item in assets if item["status"] == "warn")
    png_count = sum(1 for item in assets if item["kind"] == "png" and item["present"])
    pdf_count = sum(1 for item in assets if item["kind"] == "pdf" and item["present"])
    required_missing = sum(
        1 for item in assets if item["required"] and item["status"] == "fail" and not item["present"]
    )

    if args.min_pdfs > 0 and pdf_count < args.min_pdfs:
        root_issues.append(f"Found {pdf_count} PDF file(s), below required minimum {args.min_pdfs}.")

    status = "fail" if failures or root_issues else "pass"
    return {
        "status": status,
        "out_dir": str(out_dir),
        "recursive": args.recursive,
        "thresholds": {
            "min_width_px": args.min_width,
            "min_height_px": args.min_height,
            "min_dpi": args.min_dpi,
            "dpi_tolerance": args.dpi_tolerance,
            "allow_missing_dpi": args.allow_missing_dpi,
            "min_png_bytes": args.min_png_bytes,
            "min_pdf_bytes": args.min_pdf_bytes,
            "min_pdfs": args.min_pdfs,
        },
        "summary": {
            "required_files": len(required_paths),
            "checked_assets": len(assets),
            "checked_pngs": png_count,
            "checked_pdfs": pdf_count,
            "required_missing": required_missing,
            "failures": failures + len(root_issues),
            "warnings": warnings,
        },
        "root_issues": root_issues,
        "assets": assets,
    }


def render_text(report: dict[str, Any]) -> str:
    summary = report["summary"]
    thresholds = report["thresholds"]
    lines = [
        f"TNSRE figure QA: {report['status'].upper()}",
        f"output_dir: {report['out_dir']}",
        (
            "thresholds: "
            f"PNG >= {thresholds['min_width_px']}x{thresholds['min_height_px']} px, "
            f"DPI >= {thresholds['min_dpi']:g} "
            f"(tolerance {thresholds['dpi_tolerance']:g}), "
            f"PNG >= {thresholds['min_png_bytes']} bytes, "
            f"PDF >= {thresholds['min_pdf_bytes']} bytes"
        ),
        (
            "summary: "
            f"{summary['checked_assets']} assets checked, "
            f"{summary['checked_pngs']} PNGs, "
            f"{summary['checked_pdfs']} PDFs, "
            f"{summary['failures']} failures, "
            f"{summary['warnings']} warnings"
        ),
    ]

    if report["root_issues"]:
        lines.append("")
        lines.append("Root issues:")
        for issue in report["root_issues"]:
            lines.append(f"  FAIL {issue}")

    if not report["assets"]:
        lines.append("")
        lines.append("No PNG or PDF assets found.")
        return "\n".join(lines) + "\n"

    lines.append("")
    lines.append("Assets:")
    for item in report["assets"]:
        required = "required" if item["required"] else "extra"
        details = item["details"]
        detail_bits = []
        if "size_bytes" in details:
            detail_bits.append(f"{details['size_bytes']} bytes")
        if "width_px" in details and "height_px" in details:
            detail_bits.append(f"{details['width_px']}x{details['height_px']} px")
        if "x_dpi" in details and "y_dpi" in details:
            detail_bits.append(f"{details['x_dpi']}x{details['y_dpi']} dpi")
        suffix = f" ({', '.join(detail_bits)})" if detail_bits else ""
        lines.append(f"  {item['status'].upper():4} {required:8} {item['path']}{suffix}")
        for issue in item["issues"]:
            lines.append(f"       issue: {issue}")
        for warning in item["warnings"]:
            lines.append(f"       warning: {warning}")

    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lightweight QA gate for final TNSRE figure outputs. Checks required files, "
            "PNG dimensions/DPI metadata, and PDF presence/byte size."
        )
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/reports/publication_clean"),
        help="Directory containing final figure outputs.",
    )
    parser.add_argument(
        "--required",
        action="append",
        help=(
            "Required figure file relative to --out-dir. Repeat as needed. "
            "If omitted, uses the default publication-clean TNSRE PNG set."
        ),
    )
    parser.add_argument(
        "--required-list",
        type=Path,
        help=(
            "Text file with one required path per line, or JSON list/object with "
            "required, required_files, or required_figures."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "CSV manifest containing a file column. If omitted and "
            "--out-dir/tnsre_figure_manifest.csv exists, that manifest is used."
        ),
    )
    parser.add_argument(
        "--require-pdf-sidecars",
        action="store_true",
        help="Also require a same-stem PDF for every required PNG.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan PNG and PDF files recursively under --out-dir.",
    )
    parser.add_argument("--min-width", type=int, default=1800, help="Minimum PNG width in pixels.")
    parser.add_argument("--min-height", type=int, default=900, help="Minimum PNG height in pixels.")
    parser.add_argument("--min-dpi", type=float, default=300.0, help="Minimum PNG DPI; use 0 to disable.")
    parser.add_argument(
        "--dpi-tolerance",
        type=float,
        default=0.5,
        help="Tolerance for PNG DPI metadata rounding.",
    )
    parser.add_argument(
        "--allow-missing-dpi",
        action="store_true",
        help="Warn instead of fail when a PNG lacks pHYs DPI metadata.",
    )
    parser.add_argument(
        "--min-png-bytes",
        type=int,
        default=10_000,
        help="Minimum acceptable PNG byte size.",
    )
    parser.add_argument(
        "--min-pdf-bytes",
        type=int,
        default=10_000,
        help="Minimum acceptable PDF byte size.",
    )
    parser.add_argument(
        "--min-pdfs",
        type=int,
        default=0,
        help="Minimum number of PDF files expected in --out-dir.",
    )
    parser.add_argument("--format", choices=("text", "json"), default="text", help="Report format.")
    parser.add_argument("--report", type=Path, help="Optional path to write the report.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_report(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not build TNSRE figure QA report: {exc}") from exc

    if args.format == "json":
        rendered = json.dumps(report, indent=2) + "\n"
    else:
        rendered = render_text(report)

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")

    print(rendered, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

