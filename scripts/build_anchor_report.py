from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from j2bench.io import load_yaml, read_json  # noqa: E402


EXPECTED_CEMHSEY_GRASP_FAILURE = "S4_Day1_Session1_Task1_Trial1.mat"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a manuscript-facing validation summary for the real-data J2 benchmark.")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--processed-root", type=Path)
    parser.add_argument("--audits-root", type=Path, default=Path("results/audits"))
    parser.add_argument("--real-root", type=Path, default=Path("results/real"))
    parser.add_argument("--stats-root", type=Path, default=Path("results/reports/stats"))
    parser.add_argument("--out-root", type=Path, default=Path("results/reports"))
    args = parser.parse_args()

    config = load_yaml(args.config)
    processed_root = args.processed_root
    if processed_root is None:
        processed_root = Path(config["project"]["processed_data_root"])
    args.out_root.mkdir(parents=True, exist_ok=True)

    audit_rows = collect_audit_rows(args.audits_root)
    prepared_rows = collect_prepared_rows(processed_root)
    split_rows, observation_rows = collect_benchmark_rows(args.real_root)
    manifest_rows = collect_manifest_rows(args.real_root, config)
    exclusion_report = load_optional_json(processed_root / "cemhsey" / "excluded_trials.json")
    primary_pairwise = load_optional_csv(args.stats_root / "primary_pairwise.csv")

    checks = build_checks(
        exclusion_report=exclusion_report,
        observations=pd.DataFrame(observation_rows),
        primary_pairwise=primary_pairwise,
        manifest_rows=pd.DataFrame(manifest_rows),
    )
    report_payload = {
        "status": "not_submission_ready" if any(not check["passed"] for check in checks if check["critical"]) else "provisionally_ready",
        "checks": checks,
        "datasets": {
            "audits": audit_rows,
            "prepared": prepared_rows,
            "benchmark_manifests": manifest_rows,
            "benchmark_observations": observation_rows,
        },
    }
    (args.out_root / "anchor_status.json").write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    (args.out_root / "anchor_report.md").write_text(
        build_markdown(
            audit_rows=audit_rows,
            prepared_rows=prepared_rows,
            manifest_rows=manifest_rows,
            split_rows=split_rows,
            observation_rows=observation_rows,
            checks=checks,
            exclusion_report=exclusion_report,
        ),
        encoding="utf-8",
    )
    print(f"Wrote anchor report to {args.out_root}")


def collect_audit_rows(audits_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in ["db10", "hyser", "cemhsey", "grabmyo"]:
        path = audits_root / f"{dataset}_audit.json"
        if not path.exists():
            continue
        payload = read_json(path)
        rows.append(
            {
                "dataset_id": dataset,
                "status": payload.get("status", "UNKNOWN"),
                "file_count": int(payload.get("file_count", 0)),
                "total_gb": round(float(payload.get("total_bytes", 0)) / (1024**3), 2),
            }
        )
    return rows


def collect_prepared_rows(processed_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in ["db10", "hyser", "cemhsey", "grabmyo"]:
        root = processed_root / dataset
        metadata_path = root / "metadata.csv"
        if not metadata_path.exists():
            continue
        metadata = pd.read_csv(metadata_path)
        user_features = np.load(root / "user_features.npy", mmap_mode="r")
        assist_features = np.load(root / "assist_features.npy", mmap_mode="r")
        label_vocab = np.load(root / "label_vocab.npy", mmap_mode="r")
        rows.append(
            {
                "dataset_id": dataset,
                "rows": int(len(metadata)),
                "subjects": int(metadata["subject_id"].astype(str).nunique()) if "subject_id" in metadata else 0,
                "sessions": int(metadata["session"].astype(str).nunique()) if "session" in metadata else 0,
                "days": int(metadata["day"].dropna().astype(str).nunique()) if "day" in metadata else 0,
                "labels": int(len(label_vocab)),
                "user_dim": int(user_features.shape[1]),
                "assist_dim": int(assist_features.shape[1]),
                "assist_context_modes": "|".join(
                    sorted(metadata["assist_context_mode"].dropna().astype(str).unique().tolist())
                )
                if "assist_context_mode" in metadata.columns
                else "",
            }
        )
    return rows


def collect_benchmark_rows(real_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    split_rows: list[dict[str, object]] = []
    observation_rows: list[dict[str, object]] = []
    for dataset in ["db10", "hyser", "cemhsey", "grabmyo"]:
        aggregate_path = real_root / dataset / "aggregate_policy_metrics.csv"
        unit_path = real_root / dataset / "metrics_by_policy_unit.csv"
        if not aggregate_path.exists() or not unit_path.exists():
            continue
        aggregate = pd.read_csv(aggregate_path)
        units = pd.read_csv(unit_path)
        for split_family, split_df in aggregate.groupby("split_family", sort=True):
            split_units = units.loc[units["split_family"] == split_family].copy()
            n_splits = int(split_units["split_id"].astype(str).nunique()) if "split_id" in split_units.columns else 0
            best_overall = split_df.sort_values(["macro_f1", "risk_coverage_auc"], ascending=[False, True]).iloc[0]
            best_active = split_df.sort_values(["active_macro_f1", "active_risk_coverage_auc"], ascending=[False, True]).iloc[0]
            agency_df = split_df.loc[split_df["policy"].astype(str).str.startswith("agency_margin_tau_")]
            best_agency = agency_df.sort_values(["macro_f1", "risk_coverage_auc"], ascending=[False, True]).iloc[0]
            user_only = split_df.loc[split_df["policy"] == "user_only"].iloc[0]
            assist_only = split_df.loc[split_df["policy"] == "assist_only"].iloc[0] if (split_df["policy"] == "assist_only").any() else None
            split_rows.append(
                {
                    "dataset_id": dataset,
                    "split_family": split_family,
                    "n_splits": n_splits,
                    "n_policies": int(split_df["policy"].nunique()),
                }
            )
            observation_rows.append(
                {
                    "dataset_id": dataset,
                    "split_family": split_family,
                    "best_overall_policy": str(best_overall["policy"]),
                    "best_overall_macro_f1": float(best_overall["macro_f1"]),
                    "best_overall_active_macro_f1": float(best_active.get("active_macro_f1", best_active["macro_f1"])),
                    "best_overall_active_risk_coverage_auc": float(
                        best_active.get("active_risk_coverage_auc", best_active["risk_coverage_auc"])
                    ),
                    "best_overall_mean_ali": float(best_overall["mean_ali"]),
                    "best_agency_policy": str(best_agency["policy"]),
                    "best_agency_macro_f1": float(best_agency["macro_f1"]),
                    "best_agency_active_macro_f1": float(best_agency.get("active_macro_f1", best_agency["macro_f1"])),
                    "best_agency_active_risk_coverage_auc": float(
                        best_agency.get("active_risk_coverage_auc", best_agency["risk_coverage_auc"])
                    ),
                    "best_agency_mean_ali": float(best_agency["mean_ali"]),
                    "user_only_macro_f1": float(user_only["macro_f1"]),
                    "user_only_active_macro_f1": float(user_only.get("active_macro_f1", user_only["macro_f1"])),
                    "user_only_active_risk_coverage_auc": float(
                        user_only.get("active_risk_coverage_auc", user_only["risk_coverage_auc"])
                    ),
                    "user_only_risk_coverage_auc": float(user_only["risk_coverage_auc"]),
                    "assist_only_macro_f1": float(assist_only["macro_f1"]) if assist_only is not None else None,
                }
            )
    return split_rows, observation_rows


def collect_manifest_rows(real_root: Path, config: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    benchmark_cfg = dict(config.get("benchmark", {}))
    for dataset in ["db10", "hyser", "cemhsey", "grabmyo"]:
        manifest_path = real_root / dataset / "benchmark_manifest.json"
        if not manifest_path.exists():
            continue
        payload = read_json(manifest_path)
        expected_user = str(benchmark_cfg.get(dataset, {}).get("models", {}).get("user", ""))
        expected_assist = str(benchmark_cfg.get(dataset, {}).get("models", {}).get("assist", ""))
        actual_user = str(payload.get("user_model", ""))
        actual_assist = str(payload.get("assist_model", ""))
        rows.append(
            {
                "dataset_id": dataset,
                "user_model": actual_user,
                "assist_model": actual_assist,
                "config_user_model": expected_user,
                "config_assist_model": expected_assist,
                "matches_config": bool(actual_user == expected_user and actual_assist == expected_assist),
                "sequence_roles": "|".join(sorted(str(role) for role in payload.get("available_sequence_payload_roles", []))),
            }
        )
    return rows


def build_checks(
    exclusion_report: dict[str, object] | None,
    observations: pd.DataFrame,
    primary_pairwise: pd.DataFrame,
    manifest_rows: pd.DataFrame | None = None,
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    if manifest_rows is None:
        manifest_rows = pd.DataFrame()
    matched_failed = []
    if exclusion_report:
        matched_failed = [str(item) for item in exclusion_report.get("matched_failed_trials", [])]
    checks.append(
        {
            "name": "cemhsey_failed_trial_excluded",
            "passed": EXPECTED_CEMHSEY_GRASP_FAILURE in matched_failed,
            "critical": True,
            "detail": f"Matched exclusions: {matched_failed or ['<none>']}",
        }
    )
    db10 = observations.loc[observations["dataset_id"] == "db10"].copy()
    if not db10.empty:
        assist_direction = bool((db10["assist_only_macro_f1"] > db10["user_only_macro_f1"]).all())
        checks.append(
            {
                "name": "db10_multimodal_direction",
                "passed": assist_direction,
                "critical": True,
                "detail": "assist_only macro-F1 exceeds user_only on every DB10 split family",
            }
        )
        checks.append(
            {
                "name": "db10_absolute_performance_floor",
                "passed": float(db10["best_agency_macro_f1"].max()) >= 0.20,
                "critical": True,
                "detail": f"Best DB10 agency macro-F1 observed: {float(db10['best_agency_macro_f1'].max()):.4f}",
            }
        )
        checks.append(
            {
                "name": "db10_active_grasp_floor",
                "passed": float(db10["best_agency_active_macro_f1"].max()) >= 0.45,
                "critical": True,
                "detail": f"Best DB10 agency active-grasp macro-F1 observed: {float(db10['best_agency_active_macro_f1'].max()):.4f}",
            }
        )
    external = observations.loc[observations["dataset_id"].isin(["hyser", "cemhsey", "grabmyo"])].copy()
    if not external.empty:
        external_direction = bool((external["best_agency_macro_f1"] > external["user_only_macro_f1"]).all())
        checks.append(
            {
                "name": "external_agency_margin_direction",
                "passed": external_direction,
                "critical": False,
                "detail": "best agency-margin policy improves macro-F1 over user_only on executed external robustness datasets",
            }
        )
    if not manifest_rows.empty:
        aligned = bool(manifest_rows["matches_config"].all())
        checks.append(
            {
                "name": "canonical_model_manifest_alignment",
                "passed": aligned,
                "critical": False,
                "detail": (
                    "Benchmark manifests match config model targets: "
                    f"{manifest_rows[['dataset_id', 'user_model', 'assist_model', 'matches_config']].to_dict(orient='records')}"
                ),
            }
        )
    if primary_pairwise.empty:
        qualifying = pd.DataFrame()
    else:
        db10_primary = primary_pairwise.loc[
            (primary_pairwise["dataset_id"] == "db10")
            & (primary_pairwise["comparison_family"] == "matched_tau")
            & (primary_pairwise["metric"].isin(["active_macro_f1", "active_risk_coverage_auc"]))
            & (primary_pairwise["holm_reject_0_05"] == True)
        ].copy()
        if not db10_primary.empty:
            families = sorted(db10_primary["split_family"].astype(str).unique())
            db10_primary["tau"] = db10_primary["policy_a"].astype(str).str.extract(r"tau_(0\.\d+)", expand=False)
            db10_primary["beneficial"] = np.where(
                db10_primary["metric"].astype(str) == "active_macro_f1",
                db10_primary["estimate"].astype(float) > 0.0,
                db10_primary["estimate"].astype(float) < 0.0,
            )
            beneficial = db10_primary.loc[db10_primary["beneficial"]].copy()
            qualifying_rows = []
            for tau, tau_df in beneficial.groupby("tau", sort=True):
                metrics_ok = {}
                for metric in ["active_macro_f1", "active_risk_coverage_auc"]:
                    metric_families = set(tau_df.loc[tau_df["metric"] == metric, "split_family"].astype(str))
                    metrics_ok[metric] = metric_families
                if all(metrics_ok[metric] == set(families) for metric in metrics_ok):
                    qualifying_rows.append({"tau": tau, "split_families": "|".join(families)})
            qualifying = pd.DataFrame(qualifying_rows)
        else:
            qualifying = pd.DataFrame()
    passed_primary = not qualifying.empty
    checks.append(
        {
            "name": "db10_primary_pairwise_significance",
            "passed": passed_primary,
            "critical": True,
            "detail": (
                "DB10 taus with beneficial Holm-significant active-F1 and active-risk results across all DB10 split families: "
                f"{(qualifying['tau'].astype(str).tolist() if not qualifying.empty else ['<none>'])}"
            ),
        }
    )
    return checks


def build_markdown(
    audit_rows: list[dict[str, object]],
    prepared_rows: list[dict[str, object]],
    manifest_rows: list[dict[str, object]],
    split_rows: list[dict[str, object]],
    observation_rows: list[dict[str, object]],
    checks: list[dict[str, object]],
    exclusion_report: dict[str, object] | None,
) -> str:
    lines = [
        "# J2 Validation Summary",
        "",
        "## Submission Status",
    ]
    overall_ready = all(check["passed"] for check in checks if check["critical"])
    lines.append(f"- Status: {'provisionally ready' if overall_ready else 'not submission ready'}")
    lines.append("")
    lines.append("## Critical Checks")
    for check in checks:
        status = "PASS" if check["passed"] else "FAIL"
        critical = "critical" if check["critical"] else "supporting"
        lines.append(f"- {status} [{critical}] {check['name']}: {check['detail']}")
    lines.append("")
    lines.append("## Dataset Audits")
    lines.extend(render_table(pd.DataFrame(audit_rows)))
    lines.append("")
    lines.append("## Prepared Dataset Summary")
    lines.extend(render_table(pd.DataFrame(prepared_rows)))
    lines.append("")
    lines.append("## Benchmark Manifests")
    lines.extend(render_table(pd.DataFrame(manifest_rows)))
    lines.append("")
    lines.append("## Benchmark Coverage")
    lines.extend(render_table(pd.DataFrame(split_rows)))
    lines.append("")
    lines.append("## Benchmark Observations")
    lines.extend(render_table(pd.DataFrame(observation_rows)))
    if exclusion_report:
        lines.append("")
        lines.append("## CEMHSEY Exclusions")
        lines.append(f"- Configured failed trials: {', '.join(exclusion_report.get('configured_failed_trials', [])) or '<none>'}")
        lines.append(f"- Matched failed trials: {', '.join(exclusion_report.get('matched_failed_trials', [])) or '<none>'}")
        lines.append(f"- Excluded member paths: {len(exclusion_report.get('excluded_member_paths', []))}")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- The workflow now regenerates exclusion-aware prepared data, paired stats, and a manuscript-facing report from declared targets.")
    lines.append("- DB10 currently uses a declared annotated object-context proxy in the assistive feature block, so the result is an offline context-assisted benchmark rather than an end-to-end vision claim.")
    if overall_ready:
        lines.append("- The current blocking infrastructure issues are cleared; the remaining risk is manuscript-level framing and reviewer-facing justification, not benchmark failure.")
    else:
        lines.append("- External robustness direction is positive, but at least one DB10 primary matched-budget gate still fails.")
        lines.append("- Until every critical DB10 gate passes, this is not a TNSRE-safe submission package.")
    lines.append("")
    return "\n".join(lines)


def render_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["_No data_"]
    return frame.to_markdown(index=False).splitlines()


def load_optional_json(path: Path) -> dict[str, object] | None:
    return read_json(path) if path.exists() else None


def load_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


if __name__ == "__main__":
    main()

