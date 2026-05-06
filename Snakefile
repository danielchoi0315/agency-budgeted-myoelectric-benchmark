configfile: "config/config.yaml"

RAW_ROOT = config["project"]["raw_data_root"].replace("\\", "/")
INTERIM_ROOT = config["project"]["interim_data_root"].replace("\\", "/")
PROCESSED_ROOT = config["project"]["processed_data_root"].replace("\\", "/")
RESULTS_ROOT = config["project"]["results_root"].replace("\\", "/")
LITERATURE_ANCHORS = config["project"]["literature_anchors_path"].replace("\\", "/")
LITERATURE_COMPARABILITY = config["project"]["literature_comparability_path"].replace("\\", "/")
EXCLUSIONS_CONFIG = "config/exclusions.yaml"
J1_SOURCE_ROOT = config["benchmark"]["j1"]["prepare"]["source_root"].replace("\\", "/")
J1_USER_MODEL = config["benchmark"]["j1"]["models"]["user"]
J1_ASSIST_MODEL = config["benchmark"]["j1"]["models"]["assist"]
DB10_CONTEXT_MODE = config["benchmark"]["db10"]["prepare"]["context_mode"]
DB10_INCLUDE_REST = bool(config["benchmark"]["db10"]["prepare"]["include_rest"])
DB10_EXCLUDE_REST_FLAG = "" if DB10_INCLUDE_REST else "--exclude-rest"
DB10_USER_MODEL = config["benchmark"]["db10"]["models"]["user"]
DB10_ASSIST_MODEL = config["benchmark"]["db10"]["models"]["assist"]
HYSER_USER_MODEL = config["benchmark"]["hyser"]["models"]["user"]
HYSER_ASSIST_MODEL = config["benchmark"]["hyser"]["models"]["assist"]
CEMHSEY_USER_MODEL = config["benchmark"]["cemhsey"]["models"]["user"]
CEMHSEY_ASSIST_MODEL = config["benchmark"]["cemhsey"]["models"]["assist"]
GRABMYO_USER_MODEL = config["benchmark"]["grabmyo"]["models"]["user"]
GRABMYO_ASSIST_MODEL = config["benchmark"]["grabmyo"]["models"]["assist"]

rule all:
    input:
        f"{RESULTS_ROOT}/synthetic_smoke/aggregate_policy_metrics.csv",
        f"{RESULTS_ROOT}/synthetic_smoke/policy_tradeoff.png",
        f"{RESULTS_ROOT}/reproducibility/no_raw_data.ok",

rule real_all:
    input:
        f"{RESULTS_ROOT}/audits/db10_audit.json",
        f"{RESULTS_ROOT}/audits/hyser_audit.json",
        f"{RESULTS_ROOT}/audits/cemhsey_audit.json",
        f"{RESULTS_ROOT}/audits/grabmyo_audit.json",
        f"{RESULTS_ROOT}/reproducibility/no_raw_data.ok",

rule real_benchmarks:
    input:
        f"{RESULTS_ROOT}/real/db10/aggregate_policy_metrics.csv",
        f"{RESULTS_ROOT}/real/hyser/aggregate_policy_metrics.csv",
        f"{RESULTS_ROOT}/real/cemhsey/aggregate_policy_metrics.csv",
        f"{RESULTS_ROOT}/real/grabmyo/aggregate_policy_metrics.csv",
        f"{RESULTS_ROOT}/reproducibility/no_raw_data.ok",

rule real_reports:
    input:
        f"{RESULTS_ROOT}/reports/stats/policy_ci.csv",
        f"{RESULTS_ROOT}/reports/stats/primary_pairwise.csv",
        f"{RESULTS_ROOT}/reports/stats/secondary_pairwise.csv",
        f"{RESULTS_ROOT}/reports/anchor_report.md",
        f"{RESULTS_ROOT}/reports/anchor_status.json",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_anchor_reproductions.json",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_anchor_reproductions.md",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_publication_metrics_by_policy_unit.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_publication_aggregate_policy_metrics.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_confidence_gate_match_summary.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_confidence_gate_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_confidence_gate_iso_budget_unit_deltas.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_confidence_gate_iso_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_plain_conf_exact_budget_selection_by_unit.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_confidence_gate_exact_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_earliest_safe_by_episode.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_earliest_safe_by_unit.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_earliest_safe_summary.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_earliest_safe_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_earliest_safe_iso_budget_unit_deltas.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_earliest_safe_iso_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_earliest_safe_exact_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_matched_budget_ablation.png",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_earliest_safe.png",
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_publication_metrics_by_policy_unit.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_publication_aggregate_policy_metrics.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_confidence_gate_match_summary.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_confidence_gate_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_confidence_gate_iso_budget_unit_deltas.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_confidence_gate_iso_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_plain_conf_exact_budget_selection_by_unit.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_confidence_gate_exact_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_earliest_safe_by_episode.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_earliest_safe_by_unit.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_earliest_safe_summary.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_earliest_safe_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_earliest_safe_iso_budget_unit_deltas.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_earliest_safe_iso_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_earliest_safe_exact_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_matched_budget_ablation.png",
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_earliest_safe.png",
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_publication_metrics_by_policy_unit.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_publication_aggregate_policy_metrics.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_confidence_gate_match_summary.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_confidence_gate_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_confidence_gate_iso_budget_unit_deltas.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_confidence_gate_iso_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_plain_conf_exact_budget_selection_by_unit.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_confidence_gate_exact_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_earliest_safe_by_episode.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_earliest_safe_by_unit.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_earliest_safe_summary.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_earliest_safe_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_earliest_safe_iso_budget_unit_deltas.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_earliest_safe_iso_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_earliest_safe_exact_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_matched_budget_ablation.png",
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_earliest_safe.png",
        f"{RESULTS_ROOT}/reports/publication_clean/direct_comparability_matrix.md",
        f"{RESULTS_ROOT}/reports/publication_clean/direct_comparability_matrix.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/direct_comparability_matrix.json",
        f"{RESULTS_ROOT}/reports/publication_clean/sota_report.md",
        f"{RESULTS_ROOT}/reports/publication_clean/sota_status.json",
        f"{RESULTS_ROOT}/reproducibility/no_raw_data.ok",

rule j1_all:
    input:
        f"{RESULTS_ROOT}/real/j1/aggregate_policy_metrics.csv",
        f"{RESULTS_ROOT}/real/j1/benchmark_provenance.json",
        f"{RESULTS_ROOT}/reports/j1_open/stats/policy_ci.csv",
        f"{RESULTS_ROOT}/reports/j1_open/stats/primary_pairwise.csv",
        f"{RESULTS_ROOT}/reports/j1_open/stats/secondary_pairwise.csv",
        f"{RESULTS_ROOT}/reports/j1_open/stats/stats_provenance.json",
        f"{RESULTS_ROOT}/reports/j1_open/split_report.json",
        f"{RESULTS_ROOT}/reports/j1_open/split_report.md",
        f"{RESULTS_ROOT}/reports/j1_open/split_report_provenance.json",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_publication_metrics_by_policy_unit.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_publication_aggregate_policy_metrics.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_confidence_gate_match_summary.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_confidence_gate_pairwise.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_confidence_gate_iso_budget_unit_deltas.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_confidence_gate_iso_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_plain_conf_exact_budget_selection_by_unit.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_confidence_gate_exact_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_earliest_safe_by_episode.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_earliest_safe_by_unit.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_earliest_safe_summary.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_earliest_safe_pairwise.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_earliest_safe_iso_budget_unit_deltas.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_earliest_safe_iso_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_earliest_safe_exact_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_matched_budget_ablation.png",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_earliest_safe.png",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_publication_provenance.json",
        f"{RESULTS_ROOT}/reports/j1_open/j1_scorecard.json",
        f"{RESULTS_ROOT}/reports/j1_open/j1_scorecard.md",
        f"{RESULTS_ROOT}/reports/j1_open/j1_frontier_readout.json",
        f"{RESULTS_ROOT}/reports/j1_open/j1_frontier_readout.md",
        f"{RESULTS_ROOT}/reports/j1_open/frontier/frontier_points.csv",
        f"{RESULTS_ROOT}/reports/j1_open/frontier/pareto_frontier.csv",
        f"{RESULTS_ROOT}/reports/j1_open/frontier/frontier_overlap.csv",
        f"{RESULTS_ROOT}/reports/j1_open/frontier/non_dominated_summary.csv",
        f"{RESULTS_ROOT}/reports/j1_open/frontier/universal_tau_regret.csv",
        f"{RESULTS_ROOT}/reports/j1_open/frontier/frontier_unit_deltas.csv",
        f"{RESULTS_ROOT}/reports/j1_open/frontier/budget_match_summary.csv",
        f"{RESULTS_ROOT}/reports/j1_open/frontier/iso_budget_claim_summary.csv",
        f"{RESULTS_ROOT}/reports/j1_open/frontier/exact_budget_claim_summary.csv",
        f"{RESULTS_ROOT}/reports/j1_open/frontier/operating_point_instability.json",
        f"{RESULTS_ROOT}/reports/j1_open/frontier/frontier_atlas.png",
        f"{RESULTS_ROOT}/reports/j1_open/frontier/safety_timing_frontier.png",
        f"{RESULTS_ROOT}/reports/j1_open/frontier/j1_frontier_package.json",
        f"{RESULTS_ROOT}/reports/j1_open/frontier/j1_frontier_package.md",
        f"{RESULTS_ROOT}/reports/j1_open/frontier/j1_frontier_package_provenance.json",
        f"{RESULTS_ROOT}/reports/publication_clean/exact_budget_instability_by_tau.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/exact_budget_instability_summary.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/exact_budget_instability_status.json",
        f"{RESULTS_ROOT}/reports/publication_clean/exact_budget_instability_report.md",

rule synthetic_smoke:
    input:
        "scripts/run_synthetic_smoke.py"
    output:
        metrics=f"{RESULTS_ROOT}/synthetic_smoke/aggregate_policy_metrics.csv",
        figure=f"{RESULTS_ROOT}/synthetic_smoke/policy_tradeoff.png",
    shell:
        "python scripts/run_synthetic_smoke.py --out {RESULTS_ROOT}/synthetic_smoke"

rule no_raw_data_check:
    input:
        "scripts/check_no_raw_data.py"
    output:
        f"{RESULTS_ROOT}/reproducibility/no_raw_data.ok"
    shell:
        "python scripts/check_no_raw_data.py --root . --ok-file {output}"

rule audit_db10:
    input:
        "scripts/audit_dataset.py"
    output:
        f"{RESULTS_ROOT}/audits/db10_audit.json"
    shell:
        "python scripts/audit_dataset.py --dataset db10 --root {RAW_ROOT}/db10 --out {output} --hash-files --require-pass"

rule audit_hyser:
    input:
        "scripts/audit_dataset.py"
    output:
        f"{RESULTS_ROOT}/audits/hyser_audit.json"
    shell:
        "python scripts/audit_dataset.py --dataset hyser --root {RAW_ROOT}/hyser --out {output} --hash-files --require-pass"

rule audit_cemhsey:
    input:
        "scripts/audit_dataset.py"
    output:
        f"{RESULTS_ROOT}/audits/cemhsey_audit.json"
    shell:
        "python scripts/audit_dataset.py --dataset cemhsey --root {RAW_ROOT}/cemhsey --out {output} --hash-files --require-pass"

rule audit_grabmyo:
    input:
        "scripts/audit_dataset.py"
    output:
        f"{RESULTS_ROOT}/audits/grabmyo_audit.json"
    shell:
        "python scripts/audit_dataset.py --dataset grabmyo --root {RAW_ROOT}/grabmyo --out {output} --hash-files --require-pass"

rule audit_j1:
    input:
        "scripts/audit_dataset.py"
    output:
        f"{RESULTS_ROOT}/audits/j1_audit.json"
    shell:
        "python scripts/audit_dataset.py --dataset j1 --root {J1_SOURCE_ROOT} --out {output} --hash-files --require-pass"

rule fetch_hyser_labels:
    input:
        "scripts/fetch_hyser_pr_labels.py"
    output:
        f"{RESULTS_ROOT}/downloads/hyser_labels.ok"
    shell:
        "python scripts/fetch_hyser_pr_labels.py --root {RAW_ROOT}/hyser --ok-file {output}"

rule prepare_db10:
    input:
        f"{RESULTS_ROOT}/audits/db10_audit.json",
        "config/config.yaml",
        "scripts/prepare_real_dataset.py",
        "src/j2bench/realdata.py"
    output:
        f"{PROCESSED_ROOT}/db10/metadata.csv",
        f"{PROCESSED_ROOT}/db10/user_features.npy",
        f"{PROCESSED_ROOT}/db10/assist_features.npy",
        f"{PROCESSED_ROOT}/db10/labels.npy",
        f"{PROCESSED_ROOT}/db10/label_vocab.npy",
        directory(f"{PROCESSED_ROOT}/db10/sequence_payloads"),
        f"{PROCESSED_ROOT}/db10/_SUCCESS",
    shell:
        "python scripts/prepare_real_dataset.py --dataset db10 --root {RAW_ROOT}/db10 --out {PROCESSED_ROOT}/db10 "
        "--db10-context-mode {DB10_CONTEXT_MODE} "
        "--db10-sequences "
        "{DB10_EXCLUDE_REST_FLAG}"

rule prepare_hyser:
    input:
        f"{RESULTS_ROOT}/audits/hyser_audit.json",
        f"{RESULTS_ROOT}/downloads/hyser_labels.ok",
        "scripts/prepare_real_dataset.py",
        "src/j2bench/realdata.py"
    output:
        f"{PROCESSED_ROOT}/hyser/metadata.csv",
        f"{PROCESSED_ROOT}/hyser/user_features.npy",
        f"{PROCESSED_ROOT}/hyser/assist_features.npy",
        f"{PROCESSED_ROOT}/hyser/labels.npy",
        f"{PROCESSED_ROOT}/hyser/label_vocab.npy",
    shell:
        "python scripts/prepare_real_dataset.py --dataset hyser --root {RAW_ROOT}/hyser --out {PROCESSED_ROOT}/hyser"

rule prepare_cemhsey:
    input:
        f"{RESULTS_ROOT}/audits/cemhsey_audit.json",
        EXCLUSIONS_CONFIG,
        "scripts/prepare_cemhsey_resumable.py",
        "src/j2bench/realdata.py",
    output:
        f"{PROCESSED_ROOT}/cemhsey/metadata.csv",
        f"{PROCESSED_ROOT}/cemhsey/user_features.npy",
        f"{PROCESSED_ROOT}/cemhsey/assist_features.npy",
        f"{PROCESSED_ROOT}/cemhsey/labels.npy",
        f"{PROCESSED_ROOT}/cemhsey/label_vocab.npy",
        f"{PROCESSED_ROOT}/cemhsey/excluded_trials.json",
    shell:
        "python scripts/prepare_cemhsey_resumable.py --root {RAW_ROOT}/cemhsey --parts-dir {PROCESSED_ROOT}/cemhsey_parts --out {PROCESSED_ROOT}/cemhsey"

rule prepare_grabmyo:
    input:
        f"{RESULTS_ROOT}/audits/grabmyo_audit.json",
        "scripts/prepare_real_dataset.py",
        "src/j2bench/realdata.py"
    output:
        f"{PROCESSED_ROOT}/grabmyo/metadata.csv",
        f"{PROCESSED_ROOT}/grabmyo/user_features.npy",
        f"{PROCESSED_ROOT}/grabmyo/assist_features.npy",
        f"{PROCESSED_ROOT}/grabmyo/labels.npy",
        f"{PROCESSED_ROOT}/grabmyo/label_vocab.npy",
        f"{PROCESSED_ROOT}/grabmyo/_SUCCESS",
    shell:
        "python scripts/prepare_real_dataset.py --dataset grabmyo --root {RAW_ROOT}/grabmyo --out {PROCESSED_ROOT}/grabmyo"

rule prepare_j1:
    input:
        f"{RESULTS_ROOT}/audits/j1_audit.json",
        "config/config.yaml",
        "scripts/prepare_real_dataset.py",
        "src/j2bench/artifact_schemas.py",
        "src/j2bench/j1_contract.py",
        "src/j2bench/provenance.py",
        "src/j2bench/realdata.py"
    output:
        f"{PROCESSED_ROOT}/j1/metadata.csv",
        f"{PROCESSED_ROOT}/j1/user_features.npy",
        f"{PROCESSED_ROOT}/j1/assist_features.npy",
        f"{PROCESSED_ROOT}/j1/labels.npy",
        f"{PROCESSED_ROOT}/j1/label_vocab.npy",
        f"{PROCESSED_ROOT}/j1/prepared_summary.json",
        f"{PROCESSED_ROOT}/j1/prepared_provenance.json",
        f"{PROCESSED_ROOT}/j1/_SUCCESS",
    shell:
        "python scripts/prepare_real_dataset.py --dataset j1 --root {J1_SOURCE_ROOT} --out {PROCESSED_ROOT}/j1"

rule benchmark_db10:
    input:
        f"{PROCESSED_ROOT}/db10/metadata.csv",
        f"{PROCESSED_ROOT}/db10/user_features.npy",
        f"{PROCESSED_ROOT}/db10/assist_features.npy",
        f"{PROCESSED_ROOT}/db10/labels.npy",
        f"{PROCESSED_ROOT}/db10/label_vocab.npy",
        f"{PROCESSED_ROOT}/db10/sequence_payloads",
        f"{PROCESSED_ROOT}/db10/_SUCCESS",
        "config/config.yaml",
        "scripts/run_real_baseline.py",
        "src/j2bench/metrics.py",
        "src/j2bench/policies.py",
        "src/j2bench/real_benchmark.py",
        "src/j2bench/realdata.py",
    output:
        f"{RESULTS_ROOT}/real/db10/metrics_by_policy_unit.csv",
        f"{RESULTS_ROOT}/real/db10/aggregate_policy_metrics.csv",
        f"{RESULTS_ROOT}/real/db10/benchmark_manifest.json",
        f"{RESULTS_ROOT}/real/db10/policy_tradeoff.png",
    shell:
        "python scripts/run_real_baseline.py --dataset db10 --prepared-root {PROCESSED_ROOT}/db10 --out {RESULTS_ROOT}/real/db10 "
        "--user-model {DB10_USER_MODEL} --assist-model {DB10_ASSIST_MODEL}"

rule benchmark_hyser:
    input:
        f"{PROCESSED_ROOT}/hyser/metadata.csv",
        f"{PROCESSED_ROOT}/hyser/user_features.npy",
        f"{PROCESSED_ROOT}/hyser/assist_features.npy",
        f"{PROCESSED_ROOT}/hyser/labels.npy",
        f"{PROCESSED_ROOT}/hyser/label_vocab.npy",
        "config/config.yaml",
        "scripts/run_real_baseline.py",
        "src/j2bench/metrics.py",
        "src/j2bench/policies.py",
        "src/j2bench/real_benchmark.py",
        "src/j2bench/realdata.py",
    output:
        f"{RESULTS_ROOT}/real/hyser/metrics_by_policy_unit.csv",
        f"{RESULTS_ROOT}/real/hyser/aggregate_policy_metrics.csv",
        f"{RESULTS_ROOT}/real/hyser/benchmark_manifest.json",
        f"{RESULTS_ROOT}/real/hyser/policy_tradeoff.png",
    shell:
        "python scripts/run_real_baseline.py --dataset hyser --prepared-root {PROCESSED_ROOT}/hyser --out {RESULTS_ROOT}/real/hyser "
        "--user-model {HYSER_USER_MODEL} --assist-model {HYSER_ASSIST_MODEL}"

rule benchmark_j1:
    input:
        f"{PROCESSED_ROOT}/j1/metadata.csv",
        f"{PROCESSED_ROOT}/j1/user_features.npy",
        f"{PROCESSED_ROOT}/j1/assist_features.npy",
        f"{PROCESSED_ROOT}/j1/labels.npy",
        f"{PROCESSED_ROOT}/j1/label_vocab.npy",
        f"{PROCESSED_ROOT}/j1/prepared_summary.json",
        f"{PROCESSED_ROOT}/j1/prepared_provenance.json",
        f"{PROCESSED_ROOT}/j1/_SUCCESS",
        "config/config.yaml",
        "scripts/run_real_baseline.py",
        "src/j2bench/artifact_schemas.py",
        "src/j2bench/metrics.py",
        "src/j2bench/policies.py",
        "src/j2bench/provenance.py",
        "src/j2bench/real_benchmark.py",
        "src/j2bench/realdata.py",
    output:
        f"{RESULTS_ROOT}/real/j1/metrics_by_policy_unit.csv",
        f"{RESULTS_ROOT}/real/j1/aggregate_policy_metrics.csv",
        f"{RESULTS_ROOT}/real/j1/benchmark_manifest.json",
        f"{RESULTS_ROOT}/real/j1/policy_tradeoff.png",
        f"{RESULTS_ROOT}/real/j1/benchmark_provenance.json",
    shell:
        "python scripts/run_real_baseline.py --dataset j1 --prepared-root {PROCESSED_ROOT}/j1 --out {RESULTS_ROOT}/real/j1 "
        "--user-model {J1_USER_MODEL} --assist-model {J1_ASSIST_MODEL}"

rule benchmark_cemhsey:
    input:
        f"{PROCESSED_ROOT}/cemhsey/metadata.csv",
        f"{PROCESSED_ROOT}/cemhsey/user_features.npy",
        f"{PROCESSED_ROOT}/cemhsey/assist_features.npy",
        f"{PROCESSED_ROOT}/cemhsey/labels.npy",
        f"{PROCESSED_ROOT}/cemhsey/label_vocab.npy",
        f"{PROCESSED_ROOT}/cemhsey/excluded_trials.json",
        "config/config.yaml",
        "scripts/run_real_baseline.py",
        "src/j2bench/metrics.py",
        "src/j2bench/policies.py",
        "src/j2bench/real_benchmark.py",
        "src/j2bench/realdata.py",
    output:
        f"{RESULTS_ROOT}/real/cemhsey/metrics_by_policy_unit.csv",
        f"{RESULTS_ROOT}/real/cemhsey/aggregate_policy_metrics.csv",
        f"{RESULTS_ROOT}/real/cemhsey/benchmark_manifest.json",
        f"{RESULTS_ROOT}/real/cemhsey/policy_tradeoff.png",
    shell:
        "python scripts/run_real_baseline.py --dataset cemhsey --prepared-root {PROCESSED_ROOT}/cemhsey --out {RESULTS_ROOT}/real/cemhsey "
        "--user-model {CEMHSEY_USER_MODEL} --assist-model {CEMHSEY_ASSIST_MODEL}"

rule benchmark_grabmyo:
    input:
        f"{PROCESSED_ROOT}/grabmyo/metadata.csv",
        f"{PROCESSED_ROOT}/grabmyo/user_features.npy",
        f"{PROCESSED_ROOT}/grabmyo/assist_features.npy",
        f"{PROCESSED_ROOT}/grabmyo/labels.npy",
        f"{PROCESSED_ROOT}/grabmyo/label_vocab.npy",
        f"{PROCESSED_ROOT}/grabmyo/_SUCCESS",
        "config/config.yaml",
        "scripts/run_real_baseline.py",
        "src/j2bench/metrics.py",
        "src/j2bench/policies.py",
        "src/j2bench/real_benchmark.py",
        "src/j2bench/realdata.py",
    output:
        f"{RESULTS_ROOT}/real/grabmyo/metrics_by_policy_unit.csv",
        f"{RESULTS_ROOT}/real/grabmyo/aggregate_policy_metrics.csv",
        f"{RESULTS_ROOT}/real/grabmyo/benchmark_manifest.json",
        f"{RESULTS_ROOT}/real/grabmyo/policy_tradeoff.png",
    shell:
        "python scripts/run_real_baseline.py --dataset grabmyo --prepared-root {PROCESSED_ROOT}/grabmyo --out {RESULTS_ROOT}/real/grabmyo "
        "--user-model {GRABMYO_USER_MODEL} --assist-model {GRABMYO_ASSIST_MODEL}"

rule stats_j1:
    input:
        f"{RESULTS_ROOT}/real/j1/metrics_by_policy_unit.csv",
        "scripts/postprocess_real_stats.py",
        "src/j2bench/provenance.py",
        "src/j2bench/stats.py",
    output:
        f"{RESULTS_ROOT}/reports/j1_open/stats/policy_ci.csv",
        f"{RESULTS_ROOT}/reports/j1_open/stats/primary_pairwise.csv",
        f"{RESULTS_ROOT}/reports/j1_open/stats/secondary_pairwise.csv",
        f"{RESULTS_ROOT}/reports/j1_open/stats/stats_provenance.json",
    shell:
        "python scripts/postprocess_real_stats.py "
        "--metrics {RESULTS_ROOT}/real/j1/metrics_by_policy_unit.csv "
        "--real-root {RESULTS_ROOT}/real --out-root {RESULTS_ROOT}/reports/j1_open/stats"

rule j1_split_report:
    input:
        f"{PROCESSED_ROOT}/j1/metadata.csv",
        f"{PROCESSED_ROOT}/j1/_SUCCESS",
        "scripts/build_split_report.py",
        "src/j2bench/provenance.py",
        "src/j2bench/real_benchmark.py",
        "src/j2bench/splits.py",
    output:
        f"{RESULTS_ROOT}/reports/j1_open/split_report.json",
        f"{RESULTS_ROOT}/reports/j1_open/split_report.md",
        f"{RESULTS_ROOT}/reports/j1_open/split_report_provenance.json",
    shell:
        "python scripts/build_split_report.py "
        "--prepared-root {PROCESSED_ROOT}/j1 --out-root {RESULTS_ROOT}/reports/j1_open"

rule stats_real:
    input:
        f"{RESULTS_ROOT}/real/db10/metrics_by_policy_unit.csv",
        f"{RESULTS_ROOT}/real/hyser/metrics_by_policy_unit.csv",
        f"{RESULTS_ROOT}/real/cemhsey/metrics_by_policy_unit.csv",
        f"{RESULTS_ROOT}/real/grabmyo/metrics_by_policy_unit.csv",
        "scripts/postprocess_real_stats.py",
        "src/j2bench/stats.py",
    output:
        f"{RESULTS_ROOT}/reports/stats/policy_ci.csv",
        f"{RESULTS_ROOT}/reports/stats/primary_pairwise.csv",
        f"{RESULTS_ROOT}/reports/stats/secondary_pairwise.csv",
    shell:
        "python scripts/postprocess_real_stats.py "
        "--metrics {RESULTS_ROOT}/real/db10/metrics_by_policy_unit.csv "
        "--metrics {RESULTS_ROOT}/real/hyser/metrics_by_policy_unit.csv "
        "--metrics {RESULTS_ROOT}/real/cemhsey/metrics_by_policy_unit.csv "
        "--metrics {RESULTS_ROOT}/real/grabmyo/metrics_by_policy_unit.csv "
        "--real-root {RESULTS_ROOT}/real --out-root {RESULTS_ROOT}/reports/stats"

rule anchor_report:
    input:
        f"{RESULTS_ROOT}/audits/db10_audit.json",
        f"{RESULTS_ROOT}/audits/hyser_audit.json",
        f"{RESULTS_ROOT}/audits/cemhsey_audit.json",
        f"{RESULTS_ROOT}/audits/grabmyo_audit.json",
        f"{RESULTS_ROOT}/real/db10/aggregate_policy_metrics.csv",
        f"{RESULTS_ROOT}/real/hyser/aggregate_policy_metrics.csv",
        f"{RESULTS_ROOT}/real/cemhsey/aggregate_policy_metrics.csv",
        f"{RESULTS_ROOT}/real/grabmyo/aggregate_policy_metrics.csv",
        f"{RESULTS_ROOT}/reports/stats/policy_ci.csv",
        f"{RESULTS_ROOT}/reports/stats/primary_pairwise.csv",
        f"{RESULTS_ROOT}/reports/stats/secondary_pairwise.csv",
        f"{PROCESSED_ROOT}/db10/metadata.csv",
        f"{PROCESSED_ROOT}/db10/user_features.npy",
        f"{PROCESSED_ROOT}/db10/assist_features.npy",
        f"{PROCESSED_ROOT}/db10/label_vocab.npy",
        f"{PROCESSED_ROOT}/hyser/metadata.csv",
        f"{PROCESSED_ROOT}/hyser/user_features.npy",
        f"{PROCESSED_ROOT}/hyser/assist_features.npy",
        f"{PROCESSED_ROOT}/hyser/label_vocab.npy",
        f"{PROCESSED_ROOT}/cemhsey/metadata.csv",
        f"{PROCESSED_ROOT}/cemhsey/user_features.npy",
        f"{PROCESSED_ROOT}/cemhsey/assist_features.npy",
        f"{PROCESSED_ROOT}/cemhsey/label_vocab.npy",
        f"{PROCESSED_ROOT}/cemhsey/excluded_trials.json",
        f"{PROCESSED_ROOT}/grabmyo/metadata.csv",
        f"{PROCESSED_ROOT}/grabmyo/user_features.npy",
        f"{PROCESSED_ROOT}/grabmyo/assist_features.npy",
        f"{PROCESSED_ROOT}/grabmyo/label_vocab.npy",
        "scripts/build_anchor_report.py",
    output:
        f"{RESULTS_ROOT}/reports/anchor_report.md",
        f"{RESULTS_ROOT}/reports/anchor_status.json",
    shell:
        "python scripts/build_anchor_report.py --config config/config.yaml --processed-root {PROCESSED_ROOT} --audits-root {RESULTS_ROOT}/audits --real-root {RESULTS_ROOT}/real --stats-root {RESULTS_ROOT}/reports/stats --out-root {RESULTS_ROOT}/reports"

rule db10_publication_extension:
    input:
        f"{PROCESSED_ROOT}/db10/metadata.csv",
        f"{PROCESSED_ROOT}/db10/user_features.npy",
        f"{PROCESSED_ROOT}/db10/assist_features.npy",
        f"{PROCESSED_ROOT}/db10/labels.npy",
        f"{PROCESSED_ROOT}/db10/label_vocab.npy",
        f"{PROCESSED_ROOT}/db10/_SUCCESS",
        f"{RESULTS_ROOT}/real/db10/metrics_by_policy_unit.csv",
        "scripts/run_db10_publication_extension.py",
        "scripts/run_publication_extension.py",
        "src/j2bench/figures.py",
        "src/j2bench/metrics.py",
        "src/j2bench/policies.py",
        "src/j2bench/publication.py",
        "src/j2bench/real_benchmark.py",
        "src/j2bench/realdata.py",
        "src/j2bench/stats.py",
    output:
        f"{RESULTS_ROOT}/reports/publication_clean/db10_anchor_reproductions.json",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_anchor_reproductions.md",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_publication_metrics_by_policy_unit.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_publication_aggregate_policy_metrics.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_confidence_gate_match_summary.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_confidence_gate_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_confidence_gate_iso_budget_unit_deltas.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_confidence_gate_iso_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_plain_conf_exact_budget_selection_by_unit.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_confidence_gate_exact_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_earliest_safe_by_episode.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_earliest_safe_by_unit.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_earliest_safe_summary.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_earliest_safe_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_earliest_safe_iso_budget_unit_deltas.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_earliest_safe_iso_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_earliest_safe_exact_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_matched_budget_ablation.png",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_earliest_safe.png",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_publication_extension_manifest.json",
    shell:
        "python scripts/run_db10_publication_extension.py --prepared-root {PROCESSED_ROOT}/db10 "
        "--out-root {RESULTS_ROOT}/reports/publication_clean "
        "--user-model {DB10_USER_MODEL} --assist-model {DB10_ASSIST_MODEL}"

rule hyser_publication_extension:
    input:
        f"{PROCESSED_ROOT}/hyser/metadata.csv",
        f"{PROCESSED_ROOT}/hyser/user_features.npy",
        f"{PROCESSED_ROOT}/hyser/assist_features.npy",
        f"{PROCESSED_ROOT}/hyser/labels.npy",
        f"{PROCESSED_ROOT}/hyser/label_vocab.npy",
        f"{PROCESSED_ROOT}/hyser/_SUCCESS",
        f"{RESULTS_ROOT}/real/hyser/metrics_by_policy_unit.csv",
        "scripts/run_publication_extension.py",
        "src/j2bench/figures.py",
        "src/j2bench/metrics.py",
        "src/j2bench/policies.py",
        "src/j2bench/publication.py",
        "src/j2bench/real_benchmark.py",
        "src/j2bench/realdata.py",
        "src/j2bench/stats.py",
    output:
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_publication_metrics_by_policy_unit.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_publication_aggregate_policy_metrics.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_confidence_gate_match_summary.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_confidence_gate_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_confidence_gate_iso_budget_unit_deltas.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_confidence_gate_iso_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_plain_conf_exact_budget_selection_by_unit.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_confidence_gate_exact_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_earliest_safe_by_episode.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_earliest_safe_by_unit.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_earliest_safe_summary.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_earliest_safe_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_earliest_safe_iso_budget_unit_deltas.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_earliest_safe_iso_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_earliest_safe_exact_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_matched_budget_ablation.png",
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_earliest_safe.png",
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_publication_extension_manifest.json",
    shell:
        "python scripts/run_publication_extension.py --dataset hyser --prepared-root {PROCESSED_ROOT}/hyser "
        "--out-root {RESULTS_ROOT}/reports/publication_clean "
        "--user-model {HYSER_USER_MODEL} --assist-model {HYSER_ASSIST_MODEL}"

rule cemhsey_publication_extension:
    input:
        f"{PROCESSED_ROOT}/cemhsey/metadata.csv",
        f"{PROCESSED_ROOT}/cemhsey/user_features.npy",
        f"{PROCESSED_ROOT}/cemhsey/assist_features.npy",
        f"{PROCESSED_ROOT}/cemhsey/labels.npy",
        f"{PROCESSED_ROOT}/cemhsey/label_vocab.npy",
        f"{PROCESSED_ROOT}/cemhsey/_SUCCESS",
        f"{RESULTS_ROOT}/real/cemhsey/metrics_by_policy_unit.csv",
        "scripts/run_publication_extension.py",
        "src/j2bench/figures.py",
        "src/j2bench/metrics.py",
        "src/j2bench/policies.py",
        "src/j2bench/publication.py",
        "src/j2bench/real_benchmark.py",
        "src/j2bench/realdata.py",
        "src/j2bench/stats.py",
    output:
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_publication_metrics_by_policy_unit.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_publication_aggregate_policy_metrics.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_confidence_gate_match_summary.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_confidence_gate_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_confidence_gate_iso_budget_unit_deltas.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_confidence_gate_iso_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_plain_conf_exact_budget_selection_by_unit.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_confidence_gate_exact_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_earliest_safe_by_episode.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_earliest_safe_by_unit.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_earliest_safe_summary.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_earliest_safe_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_earliest_safe_iso_budget_unit_deltas.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_earliest_safe_iso_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_earliest_safe_exact_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_matched_budget_ablation.png",
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_earliest_safe.png",
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_publication_extension_manifest.json",
    shell:
        "python scripts/run_publication_extension.py --dataset cemhsey --prepared-root {PROCESSED_ROOT}/cemhsey "
        "--out-root {RESULTS_ROOT}/reports/publication_clean "
        "--user-model {CEMHSEY_USER_MODEL} --assist-model {CEMHSEY_ASSIST_MODEL}"

rule j1_publication_extension:
    input:
        f"{PROCESSED_ROOT}/j1/metadata.csv",
        f"{PROCESSED_ROOT}/j1/user_features.npy",
        f"{PROCESSED_ROOT}/j1/assist_features.npy",
        f"{PROCESSED_ROOT}/j1/labels.npy",
        f"{PROCESSED_ROOT}/j1/label_vocab.npy",
        f"{PROCESSED_ROOT}/j1/prepared_summary.json",
        f"{PROCESSED_ROOT}/j1/_SUCCESS",
        f"{RESULTS_ROOT}/real/j1/metrics_by_policy_unit.csv",
        "scripts/run_publication_extension.py",
        "src/j2bench/artifact_schemas.py",
        "src/j2bench/figures.py",
        "src/j2bench/metrics.py",
        "src/j2bench/policies.py",
        "src/j2bench/provenance.py",
        "src/j2bench/publication.py",
        "src/j2bench/real_benchmark.py",
        "src/j2bench/realdata.py",
        "src/j2bench/stats.py",
    output:
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_publication_metrics_by_policy_unit.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_publication_aggregate_policy_metrics.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_confidence_gate_match_summary.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_confidence_gate_pairwise.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_confidence_gate_iso_budget_unit_deltas.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_confidence_gate_iso_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_plain_conf_exact_budget_selection_by_unit.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_confidence_gate_exact_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_earliest_safe_by_episode.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_earliest_safe_by_unit.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_earliest_safe_summary.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_earliest_safe_pairwise.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_earliest_safe_iso_budget_unit_deltas.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_earliest_safe_iso_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_earliest_safe_exact_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_matched_budget_ablation.png",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_earliest_safe.png",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_publication_extension_manifest.json",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_publication_provenance.json",
    shell:
        "python scripts/run_publication_extension.py --dataset j1 --prepared-root {PROCESSED_ROOT}/j1 "
        "--out-root {RESULTS_ROOT}/reports/j1_open/publication_clean "
        "--user-model {J1_USER_MODEL} --assist-model {J1_ASSIST_MODEL}"

rule j1_scorecard:
    input:
        f"{RESULTS_ROOT}/real/j1/benchmark_manifest.json",
        f"{RESULTS_ROOT}/real/j1/aggregate_policy_metrics.csv",
        f"{RESULTS_ROOT}/real/j1/benchmark_provenance.json",
        f"{RESULTS_ROOT}/reports/j1_open/stats/primary_pairwise.csv",
        f"{RESULTS_ROOT}/reports/j1_open/stats/stats_provenance.json",
        f"{RESULTS_ROOT}/reports/j1_open/split_report.json",
        f"{RESULTS_ROOT}/reports/j1_open/split_report_provenance.json",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_publication_extension_manifest.json",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_confidence_gate_match_summary.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_confidence_gate_pairwise.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_earliest_safe_summary.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_earliest_safe_pairwise.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_publication_provenance.json",
        "scripts/build_j1_scorecard.py",
        "src/j2bench/artifact_schemas.py",
        "src/j2bench/j1_scorecard.py",
    output:
        f"{RESULTS_ROOT}/reports/j1_open/j1_scorecard.md",
        f"{RESULTS_ROOT}/reports/j1_open/j1_scorecard.json",
    shell:
        "python scripts/build_j1_scorecard.py "
        "--benchmark-root {RESULTS_ROOT}/real/j1 "
        "--stats-root {RESULTS_ROOT}/reports/j1_open/stats "
        "--comparator-root {RESULTS_ROOT}/reports/j1_open/publication_clean "
        "--report-root {RESULTS_ROOT}/reports/j1_open "
        "--benchmark-provenance {RESULTS_ROOT}/real/j1/benchmark_provenance.json "
        "--stats-provenance {RESULTS_ROOT}/reports/j1_open/stats/stats_provenance.json "
        "--comparator-provenance {RESULTS_ROOT}/reports/j1_open/publication_clean/j1_publication_provenance.json "
        "--report-provenance {RESULTS_ROOT}/reports/j1_open/split_report_provenance.json "
        "--out-root {RESULTS_ROOT}/reports/j1_open"

rule j1_frontier_readout:
    input:
        f"{RESULTS_ROOT}/reports/j1_open/j1_scorecard.json",
        f"{RESULTS_ROOT}/reports/j1_open/stats/primary_pairwise.csv",
        "scripts/build_j1_frontier_readout.py",
        "src/j2bench/j1_frontier.py",
        "src/j2bench/j1_scorecard.py",
    output:
        f"{RESULTS_ROOT}/reports/j1_open/j1_frontier_readout.md",
        f"{RESULTS_ROOT}/reports/j1_open/j1_frontier_readout.json",
    shell:
        "python scripts/build_j1_frontier_readout.py "
        "--scorecard-path {RESULTS_ROOT}/reports/j1_open/j1_scorecard.json "
        "--primary-pairwise-path {RESULTS_ROOT}/reports/j1_open/stats/primary_pairwise.csv "
        "--out-root {RESULTS_ROOT}/reports/j1_open"

rule j1_frontier_package:
    input:
        f"{RESULTS_ROOT}/reports/j1_open/j1_scorecard.json",
        f"{RESULTS_ROOT}/reports/j1_open/stats/primary_pairwise.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_publication_aggregate_policy_metrics.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_publication_metrics_by_policy_unit.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_earliest_safe_summary.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_earliest_safe_by_unit.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_confidence_gate_iso_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_earliest_safe_iso_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_confidence_gate_exact_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_earliest_safe_exact_budget_pairwise.csv",
        "config/j1_claims.yaml",
        "scripts/build_j1_frontier_package.py",
        "src/j2bench/artifact_schemas.py",
        "src/j2bench/figures.py",
        "src/j2bench/j1_claims.py",
        "src/j2bench/j1_frontier.py",
    output:
        f"{RESULTS_ROOT}/reports/j1_open/frontier/frontier_points.csv",
        f"{RESULTS_ROOT}/reports/j1_open/frontier/pareto_frontier.csv",
        f"{RESULTS_ROOT}/reports/j1_open/frontier/frontier_overlap.csv",
        f"{RESULTS_ROOT}/reports/j1_open/frontier/non_dominated_summary.csv",
        f"{RESULTS_ROOT}/reports/j1_open/frontier/universal_tau_regret.csv",
        f"{RESULTS_ROOT}/reports/j1_open/frontier/frontier_unit_deltas.csv",
        f"{RESULTS_ROOT}/reports/j1_open/frontier/budget_match_summary.csv",
        f"{RESULTS_ROOT}/reports/j1_open/frontier/iso_budget_claim_summary.csv",
        f"{RESULTS_ROOT}/reports/j1_open/frontier/exact_budget_claim_summary.csv",
        f"{RESULTS_ROOT}/reports/j1_open/frontier/operating_point_instability.json",
        f"{RESULTS_ROOT}/reports/j1_open/frontier/frontier_atlas.png",
        f"{RESULTS_ROOT}/reports/j1_open/frontier/safety_timing_frontier.png",
        f"{RESULTS_ROOT}/reports/j1_open/frontier/j1_frontier_package.json",
        f"{RESULTS_ROOT}/reports/j1_open/frontier/j1_frontier_package.md",
        f"{RESULTS_ROOT}/reports/j1_open/frontier/j1_frontier_package_provenance.json",
    shell:
        "python scripts/build_j1_frontier_package.py "
        "--scorecard-path {RESULTS_ROOT}/reports/j1_open/j1_scorecard.json "
        "--primary-pairwise-path {RESULTS_ROOT}/reports/j1_open/stats/primary_pairwise.csv "
        "--publication-aggregate-path {RESULTS_ROOT}/reports/j1_open/publication_clean/j1_publication_aggregate_policy_metrics.csv "
        "--publication-unit-path {RESULTS_ROOT}/reports/j1_open/publication_clean/j1_publication_metrics_by_policy_unit.csv "
        "--earliest-summary-path {RESULTS_ROOT}/reports/j1_open/publication_clean/j1_earliest_safe_summary.csv "
        "--earliest-unit-path {RESULTS_ROOT}/reports/j1_open/publication_clean/j1_earliest_safe_by_unit.csv "
        "--confidence-iso-budget-pairwise-path {RESULTS_ROOT}/reports/j1_open/publication_clean/j1_confidence_gate_iso_budget_pairwise.csv "
        "--earliest-iso-budget-pairwise-path {RESULTS_ROOT}/reports/j1_open/publication_clean/j1_earliest_safe_iso_budget_pairwise.csv "
        "--confidence-exact-budget-pairwise-path {RESULTS_ROOT}/reports/j1_open/publication_clean/j1_confidence_gate_exact_budget_pairwise.csv "
        "--earliest-exact-budget-pairwise-path {RESULTS_ROOT}/reports/j1_open/publication_clean/j1_earliest_safe_exact_budget_pairwise.csv "
        "--claims-config-path config/j1_claims.yaml "
        "--out-root {RESULTS_ROOT}/reports/j1_open/frontier"

rule exact_budget_instability_report:
    input:
        f"{RESULTS_ROOT}/reports/publication_clean/db10_confidence_gate_exact_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_earliest_safe_exact_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_confidence_gate_exact_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/hyser_earliest_safe_exact_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_confidence_gate_exact_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/cemhsey_earliest_safe_exact_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_confidence_gate_exact_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/j1_open/publication_clean/j1_earliest_safe_exact_budget_pairwise.csv",
        f"{RESULTS_ROOT}/reports/j1_open/j1_scorecard.json",
        "scripts/build_exact_budget_instability_report.py",
    output:
        f"{RESULTS_ROOT}/reports/publication_clean/exact_budget_instability_by_tau.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/exact_budget_instability_summary.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/exact_budget_instability_status.json",
        f"{RESULTS_ROOT}/reports/publication_clean/exact_budget_instability_report.md",
    shell:
        "python scripts/build_exact_budget_instability_report.py "
        "--publication-root {RESULTS_ROOT}/reports/publication_clean "
        "--j1-root {RESULTS_ROOT}/reports/j1_open "
        "--out-root {RESULTS_ROOT}/reports/publication_clean"

rule publication_claims:
    input:
        f"{RESULTS_ROOT}/reports/anchor_status.json",
        f"{RESULTS_ROOT}/reports/publication_clean/db10_anchor_reproductions.json",
        LITERATURE_ANCHORS,
        LITERATURE_COMPARABILITY,
        "scripts/build_comparability_matrix.py",
        "scripts/build_sota_report.py",
        "src/j2bench/literature.py",
    output:
        f"{RESULTS_ROOT}/reports/publication_clean/direct_comparability_matrix.md",
        f"{RESULTS_ROOT}/reports/publication_clean/direct_comparability_matrix.csv",
        f"{RESULTS_ROOT}/reports/publication_clean/direct_comparability_matrix.json",
        f"{RESULTS_ROOT}/reports/publication_clean/sota_report.md",
        f"{RESULTS_ROOT}/reports/publication_clean/sota_status.json",
    shell:
        "python scripts/build_comparability_matrix.py --targets {LITERATURE_ANCHORS} --comparability {LITERATURE_COMPARABILITY} --out-root {RESULTS_ROOT}/reports/publication_clean && "
        "python scripts/build_sota_report.py --config config/config.yaml --anchors {LITERATURE_ANCHORS} --comparability {LITERATURE_COMPARABILITY} --anchor-status {RESULTS_ROOT}/reports/anchor_status.json --out-root {RESULTS_ROOT}/reports/publication_clean"

rule extract_cemhsey_gesture:
    output:
        f"{INTERIM_ROOT}/cemhsey_gesture/_SUCCESS"
    shell:
        "python scripts/extract_cemhsey.py --root {RAW_ROOT}/cemhsey --out {INTERIM_ROOT}/cemhsey_gesture --subset gesture ; "
        "New-Item -ItemType File -Force -Path {output} | Out-Null"

