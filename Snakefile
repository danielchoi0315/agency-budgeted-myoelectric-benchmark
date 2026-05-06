configfile: "config/config.yaml"

RESULTS_ROOT = config["project"].get("results_root", "results")
FIGURE_QA = f"{RESULTS_ROOT}/reports/tnsre_figures/tnsre_figure_qa.json"


rule all:
    input:
        FIGURE_QA,
        "results/reports/publication_clean/exact_budget_instability_status.json",


rule exact_budget_instability:
    input:
        "results/reports/publication_clean/db10_confidence_gate_exact_budget_pairwise.csv",
        "results/reports/publication_clean/db10_earliest_safe_exact_budget_pairwise.csv",
        "results/reports/publication_clean/hyser_confidence_gate_exact_budget_pairwise.csv",
        "results/reports/publication_clean/hyser_earliest_safe_exact_budget_pairwise.csv",
        "results/reports/publication_clean/cemhsey_confidence_gate_exact_budget_pairwise.csv",
        "results/reports/publication_clean/cemhsey_earliest_safe_exact_budget_pairwise.csv",
    output:
        "results/reports/publication_clean/exact_budget_instability_status.json",
    shell:
        "python scripts/build_exact_budget_instability_report.py --out-root results/reports/publication_clean"


rule tnsre_figures:
    input:
        "results/reports/publication_clean/db10_publication_aggregate_policy_metrics.csv",
        "results/reports/publication_clean/hyser_publication_aggregate_policy_metrics.csv",
        "results/reports/publication_clean/cemhsey_publication_aggregate_policy_metrics.csv",
    output:
        FIGURE_QA,
    shell:
        "Rscript scripts/build_tnsre_figures.R --repo-root . && "
        "python scripts/check_tnsre_figures.py --out-dir results/reports/tnsre_figures --recursive "
        "--min-pdfs 11 --min-width 1800 --min-height 1800 --min-dpi 300"
