#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(ggprism)
  library(patchwork)
  library(readr)
  library(scales)
  library(stringr)
  library(tibble)
  library(tidyr)
})

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default = NULL) {
  hit <- which(args == flag)
  if (length(hit) == 0 || hit[length(hit)] == length(args)) {
    return(default)
  }
  args[[hit[length(hit)] + 1]]
}

repo_root <- normalizePath(get_arg("--repo-root", getwd()), winslash = "/", mustWork = TRUE)
input_dir <- normalizePath(
  get_arg("--input-dir", file.path(repo_root, "results", "reports", "publication_clean")),
  winslash = "/",
  mustWork = TRUE
)
out_dir <- normalizePath(
  get_arg("--out-dir", file.path(repo_root, "results", "reports", "tnsre_figures")),
  winslash = "/",
  mustWork = FALSE
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

fig_pdf_dir <- file.path(out_dir, "pdf")
fig_png_dir <- file.path(out_dir, "png")
dir.create(fig_pdf_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(fig_png_dir, recursive = TRUE, showWarnings = FALSE)

relative_to <- function(path, base) {
  norm_path <- normalizePath(path, winslash = "/", mustWork = FALSE)
  norm_base <- normalizePath(base, winslash = "/", mustWork = FALSE)
  prefix <- paste0(norm_base, "/")
  if (startsWith(norm_path, prefix)) {
    return(substr(norm_path, nchar(prefix) + 1, nchar(norm_path)))
  }
  norm_path
}

relative_to_repo <- function(path) {
  relative_to(path, repo_root)
}

relative_to_out <- function(path) {
  relative_to(path, out_dir)
}

read_pub <- function(name) {
  path <- file.path(input_dir, name)
  if (!file.exists(path)) {
    stop("Missing required input: ", path, call. = FALSE)
  }
  readr::read_csv(path, show_col_types = FALSE, progress = FALSE)
}

policy_palette <- c(
  "Agency-margin" = "#0072B2",
  "SetACSA" = "#D55E00",
  "Confidence gate" = "#009E73",
  "Confidence gate (exact)" = "#56B4E9",
  "Confidence blend" = "#CC79A7",
  "Selective prediction" = "#7F7F7F",
  "User only" = "#222222",
  "Assist only" = "#E69F00"
)

policy_shapes <- c(
  "Agency-margin" = 16,
  "SetACSA" = 17,
  "Confidence gate" = 15,
  "Confidence gate (exact)" = 18,
  "Confidence blend" = 8,
  "Selective prediction" = 4,
  "User only" = 1,
  "Assist only" = 3
)

dataset_labels <- c(
  "db10" = "DB10/MeganePro",
  "hyser" = "Hyser",
  "cemhsey" = "CEMHSEY"
)

split_labels <- c(
  "db10_able_to_amputee" = "DB10 able-to-amputee",
  "db10_amputee_loso" = "DB10 amputee LOSO",
  "db10_mixed_to_amputee" = "DB10 mixed-to-amputee",
  "hyser_subject_logo" = "Hyser subject LOSO",
  "hyser_within_subject_dayshift" = "Hyser day shift",
  "cemhsey_forward_day_logo" = "CEMHSEY forward-day"
)

metric_labels <- c(
  "active_macro_f1" = "Active macro-F1",
  "active_risk_coverage_auc" = "Active risk-coverage AUC",
  "stable_safe_episode_rate" = "Stable-safe episode rate",
  "median_earliest_stable_safe_s" = "Median earliest stable-safe time (s)",
  "final_correct_rate" = "Final-correct episode rate"
)

primary_db10 <- c("db10_amputee_loso", "db10_mixed_to_amputee")
db10_all <- c("db10_able_to_amputee", primary_db10)

classify_policy <- function(policy) {
  case_when(
    policy == "user_only" ~ "User only",
    policy == "assist_only" ~ "Assist only",
    str_starts(policy, "agency_margin_tau_") ~ "Agency-margin",
    str_starts(policy, "set_acsa_tau_") ~ "SetACSA",
    str_starts(policy, "plain_conf_threshold_exact_budget_tau_") ~ "Confidence gate (exact)",
    str_starts(policy, "plain_conf_threshold_matched_tau_") ~ "Confidence gate",
    str_starts(policy, "confidence_blend_alpha_") ~ "Confidence blend",
    policy == "selective_prediction" ~ "Selective prediction",
    TRUE ~ "Other"
  )
}

extract_tau <- function(policy) {
  as.numeric(str_match(policy, "tau_([0-9]+\\.[0-9]+)$")[, 2])
}

prepare_aggregate <- function(df, dataset_id) {
  df %>%
    mutate(
      dataset_id = dataset_id,
      policy_family = classify_policy(policy),
      tau = extract_tau(policy),
      split_label = recode(split_family, !!!split_labels),
      dataset_label = recode(dataset_id, !!!dataset_labels),
      policy_family = factor(policy_family, levels = names(policy_palette))
    )
}

theme_tnsre <- function(base_size = 8.4) {
  ggprism::theme_prism(base_size = base_size, base_family = "Arial", border = FALSE) +
    theme(
      plot.title = element_text(face = "bold", size = base_size + 0.8, margin = margin(b = 3)),
      plot.subtitle = element_text(face = "plain", size = base_size - 0.7, margin = margin(b = 4)),
      axis.title = element_text(face = "bold", size = base_size),
      axis.text = element_text(face = "plain", size = base_size - 1.2, color = "black"),
      strip.text = element_text(face = "bold", size = base_size - 0.8, color = "black"),
      strip.text.y.left = element_text(angle = 90, margin = margin(r = 2)),
      strip.text.y.right = element_text(angle = -90, margin = margin(l = 2)),
      strip.background = element_rect(fill = "#F4F4F4", color = "#BDBDBD", linewidth = 0.35),
      legend.title = element_text(face = "bold", size = base_size - 0.3),
      legend.text = element_text(face = "plain", size = base_size - 1.0),
      legend.key.height = unit(0.13, "in"),
      legend.key.width = unit(0.22, "in"),
      legend.box.spacing = unit(0.02, "in"),
      panel.grid.major = element_line(color = "#DDDDDD", linewidth = 0.18),
      panel.grid.minor = element_blank(),
      panel.spacing = unit(0.08, "in"),
      strip.placement = "outside",
      plot.margin = margin(4, 5, 3, 4)
    )
}

scale_policy_color <- function(...) {
  scale_color_manual(values = policy_palette, drop = TRUE, ...)
}

scale_policy_shape <- function(...) {
  scale_shape_manual(values = policy_shapes, drop = TRUE, ...)
}

save_figure <- function(plot, name, width, height, dpi = 600) {
  pdf_path <- file.path(fig_pdf_dir, paste0(name, ".pdf"))
  png_path <- file.path(fig_png_dir, paste0(name, ".png"))
  tryCatch(
    {
      ggsave(
        filename = pdf_path,
        plot = plot,
        width = width,
        height = height,
        units = "in",
        device = grDevices::cairo_pdf,
        bg = "white"
      )
    },
    error = function(err) {
      message("cairo_pdf failed for ", name, "; falling back to pdf(): ", conditionMessage(err))
      ggsave(
        filename = pdf_path,
        plot = plot,
        width = width,
        height = height,
        units = "in",
        device = "pdf",
        bg = "white"
      )
    }
  )
  ggsave(
    filename = png_path,
    plot = plot,
    width = width,
    height = height,
    units = "in",
    device = ragg::agg_png,
    dpi = dpi,
    bg = "white"
  )
  invisible(tibble(file = c(relative_to_out(pdf_path), relative_to_out(png_path)), width_in = width, height_in = height, dpi = c(NA, dpi)))
}

format_tau <- function(x) {
  sprintf("%.2f", as.numeric(x))
}

policy_ordered <- function(df) {
  df %>%
    filter(policy_family %in% names(policy_palette)) %>%
    mutate(policy_family = factor(as.character(policy_family), levels = names(policy_palette)))
}

make_flow_figure <- function() {
  nodes <- tribble(
    ~id, ~x, ~y, ~w, ~h, ~label, ~fill,
    "datasets", 0.7, 2.65, 1.25, 0.62, "Public datasets\nDB10 primary\nexternal checks", "#E8F1FA",
    "decoders", 2.2, 2.65, 1.25, 0.62, "Dataset-specific\nuser and assistive\ndecoders", "#F7F7F7",
    "calibration", 3.7, 2.65, 1.25, 0.62, "Calibrated\nposterior traces", "#F7F7F7",
    "policy", 5.2, 2.65, 1.25, 0.62, "Policy layer\nconfidence gate\nSetACSA\nagency-margin", "#EAF6EF",
    "evaluation", 6.7, 2.65, 1.25, 0.62, "Unit-level\ninference", "#F7F7F7",
    "outputs", 8.2, 2.65, 1.25, 0.62, "Performance-\nagency\nfrontier", "#FFF3D8",
    "splits", 2.2, 1.20, 1.7, 0.62, "Declared shifts\namputee LOSO\nmixed-to-amputee\nday/session transfer", "#F8E9E6",
    "audit", 5.2, 1.20, 1.7, 0.62, "Comparator audits\nmatched budget\nexact-budget gates", "#F8E9E6",
    "claim", 8.2, 1.20, 1.7, 0.62, "Claim boundary\noffline benchmark\nno online/clinical\nvalidation", "#F8E9E6"
  )
  arrows <- tribble(
    ~x, ~y, ~xend, ~yend,
    1.35, 2.55, 1.55, 2.55,
    2.85, 2.55, 3.05, 2.55,
    4.35, 2.55, 4.55, 2.55,
    5.85, 2.55, 6.05, 2.55,
    7.35, 2.55, 7.55, 2.55,
    2.2, 2.25, 2.2, 1.55,
    5.2, 2.25, 5.2, 1.55,
    8.2, 2.25, 8.2, 1.55,
    3.10, 1.32, 4.25, 1.32,
    6.10, 1.32, 7.25, 1.32
  )

  ggplot() +
    geom_segment(
      data = arrows,
      aes(x = x, y = y, xend = xend, yend = yend),
      linewidth = 0.38,
      color = "#4A4A4A",
      arrow = arrow(type = "closed", length = unit(0.07, "in"))
    ) +
    geom_rect(
      data = nodes,
      aes(xmin = x - w / 2, xmax = x + w / 2, ymin = y - h / 2, ymax = y + h / 2, fill = fill),
      color = "#333333",
      linewidth = 0.35
    ) +
    geom_text(
      data = nodes,
      aes(x = x, y = y, label = label),
      family = "Arial",
      size = 1.95,
      lineheight = 0.88,
      fontface = "bold",
      color = "#111111"
    ) +
    annotate(
      "text",
      x = 0.08,
      y = 3.12,
      hjust = 0,
      label = "J2 benchmark architecture",
      family = "Arial",
      fontface = "bold",
      size = 2.75
    ) +
    annotate(
      "text",
      x = 0.08,
      y = 0.72,
      hjust = 0,
      label = "Frozen traces keep dataset-specific decoding separate from the policy-layer comparison.",
      family = "Arial",
      size = 1.95,
      color = "#444444"
    ) +
    scale_fill_identity() +
    coord_cartesian(xlim = c(-0.05, 9.15), ylim = c(0.42, 3.35), expand = FALSE, clip = "off") +
    theme_void(base_family = "Arial") +
    theme(plot.margin = margin(4, 5, 4, 5))
}

make_frontier_plot <- function(aggregate, title, subtitle = NULL, splits = NULL) {
  data <- aggregate %>%
    { if (!is.null(splits)) filter(., split_family %in% splits) else . } %>%
    policy_ordered() %>%
    filter(policy_family %in% c("User only", "Assist only", "Confidence blend", "Confidence gate", "SetACSA", "Agency-margin")) %>%
    pivot_longer(
      cols = c(active_macro_f1, active_risk_coverage_auc),
      names_to = "metric",
      values_to = "value"
    ) %>%
    mutate(
      metric = factor(metric, levels = names(metric_labels)[1:2], labels = metric_labels[names(metric_labels)[1:2]]),
      line_group = if_else(is.na(tau), as.character(policy), paste(split_family, metric, policy_family, sep = "|"))
    )

  build_panel <- function(metric_value, panel_title, show_x = FALSE) {
    panel_data <- data %>% filter(metric == metric_value)
    line_data <- panel_data %>%
      filter(!is.na(tau), policy_family %in% c("Confidence gate", "SetACSA", "Agency-margin"))
    ggplot(panel_data, aes(x = mean_ali, y = value, color = policy_family, shape = policy_family)) +
      geom_line(
        data = line_data,
        aes(group = line_group),
        linewidth = 0.44,
        alpha = 0.85
      ) +
      geom_point(size = 1.65, stroke = 0.35, alpha = 0.95) +
      facet_wrap(~ split_label, nrow = 1, scales = "free_y") +
      scale_policy_color(name = "Policy") +
      scale_policy_shape(name = "Policy") +
      scale_x_continuous(labels = label_number(accuracy = 0.001), expand = expansion(mult = c(0.04, 0.08))) +
      scale_y_continuous(labels = label_number(accuracy = 0.01), expand = expansion(mult = c(0.08, 0.10))) +
      labs(
        title = panel_title,
        x = if (show_x) "Mean model-implied agency loss (ALI)" else NULL,
        y = NULL
      ) +
      guides(color = guide_legend(nrow = 1, byrow = TRUE), shape = guide_legend(nrow = 1, byrow = TRUE)) +
      theme_tnsre(7.5) +
      theme(
        legend.position = "bottom",
        axis.text.x = if (show_x) element_text(face = "plain", size = 6.3, color = "black") else element_blank(),
        axis.ticks.x = if (show_x) element_line(color = "black", linewidth = 0.3) else element_blank()
      )
  }

  p_f1 <- build_panel("Active macro-F1", "A. Active macro-F1", show_x = FALSE)
  p_auc <- build_panel("Active risk-coverage AUC", "B. Active risk-coverage AUC (lower is better)", show_x = TRUE)

  (p_f1 / p_auc) +
    plot_layout(guides = "collect", heights = c(1, 1)) +
    plot_annotation(
      title = title,
      subtitle = subtitle,
      theme = theme(
        plot.title = element_text(family = "Arial", face = "bold", size = 10.0, hjust = 0.5),
        plot.subtitle = element_text(family = "Arial", face = "plain", size = 8.3, hjust = 0.5)
      )
    ) &
    theme(legend.position = "bottom")
}

make_pairwise_setacsa <- function(pairwise) {
  data <- pairwise %>%
    filter(
      split_family %in% primary_db10,
      comparison_family == "agency_vs_set_acsa",
      metric %in% c("active_macro_f1", "active_risk_coverage_auc")
    ) %>%
    mutate(
      tau = extract_tau(policy_a),
      metric_label = recode(metric, !!!metric_labels),
      split_label = recode(split_family, !!!split_labels),
      benefit = if_else(metric == "active_risk_coverage_auc", -estimate, estimate),
      benefit_low = if_else(metric == "active_risk_coverage_auc", -ci_high, ci_low),
      benefit_high = if_else(metric == "active_risk_coverage_auc", -ci_low, ci_high),
      holm_status = if_else(beneficial_and_holm_significant, "Holm significant", "Not significant")
    )

  ggplot(data, aes(x = tau, y = benefit, color = split_label, shape = holm_status)) +
    geom_hline(yintercept = 0, linewidth = 0.32, color = "#555555") +
    geom_errorbar(aes(ymin = benefit_low, ymax = benefit_high), width = 0.006, linewidth = 0.42, alpha = 0.95) +
    geom_point(size = 2.05, stroke = 0.45) +
    facet_wrap(~ metric_label, nrow = 1, scales = "free_y") +
    scale_color_manual(values = c("DB10 amputee LOSO" = "#0072B2", "DB10 mixed-to-amputee" = "#D55E00"), name = "DB10 split") +
    scale_shape_manual(values = c("Holm significant" = 16, "Not significant" = 1), name = "Inference") +
    scale_x_continuous(breaks = sort(unique(data$tau)), labels = format_tau, expand = expansion(mult = c(0.05, 0.08))) +
    scale_y_continuous(labels = label_number(accuracy = 0.001), expand = expansion(mult = c(0.12, 0.12))) +
    labs(
      title = "Agency-margin benefit over matched SetACSA",
      x = "Agency budget tau",
      y = "Paired benefit"
    ) +
    theme_tnsre(7.8) +
    theme(legend.position = "bottom")
}

make_exact_budget_audit <- function() {
  pairwise <- bind_rows(
    read_pub("db10_confidence_gate_exact_budget_pairwise.csv") %>% mutate(dataset_id = "db10"),
    read_pub("hyser_confidence_gate_exact_budget_pairwise.csv") %>% mutate(dataset_id = "hyser"),
    read_pub("cemhsey_confidence_gate_exact_budget_pairwise.csv") %>% mutate(dataset_id = "cemhsey")
  ) %>%
    filter(
      comparison_family == "agency_vs_plain_conf_exact_budget",
      metric %in% c("active_macro_f1", "active_risk_coverage_auc")
    ) %>%
    mutate(
      dataset_label = recode(dataset_id, !!!dataset_labels),
      split_label = recode(split_family, !!!split_labels),
      metric_label = recode(metric, !!!metric_labels),
      benefit = if_else(metric == "active_risk_coverage_auc", -estimate, estimate),
      benefit_low = if_else(metric == "active_risk_coverage_auc", -ci_high, ci_low),
      benefit_high = if_else(metric == "active_risk_coverage_auc", -ci_low, ci_high)
    )

  p_delta <- ggplot(pairwise, aes(x = tau, y = benefit, color = dataset_label)) +
    geom_hline(yintercept = 0, linewidth = 0.32, color = "#555555") +
    geom_errorbar(aes(ymin = benefit_low, ymax = benefit_high), width = 0.006, linewidth = 0.34, alpha = 0.55) +
    geom_point(aes(shape = dataset_label), size = 1.75, stroke = 0.35, alpha = 0.92) +
    facet_wrap(~ metric_label, ncol = 1, scales = "free_y") +
    scale_color_manual(values = c("DB10/MeganePro" = "#0072B2", "Hyser" = "#009E73", "CEMHSEY" = "#D55E00"), name = "Dataset") +
    scale_shape_manual(values = c("DB10/MeganePro" = 16, "Hyser" = 17, "CEMHSEY" = 15), name = "Dataset") +
    scale_x_continuous(breaks = sort(unique(pairwise$tau)), labels = format_tau, expand = expansion(mult = c(0.05, 0.08))) +
    scale_y_continuous(labels = label_number(accuracy = 0.01), expand = expansion(mult = c(0.12, 0.12))) +
    labs(
      title = "A. Exact-budget paired benefit",
      x = "Agency budget tau",
      y = NULL
    ) +
    theme_tnsre(7.9) +
    theme(legend.position = "bottom")

  heat <- read_pub("exact_budget_instability_by_tau.csv") %>%
    filter(dataset_id %in% c("db10", "hyser", "cemhsey"), scope == "required") %>%
    mutate(
      dataset_label = recode(dataset_id, !!!dataset_labels),
      split_label = recode(split_family, !!!split_labels),
      tau_label = format_tau(tau)
    ) %>%
    mutate(
      split_label = factor(
        split_label,
        levels = rev(c(
          "DB10 able-to-amputee",
          "DB10 amputee LOSO",
          "DB10 mixed-to-amputee",
          "Hyser subject LOSO",
          "Hyser day shift",
          "CEMHSEY forward-day"
        ))
      )
    )

  p_heat <- ggplot(heat, aes(x = tau_label, y = split_label, fill = confidence_exact_match_fraction)) +
    geom_tile(color = "white", linewidth = 0.45) +
    geom_text(aes(label = sprintf("%.2f", confidence_exact_match_fraction)), family = "Arial", size = 2.35, color = "#111111") +
    scale_fill_gradient(
      low = "#F6E7E2",
      high = "#0072B2",
      limits = c(0, 1),
      labels = label_number(accuracy = 0.1)
    ) +
    labs(
      title = "B. Exact intervention-budget match fraction",
      x = "Agency budget tau",
      y = NULL
    ) +
    theme_tnsre(7.9) +
    theme(
      legend.position = "none",
      axis.text.y = element_text(size = 6.3),
      panel.grid.major = element_blank()
    )

  (p_delta | p_heat) + plot_layout(widths = c(1.03, 1.0))
}

make_earliest_safe <- function(summary, dataset_title, splits = NULL) {
  data <- summary %>%
    { if (!is.null(splits)) filter(., split_family %in% splits) else . } %>%
    mutate(
      policy_family = classify_policy(policy),
      tau = extract_tau(policy),
      split_label = recode(split_family, !!!split_labels),
      policy_family = factor(policy_family, levels = names(policy_palette))
    ) %>%
    filter(
      !is.na(tau),
      policy_family %in% c("Confidence gate", "SetACSA", "Agency-margin")
    ) %>%
    pivot_longer(
      cols = c(stable_safe_episode_rate, median_earliest_stable_safe_s),
      names_to = "metric",
      values_to = "value"
    ) %>%
    mutate(metric = factor(metric, levels = c("stable_safe_episode_rate", "median_earliest_stable_safe_s"), labels = metric_labels[c("stable_safe_episode_rate", "median_earliest_stable_safe_s")]))

  build_panel <- function(metric_value, panel_title, show_x = FALSE) {
    panel_data <- data %>% filter(metric == metric_value)
    ggplot(panel_data, aes(x = tau, y = value, color = policy_family, shape = policy_family, group = policy_family)) +
      geom_line(linewidth = 0.45, alpha = 0.9) +
      geom_point(size = 1.75, stroke = 0.35) +
      facet_wrap(~ split_label, nrow = 1, scales = "free_y") +
      scale_policy_color(name = "Policy") +
      scale_policy_shape(name = "Policy") +
      scale_x_continuous(breaks = sort(unique(panel_data$tau)), labels = format_tau, expand = expansion(mult = c(0.05, 0.08))) +
      scale_y_continuous(labels = label_number(accuracy = 0.01), expand = expansion(mult = c(0.10, 0.12))) +
      labs(
        title = panel_title,
        x = if (show_x) "Agency budget tau" else NULL,
        y = NULL
      ) +
      guides(color = guide_legend(nrow = 1), shape = guide_legend(nrow = 1)) +
      theme_tnsre(7.5) +
      theme(
        legend.position = "bottom",
        axis.text.x = if (show_x) element_text(face = "plain", size = 6.3, color = "black") else element_blank(),
        axis.ticks.x = if (show_x) element_line(color = "black", linewidth = 0.3) else element_blank()
      )
  }

  p_rate <- build_panel("Stable-safe episode rate", "A. Stable-safe episode rate", show_x = FALSE)
  p_time <- build_panel("Median earliest stable-safe time (s)", "B. Median earliest stable-safe time (s; lower is better)", show_x = TRUE)
  (p_rate / p_time) +
    plot_layout(guides = "collect", heights = c(1, 1)) +
    plot_annotation(
      title = paste0(dataset_title, " timing diagnostic"),
      theme = theme(
        plot.title = element_text(family = "Arial", face = "bold", size = 9.0, hjust = 0.5, margin = margin(b = 4))
      )
    ) &
    theme(legend.position = "bottom")
}

make_matched_budget <- function(aggregate, dataset_title, splits = NULL) {
  data <- aggregate %>%
    { if (!is.null(splits)) filter(., split_family %in% splits) else . } %>%
    policy_ordered() %>%
    filter(policy_family %in% c("Confidence gate", "Agency-margin")) %>%
    pivot_longer(
      cols = c(active_macro_f1, active_risk_coverage_auc),
      names_to = "metric",
      values_to = "value"
    ) %>%
    mutate(
      metric = factor(metric, levels = names(metric_labels)[1:2], labels = metric_labels[names(metric_labels)[1:2]])
    )

  build_panel <- function(metric_value, panel_title, show_x = FALSE) {
    panel_data <- data %>% filter(metric == metric_value)
    ggplot(panel_data, aes(x = tau, y = value, color = policy_family, shape = policy_family, group = policy_family)) +
      geom_line(linewidth = 0.45, alpha = 0.9) +
      geom_point(size = 1.75, stroke = 0.35) +
      facet_wrap(~ split_label, nrow = 1, scales = "free_y") +
      scale_policy_color(name = "Policy") +
      scale_policy_shape(name = "Policy") +
      scale_x_continuous(breaks = sort(unique(panel_data$tau)), labels = format_tau, expand = expansion(mult = c(0.05, 0.08))) +
      scale_y_continuous(labels = label_number(accuracy = 0.01), expand = expansion(mult = c(0.10, 0.12))) +
      labs(
        title = panel_title,
        x = if (show_x) "Agency budget tau" else NULL,
        y = NULL
      ) +
      guides(color = guide_legend(nrow = 1), shape = guide_legend(nrow = 1)) +
      theme_tnsre(7.5) +
      theme(
        legend.position = "bottom",
        axis.text.x = if (show_x) element_text(face = "plain", size = 6.3, color = "black") else element_blank(),
        axis.ticks.x = if (show_x) element_line(color = "black", linewidth = 0.3) else element_blank()
      )
  }

  p_f1 <- build_panel("Active macro-F1", "A. Active macro-F1", show_x = FALSE)
  p_auc <- build_panel("Active risk-coverage AUC", "B. Active risk-coverage AUC (lower is better)", show_x = TRUE)
  (p_f1 / p_auc) +
    plot_layout(guides = "collect", heights = c(1, 1)) +
    plot_annotation(
      title = paste0(dataset_title, " matched budget"),
      theme = theme(
        plot.title = element_text(family = "Arial", face = "bold", size = 9.0, hjust = 0.5, margin = margin(b = 4))
      )
    ) &
    theme(legend.position = "bottom")
}

db10_aggregate <- prepare_aggregate(read_pub("db10_publication_aggregate_policy_metrics.csv"), "db10")
hyser_aggregate <- prepare_aggregate(read_pub("hyser_publication_aggregate_policy_metrics.csv"), "hyser")
cemhsey_aggregate <- prepare_aggregate(read_pub("cemhsey_publication_aggregate_policy_metrics.csv"), "cemhsey")

manifest <- bind_rows(
  save_figure(make_flow_figure(), "fig1_benchmark_flow", width = 7.16, height = 3.15),
  save_figure(
    make_frontier_plot(
      db10_aggregate,
      title = "DB10 primary performance-agency frontier",
      subtitle = "Primary claim-bearing split families; risk-coverage AUC is better when lower.",
      splits = primary_db10
    ),
    "fig2_db10_frontier",
    width = 7.16,
    height = 4.65
  ),
  save_figure(make_pairwise_setacsa(read_pub("db10_confidence_gate_pairwise.csv")), "fig3_setacsa_pairwise", width = 7.16, height = 3.05),
  save_figure(make_exact_budget_audit(), "fig4_exact_budget_audit", width = 7.16, height = 4.25),
  save_figure(
    make_frontier_plot(
      bind_rows(hyser_aggregate, cemhsey_aggregate),
      title = "External high-density EMG robustness checks",
      subtitle = "Hyser and CEMHSEY support the benchmark stress-test story but are not prosthetic clinical evidence."
    ),
    "fig5_external_robustness",
    width = 7.16,
    height = 4.65
  ),
  save_figure(make_earliest_safe(read_pub("db10_earliest_safe_summary.csv"), "DB10", splits = primary_db10), "fig6_db10_earliest_safe", width = 7.16, height = 4.35),
  save_figure(make_matched_budget(db10_aggregate, "DB10", splits = db10_all), "figS_db10_matched_budget_ablation", width = 7.16, height = 4.65),
  save_figure(make_matched_budget(hyser_aggregate, "Hyser"), "figS_hyser_matched_budget_ablation", width = 7.16, height = 4.25),
  save_figure(make_matched_budget(cemhsey_aggregate, "CEMHSEY"), "figS_cemhsey_matched_budget_ablation", width = 3.5, height = 4.25),
  save_figure(make_earliest_safe(read_pub("hyser_earliest_safe_summary.csv"), "Hyser"), "figS_hyser_earliest_safe", width = 7.16, height = 4.25),
  save_figure(make_earliest_safe(read_pub("cemhsey_earliest_safe_summary.csv"), "CEMHSEY"), "figS_cemhsey_earliest_safe", width = 3.5, height = 4.25)
)

manifest <- manifest %>%
  mutate(
    created_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
    source_dir = relative_to_repo(input_dir)
  )
readr::write_csv(manifest, file.path(out_dir, "tnsre_figure_manifest.csv"))

message("Wrote ", nrow(manifest), " figure assets to ", out_dir)

