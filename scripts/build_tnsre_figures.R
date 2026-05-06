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

figure_font_family <- "Arial"
if (requireNamespace("systemfonts", quietly = TRUE)) {
  installed_families <- unique(systemfonts::system_fonts()$family)
  if (!figure_font_family %in% installed_families) {
    stop("Publication font '", figure_font_family, "' is not installed; install it before building figures.", call. = FALSE)
  }
}

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
  "Confidence gate (exact)" = "#6A51A3",
  "Confidence blend" = "#CC79A7",
  "Selective prediction" = "#7F7F7F",
  "User only" = "#2B2B2B",
  "Assist only" = "#E69F00"
)

policy_shapes <- c(
  "Agency-margin" = 16,
  "SetACSA" = 17,
  "Confidence gate" = 15,
  "Confidence gate (exact)" = 18,
  "Confidence blend" = 5,
  "Selective prediction" = 4,
  "User only" = 1,
  "Assist only" = 3
)

policy_linetypes <- c(
  "Agency-margin" = "solid",
  "SetACSA" = "longdash",
  "Confidence gate" = "dotdash",
  "Confidence gate (exact)" = "dotted",
  "Confidence blend" = "dotted",
  "Selective prediction" = "dotted",
  "User only" = "blank",
  "Assist only" = "blank"
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

split_order <- c(
  "DB10 able-to-amputee",
  "DB10 amputee LOSO",
  "DB10 mixed-to-amputee",
  "Hyser day shift",
  "Hyser subject LOSO",
  "CEMHSEY forward-day"
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
frontier_families <- c("Agency-margin", "SetACSA", "Confidence gate")
reference_families <- c("Confidence blend", "User only", "Assist only")

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
      split_label = factor(split_label, levels = split_order),
      dataset_label = recode(dataset_id, !!!dataset_labels),
      policy_family = factor(policy_family, levels = names(policy_palette))
    )
}

pad_range <- function(x, mult = 0.08) {
  x <- x[is.finite(x)]
  if (length(x) == 0) {
    return(c(0, 1))
  }
  rng <- range(x)
  span <- diff(rng)
  if (span == 0) {
    span <- max(abs(rng), 1) * 0.05
  }
  c(rng[1] - span * mult, rng[2] + span * mult)
}

theme_tnsre <- function(base_size = 9.0) {
  ggprism::theme_prism(
    base_size = base_size,
    base_family = figure_font_family,
    base_fontface = "plain",
    base_line_size = 0.30,
    base_rect_size = 0.28,
    border = FALSE
  ) +
    theme(
      text = element_text(family = figure_font_family, color = "black", lineheight = 0.95),
      plot.title = element_text(face = "bold", size = base_size + 0.8, margin = margin(b = 3, unit = "pt")),
      plot.subtitle = element_text(face = "plain", size = base_size - 0.3, color = "#333333", margin = margin(b = 5, unit = "pt")),
      axis.title.x = element_text(face = "plain", size = base_size, margin = margin(t = 3, unit = "pt")),
      axis.title.y = element_text(face = "plain", size = base_size, margin = margin(r = 3, unit = "pt")),
      axis.text.x = element_text(face = "plain", size = base_size - 0.9, color = "black", margin = margin(t = 1.5, unit = "pt")),
      axis.text.y = element_text(face = "plain", size = base_size - 0.9, color = "black", margin = margin(r = 1.5, unit = "pt")),
      strip.text = element_text(face = "bold", size = base_size - 0.4, color = "black", margin = margin(2.5, 3.5, 2.5, 3.5, unit = "pt")),
      strip.background = element_rect(fill = "#F5F5F5", color = "#B8B8B8", linewidth = 0.25),
      legend.title = element_text(face = "bold", size = base_size - 0.3),
      legend.text = element_text(face = "plain", size = base_size - 0.8),
      legend.key.height = unit(10, "pt"),
      legend.key.width = unit(17, "pt"),
      legend.spacing.x = unit(3, "pt"),
      legend.spacing.y = unit(1, "pt"),
      legend.box.spacing = unit(2, "pt"),
      legend.margin = margin(0, 0, 0, 0, unit = "pt"),
      axis.line = element_line(linewidth = 0.30, color = "black"),
      axis.ticks = element_line(linewidth = 0.28, color = "black"),
      axis.ticks.length = unit(2.2, "pt"),
      panel.grid.major = element_line(color = "#E6E6E6", linewidth = 0.18),
      panel.grid.minor = element_blank(),
      panel.spacing = unit(9, "pt"),
      plot.margin = margin(5, 7, 5, 6, unit = "pt")
    )
}

scale_policy_color <- function(...) {
  scale_color_manual(values = policy_palette, drop = TRUE, ...)
}

scale_policy_shape <- function(...) {
  scale_shape_manual(values = policy_shapes, drop = TRUE, ...)
}

scale_policy_linetype <- function(...) {
  scale_linetype_manual(values = policy_linetypes, drop = TRUE, ...)
}

save_figure <- function(plot, name, width, height, dpi = 600) {
  pdf_path <- file.path(fig_pdf_dir, paste0(name, ".pdf"))
  png_path <- file.path(fig_png_dir, paste0(name, ".png"))
  ggsave(
    filename = pdf_path,
    plot = plot,
    width = width,
    height = height,
    units = "in",
    device = grDevices::cairo_pdf,
    bg = "white",
    limitsize = FALSE
  )
  ggsave(
    filename = png_path,
    plot = plot,
    width = width,
    height = height,
    units = "in",
    device = ragg::agg_png,
    dpi = dpi,
    bg = "white",
    limitsize = FALSE
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
  top_nodes <- tribble(
    ~x, ~heading, ~body, ~fill,
    0.85, "Public\ndatasets", "DB10 primary\nexternal EMG checks", "#E8F1FA",
    2.35, "Fixed\ndecoders", "dataset-specific\nuser + assistive", "#F7F7F7",
    3.85, "Frozen\nposteriors", "calibrated traces\nfor policy replay", "#F7F7F7",
    5.35, "Policy\ncomparison", "confidence gate\nSetACSA\nagency-margin", "#EAF6EF",
    6.85, "Offline\nevaluation", "unit inference\nperformance + ALI", "#FFF3D8"
  ) %>%
    mutate(y = 2.38, w = 1.25, h = 1.00)

  lower_nodes <- tribble(
    ~x, ~w, ~heading, ~body, ~fill,
    1.55, 2.35, "Declared shifts", "amputee LOSO, mixed-to-amputee,\nday/session transfer", "#F8E9E6",
    4.05, 2.35, "Comparator audits", "matched-budget SetACSA and\nexact-budget confidence gates", "#F8E9E6",
    6.55, 2.35, "Evidence boundary", "offline benchmark; no online,\nclinical, haptic, or user-study claim", "#F8E9E6"
  ) %>%
    mutate(y = 0.82, h = 0.92)

  arrows <- tribble(
    ~x, ~y, ~xend, ~yend,
    1.46, 2.22, 1.74, 2.22,
    2.96, 2.22, 3.24, 2.22,
    4.46, 2.22, 4.74, 2.22,
    5.96, 2.22, 6.24, 2.22,
    2.35, 1.88, 2.35, 1.34,
    5.35, 1.88, 5.35, 1.34,
    6.85, 1.88, 6.85, 1.34
  )

  ggplot() +
    geom_segment(
      data = arrows,
      aes(x = x, y = y, xend = xend, yend = yend),
      linewidth = 0.36,
      color = "#4A4A4A",
      arrow = arrow(type = "closed", length = unit(0.07, "in"))
    ) +
    geom_rect(
      data = top_nodes,
      aes(xmin = x - w / 2, xmax = x + w / 2, ymin = y - h / 2, ymax = y + h / 2),
      fill = top_nodes$fill,
      color = "#303030",
      linewidth = 0.30
    ) +
    geom_rect(
      data = lower_nodes,
      aes(xmin = x - w / 2, xmax = x + w / 2, ymin = y - h / 2, ymax = y + h / 2),
      fill = lower_nodes$fill,
      color = "#303030",
      linewidth = 0.30
    ) +
    geom_text(
      data = top_nodes,
      aes(x = x, y = y + 0.17, label = heading),
      family = figure_font_family,
      size = 3.08,
      lineheight = 0.86,
      fontface = "bold",
      color = "#111111"
    ) +
    geom_text(
      data = top_nodes,
      aes(x = x, y = y - 0.22, label = body),
      family = figure_font_family,
      size = 2.72,
      lineheight = 0.90,
      color = "#222222"
    ) +
    geom_text(
      data = lower_nodes,
      aes(x = x, y = y + 0.18, label = heading),
      family = figure_font_family,
      size = 3.02,
      fontface = "bold",
      color = "#111111"
    ) +
    geom_text(
      data = lower_nodes,
      aes(x = x, y = y - 0.17, label = body),
      family = figure_font_family,
      size = 2.68,
      lineheight = 0.90,
      color = "#222222"
    ) +
    annotate(
      "text",
      x = 0.28,
      y = 3.03,
      hjust = 0,
      label = "Offline open-data policy benchmark",
      family = figure_font_family,
      fontface = "bold",
      size = 3.25
    ) +
    coord_cartesian(xlim = c(0.12, 7.85), ylim = c(0.16, 3.32), expand = FALSE, clip = "off") +
    theme_void(base_family = figure_font_family) +
    theme(plot.margin = margin(5, 6, 5, 6))
}

frontier_panel <- function(data, metric_col, panel_title, y_title, show_x, include_references = TRUE, fixed_x = NULL) {
  plot_data <- data %>%
    mutate(value = .data[[metric_col]]) %>%
    filter(policy_family %in% c(frontier_families, if (include_references) reference_families else character(0)))

  sweep_data <- plot_data %>%
    filter(policy_family %in% frontier_families, !is.na(tau)) %>%
    arrange(split_label, policy_family, tau)

  ref_data <- plot_data %>%
    filter(policy_family %in% reference_families)

  x_limits <- if (is.null(fixed_x)) pad_range(plot_data$mean_ali, 0.06) else fixed_x
  y_limits <- pad_range(plot_data$value, 0.09)
  legend_rows <- if (include_references) 2 else 1

  ggplot() +
    geom_line(
      data = sweep_data,
      aes(x = mean_ali, y = value, color = policy_family, linetype = policy_family, group = policy_family),
      linewidth = 0.56,
      alpha = 0.96
    ) +
    geom_point(
      data = sweep_data,
      aes(x = mean_ali, y = value, color = policy_family, shape = policy_family),
      size = 2.0,
      stroke = 0.36
    ) +
    geom_point(
      data = ref_data,
      aes(x = mean_ali, y = value, color = policy_family, shape = policy_family),
      size = 1.85,
      stroke = 0.36,
      alpha = 0.56
    ) +
    facet_wrap(~ split_label, nrow = 1) +
    scale_policy_color(name = "Policy") +
    scale_policy_shape(name = "Policy") +
    scale_policy_linetype(name = "Policy") +
    scale_x_continuous(labels = label_number(accuracy = 0.001), limits = x_limits, expand = expansion(mult = c(0.02, 0.03))) +
    scale_y_continuous(labels = label_number(accuracy = 0.01), limits = y_limits, expand = expansion(mult = c(0.02, 0.04))) +
      labs(
        title = panel_title,
        x = if (show_x) "Mean ALI (lower = less model-implied agency loss)" else NULL,
        y = y_title
    ) +
    guides(
      color = guide_legend(nrow = legend_rows, byrow = TRUE),
      shape = guide_legend(nrow = legend_rows, byrow = TRUE),
      linetype = "none"
    ) +
    theme_tnsre(8.6) +
    theme(
      legend.position = "bottom",
      axis.text.x = if (show_x) element_text(face = "plain", size = 7.3, color = "black") else element_blank(),
      axis.ticks.x = if (show_x) element_line(color = "black", linewidth = 0.32) else element_blank()
    )
}

make_frontier_plot <- function(aggregate, title, subtitle = NULL, splits = NULL, include_references = TRUE, fixed_x = NULL) {
  data <- aggregate %>%
    { if (!is.null(splits)) filter(., split_family %in% splits) else . } %>%
    policy_ordered() %>%
    mutate(split_label = droplevels(split_label))

  p_f1 <- frontier_panel(
    data,
    metric_col = "active_macro_f1",
    panel_title = "A. Active macro-F1 (higher is better)",
    y_title = "Active macro-F1",
    show_x = FALSE,
    include_references = include_references,
    fixed_x = fixed_x
  )
  p_auc <- frontier_panel(
    data,
    metric_col = "active_risk_coverage_auc",
    panel_title = "B. Active risk-coverage AUC (lower is better)",
    y_title = "Risk-coverage AUC",
    show_x = TRUE,
    include_references = include_references,
    fixed_x = fixed_x
  )

  (p_f1 / p_auc) +
    plot_layout(guides = "collect", heights = c(1, 1)) +
    plot_annotation(
      title = title,
      subtitle = subtitle,
      theme = theme(
        plot.title = element_text(family = figure_font_family, face = "bold", size = 10.0, hjust = 0.5, margin = margin(b = 2)),
        plot.subtitle = element_text(family = figure_font_family, face = "plain", size = 8.2, hjust = 0.5, color = "#3A3A3A", margin = margin(b = 4))
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
      metric_label = if_else(metric == "active_risk_coverage_auc", "Risk-coverage AUC benefit", "Active macro-F1 benefit"),
      split_label = recode(split_family, !!!split_labels),
      split_label = recode(split_label, "DB10 amputee LOSO" = "Amputee LOSO", "DB10 mixed-to-amputee" = "Mixed-to-amp."),
      split_label = factor(split_label, levels = c("Amputee LOSO", "Mixed-to-amp.")),
      tau_label = factor(format_tau(tau), levels = format_tau(sort(unique(tau)))),
      benefit = if_else(metric == "active_risk_coverage_auc", -estimate, estimate),
      benefit_low = if_else(metric == "active_risk_coverage_auc", -ci_high, ci_low),
      benefit_high = if_else(metric == "active_risk_coverage_auc", -ci_low, ci_high),
      holm_status = if_else(beneficial_and_holm_significant, "Holm-significant benefit", "No Holm-significant benefit")
    ) %>%
    group_by(metric_label) %>%
    mutate(star_y = benefit_high + diff(range(c(benefit_low, benefit_high))) * 0.08) %>%
    ungroup()

  dodge <- position_dodge(width = 0.48)

  ggplot(data, aes(x = tau_label, y = benefit, color = split_label, shape = holm_status)) +
    annotate("rect", xmin = -Inf, xmax = Inf, ymin = 0, ymax = Inf, fill = "#EAF6EF", alpha = 0.45) +
    geom_hline(yintercept = 0, linewidth = 0.38, color = "#383838") +
    geom_linerange(aes(ymin = benefit_low, ymax = benefit_high), linewidth = 0.52, alpha = 0.95, position = dodge) +
    geom_point(size = 2.35, stroke = 0.50, position = dodge) +
    geom_text(
      data = data %>% filter(beneficial_and_holm_significant),
      aes(y = star_y, label = "*", group = split_label),
      position = dodge,
      family = figure_font_family,
      fontface = "bold",
      size = 3.0,
      show.legend = FALSE
    ) +
    facet_wrap(~ metric_label, nrow = 1, scales = "free_y") +
    scale_color_manual(values = c("Amputee LOSO" = "#0072B2", "Mixed-to-amp." = "#D55E00"), name = "DB10 split") +
    scale_shape_manual(values = c("Holm-significant benefit" = 16, "No Holm-significant benefit" = 1), name = "Inference") +
    scale_y_continuous(labels = label_number(accuracy = 0.001), expand = expansion(mult = c(0.10, 0.14))) +
    labs(
      title = "Agency-margin benefit over matched SetACSA",
      subtitle = "Positive favors agency-margin; bars are nominal paired 95% CIs; stars and filled points mark Holm-adjusted beneficial tests.",
      x = "Matched budget tau",
      y = "Agency-margin - SetACSA"
    ) +
    guides(color = guide_legend(nrow = 1, byrow = TRUE), shape = guide_legend(nrow = 1, byrow = TRUE)) +
    theme_tnsre(8.8) +
    theme(legend.position = "bottom", legend.box = "vertical")
}

make_exact_budget_audit <- function() {
  compact_split_labels <- c(
    "DB10 able-to-amputee" = "DB10 able",
    "DB10 amputee LOSO" = "DB10 amputee",
    "DB10 mixed-to-amputee" = "DB10 mixed",
    "Hyser day shift" = "Hyser day",
    "Hyser subject LOSO" = "Hyser subject",
    "CEMHSEY forward-day" = "CEMHSEY"
  )

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
      dataset_label = factor(dataset_label, levels = c("DB10/MeganePro", "Hyser", "CEMHSEY")),
      split_label = recode(split_family, !!!split_labels),
      split_label = recode(split_label, !!!compact_split_labels),
      split_label = factor(split_label, levels = rev(unname(compact_split_labels))),
      tau_label = factor(format_tau(tau), levels = format_tau(sort(unique(tau)))),
      metric_label = if_else(metric == "active_risk_coverage_auc", "Risk-coverage AUC benefit", "Macro-F1 benefit"),
      benefit = if_else(metric == "active_risk_coverage_auc", -estimate, estimate),
      benefit_low = if_else(metric == "active_risk_coverage_auc", -ci_high, ci_low),
      benefit_high = if_else(metric == "active_risk_coverage_auc", -ci_low, ci_high),
      benefit_label = paste0(sprintf("%+.2f", benefit), if_else(beneficial_and_holm_significant, "*", ""))
    )

  max_abs_benefit <- max(abs(pairwise$benefit), na.rm = TRUE)

  p_delta <- ggplot(pairwise, aes(x = tau_label, y = split_label, fill = benefit)) +
    geom_tile(color = "white", linewidth = 0.35) +
    geom_text(aes(label = benefit_label), family = figure_font_family, size = 2.74, color = "#111111") +
    facet_wrap(~ metric_label, nrow = 1) +
    scale_fill_gradient2(
      low = "#B2182B",
      mid = "white",
      high = "#2166AC",
      midpoint = 0,
      limits = c(-max_abs_benefit, max_abs_benefit),
      labels = label_number(accuracy = 0.01),
      name = "Benefit"
    ) +
    labs(
      title = "A. Benefit vs exact confidence gate",
      x = "Agency budget tau",
      y = NULL
    ) +
    theme_tnsre(8.6) +
    theme(
      legend.position = "right",
      legend.key.height = unit(0.28, "in"),
      axis.text.y = element_text(size = 7.8),
      panel.grid.major = element_blank()
    )

  match_counts <- bind_rows(
    read_pub("db10_plain_conf_exact_budget_selection_by_unit.csv"),
    read_pub("hyser_plain_conf_exact_budget_selection_by_unit.csv"),
    read_pub("cemhsey_plain_conf_exact_budget_selection_by_unit.csv")
  ) %>%
    filter(comparison_family == "agency_vs_plain_conf_exact_budget") %>%
    group_by(dataset_id, split_family, tau) %>%
    summarise(
      exact_n = sum(exact_budget_match, na.rm = TRUE),
      total_n = n(),
      .groups = "drop"
    )

  heat <- read_pub("exact_budget_instability_by_tau.csv") %>%
    filter(dataset_id %in% c("db10", "hyser", "cemhsey"), scope == "required") %>%
    left_join(match_counts, by = c("dataset_id", "split_family", "tau")) %>%
    mutate(
      dataset_label = recode(dataset_id, !!!dataset_labels),
      split_label = recode(split_family, !!!split_labels),
      split_label = recode(split_label, !!!compact_split_labels),
      tau_label = format_tau(tau),
      split_label = factor(
        split_label,
        levels = rev(unname(compact_split_labels))
      ),
      tau_label = factor(tau_label, levels = format_tau(sort(unique(tau)))),
      text_color = if_else(confidence_exact_match_fraction >= 0.60, "white", "#111111"),
      count_label = paste0(exact_n, "\n/", total_n)
    )

  p_heat <- ggplot(heat, aes(x = tau_label, y = split_label, fill = confidence_exact_match_fraction)) +
    geom_tile(color = "white", linewidth = 0.35) +
    geom_text(aes(label = count_label, color = text_color), family = figure_font_family, size = 2.74, lineheight = 0.82) +
    scale_color_identity() +
    scale_fill_gradientn(
      colors = c("#F7FBFF", "#C6DBEF", "#6BAED6", "#08519C"),
      limits = c(0, 1),
      labels = label_number(accuracy = 0.1),
      name = "Exact match"
    ) +
    labs(
      title = "B. Exact matches among audit units",
      x = "Agency budget tau",
      y = NULL
    ) +
    theme_tnsre(8.6) +
    theme(
      legend.position = "right",
      legend.key.height = unit(0.28, "in"),
      axis.text.y = element_text(size = 7.8),
      panel.grid.major = element_blank()
    )

  (p_delta / p_heat) +
    plot_layout(heights = c(1.03, 0.97), guides = "collect") +
    plot_annotation(
      title = "Exact-budget confidence-threshold audit",
      subtitle = "Positive benefit favors agency-margin; risk-coverage AUC is sign-inverted. Stars mark Holm-adjusted beneficial tests.",
      theme = theme(
        plot.title = element_text(family = figure_font_family, face = "bold", size = 9.6, hjust = 0.5, margin = margin(b = 2)),
        plot.subtitle = element_text(family = figure_font_family, size = 7.8, hjust = 0.5, color = "#3A3A3A", margin = margin(b = 4))
      )
    )
}

make_earliest_safe <- function(summary, dataset_title, splits = NULL) {
  data <- summary %>%
    { if (!is.null(splits)) filter(., split_family %in% splits) else . } %>%
    mutate(
      policy_family = classify_policy(policy),
      tau = extract_tau(policy),
      split_label = recode(split_family, !!!split_labels),
      split_label = factor(split_label, levels = split_order),
      policy_family = factor(policy_family, levels = names(policy_palette))
    ) %>%
    filter(
      !is.na(tau),
      policy_family %in% frontier_families
    )

  timing_panel <- function(metric_col, panel_title, y_title, show_x = FALSE, accuracy = 0.01) {
    panel_data <- data %>% mutate(value = .data[[metric_col]])
    if (metric_col == "stable_safe_episode_rate") {
      y_limits <- c(0, min(1, max(panel_data$value, na.rm = TRUE) * 1.12))
      y_labels <- label_percent(accuracy = 1)
    } else {
      y_limits <- c(0, max(panel_data$value, na.rm = TRUE) * 1.08)
      y_labels <- label_number(accuracy = accuracy)
    }
    ggplot(panel_data, aes(x = tau, y = value, color = policy_family, shape = policy_family, linetype = policy_family, group = policy_family)) +
      geom_line(linewidth = 0.56, alpha = 0.96) +
      geom_point(size = 2.0, stroke = 0.38) +
      facet_wrap(~ split_label, nrow = 1) +
      scale_policy_color(name = "Policy") +
      scale_policy_shape(name = "Policy") +
      scale_policy_linetype(name = "Policy") +
      scale_x_continuous(breaks = sort(unique(panel_data$tau)), labels = format_tau, expand = expansion(mult = c(0.05, 0.08))) +
      scale_y_continuous(labels = y_labels, limits = y_limits, expand = expansion(mult = c(0.00, 0.04))) +
      labs(
        title = panel_title,
        x = if (show_x) "Agency budget tau" else NULL,
        y = y_title
      ) +
      guides(color = guide_legend(nrow = 1), shape = guide_legend(nrow = 1), linetype = "none") +
      theme_tnsre(8.6) +
      theme(
        legend.position = "bottom",
        axis.text.x = if (show_x) element_text(face = "plain", size = 7.2, color = "black") else element_blank(),
        axis.ticks.x = if (show_x) element_line(color = "black", linewidth = 0.32) else element_blank()
      )
  }

  p_rate <- timing_panel(
    "stable_safe_episode_rate",
    "A. Stable-safe episode rate (higher is better)",
    "Episode rate (%)",
    show_x = FALSE,
    accuracy = 0.01
  )
  p_time <- timing_panel(
    "median_earliest_stable_safe_s",
    "B. Earliest stable-safe time (lower is better)",
    "Time (s)",
    show_x = TRUE,
    accuracy = 0.1
  )

  (p_rate / p_time) +
    plot_layout(guides = "collect", heights = c(1, 1)) +
    plot_annotation(
      title = paste0(dataset_title, " timing diagnostic"),
      subtitle = "Offline prefix analysis; Panel B is conditional on episodes counted as stable-safe in Panel A.",
      theme = theme(
        plot.title = element_text(family = figure_font_family, face = "bold", size = 9.6, hjust = 0.5, margin = margin(b = 2)),
        plot.subtitle = element_text(family = figure_font_family, face = "plain", size = 8.0, hjust = 0.5, color = "#3A3A3A", margin = margin(b = 4))
      )
    ) &
    theme(legend.position = "bottom")
}

make_matched_budget <- function(aggregate, dataset_title, splits = NULL) {
  data <- aggregate %>%
    { if (!is.null(splits)) filter(., split_family %in% splits) else . } %>%
    policy_ordered() %>%
    filter(policy_family %in% c("Agency-margin", "Confidence gate")) %>%
    mutate(split_label = droplevels(split_label))

  matched_panel <- function(metric_col, panel_title, y_title, show_x = FALSE) {
    panel_data <- data %>% mutate(value = .data[[metric_col]])
    ggplot(panel_data, aes(x = tau, y = value, color = policy_family, shape = policy_family, linetype = policy_family, group = policy_family)) +
      geom_line(linewidth = 0.56, alpha = 0.96) +
      geom_point(size = 2.0, stroke = 0.38) +
      facet_wrap(~ split_label, nrow = 1) +
      scale_policy_color(name = "Policy") +
      scale_policy_shape(name = "Policy") +
      scale_policy_linetype(name = "Policy") +
      scale_x_continuous(breaks = sort(unique(panel_data$tau)), labels = format_tau, expand = expansion(mult = c(0.05, 0.08))) +
      scale_y_continuous(labels = label_number(accuracy = 0.01), limits = pad_range(panel_data$value, 0.09), expand = expansion(mult = c(0.02, 0.04))) +
      labs(
        title = panel_title,
        x = if (show_x) "Agency budget tau" else NULL,
        y = y_title
      ) +
      guides(color = guide_legend(nrow = 1), shape = guide_legend(nrow = 1), linetype = "none") +
      theme_tnsre(8.6) +
      theme(
        legend.position = "bottom",
        axis.text.x = if (show_x) element_text(face = "plain", size = 7.2, color = "black") else element_blank(),
        axis.ticks.x = if (show_x) element_line(color = "black", linewidth = 0.32) else element_blank()
      )
  }

  p_f1 <- matched_panel("active_macro_f1", "A. Active macro-F1 (higher is better)", "Active macro-F1", show_x = FALSE)
  p_auc <- matched_panel("active_risk_coverage_auc", "B. Active risk-coverage AUC (lower is better)", "Risk-coverage AUC", show_x = TRUE)

  (p_f1 / p_auc) +
    plot_layout(guides = "collect", heights = c(1, 1)) +
    plot_annotation(
      title = paste0(dataset_title, " matched-budget ablation"),
      theme = theme(plot.title = element_text(family = figure_font_family, face = "bold", size = 9.6, hjust = 0.5, margin = margin(b = 4)))
    ) &
    theme(legend.position = "bottom")
}

db10_aggregate <- prepare_aggregate(read_pub("db10_publication_aggregate_policy_metrics.csv"), "db10")
hyser_aggregate <- prepare_aggregate(read_pub("hyser_publication_aggregate_policy_metrics.csv"), "hyser")
cemhsey_aggregate <- prepare_aggregate(read_pub("cemhsey_publication_aggregate_policy_metrics.csv"), "cemhsey")

manifest <- bind_rows(
  save_figure(make_flow_figure(), "fig1_benchmark_flow", width = 7.16, height = 3.55),
  save_figure(
    make_frontier_plot(
      db10_aggregate,
      title = "DB10 primary performance-agency trade-off",
      subtitle = "Descriptive aggregate policy sweeps in the two claim-bearing split families.",
      splits = primary_db10,
      include_references = TRUE
    ),
    "fig2_db10_frontier",
    width = 7.16,
    height = 5.05
  ),
  save_figure(make_pairwise_setacsa(read_pub("db10_confidence_gate_pairwise.csv")), "fig3_setacsa_pairwise", width = 7.16, height = 3.85),
  save_figure(make_exact_budget_audit(), "fig4_exact_budget_audit", width = 7.16, height = 5.10),
  save_figure(
    make_frontier_plot(
      bind_rows(hyser_aggregate, cemhsey_aggregate),
      title = "External EMG stress-test trade-offs",
      subtitle = "Hyser and CEMHSEY probe policy-layer behavior; they are not primary prosthetic clinical evidence.",
      include_references = FALSE
    ),
    "fig5_external_robustness",
    width = 7.16,
    height = 5.05
  ),
  save_figure(make_earliest_safe(read_pub("db10_earliest_safe_summary.csv"), "DB10", splits = primary_db10), "fig6_db10_earliest_safe", width = 7.16, height = 5.00),
  save_figure(make_matched_budget(db10_aggregate, "DB10", splits = db10_all), "figS_db10_matched_budget_ablation", width = 7.16, height = 4.95),
  save_figure(make_matched_budget(hyser_aggregate, "Hyser"), "figS_hyser_matched_budget_ablation", width = 7.16, height = 4.65),
  save_figure(make_matched_budget(cemhsey_aggregate, "CEMHSEY"), "figS_cemhsey_matched_budget_ablation", width = 7.16, height = 4.65),
  save_figure(make_earliest_safe(read_pub("hyser_earliest_safe_summary.csv"), "Hyser"), "figS_hyser_earliest_safe", width = 7.16, height = 5.00),
  save_figure(make_earliest_safe(read_pub("cemhsey_earliest_safe_summary.csv"), "CEMHSEY"), "figS_cemhsey_earliest_safe", width = 7.16, height = 5.00)
)

manifest <- manifest %>%
  mutate(
    created_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
    source_dir = relative_to_repo(input_dir)
  )
readr::write_csv(manifest, file.path(out_dir, "tnsre_figure_manifest.csv"))

message("Wrote ", nrow(manifest), " figure assets to ", out_dir)
