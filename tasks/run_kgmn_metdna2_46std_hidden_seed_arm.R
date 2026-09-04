#!/usr/bin/env Rscript

# Continue one already whitelist-filtered hidden-seed state with the frozen
# DreaMS edge overlay.  Candidate generation, MRN, propagation depth,
# credential, ion-form handling and final scoring remain author MetDNA2.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
  stop(
    "usage: run_kgmn_metdna2_46std_hidden_seed_arm.R ",
    "<overlay_source> <prepared_state_dir> <embedding_csv_gz> <calibration_json> ",
    "<noop_author|official_dreams|author_official_intersection>"
  )
}

overlay_source <- normalizePath(args[[1]], mustWork = TRUE)
state_dir <- normalizePath(args[[2]], mustWork = TRUE)
embedding_file <- normalizePath(args[[3]], mustWork = TRUE)
calibration_file <- normalizePath(args[[4]], mustWork = TRUE)
arm <- args[[5]]
allowed_arms <- c("noop_author", "official_dreams", "author_official_intersection")
if (!(arm %in% allowed_arms)) stop(paste("unsupported edge arm:", arm))

if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite is required")
if (!requireNamespace("MetDNA2", quietly = TRUE)) stop("MetDNA2 overlay is not installed")
if (as.character(utils::packageVersion("MetDNA2")) != "1.2.10") stop("expected MetDNA2 1.2.10 overlay")

audit_path <- file.path(state_dir, "hidden_seed_state_audit.json")
if (!file.exists(audit_path)) stop("prepared hidden-seed state audit is missing")
audit <- jsonlite::fromJSON(audit_path, simplifyVector = FALSE)
if (!identical(audit$status, "kgmn_hidden_seed_initial_state_prepared")) stop("hidden-seed state status mismatch")
if (!identical(as.integer(audit$hidden_identity_leakage), 0L)) stop("hidden-seed state reports identity leakage")
polarity <- as.character(audit$polarity)
if (!(polarity %in% c("positive", "negative"))) stop("hidden-seed polarity is invalid")

for (path in c("02_result_MRN_annotation", "03_annotation_credential", "00_annotation_table")) {
  candidate <- file.path(state_dir, path)
  if (dir.exists(candidate) && length(list.files(candidate, all.files = TRUE, no.. = TRUE)) > 0) {
    stop(paste("refusing to reuse a state with recursive/final output:", candidate))
  }
}

overlay_manifest_path <- file.path(overlay_source, "DREAMS_EDGE_OVERLAY_MANIFEST.json")
if (!file.exists(overlay_manifest_path)) stop("missing DreaMS edge overlay manifest")
overlay_manifest <- jsonlite::fromJSON(overlay_manifest_path, simplifyVector = FALSE)
if (!identical(overlay_manifest$status, "kgmn_metdna2_dreams_edge_overlay_built")) {
  stop("overlay manifest status mismatch")
}
if (!isTRUE(overlay_manifest$contracts$hook_is_explicitly_exported_to_psock_workers)) {
  stop("overlay does not guarantee PSOCK hook export")
}

ms2_files <- list.files(state_dir, pattern = "\\.(mgf|msp|mzXML|cef)$", full.names = FALSE, ignore.case = TRUE)
if (length(ms2_files) < 1) stop("DreaMS hidden-seed arms require at least one staged MS2 file")
ms2_extensions <- unique(tolower(tools::file_ext(ms2_files)))
if (length(ms2_extensions) != 1) stop("DreaMS hidden-seed arm MS2 inputs must be homogeneous")
ms2_type <- switch(
  ms2_extensions[[1]],
  mgf = "mgf", msp = "msp", mzxml = "mzxml", cef = "cef",
  stop("unsupported staged MS2 type")
)
for (name in c("data.csv", "sample.info.csv")) {
  if (!file.exists(file.path(state_dir, name))) stop(paste("prepared state lacks", name))
}

embedding_frame <- utils::read.csv(gzfile(embedding_file), check.names = FALSE)
if (!identical(names(embedding_frame)[1], "feature_name")) stop("embedding table must start with feature_name")
feature_names <- trimws(as.character(embedding_frame$feature_name))
if (anyNA(feature_names) || any(feature_names == "") || anyDuplicated(feature_names)) {
  stop("embedding feature names must be unique and non-empty")
}
embedding_matrix <- as.matrix(embedding_frame[, -1, drop = FALSE])
storage.mode(embedding_matrix) <- "double"
if (ncol(embedding_matrix) < 2 || any(!is.finite(embedding_matrix))) stop("invalid embedding matrix")
norm_error <- max(abs(sqrt(rowSums(embedding_matrix^2)) - 1))
if (!is.finite(norm_error) || norm_error > 2e-6) stop("embedding table is not unit-normalised")
feature_index <- stats::setNames(seq_along(feature_names), feature_names)

calibration <- jsonlite::fromJSON(calibration_file, simplifyVector = FALSE)
threshold <- 0.5
official_coefficient <- official_intercept <- author_coefficient <- author_intercept <- NA_real_
if (arm != "noop_author") {
  official_coefficient <- as.numeric(calibration$official_dreams$coefficient)
  official_intercept <- as.numeric(calibration$official_dreams$intercept)
  threshold <- as.numeric(calibration$deployment_thresholds_full_refit[[arm]][["fdr_0.05"]])
  if (arm == "author_official_intersection") {
    author_coefficient <- as.numeric(calibration$author_dp$coefficient)
    author_intercept <- as.numeric(calibration$author_dp$intercept)
  }
}
if (length(threshold) != 1 || !is.finite(threshold) || threshold < 0 || threshold > 1) {
  stop(paste("arm lacks a finite frozen 5% FDR threshold:", arm))
}

hook_log_dir <- file.path(state_dir, "_hook_worker_logs")
dir.create(hook_log_dir, recursive = TRUE, showWarnings = FALSE)
recursive_hook <- local({
  z <- embedding_matrix
  index <- feature_index
  selected_arm <- arm
  official_a <- official_coefficient
  official_b <- official_intercept
  author_a <- author_coefficient
  author_b <- author_intercept
  log_dir <- hook_log_dir
  function(edge_context) {
    source_index <- unname(index[as.character(edge_context$seed_peak_name)])
    neighbor_index <- unname(index[as.character(edge_context$neighbor_peak_name)])
    if (anyNA(source_index) || anyNA(neighbor_index)) {
      missing <- unique(c(
        as.character(edge_context$seed_peak_name)[is.na(source_index)],
        as.character(edge_context$neighbor_peak_name)[is.na(neighbor_index)]
      ))
      stop(paste("DreaMS hook lacks feature embeddings:", paste(head(missing, 10), collapse = ",")))
    }
    if (selected_arm == "noop_author") {
      result <- as.numeric(edge_context$author_score)
    } else {
      cosine <- rowSums(z[source_index, , drop = FALSE] * z[neighbor_index, , drop = FALSE])
      official_probability <- stats::plogis(official_a * cosine + official_b)
      if (selected_arm == "official_dreams") {
        result <- official_probability
      } else {
        author_probability <- stats::plogis(author_a * as.numeric(edge_context$author_score) + author_b)
        result <- pmin(author_probability, official_probability)
      }
    }
    if (any(!is.finite(result)) || any(result < 0 | result > 1)) stop("invalid hook score")
    log_path <- file.path(log_dir, paste0("worker_", Sys.getpid(), ".csv"))
    cat(length(result), min(result), max(result), sep = ",", file = log_path, append = TRUE)
    cat("\n", file = log_path, append = TRUE)
    result
  }
})
options(MetDNA2.recursive_edge_score_hook = recursive_hook)
on.exit(options(MetDNA2.recursive_edge_score_hook = NULL), add = TRUE)

genform_source <- file.path(overlay_source, "inst", "extdata", "GenForm")
if (!file.exists(genform_source)) stop("overlay lacks bundled GenForm")
genform_dir <- file.path(state_dir, "_runtime_genform")
dir.create(genform_dir, recursive = TRUE, showWarnings = FALSE)
genform_runtime <- file.path(genform_dir, "GenForm")
if (!file.copy(genform_source, genform_runtime, overwrite = FALSE)) stop("failed to stage GenForm")
Sys.chmod(genform_runtime, mode = "0755")
if (.Platform$OS.type != "unix" || file.access(genform_runtime, mode = 1) != 0) stop("executable Linux GenForm required")

parameters <- list(
  status = "kgmn_46std_hidden_seed_edge_arm_parameters_frozen",
  package_version = as.character(utils::packageVersion("MetDNA2")),
  overlay_author_commit = overlay_manifest$author_commit,
  repeat = as.integer(audit$repeat), polarity = polarity, arm = arm,
  initial_seed_annotation = FALSE, edge_threshold = threshold, edge_threshold_fdr = 0.05,
  candidate_generation = "author", reaction_network = "author", propagation_depth = "author",
  credential = TRUE, formula_filter = TRUE,
  ms2_input_files = length(ms2_files),
  embedding_file = embedding_file, calibration_file = calibration_file
)
dput(parameters, file = file.path(state_dir, "hidden_seed_edge_arm_parameters.R"))

MetDNA2::MetDNA2(
  ms1_file = "data.csv", ms2_file = ms2_files[[1]], sample_info_file = "sample.info.csv",
  metdna_version = "version2", ms2_type = ms2_type, path = state_dir, thread = 8,
  is_check_data = TRUE, is_anno_initial_seed = FALSE, is_anno_mrn = TRUE,
  is_credential = TRUE, is_bio_interpret = FALSE, is_exported_report = FALSE,
  is_cred_pg_filter = TRUE, is_cred_formula_filter = TRUE, is_rm_intermediate_data = FALSE,
  lib = "zhumetlib_qtof", polarity = polarity, instrument = "SciexTripleTOF",
  column = "hilic", ce = "30", method_lc = "Amide23min", is_rt_calibration = FALSE,
  direction = "reverse", extension_step = "2", dp_tol = threshold, max_step = 3,
  score_cutoff = 0, seed_neighbor_match_plot = FALSE, candidate_num = 5,
  scoring_approach_recursive = "dp", matched_frag_cutoff = 1, whether_link_frag = FALSE,
  dir_GenForm = genform_dir, is_pred_formula_all = FALSE, platform = "linux",
  is_plot_pseudo_MS1 = FALSE, test_evaluation = "46STD"
)

required_outputs <- c(
  file.path(state_dir, "00_annotation_table", "00_intermediate_data", "list_identification"),
  file.path(state_dir, "00_annotation_table", "00_intermediate_data", "table_identification"),
  file.path(state_dir, "00_annotation_table", "table3_identification_pair.csv")
)
missing <- required_outputs[!file.exists(required_outputs) | file.info(required_outputs)$size <= 0]
if (length(missing) > 0) stop(paste("hidden-seed edge arm lacks outputs:", paste(missing, collapse = ",")))
log_files <- list.files(hook_log_dir, full.names = TRUE)
if (length(log_files) == 0) stop("recursive hook produced no worker execution logs")
log_rows <- do.call(rbind, lapply(log_files, function(path) {
  utils::read.csv(path, header = FALSE, col.names = c("edges", "minimum", "maximum"))
}))
if (nrow(log_rows) == 0 || sum(log_rows$edges) <= 0) stop("recursive hook scored no dynamic edges")

completion <- list(
  status = "kgmn_46std_hidden_seed_edge_arm_complete", formal = TRUE,
  repeat = as.integer(audit$repeat), polarity = polarity, arm = arm,
  hook_calls = nrow(log_rows), dynamic_edges_scored = sum(log_rows$edges),
  minimum_score = min(log_rows$minimum), maximum_score = max(log_rows$maximum),
  frozen_threshold = threshold,
  contracts = list(
    initial_seed_rerun = FALSE, hidden_identity_leakage = 0,
    candidate_generation_unchanged = TRUE, reaction_network_unchanged = TRUE,
    credential_unchanged = TRUE, inference_uses_no_truth = TRUE
  ),
  claim_limit = "One frozen hidden-seed arm; matched cross-repeat evaluation is required."
)
jsonlite::write_json(completion, file.path(state_dir, "hidden_seed_edge_arm_completion.json"), pretty = TRUE, auto_unbox = TRUE)
cat(jsonlite::toJSON(completion, pretty = TRUE, auto_unbox = TRUE), "\n")
