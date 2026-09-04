#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
  stop(
    "usage: run_kgmn_metdna2_200std_dreams_arm.R ",
    "<overlay_source> <output_dir> <embedding_csv_gz> <calibration_json> ",
    "<noop_author|official_dreams|author_official_intersection>"
  )
}

overlay_source <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- args[[2]]
embedding_file <- normalizePath(args[[3]], mustWork = TRUE)
calibration_file <- normalizePath(args[[4]], mustWork = TRUE)
arm <- args[[5]]
allowed_arms <- c("noop_author", "official_dreams", "author_official_intersection")
if (!(arm %in% allowed_arms)) stop(paste("unsupported edge arm:", arm))

if (dir.exists(output_dir) && length(list.files(output_dir, all.files = TRUE, no.. = TRUE)) > 0) {
  stop(paste("refusing to overwrite non-empty arm directory:", output_dir))
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite is required")
if (!requireNamespace("MetDNA2", quietly = TRUE)) stop("MetDNA2 overlay is not installed")
if (as.character(utils::packageVersion("MetDNA2")) != "1.2.10") {
  stop("expected MetDNA2 1.2.10 overlay")
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

extdata <- file.path(overlay_source, "inst", "extdata")
input_files <- c(
  data = "peak_table_200STD_neg_200805.csv",
  spectra = "spectra_200STD_neg_200805.msp",
  annotated_features = "peak_table_annotated_200STD_neg_200805.csv",
  truth = "annotation_initial.csv"
)
for (filename in unname(input_files)) {
  path <- file.path(extdata, filename)
  if (!file.exists(path) || file.info(path)$size <= 0) stop(paste("missing 200STD input:", path))
}

copy_contract <- c(
  data = file.copy(file.path(extdata, input_files[["data"]]), file.path(output_dir, "data.csv"), overwrite = FALSE),
  spectra = file.copy(file.path(extdata, input_files[["spectra"]]), file.path(output_dir, "spectra.msp"), overwrite = FALSE),
  annotated_features = file.copy(
    file.path(extdata, input_files[["annotated_features"]]),
    file.path(output_dir, "peak_table_annotated_200STD_neg_200805.csv"), overwrite = FALSE
  ),
  truth = file.copy(
    file.path(extdata, input_files[["truth"]]), file.path(output_dir, "annotation_initial.csv"), overwrite = FALSE
  )
)
if (!all(copy_contract)) stop("failed to freeze one or more 200STD inputs")

peak_table <- utils::read.csv(file.path(output_dir, "data.csv"), check.names = FALSE)
if (!identical(colnames(peak_table)[1:3], c("name", "mz", "rt"))) stop("invalid 200STD peak table")
sample_names <- colnames(peak_table)[-(1:3)]
if (length(sample_names) != 4 || anyDuplicated(sample_names)) stop("expected four unique 200STD samples")
sample_info <- data.frame(sample.name = sample_names, group = rep("200STD", length(sample_names)))
utils::write.csv(sample_info, file.path(output_dir, "sample.info.csv"), row.names = FALSE, quote = FALSE)

genform_source <- file.path(extdata, "GenForm")
genform_magic <- as.integer(readBin(genform_source, what = "raw", n = 4))
if (!identical(genform_magic, c(127L, 69L, 76L, 70L))) stop("bundled GenForm is not ELF")
genform_dir <- file.path(output_dir, "_runtime_genform")
dir.create(genform_dir, recursive = TRUE, showWarnings = FALSE)
genform_runtime <- file.path(genform_dir, "GenForm")
if (!file.copy(genform_source, genform_runtime, overwrite = FALSE)) stop("failed to stage GenForm")
Sys.chmod(genform_runtime, mode = "0755")
if (.Platform$OS.type != "unix" || file.access(genform_runtime, mode = 1) != 0) {
  stop("arm evaluation requires executable Linux GenForm")
}

embedding_frame <- utils::read.csv(gzfile(embedding_file), check.names = FALSE)
if (!("feature_name" %in% names(embedding_frame))) stop("embedding table lacks feature_name")
feature_names <- as.character(embedding_frame$feature_name)
if (anyNA(feature_names) || anyDuplicated(feature_names) || any(feature_names == "")) {
  stop("embedding feature names must be unique and non-empty")
}
embedding_matrix <- as.matrix(embedding_frame[, setdiff(names(embedding_frame), "feature_name"), drop = FALSE])
storage.mode(embedding_matrix) <- "double"
if (any(!is.finite(embedding_matrix))) stop("embedding table contains non-finite values")
norm_error <- max(abs(sqrt(rowSums(embedding_matrix^2)) - 1))
if (!is.finite(norm_error) || norm_error > 2e-6) stop("embedding table is not unit-normalised")
feature_index <- stats::setNames(seq_along(feature_names), feature_names)

calibration <- jsonlite::fromJSON(calibration_file, simplifyVector = FALSE)
threshold <- 0.5
official_coefficient <- official_intercept <- author_coefficient <- author_intercept <- NA_real_
if (arm != "noop_author") {
  official_coefficient <- as.numeric(calibration$official_dreams$coefficient)
  official_intercept <- as.numeric(calibration$official_dreams$intercept)
  threshold <- as.numeric(
    calibration$deployment_thresholds_full_refit[[arm]][["fdr_0.05"]]
  )
  if (arm == "author_official_intersection") {
    author_coefficient <- as.numeric(calibration$author_dp$coefficient)
    author_intercept <- as.numeric(calibration$author_dp$intercept)
  }
}
if (length(threshold) != 1 || !is.finite(threshold) || threshold < 0 || threshold > 1) {
  stop(paste("arm lacks a finite frozen 5% FDR threshold:", arm))
}

hook_log_dir <- file.path(output_dir, "_hook_worker_logs")
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

parameters <- list(
  package = "MetDNA2", package_version = "1.2.10", arm = arm,
  overlay_author_commit = overlay_manifest$author_commit,
  edge_threshold = threshold, edge_threshold_fdr = 0.05,
  embedding_file = embedding_file, calibration_file = calibration_file,
  recursive_similarity = if (arm == "noop_author") "author_dp_noop_hook" else arm,
  candidate_generation = "author", reaction_network = "author", propagation = "author",
  credential = TRUE, rt_calibration = FALSE
)
dput(parameters, file = file.path(output_dir, "frozen_parameters.R"))

MetDNA2::MetDNA2(
  ms1_file = "data.csv", ms2_file = "spectra.msp", sample_info_file = "sample.info.csv",
  metdna_version = "version2", ms2_type = "msp", path = output_dir, thread = 8,
  is_check_data = TRUE, is_anno_initial_seed = TRUE, is_anno_mrn = TRUE,
  is_credential = TRUE, is_bio_interpret = FALSE, is_exported_report = FALSE,
  is_cred_pg_filter = TRUE, is_cred_formula_filter = FALSE, is_rm_intermediate_data = FALSE,
  lib = "zhumetlib_qtof", polarity = "negative", instrument = "SciexTripleTOF",
  column = "hilic", ce = "30", method_lc = "Amide23min", is_rt_calibration = FALSE,
  dp_cutoff = 0.8, direction = "reverse", is_plot_ms2 = FALSE, extension_step = "2",
  dp_tol = threshold, max_step = 3, score_cutoff = 0, seed_neighbor_match_plot = FALSE,
  candidate_num = 5, scoring_approach_recursive = "dp", matched_frag_cutoff = 1,
  whether_link_frag = FALSE, dir_GenForm = genform_dir, is_pred_formula_all = FALSE,
  platform = "linux", is_plot_pseudo_MS1 = FALSE, test_evaluation = "200STD"
)

log_files <- list.files(hook_log_dir, full.names = TRUE)
if (length(log_files) == 0) stop("recursive hook produced no worker execution logs")
log_rows <- do.call(rbind, lapply(log_files, function(path) {
  utils::read.csv(path, header = FALSE, col.names = c("edges", "minimum", "maximum"))
}))
if (nrow(log_rows) == 0 || sum(log_rows$edges) <= 0) stop("recursive hook scored no dynamic edges")
hook_summary <- list(
  status = "kgmn_metdna2_recursive_hook_executed",
  arm = arm,
  worker_processes = length(log_files),
  hook_calls = nrow(log_rows),
  dynamic_edges_scored = sum(log_rows$edges),
  minimum_score = min(log_rows$minimum),
  maximum_score = max(log_rows$maximum),
  frozen_threshold = threshold,
  psock_worker_execution_proven = TRUE
)
jsonlite::write_json(hook_summary, file.path(output_dir, "recursive_hook_summary.json"), pretty = TRUE, auto_unbox = TRUE)
cat(jsonlite::toJSON(hook_summary, pretty = TRUE, auto_unbox = TRUE), "\n")
cat("[kgmn-metdna2] DreaMS edge arm completed\n")
