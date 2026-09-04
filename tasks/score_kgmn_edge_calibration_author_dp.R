#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("usage: score_kgmn_edge_calibration_author_dp.R <input_dir> <output_dir>")
}
input_dir <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- args[[2]]
if (dir.exists(output_dir) && length(list.files(output_dir, all.files = TRUE, no.. = TRUE)) > 0) {
  stop(paste("refusing to overwrite non-empty author score directory:", output_dir))
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
if (!requireNamespace("MetDNA2", quietly = TRUE)) stop("MetDNA2 is required")
if (!requireNamespace("tibble", quietly = TRUE)) stop("tibble is required")
if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite is required")
if (as.character(utils::packageVersion("MetDNA2")) != "1.2.10") stop("expected MetDNA2 1.2.10")

sha256_file <- function(path) {
  output <- system2("sha256sum", args = path, stdout = TRUE, stderr = TRUE)
  status <- attr(output, "status")
  if (!is.null(status) && status != 0L) stop(paste("sha256sum failed for", path))
  fields <- strsplit(output[[1]], "[[:space:]]+")[[1]]
  fields <- fields[nzchar(fields)]
  if (length(fields) < 1L || !grepl("^[0-9a-f]{64}$", fields[[1]])) {
    stop(paste("invalid sha256sum output for", path))
  }
  fields[[1]]
}

spectra_path <- file.path(input_dir, "spectra_long.csv.gz")
pairs_path <- file.path(input_dir, "pairs.csv.gz")
input_report_path <- file.path(input_dir, "report.json")
for (path in c(spectra_path, pairs_path, input_report_path)) {
  if (!file.exists(path) || file.info(path)$size <= 0) stop(paste("missing exact-author input:", path))
}
input_report <- jsonlite::fromJSON(input_report_path, simplifyVector = FALSE)
if (!identical(input_report$status, "kgmn_edge_calibration_exact_author_input_frozen")) {
  stop("exact-author input report status mismatch")
}
if (!identical(input_report$provenance$spectra_long_sha256, sha256_file(spectra_path)) ||
    !identical(input_report$provenance$pairs_sha256, sha256_file(pairs_path))) {
  stop("exact-author input hash mismatch")
}

spectra <- utils::read.csv(gzfile(spectra_path), check.names = FALSE)
pairs <- utils::read.csv(gzfile(pairs_path), check.names = FALSE)
required_spectra <- c("hdf5_row", "precursor_mz", "fragment_mz", "intensity")
required_pairs <- c("triple_index", "source_row", "positive_row", "decoy_row")
if (!all(required_spectra %in% names(spectra))) stop("spectra input columns are incomplete")
if (!all(required_pairs %in% names(pairs))) stop("pair input columns are incomplete")
if (any(!is.finite(as.matrix(spectra[, c("precursor_mz", "fragment_mz", "intensity")]))) ||
    any(spectra$precursor_mz <= 0) || any(spectra$fragment_mz <= 0) || any(spectra$intensity <= 0)) {
  stop("spectra input has invalid numerical values")
}
if (!identical(as.integer(pairs$triple_index), seq.int(0L, nrow(pairs) - 1L))) {
  stop("pair triple order is not contiguous and zero-based")
}

split_spectra <- split(spectra, as.character(spectra$hdf5_row))
objects <- lapply(names(split_spectra), function(row_name) {
  frame <- split_spectra[[row_name]]
  precursor <- unique(frame$precursor_mz)
  if (length(precursor) != 1 || !is.finite(precursor)) stop("one HDF5 row has inconsistent precursor")
  spec <- as.matrix(frame[, c("fragment_mz", "intensity"), drop = FALSE])
  colnames(spec) <- c("mz", "intensity")
  MetDNA2::convertSpectraData(
    ms2_data = list(
      info = tibble::tibble(NAME = paste0("row_", row_name), PRECURSORMZ = as.numeric(precursor)),
      spec = spec
    )
  )
})
names(objects) <- names(split_spectra)
precursor_by_row <- vapply(split_spectra, function(frame) unique(frame$precursor_mz), numeric(1))

score_pair <- function(left_row, right_row) {
  left_name <- as.character(left_row)
  right_name <- as.character(right_row)
  if (is.null(objects[[left_name]]) || is.null(objects[[right_name]])) stop("pair references missing spectrum")
  if (precursor_by_row[[left_name]] >= precursor_by_row[[right_name]]) {
    experimental <- objects[[left_name]]
    reference <- objects[[right_name]]
  } else {
    experimental <- objects[[right_name]]
    reference <- objects[[left_name]]
  }
  result <- try(
    MetDNA2::runSpecMatch(
      obj_ms2_cpd1 = experimental, obj_ms2_cpd2 = reference,
      mz_tol_ms2 = 25, scoring_approach = "dp"
    ), silent = TRUE
  )
  if (inherits(result, "try-error") || length(result) == 0) return(0)
  score <- as.numeric(result@info$scoreReverse)
  if (length(score) != 1 || !is.finite(score) || score < 0 || score > 1) {
    stop("MetDNA2 returned an invalid reverse DP score")
  }
  score
}

author_positive <- numeric(nrow(pairs))
author_decoy <- numeric(nrow(pairs))
for (index in seq_len(nrow(pairs))) {
  author_positive[[index]] <- score_pair(pairs$source_row[[index]], pairs$positive_row[[index]])
  author_decoy[[index]] <- score_pair(pairs$source_row[[index]], pairs$decoy_row[[index]])
  if (index %% 100 == 0 || index == nrow(pairs)) {
    cat(sprintf("[exact author DP] %d/%d triples\n", index, nrow(pairs)))
  }
}
scores <- pairs
scores$author_positive <- author_positive
scores$author_decoy <- author_decoy
scores_path <- file.path(output_dir, "author_dp_scores.csv.gz")
utils::write.csv(scores, gzfile(scores_path), row.names = FALSE, quote = FALSE)
report <- list(
  status = "kgmn_edge_calibration_exact_author_dp_complete",
  formal = TRUE,
  triples = nrow(scores),
  positive_mean = mean(author_positive),
  decoy_mean = mean(author_decoy),
  positive_beats_decoy = mean(author_positive > author_decoy),
  scorer = "MetDNA2 1.2.10 runSpecMatch scoreReverse; 25 ppm; smaller precursor as reference",
  contracts = list(
    python_similarity_proxy_used = FALSE,
    precursor_handling = "author convertSpectraData and includePrecursor logic",
    triple_order_preserved = TRUE
  ),
  provenance = list(
    input_report_sha256 = sha256_file(input_report_path),
    spectra_long_sha256 = sha256_file(spectra_path),
    pairs_sha256 = sha256_file(pairs_path),
    scores_sha256 = sha256_file(scores_path)
  ),
  claim_limit = "Exact author edge scores for calibration; no propagated annotation result."
)
jsonlite::write_json(report, file.path(output_dir, "report.json"), pretty = TRUE, auto_unbox = TRUE)
cat(jsonlite::toJSON(report, pretty = TRUE, auto_unbox = TRUE), "\n")
