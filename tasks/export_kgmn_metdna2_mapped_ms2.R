#!/usr/bin/env Rscript

# Export MetDNA2's own MS1-feature-to-MS2 mapping as deterministic MSP.
# This is the only admissible bridge for raw mzXML inputs: it preserves the
# exact feature names and representative spectra selected by combineMs1Ms2.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop(
    "usage: export_kgmn_metdna2_mapped_ms2.R ",
    "<initial_seed_dir> <output_msp> <report_json>"
  )
}

initial_dir <- normalizePath(args[[1]], mustWork = TRUE)
output_msp <- args[[2]]
report_json <- args[[3]]

if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite is required")
if (file.exists(output_msp) || file.exists(report_json)) {
  stop("refusing to overwrite mapped-MS2 output")
}
dir.create(dirname(output_msp), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(report_json), recursive = TRUE, showWarnings = FALSE)

cache_path <- file.path(
  initial_dir,
  "01_result_initial_seed_annotation",
  "00_intermediate_data",
  "ms2"
)
ms1_path <- file.path(initial_dir, "data.csv")
if (!file.exists(cache_path) || file.info(cache_path)$size <= 0) {
  stop(paste("missing MetDNA2 mapped-MS2 cache:", cache_path))
}
if (!file.exists(ms1_path) || file.info(ms1_path)$size <= 0) {
  stop(paste("missing staged MS1 feature table:", ms1_path))
}

cache_env <- new.env(parent = emptyenv())
loaded <- load(cache_path, envir = cache_env)
if (!identical(loaded, "ms2") || !exists("ms2", envir = cache_env, inherits = FALSE)) {
  stop("mapped-MS2 cache must contain exactly the object named ms2")
}
ms2 <- get("ms2", envir = cache_env, inherits = FALSE)
if (!is.list(ms2) || length(ms2) == 0) stop("mapped-MS2 cache is empty")

feature_names <- names(ms2)
if (is.null(feature_names) || any(!nzchar(trimws(feature_names)))) {
  stop("mapped-MS2 entries require non-empty feature names")
}
feature_names <- trimws(feature_names)
if (anyDuplicated(feature_names)) stop("mapped-MS2 feature names are not unique")

ms1 <- utils::read.csv(ms1_path, check.names = FALSE, stringsAsFactors = FALSE)
if (ncol(ms1) < 3 || !identical(tolower(colnames(ms1)[1:3]), c("name", "mz", "rt"))) {
  stop("staged MS1 table must start with name,mz,rt")
}
ms1_names <- trimws(as.character(ms1[[1]]))
if (any(!nzchar(ms1_names)) || anyDuplicated(ms1_names)) {
  stop("MS1 feature names must be non-empty and unique")
}
if (!all(feature_names %in% ms1_names)) {
  missing <- feature_names[!(feature_names %in% ms1_names)]
  stop(paste("mapped-MS2 names absent from MS1 table:", paste(head(missing, 10), collapse = ",")))
}

extract_info_value <- function(info, key) {
  if (is.null(info)) return(NA_character_)
  if (!is.null(rownames(info)) && key %in% rownames(info)) {
    return(as.character(info[key, 1]))
  }
  if (!is.null(colnames(info)) && key %in% colnames(info)) {
    return(as.character(info[1, key]))
  }
  if (!is.null(names(info)) && key %in% names(info)) {
    return(as.character(info[[key]][[1]]))
  }
  NA_character_
}

connection <- file(output_msp, open = "wt", encoding = "UTF-8")
on.exit(close(connection), add = TRUE)
peak_counts <- integer(length(ms2))
precursors <- numeric(length(ms2))

for (index in seq_along(ms2)) {
  entry <- ms2[[index]]
  if (!is.list(entry) || is.null(entry$info) || is.null(entry$spec)) {
    stop(paste("invalid mapped-MS2 entry at index", index))
  }
  cache_name <- trimws(extract_info_value(entry$info, "NAME"))
  if (!identical(cache_name, feature_names[[index]])) {
    stop(paste("mapped-MS2 list name/info NAME mismatch at index", index))
  }
  precursor <- suppressWarnings(as.numeric(extract_info_value(entry$info, "PRECURSORMZ")))
  if (length(precursor) != 1 || !is.finite(precursor) || precursor <= 0) {
    stop(paste("invalid precursor m/z at index", index))
  }
  spectrum <- as.matrix(entry$spec)
  if (!is.numeric(spectrum) || ncol(spectrum) != 2 || nrow(spectrum) < 1) {
    stop(paste("invalid mapped spectrum at index", index))
  }
  storage.mode(spectrum) <- "double"
  if (any(!is.finite(spectrum)) || any(spectrum[, 1] <= 0) || any(spectrum[, 2] < 0)) {
    stop(paste("non-finite or invalid mapped spectrum values at index", index))
  }
  spectrum <- spectrum[order(spectrum[, 1], spectrum[, 2]), , drop = FALSE]
  peak_counts[[index]] <- nrow(spectrum)
  precursors[[index]] <- precursor

  writeLines(paste0("NAME: ", feature_names[[index]]), connection)
  writeLines(paste0("PRECURSORMZ: ", format(precursor, digits = 15, scientific = FALSE, trim = TRUE)), connection)
  writeLines(paste0("Num Peaks: ", nrow(spectrum)), connection)
  peak_lines <- paste(
    format(spectrum[, 1], digits = 15, scientific = FALSE, trim = TRUE),
    format(spectrum[, 2], digits = 15, scientific = FALSE, trim = TRUE)
  )
  writeLines(peak_lines, connection)
  writeLines("", connection)
}
close(connection)
on.exit(NULL, add = FALSE)

if (!file.exists(output_msp) || file.info(output_msp)$size <= 0) {
  stop("mapped MSP export is empty")
}

report <- list(
  status = "kgmn_metdna2_author_mapped_ms2_export_complete",
  formal = TRUE,
  spectra = length(ms2),
  ms1_features = nrow(ms1),
  median_peaks = as.numeric(stats::median(peak_counts)),
  precursor_range = c(min(precursors), max(precursors)),
  contracts = list(
    source = "MetDNA2 combineMs1Ms2 mapped cache",
    representative_spectrum_selection = "author implementation",
    feature_names_preserved = TRUE,
    identity_labels_used = FALSE,
    phenotype_used = FALSE,
    P2b_used = FALSE
  ),
  provenance = list(
    mapped_cache_md5 = unname(tools::md5sum(cache_path)),
    ms1_table_md5 = unname(tools::md5sum(ms1_path)),
    output_msp_md5 = unname(tools::md5sum(output_msp)),
    script_md5 = NA_character_
  ),
  claim_limit = "Identifier-preserving execution bridge only; no annotation or performance result."
)

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) == 1) {
  script_path <- sub("^--file=", "", script_arg)
  if (file.exists(script_path)) report$provenance$script_md5 <- unname(tools::md5sum(script_path))
}
jsonlite::write_json(report, report_json, pretty = TRUE, auto_unbox = TRUE)
cat(jsonlite::toJSON(report, pretty = TRUE, auto_unbox = TRUE), "\n")
