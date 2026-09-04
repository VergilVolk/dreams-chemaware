#!/usr/bin/env Rscript

# Build the full author initial-seed state once per polarity.  This is an
# execution cache for later whitelist filtering; it is not itself a hidden-seed
# result.  Recursive annotation and credential are intentionally disabled.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop(
    "usage: run_kgmn_metdna2_46std_initial_seed.R ",
    "<MetDNA2_source> <input_polarity_dir> <output_dir>"
  )
}

source_dir <- normalizePath(args[[1]], mustWork = TRUE)
input_dir <- normalizePath(args[[2]], mustWork = TRUE)
output_dir <- args[[3]]

if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite is required")
if (!requireNamespace("MetDNA2", quietly = TRUE)) stop("MetDNA2 is not installed")
if (as.character(utils::packageVersion("MetDNA2")) != "1.2.10") {
  stop("expected MetDNA2 1.2.10")
}
if (dir.exists(output_dir) && length(list.files(output_dir, all.files = TRUE, no.. = TRUE)) > 0) {
  stop(paste("refusing to overwrite non-empty initial-seed directory:", output_dir))
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

source_commit <- trimws(system2("git", c("-C", source_dir, "rev-parse", "HEAD"), stdout = TRUE, stderr = TRUE))
if (length(source_commit) != 1 || !grepl("^[0-9a-f]{40}$", source_commit)) {
  stop("cannot resolve MetDNA2 source commit")
}
source_changes <- system2("git", c("-C", source_dir, "status", "--porcelain"), stdout = TRUE, stderr = TRUE)
if (length(source_changes) > 0) stop("initial-seed replay requires a clean MetDNA2 checkout")

ms1_candidates <- list.files(input_dir, full.names = TRUE)
ms1_candidates <- ms1_candidates[tolower(basename(ms1_candidates)) %in% c("data.csv", "peak_table.csv")]
sample_candidates <- list.files(input_dir, full.names = TRUE)
sample_candidates <- sample_candidates[tolower(basename(sample_candidates)) %in% c("sample.info.csv", "sample_info.csv")]
if (length(ms1_candidates) != 1) stop("expected one data.csv/peak_table.csv")
if (length(sample_candidates) != 1) stop("expected one sample.info.csv/sample_info.csv")
ms2_files <- list.files(
  input_dir,
  pattern = "\\.(mgf|msp|mzXML|cef)$",
  full.names = TRUE,
  ignore.case = TRUE
)
if (length(ms2_files) < 1) stop("expected at least one MetDNA2-supported MS2 file")
ms2_extensions <- unique(tolower(tools::file_ext(ms2_files)))
if (length(ms2_extensions) != 1) {
  stop(paste("mixed MS2 formats are forbidden:", paste(ms2_extensions, collapse = ",")))
}
ms2_extension <- ms2_extensions[[1]]
ms2_type <- switch(ms2_extension, mgf = "mgf", msp = "msp", mzxml = "mzxml", cef = "cef", stop("unsupported MS2 type"))

peak_table <- utils::read.csv(ms1_candidates[[1]], check.names = FALSE)
if (!identical(tolower(colnames(peak_table)[1:3]), c("name", "mz", "rt"))) {
  stop("MS1 table must start with name,mz,rt")
}
forbidden <- intersect(
  tolower(colnames(peak_table)),
  c("annotation", "candidate", "compound", "compound_id", "inchikey", "inchikey1", "truth", "validation_standard")
)
if (length(forbidden) > 0) stop(paste("truth-like MS1 columns are forbidden:", paste(forbidden, collapse = ",")))
sample_info <- utils::read.csv(sample_candidates[[1]], check.names = FALSE, stringsAsFactors = FALSE)
if (!identical(tolower(colnames(sample_info)[1:2]), c("sample.name", "group"))) {
  stop("sample metadata must start with sample.name,group")
}
if (!setequal(as.character(sample_info[[1]]), colnames(peak_table)[-(1:3)])) {
  stop("MS1 abundance columns do not match sample metadata")
}

ms2_destinations <- file.path(output_dir, basename(ms2_files))
if (anyDuplicated(ms2_destinations)) stop("MS2 filenames collide while staging")
copy_contract <- c(
  ms1 = file.copy(ms1_candidates[[1]], file.path(output_dir, "data.csv"), overwrite = FALSE),
  sample = file.copy(sample_candidates[[1]], file.path(output_dir, "sample.info.csv"), overwrite = FALSE),
  ms2 = file.symlink(normalizePath(ms2_files, mustWork = TRUE), ms2_destinations)
)
if (!all(copy_contract)) stop("failed to stage one or more 46STD inputs")

polarity_name <- tolower(basename(input_dir))
if (polarity_name %in% c("positive", "pos", "pos_hilic", "hilic_pos")) {
  polarity <- "positive"
} else if (polarity_name %in% c("negative", "neg", "neg_hilic", "hilic_neg")) {
  polarity <- "negative"
} else {
  stop(paste("cannot infer polarity from input directory name:", basename(input_dir)))
}

parameters <- list(
  status = "kgmn_46std_full_initial_seed_parameters_frozen",
  package_version = as.character(utils::packageVersion("MetDNA2")),
  source_commit = source_commit,
  polarity = polarity,
  ms2_type = ms2_type,
  ms2_files = length(ms2_files),
  initial_seed_annotation = TRUE,
  recursive_annotation = FALSE,
  credential = FALSE,
  test_evaluation = "46STD",
  instrument = "SciexTripleTOF",
  column = "hilic",
  library = "zhumetlib_qtof",
  collision_energy = "30",
  lc_method = "Amide23min",
  rt_calibration = FALSE
)
dput(parameters, file = file.path(output_dir, "initial_seed_parameters.R"))

MetDNA2::MetDNA2(
  ms1_file = "data.csv",
  ms2_file = basename(ms2_files[[1]]),
  sample_info_file = "sample.info.csv",
  metdna_version = "version2",
  ms2_type = ms2_type,
  path = output_dir,
  thread = 8,
  is_check_data = TRUE,
  is_anno_initial_seed = TRUE,
  is_anno_mrn = FALSE,
  is_credential = FALSE,
  is_bio_interpret = FALSE,
  is_exported_report = FALSE,
  is_rm_intermediate_data = FALSE,
  lib = "zhumetlib_qtof",
  polarity = polarity,
  instrument = "SciexTripleTOF",
  column = "hilic",
  ce = "30",
  method_lc = "Amide23min",
  is_rt_calibration = FALSE,
  dp_cutoff = 0.8,
  direction = "reverse",
  is_plot_ms2 = FALSE,
  is_pred_formula_all = FALSE,
  test_evaluation = "46STD"
)

required <- c(
  file.path(output_dir, "01_result_initial_seed_annotation", "ms2_match_annotation_result.csv"),
  file.path(output_dir, "01_result_initial_seed_annotation", "00_intermediate_data", "result_annotation"),
  file.path(output_dir, "01_result_initial_seed_annotation", "00_intermediate_data", "ms2"),
  file.path(output_dir, "01_result_initial_seed_annotation", "00_intermediate_data", "ms1_data")
)
missing <- required[!file.exists(required) | file.info(required)$size <= 0]
if (length(missing) > 0) stop(paste("full initial-seed run lacks required cache:", paste(missing, collapse = ",")))

seed_table <- utils::read.csv(required[[1]], stringsAsFactors = FALSE, check.names = FALSE)
identity_column <- if ("inchikey1" %in% names(seed_table)) "inchikey1" else if ("inchikey" %in% names(seed_table)) "inchikey" else NA_character_
if (is.na(identity_column)) stop("author initial seed table lacks InChIKey identity")
seed_identities <- unique(substr(trimws(as.character(seed_table[[identity_column]])), 1, 14))
seed_identities <- seed_identities[nzchar(seed_identities)]
if (length(seed_identities) == 0) stop("author initial seed run produced zero identities")

completion <- list(
  status = "kgmn_46std_full_initial_seed_complete",
  formal = TRUE,
  polarity = polarity,
  source_commit = source_commit,
  seed_rows = nrow(seed_table),
  seed_identities = length(seed_identities),
  ms2_input_files = length(ms2_files),
  contracts = list(
    recursive_annotation_used = FALSE,
    credential_used = FALSE,
    hidden_seed_selection_used = FALSE,
    purpose = "execution cache to be whitelist-filtered before hidden-seed replay"
  ),
  claim_limit = "Full initial-seed execution cache only; no hidden-seed or annotation-performance result."
)
jsonlite::write_json(completion, file.path(output_dir, "initial_seed_completion.json"), pretty = TRUE, auto_unbox = TRUE)
cat(jsonlite::toJSON(completion, pretty = TRUE, auto_unbox = TRUE), "\n")
