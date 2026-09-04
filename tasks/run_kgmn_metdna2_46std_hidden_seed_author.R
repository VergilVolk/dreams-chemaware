#!/usr/bin/env Rscript

# Continue one leakage-safe 46STD hidden-seed state through the untouched
# MetDNA2/KGMN recursive, credential, and export stages.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("usage: run_kgmn_metdna2_46std_hidden_seed_author.R <MetDNA2_source> <prepared_state_dir>")
}

source_dir <- normalizePath(args[[1]], mustWork = TRUE)
state_dir <- normalizePath(args[[2]], mustWork = TRUE)
if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite is required")
if (!requireNamespace("MetDNA2", quietly = TRUE)) stop("MetDNA2 is not installed")
if (as.character(utils::packageVersion("MetDNA2")) != "1.2.10") stop("expected MetDNA2 1.2.10")

audit_path <- file.path(state_dir, "hidden_seed_state_audit.json")
if (!file.exists(audit_path)) stop("prepared hidden-seed state audit is missing")
audit <- jsonlite::fromJSON(audit_path, simplifyVector = FALSE)
if (!identical(audit$status, "kgmn_hidden_seed_initial_state_prepared")) stop("hidden-seed state status mismatch")
if (!identical(as.integer(audit$hidden_identity_leakage), 0L)) stop("hidden-seed state reports identity leakage")
polarity <- as.character(audit$polarity)
if (!(polarity %in% c("positive", "negative"))) stop("hidden-seed state polarity is invalid")

for (path in c("02_result_MRN_annotation", "03_annotation_credential", "00_annotation_table")) {
  candidate <- file.path(state_dir, path)
  if (dir.exists(candidate) && length(list.files(candidate, all.files = TRUE, no.. = TRUE)) > 0) {
    stop(paste("refusing to reuse a state with recursive/final output:", candidate))
  }
}

source_commit <- trimws(system2("git", c("-C", source_dir, "rev-parse", "HEAD"), stdout = TRUE, stderr = TRUE))
if (length(source_commit) != 1 || !grepl("^[0-9a-f]{40}$", source_commit)) stop("cannot resolve MetDNA2 commit")
source_changes <- system2("git", c("-C", source_dir, "status", "--porcelain"), stdout = TRUE, stderr = TRUE)
if (length(source_changes) > 0) stop("author hidden-seed arm requires a clean MetDNA2 source checkout")

ms2_files <- list.files(state_dir, pattern = "\\.(mgf|msp|mzXML|cef)$", full.names = FALSE, ignore.case = TRUE)
if (length(ms2_files) < 1) stop("expected at least one staged MS2 file")
extensions <- unique(tolower(tools::file_ext(ms2_files)))
if (length(extensions) != 1) stop("staged MS2 files must use one homogeneous format")
extension <- extensions[[1]]
ms2_type <- switch(extension, mgf = "mgf", msp = "msp", mzxml = "mzxml", cef = "cef", stop("unsupported MS2 type"))
for (name in c("data.csv", "sample.info.csv")) {
  if (!file.exists(file.path(state_dir, name))) stop(paste("prepared state lacks", name))
}

genform_source <- file.path(source_dir, "inst", "extdata", "GenForm")
if (!file.exists(genform_source) || file.info(genform_source)$size <= 0) stop("bundled GenForm is missing")
genform_magic <- as.integer(readBin(genform_source, what = "raw", n = 4))
if (!identical(genform_magic, c(127L, 69L, 76L, 70L))) stop("bundled GenForm is not ELF")
genform_dir <- file.path(state_dir, "_runtime_genform")
dir.create(genform_dir, recursive = TRUE, showWarnings = FALSE)
genform_runtime <- file.path(genform_dir, "GenForm")
if (!file.copy(genform_source, genform_runtime, overwrite = FALSE)) stop("failed to stage GenForm")
Sys.chmod(genform_runtime, mode = "0755")
if (.Platform$OS.type != "unix" || file.access(genform_runtime, mode = 1) != 0) {
  stop("46STD hidden-seed author arm requires executable Linux GenForm")
}

parameters <- list(
  status = "kgmn_46std_hidden_seed_author_parameters_frozen",
  package_version = as.character(utils::packageVersion("MetDNA2")),
  source_commit = source_commit,
  state_audit_md5 = unname(tools::md5sum(audit_path)),
  repeat = as.integer(audit$repeat),
  polarity = polarity,
  initial_seed_annotation = FALSE,
  seed_source = "pre-registered whitelist-filtered initial state",
  test_evaluation = "46STD",
  metdna_version = "version2",
  instrument = "SciexTripleTOF",
  column = "hilic",
  library = "zhumetlib_qtof",
  collision_energy = "30",
  ms2_input_files = length(ms2_files),
  lc_method = "Amide23min",
  extension_step = "2",
  recursive_similarity = "author_dp",
  recursive_threshold = 0.5,
  credential = TRUE,
  credential_peak_group_filter = TRUE,
  credential_formula_filter = TRUE,
  rt_calibration = FALSE
)
dput(parameters, file = file.path(state_dir, "hidden_seed_author_parameters.R"))

MetDNA2::MetDNA2(
  ms1_file = "data.csv",
  ms2_file = ms2_files[[1]],
  sample_info_file = "sample.info.csv",
  metdna_version = "version2",
  ms2_type = ms2_type,
  path = state_dir,
  thread = 8,
  is_check_data = TRUE,
  is_anno_initial_seed = FALSE,
  is_anno_mrn = TRUE,
  is_credential = TRUE,
  is_bio_interpret = FALSE,
  is_exported_report = FALSE,
  is_cred_pg_filter = TRUE,
  is_cred_formula_filter = TRUE,
  is_rm_intermediate_data = FALSE,
  lib = "zhumetlib_qtof",
  polarity = polarity,
  instrument = "SciexTripleTOF",
  column = "hilic",
  ce = "30",
  method_lc = "Amide23min",
  is_rt_calibration = FALSE,
  direction = "reverse",
  extension_step = "2",
  dp_tol = 0.5,
  max_step = 3,
  score_cutoff = 0,
  seed_neighbor_match_plot = FALSE,
  candidate_num = 5,
  scoring_approach_recursive = "dp",
  matched_frag_cutoff = 1,
  whether_link_frag = FALSE,
  dir_GenForm = genform_dir,
  is_pred_formula_all = FALSE,
  platform = "linux",
  is_plot_pseudo_MS1 = FALSE,
  test_evaluation = "46STD"
)

required_outputs <- c(
  file.path(state_dir, "00_annotation_table", "table1_identification.csv"),
  file.path(state_dir, "00_annotation_table", "table3_identification_pair.csv"),
  file.path(state_dir, "03_annotation_credential", "annotation_initial.csv")
)
missing <- required_outputs[!file.exists(required_outputs) | file.info(required_outputs)$size <= 0]
if (length(missing) > 0) stop(paste("author hidden-seed run lacks outputs:", paste(missing, collapse = ",")))

completion <- list(
  status = "kgmn_46std_hidden_seed_author_complete",
  formal = TRUE,
  repeat = as.integer(audit$repeat),
  polarity = polarity,
  source_commit = source_commit,
  outputs = lapply(required_outputs, function(path) list(path = path, md5 = unname(tools::md5sum(path)))),
  contracts = list(
    author_source_unmodified = TRUE,
    initial_seed_rerun = FALSE,
    seed_whitelist_audit_required = TRUE,
    hidden_identity_available_only_as_recursive_candidate = TRUE,
    dreams_edge_used = FALSE
  ),
  claim_limit = "One author KGMN hidden-seed arm; cross-repeat evaluation is required for performance."
)
jsonlite::write_json(completion, file.path(state_dir, "hidden_seed_author_completion.json"), pretty = TRUE, auto_unbox = TRUE)
cat(jsonlite::toJSON(completion, pretty = TRUE, auto_unbox = TRUE), "\n")
