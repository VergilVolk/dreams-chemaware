#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("usage: run_kgmn_metdna2_200std_baseline.R <MetDNA2_source> <output_dir>")
}

source_dir <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- args[[2]]

if (dir.exists(output_dir) && length(list.files(output_dir, all.files = TRUE, no.. = TRUE)) > 0) {
  stop(paste("refusing to overwrite non-empty baseline directory:", output_dir))
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

required_package <- "MetDNA2"
if (!requireNamespace(required_package, quietly = TRUE)) {
  stop(
    paste0(
      "MetDNA2 is not installed in this R runtime. Install the frozen source at ",
      source_dir,
      " into an isolated R library before running the author baseline."
    )
  )
}

package_version <- as.character(utils::packageVersion(required_package))
if (package_version != "1.2.10") {
  stop(paste("expected MetDNA2 1.2.10, found", package_version))
}

source_commit <- trimws(system2("git", c("-C", source_dir, "rev-parse", "HEAD"), stdout = TRUE, stderr = TRUE))
if (length(source_commit) != 1 || !grepl("^[0-9a-f]{40}$", source_commit)) {
  stop("could not resolve one immutable MetDNA2 source commit")
}
source_changes <- system2("git", c("-C", source_dir, "status", "--porcelain"), stdout = TRUE, stderr = TRUE)
if (length(source_changes) > 0) {
  stop("MetDNA2 author baseline requires a clean source checkout")
}
writeLines(source_commit, file.path(output_dir, "source_commit.txt"))

extdata <- file.path(source_dir, "inst", "extdata")
input_files <- c(
  data = "peak_table_200STD_neg_200805.csv",
  spectra = "spectra_200STD_neg_200805.msp",
  annotated_features = "peak_table_annotated_200STD_neg_200805.csv",
  truth = "annotation_initial.csv"
)
for (filename in unname(input_files)) {
  path <- file.path(extdata, filename)
  if (!file.exists(path) || file.info(path)$size <= 0) {
    stop(paste("missing frozen MetDNA2 demo input:", path))
  }
}

copy_contract <- c(
  data = file.copy(file.path(extdata, input_files[["data"]]), file.path(output_dir, "data.csv"), overwrite = FALSE),
  spectra = file.copy(file.path(extdata, input_files[["spectra"]]), file.path(output_dir, "spectra.msp"), overwrite = FALSE),
  annotated_features = file.copy(
    file.path(extdata, input_files[["annotated_features"]]),
    file.path(output_dir, "peak_table_annotated_200STD_neg_200805.csv"),
    overwrite = FALSE
  ),
  truth = file.copy(
    file.path(extdata, input_files[["truth"]]),
    file.path(output_dir, "annotation_initial.csv"),
    overwrite = FALSE
  )
)
if (!all(copy_contract)) {
  stop(paste("failed to freeze one or more 200STD input files:", paste(names(copy_contract)[!copy_contract], collapse = ", ")))
}

peak_table <- utils::read.csv(file.path(output_dir, "data.csv"), check.names = FALSE)
if (!identical(colnames(peak_table)[1:3], c("name", "mz", "rt"))) {
  stop("frozen 200STD peak table does not start with name,mz,rt")
}
sample_names <- colnames(peak_table)[-(1:3)]
if (length(sample_names) != 4 || anyDuplicated(sample_names)) {
  stop("expected four unique 200STD sample columns")
}
sample_info <- data.frame(sample.name = sample_names, group = rep("200STD", length(sample_names)))
utils::write.csv(sample_info, file.path(output_dir, "sample.info.csv"), row.names = FALSE, quote = FALSE)

# MetDNA2's full annotation-credential stage calls GenForm even when formula
# prediction is restricted to credential candidates.  Stage the exact bundled
# Linux binary inside this immutable run directory instead of relying on an
# undocumented machine-global path.
genform_source <- file.path(extdata, "GenForm")
if (!file.exists(genform_source) || file.info(genform_source)$size <= 0) {
  stop(paste("missing bundled GenForm executable:", genform_source))
}
genform_magic <- as.integer(readBin(genform_source, what = "raw", n = 4))
if (!identical(genform_magic, c(127L, 69L, 76L, 70L))) {
  stop("bundled GenForm is not an ELF executable")
}
genform_dir <- file.path(output_dir, "_runtime_genform")
dir.create(genform_dir, recursive = TRUE, showWarnings = FALSE)
genform_runtime <- file.path(genform_dir, "GenForm")
if (!file.copy(genform_source, genform_runtime, overwrite = FALSE)) {
  stop("failed to stage bundled GenForm executable")
}
Sys.chmod(genform_runtime, mode = "0755")
if (.Platform$OS.type != "unix" || file.access(genform_runtime, mode = 1) != 0) {
  stop("frozen 200STD credential baseline requires an executable Linux GenForm binary")
}

parameters <- list(
  package = "MetDNA2",
  package_version = package_version,
  source_commit = source_commit,
  source_dir = source_dir,
  ms1_file = "data.csv",
  ms2_file = "spectra.msp",
  sample_info_file = "sample.info.csv",
  truth_file = "annotation_initial.csv",
  metdna_version = "version2",
  ms2_type = "msp",
  polarity = "negative",
  instrument = "SciexTripleTOF",
  column = "hilic",
  library = "zhumetlib_qtof",
  collision_energy = "30",
  lc_method = "Amide23min",
  extension_step = "2",
  recursive_similarity = "dp",
  test_evaluation = "200STD",
  credential = TRUE,
  credential_peak_group_filter = TRUE,
  credential_formula_filter = FALSE,
  predict_formula_for_all = FALSE,
  genform_directory = genform_dir,
  rt_calibration = FALSE,
  biology_interpretation = FALSE
)
dput(parameters, file = file.path(output_dir, "frozen_parameters.R"))

MetDNA2::MetDNA2(
  ms1_file = "data.csv",
  ms2_file = "spectra.msp",
  sample_info_file = "sample.info.csv",
  metdna_version = "version2",
  ms2_type = "msp",
  path = output_dir,
  thread = 8,
  is_check_data = TRUE,
  is_anno_initial_seed = TRUE,
  is_anno_mrn = TRUE,
  is_credential = TRUE,
  is_bio_interpret = FALSE,
  is_exported_report = FALSE,
  is_cred_pg_filter = TRUE,
  is_cred_formula_filter = FALSE,
  is_rm_intermediate_data = FALSE,
  lib = "zhumetlib_qtof",
  polarity = "negative",
  instrument = "SciexTripleTOF",
  column = "hilic",
  ce = "30",
  method_lc = "Amide23min",
  is_rt_calibration = FALSE,
  dp_cutoff = 0.8,
  direction = "reverse",
  is_plot_ms2 = FALSE,
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
  test_evaluation = "200STD"
)

cat("[kgmn-metdna2] author 200STD baseline completed\n")
