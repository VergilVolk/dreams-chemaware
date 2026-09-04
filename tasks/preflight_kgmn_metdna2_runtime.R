#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("usage: preflight_kgmn_metdna2_runtime.R <MetDNA2_source>")
source_dir <- normalizePath(args[[1]], mustWork = TRUE)
description_path <- file.path(source_dir, "DESCRIPTION")
if (!file.exists(description_path)) stop("MetDNA2 DESCRIPTION is missing")

description <- read.dcf(description_path)
if (description[1, "Package"] != "MetDNA2") stop("source package is not MetDNA2")
if (description[1, "Version"] != "1.2.10") stop("expected frozen MetDNA2 version 1.2.10")

dependency_fields <- intersect(c("Depends", "Imports", "LinkingTo"), colnames(description))
dependency_text <- paste(description[1, dependency_fields], collapse = ",")
dependencies <- trimws(unlist(strsplit(dependency_text, ",")))
dependencies <- sub("\\s*\\(.*$", "", dependencies)
dependencies <- sort(unique(dependencies[nzchar(dependencies) & dependencies != "R"]))
available <- vapply(dependencies, requireNamespace, logical(1), quietly = TRUE)
missing <- dependencies[!available]

cat("MetDNA2 source:", source_dir, "\n")
cat("MetDNA2 version:", description[1, "Version"], "\n")
cat("Dependencies checked:", length(dependencies), "\n")
if (length(missing) > 0) {
  cat("Missing R dependencies:", paste(missing, collapse = ", "), "\n")
  quit(status = 2)
}

# The public MetDNA2 repository intentionally omits the proprietary Zhu MS/MS
# library objects.  Merely importing MetLib is therefore not a sufficient
# runtime check: the recursive workflow can start successfully and fail much
# later when initial-seed annotation requests zhuMetlib.  Verify the exact
# object and API used by MetDNA2 before allocating a multi-hour job.
metlib_namespace <- asNamespace("MetLib")
if (!exists("loadLibData", envir = metlib_namespace, inherits = FALSE)) {
  stop("MetLib is installed but does not export the loadLibData runtime required by MetDNA2")
}
metlib_data <- utils::data(package = "MetLib")$results
if (is.null(metlib_data) || nrow(metlib_data) == 0) {
  stop("MetLib is installed but exposes no packaged data objects")
}
metlib_objects <- unique(as.character(metlib_data[, "Item"]))
if (!("zhuMetlib" %in% metlib_objects)) {
  stop(
    paste0(
      "MetLib does not contain the zhuMetlib object required by the frozen ",
      "SciexTripleTOF 200STD protocol. Available objects: ",
      paste(sort(metlib_objects), collapse = ", ")
    )
  )
}
library_env <- new.env(parent = emptyenv())
suppressWarnings(utils::data("zhuMetlib", package = "MetLib", envir = library_env))
if (!exists("zhuMetlib", envir = library_env, inherits = FALSE)) {
  stop("MetLib lists zhuMetlib but the object cannot be loaded")
}
zhu_library <- get("zhuMetlib", envir = library_env, inherits = FALSE)
if (!is.list(zhu_library) || !all(c("meta", "spectra") %in% names(zhu_library))) {
  stop("MetLib zhuMetlib has an unexpected schema; expected list entries meta and spectra")
}
if (length(zhu_library$spectra) == 0 || nrow(zhu_library$meta$compound) == 0) {
  stop("MetLib zhuMetlib is present but empty")
}

required_source_files <- c(
  "inst/extdata/peak_table_200STD_neg_200805.csv",
  "inst/extdata/peak_table_annotated_200STD_neg_200805.csv",
  "inst/extdata/annotation_initial.csv",
  "inst/extdata/spectra_200STD_neg_200805.msp",
  "inst/extdata/GenForm",
  "data/reaction_pair_network.rda",
  "data/md_mrn_emrn.rda",
  "data/lib_adduct_nl.rda"
)
missing_source_files <- required_source_files[
  !file.exists(file.path(source_dir, required_source_files)) |
    file.info(file.path(source_dir, required_source_files))$size <= 0
]
if (length(missing_source_files) > 0) {
  stop(paste("MetDNA2 frozen source assets are incomplete:", paste(missing_source_files, collapse = ", ")))
}

genform_path <- file.path(source_dir, "inst", "extdata", "GenForm")
genform_magic <- as.integer(readBin(genform_path, what = "raw", n = 4))
if (!identical(genform_magic, c(127L, 69L, 76L, 70L))) {
  stop("MetDNA2 bundled GenForm asset is not an ELF executable")
}
if (.Platform$OS.type != "unix") {
  stop("the frozen full-credential 200STD baseline requires a Unix/Linux R runtime")
}

cat("[kgmn-runtime-preflight] PASS\n")
cat("MetLib zhuMetlib compounds:", nrow(zhu_library$meta$compound), "\n")
cat("MetLib zhuMetlib spectra:", length(zhu_library$spectra), "\n")
cat("Bundled GenForm ELF bytes:", file.info(genform_path)$size, "\n")
