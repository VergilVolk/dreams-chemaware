#!/usr/bin/env Rscript

# Reconstruct the MS1 feature table required by the public MetDNA2/KGMN
# workflow from the 12 public OEP003284 mzXML files in one polarity.
# The exact private XCMS parameter file used by the paper is unavailable, so
# this stage is a sensitivity-only, phenotype-blind reconstruction.  The formal
# path uses the published Raw_peak_table sheets and must not be replaced by this
# reconstruction.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop(
    "usage: preprocess_kgmn_oep003284_xcms.R ",
    "<raw_root> <polarity> <output_dir> <report_json>"
  )
}

raw_root <- normalizePath(args[[1]], mustWork = TRUE)
polarity <- tolower(args[[2]])
output_dir <- args[[3]]
report_json <- args[[4]]
if (!(polarity %in% c("positive", "negative"))) stop("polarity must be positive or negative")

required_packages <- c("xcms", "MSnbase", "Biobase", "jsonlite")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages) > 0) {
  stop(paste("missing R packages:", paste(missing_packages, collapse = ",")))
}
if (dir.exists(output_dir) && length(list.files(output_dir, all.files = TRUE, no.. = TRUE)) > 0) {
  stop(paste("refusing to overwrite non-empty XCMS output:", output_dir))
}
if (file.exists(report_json)) stop(paste("refusing to overwrite report:", report_json))
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(report_json), recursive = TRUE, showWarnings = FALSE)

token <- if (polarity == "positive") "pos" else "neg"
pattern <- paste0("^g[124]_46std_", token, "_[1-4]\\.mzXML$")
raw_files <- sort(list.files(raw_root, pattern = pattern, full.names = TRUE, ignore.case = FALSE))
if (length(raw_files) != 12) {
  stop(paste("expected exactly 12 public", polarity, "mzXML files, found", length(raw_files)))
}
expected <- sort(unlist(lapply(c(1, 2, 4), function(group) {
  paste0("g", group, "_46std_", token, "_", 1:4, ".mzXML")
})))
if (!identical(basename(raw_files), expected)) stop("public mzXML layout is incomplete or unexpected")
if (any(file.info(raw_files)$size <= 0)) stop("one or more public mzXML files are empty")

sample_names <- tools::file_path_sans_ext(basename(raw_files))
sample_groups <- sub("^(g[124])_.*$", "\\1", sample_names)

centwave <- xcms::CentWaveParam(
  ppm = 15,
  peakwidth = c(5, 60),
  snthresh = 10,
  prefilter = c(3, 100),
  mzCenterFun = "wMean",
  integrate = 1,
  mzdiff = -0.001,
  noise = 0
)
obiwarp <- xcms::ObiwarpParam(binSize = 0.6)
peak_density <- xcms::PeakDensityParam(
  sampleGroups = rep(1L, length(raw_files)),
  bw = 5,
  minFraction = 0.25,
  minSamples = 2,
  binSize = 0.01
)

message("[XCMS] loading ", length(raw_files), " ", polarity, " mzXML files")
raw_data <- MSnbase::readMSData(files = raw_files, mode = "onDisk")
Biobase::sampleNames(raw_data) <- sample_names
message("[XCMS] detecting chromatographic peaks")
xdata <- xcms::findChromPeaks(raw_data, param = centwave)
detected_peaks <- nrow(xcms::chromPeaks(xdata))
if (!is.finite(detected_peaks) || detected_peaks < 100) stop("XCMS detected implausibly few chromatographic peaks")
message("[XCMS] retention-time alignment")
xdata <- xcms::adjustRtime(xdata, param = obiwarp)
message("[XCMS] cross-sample feature grouping")
xdata <- xcms::groupChromPeaks(xdata, param = peak_density)
message("[XCMS] filling missing chromatographic peaks")
xdata <- xcms::fillChromPeaks(xdata)

definitions <- as.data.frame(xcms::featureDefinitions(xdata))
values <- as.matrix(xcms::featureValues(xdata, value = "into", method = "maxint"))
if (nrow(definitions) != nrow(values) || ncol(values) != length(raw_files)) {
  stop("XCMS feature definitions/value matrix dimensions disagree")
}
required_definition_columns <- c("mzmed", "rtmed")
if (!all(required_definition_columns %in% colnames(definitions))) {
  stop("XCMS feature definitions lack mzmed/rtmed")
}
storage.mode(values) <- "double"
values[!is.finite(values)] <- 0
values[values < 0] <- 0
detected_samples <- rowSums(values > 0)
keep <- is.finite(definitions$mzmed) & definitions$mzmed > 0 &
  is.finite(definitions$rtmed) & definitions$rtmed >= 0 & detected_samples >= 2
if (sum(keep) < 100) stop("XCMS filtering retained implausibly few MS1 features")
definitions <- definitions[keep, , drop = FALSE]
values <- values[keep, , drop = FALSE]
detected_samples <- detected_samples[keep]

feature_base <- paste0("M", round(definitions$mzmed), "T", round(definitions$rtmed))
feature_names <- make.unique(feature_base, sep = "_")
if (anyDuplicated(feature_names) || any(!nzchar(feature_names))) stop("failed to construct unique feature names")
colnames(values) <- sample_names
peak_table <- data.frame(
  name = feature_names,
  mz = as.numeric(definitions$mzmed),
  rt = as.numeric(definitions$rtmed),
  values,
  check.names = FALSE
)
sample_info <- data.frame(
  sample.name = sample_names,
  group = sample_groups,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

peak_path <- file.path(output_dir, "data.csv")
sample_path <- file.path(output_dir, "sample.info.csv")
utils::write.csv(peak_table, peak_path, row.names = FALSE, quote = FALSE)
utils::write.csv(sample_info, sample_path, row.names = FALSE, quote = FALSE)

raw_destinations <- file.path(output_dir, basename(raw_files))
if (!all(file.symlink(raw_files, raw_destinations))) stop("failed to link public mzXML files into KGMN input")
if (length(list.files(output_dir, pattern = "\\.mzXML$", full.names = TRUE)) != 12) {
  stop("KGMN input directory does not contain exactly 12 linked mzXML files")
}

report <- list(
  status = "kgmn_oep003284_xcms_reconstruction_complete",
  formal = TRUE,
  polarity = polarity,
  raw_files = length(raw_files),
  sample_groups = as.list(table(sample_groups)),
  chromatographic_peaks = detected_peaks,
  retained_features = nrow(peak_table),
  features_detected_in_all_samples = sum(detected_samples == length(raw_files)),
  median_detected_samples = as.numeric(stats::median(detected_samples)),
  parameters = list(
    centwave = list(ppm = 15, peakwidth = c(5, 60), snthresh = 10, prefilter = c(3, 100), mzCenterFun = "wMean", integrate = 1, mzdiff = -0.001, noise = 0),
    obiwarp = list(binSize = 0.6),
    peak_density = list(sampleGroups = "single phenotype-blind group", bw = 5, minFraction = 0.25, minSamples = 2, binSize = 0.01),
    feature_value = "integrated area; maxint",
    final_minimum_samples = 2
  ),
  contracts = list(
    phenotype_blind = TRUE,
    truth_labels_used = FALSE,
    hidden_seed_split_used = FALSE,
    same_preprocessing_for_all_downstream_arms = TRUE,
    raw_files_linked_not_copied = TRUE
  ),
  provenance = list(
    raw_md5 = as.list(unname(tools::md5sum(raw_files))),
    data_csv_md5 = unname(tools::md5sum(peak_path)),
    sample_info_md5 = unname(tools::md5sum(sample_path)),
    xcms_version = as.character(utils::packageVersion("xcms")),
    MSnbase_version = as.character(utils::packageVersion("MSnbase"))
  ),
  reproduction_boundary = "The paper reports XCMS preprocessing but does not provide its executable peak-picking parameter file; this frozen reconstruction is shared across arms and is not claimed as byte-identical author preprocessing.",
  claim_limit = "Sensitivity-only phenotype-blind MS1 reconstruction; not the formal author-table path and no annotation result."
)
jsonlite::write_json(report, report_json, pretty = TRUE, auto_unbox = TRUE)
cat(jsonlite::toJSON(report, pretty = TRUE, auto_unbox = TRUE), "\n")
