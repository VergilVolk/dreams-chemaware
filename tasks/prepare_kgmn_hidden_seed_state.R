#!/usr/bin/env Rscript

# Create one leakage-safe MetDNA2 initial state for a hidden-seed repeat.
# Hidden identities remain in the MRN and raw feature spectra, but only the
# pre-registered seed whitelist remains in identity-bearing seed objects.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
  stop(
    "usage: prepare_kgmn_hidden_seed_state.R ",
    "<full_initial_seed_run> <contract_dir> <repeat> <positive|negative> <output_dir>"
  )
}

source_run <- normalizePath(args[[1]], mustWork = TRUE)
contract_dir <- normalizePath(args[[2]], mustWork = TRUE)
repeat_id <- suppressWarnings(as.integer(args[[3]]))
polarity <- args[[4]]
output_dir <- args[[5]]
if (is.na(repeat_id) || repeat_id < 0) stop("repeat must be a non-negative integer")
if (!(polarity %in% c("positive", "negative"))) stop("polarity must be positive or negative")
if (dir.exists(output_dir) && length(list.files(output_dir, all.files = TRUE, no.. = TRUE)) > 0) {
  stop(paste("refusing to overwrite non-empty hidden-seed state:", output_dir))
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite is required")
if (!requireNamespace("MetDNA2", quietly = TRUE)) stop("MetDNA2 is required to load author S4 caches")
if (as.character(utils::packageVersion("MetDNA2")) != "1.2.10") stop("expected MetDNA2 1.2.10")

split_path <- file.path(contract_dir, "hidden_seed_splits.csv.gz")
universe_path <- file.path(contract_dir, "level1_seed_universe.csv.gz")
if (!file.exists(split_path) || !file.exists(universe_path)) stop("hidden-seed contract files are missing")
splits <- utils::read.csv(gzfile(split_path), stringsAsFactors = FALSE, check.names = FALSE)
universe <- utils::read.csv(gzfile(universe_path), stringsAsFactors = FALSE, check.names = FALSE)
needed_split <- c("repeat", "inchikey1", "role")
needed_universe <- c("polarity", "inchikey1")
if (!all(needed_split %in% names(splits))) stop("hidden_seed_splits schema mismatch")
if (!all(needed_universe %in% names(universe))) stop("level1_seed_universe schema mismatch")

selected <- unique(as.character(splits$inchikey1[splits$repeat == repeat_id & splits$role == "seed"]))
hidden <- unique(as.character(splits$inchikey1[splits$repeat == repeat_id & splits$role == "hidden_validation"]))
if (length(selected) == 0 || length(hidden) == 0 || length(intersect(selected, hidden)) > 0) {
  stop("invalid selected/hidden identity partition")
}
polarity_universe <- unique(as.character(universe$inchikey1[universe$polarity == polarity]))
selected_polarity <- intersect(selected, polarity_universe)
hidden_polarity <- intersect(hidden, polarity_universe)
if (length(selected_polarity) == 0 || length(hidden_polarity) == 0) {
  stop("repeat has no selected or hidden identities for this polarity")
}

input_dir <- file.path(source_run, "01_result_initial_seed_annotation")
intermediate_dir <- file.path(input_dir, "00_intermediate_data")
seed_csv_path <- file.path(input_dir, "ms2_match_annotation_result.csv")
result_path <- file.path(intermediate_dir, "result_annotation")
required_intermediate <- c("ms2", "ms1_data", "result_annotation")
if (!file.exists(seed_csv_path)) stop("source initial seed CSV is missing")
for (name in required_intermediate) {
  if (!file.exists(file.path(intermediate_dir, name))) stop(paste("source initial cache is missing:", name))
}

# The two execution caches are expected to contain only observed peak/spectrum
# data.  Do not trust that assumption implicitly: recursively reject any exact
# hidden InChIKey block in values, names or S4 slot names before copying them.
contains_hidden_identity <- function(value, hidden_ids, depth = 0L) {
  if (depth > 40L || is.null(value)) return(FALSE)
  check_text <- function(x) {
    text <- trimws(as.character(x))
    any(vapply(hidden_ids, function(id) any(grepl(id, text, fixed = TRUE)), logical(1)))
  }
  value_names <- names(value)
  if (!is.null(value_names) && check_text(value_names)) return(TRUE)
  if (is.character(value) || is.factor(value)) return(check_text(value))
  if (isS4(value)) {
    slots <- methods::slotNames(value)
    if (check_text(slots)) return(TRUE)
    return(any(vapply(slots, function(slot) {
      contains_hidden_identity(methods::slot(value, slot), hidden_ids, depth + 1L)
    }, logical(1))))
  }
  if (is.list(value) || is.data.frame(value)) {
    return(any(vapply(value, contains_hidden_identity, logical(1), hidden_ids = hidden_ids, depth = depth + 1L)))
  }
  FALSE
}

cache_identity_scan <- list()
for (cache_name in c("ms2", "ms1_data")) {
  cache_path <- file.path(intermediate_dir, cache_name)
  cache_env <- new.env(parent = emptyenv())
  loaded_names <- load(cache_path, envir = cache_env)
  if (length(loaded_names) < 1) stop(paste("empty execution cache:", cache_name))
  leaked <- any(vapply(loaded_names, function(object_name) {
    contains_hidden_identity(get(object_name, envir = cache_env, inherits = FALSE), hidden_polarity)
  }, logical(1)))
  if (leaked) stop(paste("hidden identity appears in label-free execution cache:", cache_name))
  cache_identity_scan[[cache_name]] <- list(objects = loaded_names, hidden_identity_matches = 0L)
}

identity_vector <- function(frame, label) {
  if ("inchikey1" %in% names(frame)) return(substr(trimws(as.character(frame$inchikey1)), 1, 14))
  if ("inchikey" %in% names(frame)) return(substr(trimws(as.character(frame$inchikey)), 1, 14))
  if (nrow(frame) == 0) return(character())
  stop(paste(label, "has annotated rows but no inchikey/inchikey1 column"))
}

seed_table <- utils::read.csv(seed_csv_path, stringsAsFactors = FALSE, check.names = FALSE)
seed_ik14 <- identity_vector(seed_table, "seed CSV")
if (any(is.na(seed_ik14) | seed_ik14 == "")) stop("seed CSV contains blank identity labels")
available_seed_identities <- unique(seed_ik14)
missing_selected <- setdiff(selected_polarity, available_seed_identities)
if (length(missing_selected) > 0) {
  stop(paste("pre-registered seeds absent from source initial annotation:", paste(missing_selected, collapse = ",")))
}
keep_csv <- seed_ik14 %in% selected_polarity
filtered_seed_table <- seed_table[keep_csv, , drop = FALSE]
if (nrow(filtered_seed_table) == 0) stop("seed whitelist removed every initial annotation")

load(result_path)
if (!exists("result_annotation") || !is.list(result_annotation)) stop("invalid result_annotation cache")
before_object_rows <- 0L
after_object_rows <- 0L
object_identities <- character()
for (index in seq_along(result_annotation)) {
  object <- result_annotation[[index]]
  if (!("annotation_result" %in% methods::slotNames(object))) {
    stop(paste("result_annotation object lacks annotation_result slot at", index))
  }
  frame <- methods::slot(object, "annotation_result")
  before_object_rows <- before_object_rows + nrow(frame)
  if (nrow(frame) > 0) {
    ids <- identity_vector(frame, paste("result_annotation object", index))
    keep <- ids %in% selected_polarity
    frame <- frame[keep, , drop = FALSE]
    if (nrow(frame) > 0) object_identities <- c(object_identities, identity_vector(frame, "filtered object"))
    methods::slot(object, "annotation_result") <- frame
  }
  after_object_rows <- after_object_rows + nrow(frame)
  result_annotation[[index]] <- object
}

remaining <- unique(c(identity_vector(filtered_seed_table, "filtered seed CSV"), object_identities))
if (length(setdiff(remaining, selected_polarity)) > 0) stop("non-whitelisted identities remain in initial state")
if (length(intersect(remaining, hidden_polarity)) > 0) stop("hidden identity leaked into initial state")
if (length(setdiff(selected_polarity, remaining)) > 0) stop("selected identity was lost from both initial seed objects")

for (name in c("data.csv", "sample.info.csv")) {
  source <- file.path(source_run, name)
  if (!file.exists(source) || !file.copy(source, file.path(output_dir, name), overwrite = FALSE)) {
    stop(paste("failed to stage required input:", name))
  }
}
supported <- sort(list.files(source_run, pattern = "\\.(mgf|msp|mzXML|cef)$", full.names = TRUE, ignore.case = TRUE))
if (length(supported) < 1) stop("prepared source has no supported MS2 input")
supported_extensions <- unique(tolower(tools::file_ext(supported)))
if (length(supported_extensions) != 1) {
  stop(paste("prepared source contains mixed MS2 formats:", paste(supported_extensions, collapse = ",")))
}
supported_destinations <- file.path(output_dir, basename(supported))
if (anyDuplicated(supported_destinations)) stop("MS2 filenames collide while preparing hidden-seed state")
if (!all(file.symlink(normalizePath(supported, mustWork = TRUE), supported_destinations))) {
  stop("failed to stage all MS2 inputs")
}

target_initial <- file.path(output_dir, "01_result_initial_seed_annotation")
target_intermediate <- file.path(target_initial, "00_intermediate_data")
dir.create(target_intermediate, recursive = TRUE, showWarnings = FALSE)
utils::write.csv(filtered_seed_table, file.path(target_initial, "ms2_match_annotation_result.csv"), row.names = FALSE)
save(result_annotation, file = file.path(target_intermediate, "result_annotation"), compress = "xz", version = 2)
for (name in c("ms2", "ms1_data")) {
  if (!file.copy(file.path(intermediate_dir, name), file.path(target_intermediate, name), overwrite = FALSE)) {
    stop(paste("failed to stage label-free execution cache:", name))
  }
}

audit <- list(
  status = "kgmn_hidden_seed_initial_state_prepared",
  formal = TRUE,
  repeat = repeat_id,
  polarity = polarity,
  selected_identities_all_polarities = length(selected),
  hidden_identities_all_polarities = length(hidden),
  selected_identities_this_polarity = length(selected_polarity),
  hidden_identities_this_polarity = length(hidden_polarity),
  source_seed_rows = nrow(seed_table),
  retained_seed_rows = nrow(filtered_seed_table),
  source_object_annotation_rows = before_object_rows,
  retained_object_annotation_rows = after_object_rows,
  staged_ms2_files = length(supported),
  staged_ms2_type = supported_extensions[[1]],
  hidden_identity_leakage = 0,
  label_free_cache_identity_scan = cache_identity_scan,
  identity_bearing_files = c(
    "01_result_initial_seed_annotation/ms2_match_annotation_result.csv",
    "01_result_initial_seed_annotation/00_intermediate_data/result_annotation"
  ),
  label_free_execution_caches = c("ms2", "ms1_data"),
  contract = list(
    seed_policy = "whitelist selected identities; do not merely blacklist hidden identities",
    hidden_features_remain_observed = TRUE,
    hidden_identities_remain_available_in_mrn = TRUE,
    hidden_identity_labels_in_seed_state = FALSE,
    hidden_identity_labels_in_execution_caches = FALSE,
    all_homogeneous_ms2_inputs_preserved = TRUE
  ),
  claim_limit = "Leakage-safe initial state only; no recursive annotation or performance result."
)
jsonlite::write_json(audit, file.path(output_dir, "hidden_seed_state_audit.json"), pretty = TRUE, auto_unbox = TRUE)
cat(jsonlite::toJSON(audit, pretty = TRUE, auto_unbox = TRUE), "\n")
