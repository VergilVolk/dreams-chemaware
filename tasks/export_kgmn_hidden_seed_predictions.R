#!/usr/bin/env Rscript

# Export one hidden-seed arm as a candidate-identity long table.  Final scores
# come from table_identification; propagation round comes from the separately
# saved list_identification because the public CSV drops this field.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
  stop(
    "usage: export_kgmn_hidden_seed_predictions.R ",
    "<run_dir> <contract_dir> <repeat> <positive|negative> <output_csv>"
  )
}
run_dir <- normalizePath(args[[1]], mustWork = TRUE)
contract_dir <- normalizePath(args[[2]], mustWork = TRUE)
repeat_id <- suppressWarnings(as.integer(args[[3]]))
polarity <- args[[4]]
output_csv <- args[[5]]
if (is.na(repeat_id) || repeat_id < 0) stop("invalid repeat")
if (!(polarity %in% c("positive", "negative"))) stop("invalid polarity")
if (file.exists(output_csv)) stop(paste("refusing to overwrite:", output_csv))

split_path <- file.path(contract_dir, "hidden_seed_splits.csv.gz")
universe_path <- file.path(contract_dir, "level1_seed_universe.csv.gz")
if (!file.exists(split_path) || !file.exists(universe_path)) stop("contract files are missing")
splits <- utils::read.csv(gzfile(split_path), stringsAsFactors = FALSE, check.names = FALSE)
universe <- utils::read.csv(gzfile(universe_path), stringsAsFactors = FALSE, check.names = FALSE)
hidden <- unique(substr(trimws(as.character(
  splits$inchikey1[splits$repeat == repeat_id & splits$role == "hidden_validation"]
)), 1, 14))
truth_peaks <- universe[
  universe$polarity == polarity & substr(trimws(as.character(universe$inchikey1)), 1, 14) %in% hidden,
  c("peak_name", "inchikey1"), drop = FALSE
]
truth_peaks$truth_inchikey1 <- substr(trimws(as.character(truth_peaks$inchikey1)), 1, 14)
truth_peaks$peak_name <- trimws(as.character(truth_peaks$peak_name))
truth_peaks <- unique(truth_peaks[, c("peak_name", "truth_inchikey1"), drop = FALSE])
if (nrow(truth_peaks) == 0) stop("no hidden truth peaks for this repeat/polarity")

list_path <- file.path(run_dir, "00_annotation_table", "00_intermediate_data", "list_identification")
table_path <- file.path(run_dir, "00_annotation_table", "00_intermediate_data", "table_identification")
if (!file.exists(list_path) || !file.exists(table_path)) stop("MetDNA2 internal result caches are missing")
load(list_path)
if (!exists("list_identification") || !is.list(list_identification)) stop("invalid list_identification cache")
load(table_path)
if (!exists("table_identification") || !is.data.frame(table_identification)) stop("invalid table_identification cache")

nonempty_identification <- Filter(function(x) is.data.frame(x) && nrow(x) > 0, list_identification)
raw_paths <- if (length(nonempty_identification) > 0) {
  do.call(rbind, nonempty_identification)
} else {
  data.frame(
    peak_name = character(), inchikey = character(), adduct = character(),
    source = character(), round = numeric(), stringsAsFactors = FALSE
  )
}
needed_paths <- c("peak_name", "inchikey", "adduct", "source", "round")
if (!all(needed_paths %in% names(raw_paths))) stop("list_identification lacks propagation fields")
paths <- raw_paths[, needed_paths, drop = FALSE]
paths$candidate_inchikey1 <- substr(trimws(as.character(paths$inchikey)), 1, 14)
paths$propagation_depth <- ifelse(paths$source == "initial_seed", 0, as.numeric(paths$round))
paths$propagation_depth[is.na(paths$propagation_depth)] <- 0
if (nrow(paths) > 0) {
  paths <- aggregate(
    propagation_depth ~ peak_name + candidate_inchikey1 + adduct,
    data = paths, FUN = min
  )
}

needed_scores <- c("peak_name", "inchikey", "adduct", "total_score")
if (!all(needed_scores %in% names(table_identification))) stop("table_identification lacks score fields")
scores <- table_identification[, needed_scores, drop = FALSE]
scores$candidate_inchikey1 <- substr(trimws(as.character(scores$inchikey)), 1, 14)
scores$candidate_score <- as.numeric(scores$total_score)
scores <- scores[!is.na(scores$candidate_score) & scores$candidate_inchikey1 != "", , drop = FALSE]
if (nrow(scores) > 0) {
  scores <- merge(scores, paths, by = c("peak_name", "candidate_inchikey1", "adduct"), all.x = TRUE)
  if (any(is.na(scores$propagation_depth))) stop("failed to reconcile final score with propagation depth")
  scores <- merge(truth_peaks, scores, by = "peak_name", all = FALSE)
} else {
  scores <- data.frame(
    peak_name = character(), truth_inchikey1 = character(), candidate_inchikey1 = character(),
    candidate_score = numeric(), propagation_depth = integer(), adduct = character(),
    stringsAsFactors = FALSE
  )
}

result <- data.frame(
  repeat = repeat_id,
  truth_inchikey1 = scores$truth_inchikey1,
  candidate_inchikey1 = scores$candidate_inchikey1,
  candidate_score = scores$candidate_score,
  propagation_depth = as.integer(scores$propagation_depth),
  polarity = polarity,
  peak_name = scores$peak_name,
  adduct = scores$adduct,
  stringsAsFactors = FALSE
)
result <- result[order(
  result$truth_inchikey1, -result$candidate_score,
  result$propagation_depth, result$candidate_inchikey1
), , drop = FALSE]
dir.create(dirname(output_csv), recursive = TRUE, showWarnings = FALSE)
utils::write.csv(result, output_csv, row.names = FALSE, quote = TRUE)
cat(sprintf(
  "[export] repeat=%d polarity=%s hidden_truths=%d rows=%d recovered_truths=%d\n",
  repeat_id, polarity, length(unique(truth_peaks$truth_inchikey1)), nrow(result),
  length(intersect(unique(result$candidate_inchikey1), unique(result$truth_inchikey1)))
))
