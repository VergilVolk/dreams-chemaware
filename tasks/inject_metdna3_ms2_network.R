#!/usr/bin/env Rscript

# Validate a frozen, calibrated score file against the exact author edge set and
# create the table_ms2_network.rda consumed by unmodified MrnAnnoAlgo3.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("usage: inject_metdna3_ms2_network.R <workdir> <calibrated_scores.csv> <output_workdir>")
}
source_wd <- normalizePath(args[[1]], mustWork = TRUE)
scores_path <- normalizePath(args[[2]], mustWork = TRUE)
output_wd <- args[[3]]
source_dir <- file.path(source_wd, "02_result_MRN_annotation")
target_dir <- file.path(output_wd, "02_result_MRN_annotation")
edge_path <- file.path(source_dir, "table_ms2_edges.rda")
if (!file.exists(edge_path)) stop(paste("missing", edge_path))
dir.create(target_dir, recursive = TRUE, showWarnings = FALSE)
target_path <- file.path(target_dir, "table_ms2_network.rda")
if (file.exists(target_path)) stop(paste("fail-closed: target exists:", target_path))

edge_env <- new.env(parent = emptyenv())
load(edge_path, envir = edge_env)
edges <- as.data.frame(edge_env$table_ms2_edges, stringsAsFactors = FALSE)
scores <- read.csv(scores_path, stringsAsFactors = FALSE, check.names = FALSE)
required <- c("edge_index", "from", "to", "ms2_score")
if (!all(required %in% colnames(scores))) stop("calibrated score file misses required columns")
if (nrow(edges) != nrow(scores)) stop("edge count mismatch")
if (!identical(as.character(edges$from), as.character(scores$from)) ||
    !identical(as.character(edges$to), as.character(scores$to))) {
  stop("from/to order mismatch; refusing score injection")
}
if (!identical(as.integer(scores$edge_index), seq_len(nrow(edges)) - 1L)) {
  stop("edge_index mismatch")
}
if (any(!is.finite(scores$ms2_score)) || any(scores$ms2_score < 0) ||
    any(scores$ms2_score > 1)) stop("ms2_score must be finite in [0,1]")

table_ms2_network <- data.frame(
  from = as.character(edges$from),
  to = as.character(edges$to),
  ms2_score = as.numeric(scores$ms2_score),
  stringsAsFactors = FALSE
)
save(table_ms2_network, file = target_path)
writeLines(c(
  paste0("source_workdir\t", source_wd),
  paste0("score_file\t", scores_path),
  paste0("edges\t", nrow(table_ms2_network)),
  paste0("score_min\t", min(table_ms2_network$ms2_score)),
  paste0("score_max\t", max(table_ms2_network$ms2_score)),
  "author_ms2_cutoff\t0.5",
  "author_source_commit\t978ae62b33bde75a066032953ed912a716274288"
), file.path(target_dir, "dreams_ms2_network_provenance.tsv"))
cat(sprintf("[MetDNA3 bridge] injected %d calibrated edge scores\n", nrow(table_ms2_network)))
