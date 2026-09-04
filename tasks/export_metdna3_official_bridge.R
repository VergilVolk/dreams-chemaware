#!/usr/bin/env Rscript

# Export the exact MrnAnnoAlgo3 feature-edge set and its feature spectra.
# This script does not alter the author package or calculate an annotation score.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("usage: export_metdna3_official_bridge.R <metdna3_workdir> <output_dir>")
}

wd <- normalizePath(args[[1]], mustWork = TRUE)
out <- args[[2]]
source_dir <- file.path(wd, "02_result_MRN_annotation")
edge_path <- file.path(source_dir, "table_ms2_edges.rda")
spectra_path <- file.path(source_dir, "ms2_data.rda")
if (!file.exists(edge_path)) stop(paste("missing", edge_path))
if (!file.exists(spectra_path)) stop(paste("missing", spectra_path))
if (dir.exists(out) && length(list.files(out, all.files = TRUE, no.. = TRUE)) > 0) {
  stop(paste("fail-closed: output directory is non-empty:", out))
}
dir.create(out, recursive = TRUE, showWarnings = FALSE)

edge_env <- new.env(parent = emptyenv())
load(edge_path, envir = edge_env)
if (!exists("table_ms2_edges", envir = edge_env, inherits = FALSE)) {
  stop("table_ms2_edges.rda does not contain table_ms2_edges")
}
edges <- as.data.frame(edge_env$table_ms2_edges, stringsAsFactors = FALSE)
if (!all(c("from", "to") %in% colnames(edges))) stop("edge table misses from/to")
edges <- edges[, c("from", "to"), drop = FALSE]
edges$from <- as.character(edges$from)
edges$to <- as.character(edges$to)
if (any(is.na(edges$from) | is.na(edges$to) | edges$from == "" | edges$to == "")) {
  stop("edge table contains missing feature names")
}
if (any(edges$from == edges$to)) stop("edge table contains self edges")
edge_key <- vapply(seq_len(nrow(edges)), function(i) {
  paste(sort(c(edges$from[[i]], edges$to[[i]])), collapse = "\037")
}, character(1))
if (anyDuplicated(edge_key)) stop("edge table contains duplicate undirected edges")
edges$edge_index <- seq_len(nrow(edges)) - 1L
edges$edge_key <- edge_key
write.csv(edges[, c("edge_index", "from", "to", "edge_key")],
          file.path(out, "table_ms2_edges.csv"), row.names = FALSE, na = "")

spectra_env <- new.env(parent = emptyenv())
load(spectra_path, envir = spectra_env)
if (!exists("ms2_data", envir = spectra_env, inherits = FALSE)) {
  stop("ms2_data.rda does not contain ms2_data")
}
ms2_data <- spectra_env$ms2_data
feature_names <- vapply(ms2_data, function(x) as.character(x$info[1, 1]), character(1))
if (anyDuplicated(feature_names)) stop("ms2_data contains duplicate feature names")
needed <- unique(c(edges$from, edges$to))
missing <- setdiff(needed, feature_names)
if (length(missing) > 0) stop(paste("ms2_data misses edge features:", length(missing)))

mgf_path <- file.path(out, "feature_spectra.mgf")
manifest <- vector("list", length(needed))
con <- file(mgf_path, open = "wt", encoding = "UTF-8")
on.exit(close(con), add = TRUE)
for (i in seq_along(needed)) {
  feature <- needed[[i]]
  item <- ms2_data[[match(feature, feature_names)]]
  info <- as.data.frame(item$info, stringsAsFactors = FALSE)
  precursor <- suppressWarnings(as.numeric(info$PRECURSORMZ[[1]]))
  spec <- as.data.frame(item$spec)
  if (!is.finite(precursor)) stop(paste("non-finite precursor for", feature))
  if (ncol(spec) < 2) stop(paste("invalid peak matrix for", feature))
  mz <- suppressWarnings(as.numeric(spec[[1]]))
  intensity <- suppressWarnings(as.numeric(spec[[2]]))
  keep <- is.finite(mz) & is.finite(intensity) & mz > 0 & intensity > 0
  mz <- mz[keep]
  intensity <- intensity[keep]
  if (length(mz) == 0) stop(paste("no valid peaks for", feature))
  writeLines(c("BEGIN IONS", paste0("TITLE=", feature), paste0("PEPMASS=", precursor)), con)
  writeLines(sprintf("%.10f %.10f", mz, intensity), con)
  writeLines(c("END IONS", ""), con)
  manifest[[i]] <- data.frame(feature_name = feature, precursor_mz = precursor,
                              peaks = length(mz), stringsAsFactors = FALSE)
}
close(con)
on.exit(NULL, add = FALSE)
write.csv(do.call(rbind, manifest), file.path(out, "feature_spectra.csv"),
          row.names = FALSE, na = "")
writeLines(c(
  paste0("source_workdir\t", wd),
  paste0("edges\t", nrow(edges)),
  paste0("features\t", length(needed)),
  paste0("author_source_commit\t978ae62b33bde75a066032953ed912a716274288")
), file.path(out, "bridge_provenance.tsv"))
cat(sprintf("[MetDNA3 bridge] exported %d edges and %d spectra\n", nrow(edges), length(needed)))
