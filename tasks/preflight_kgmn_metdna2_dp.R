#!/usr/bin/env Rscript

if (!requireNamespace("MetDNA2", quietly = TRUE)) stop("MetDNA2 is unavailable")
if (!requireNamespace("tibble", quietly = TRUE)) stop("tibble is unavailable")
if (as.character(utils::packageVersion("MetDNA2")) != "1.2.10") {
  stop("expected MetDNA2 1.2.10")
}

make_spectrum <- function(name, precursor, intensity_scale = 1) {
  spec <- cbind(
    mz = c(40.0, 55.0, 72.0),
    intensity = intensity_scale * c(100.0, 50.0, 20.0)
  )
  MetDNA2::convertSpectraData(
    list(
      info = tibble::tibble(NAME = name, PRECURSORMZ = precursor),
      spec = spec
    )
  )
}

larger <- make_spectrum("larger", 101.0, 0.8)
smaller <- make_spectrum("smaller", 100.0, 1.0)
result <- try(
  MetDNA2::runSpecMatch(
    obj_ms2_cpd1 = larger,
    obj_ms2_cpd2 = smaller,
    mz_tol_ms2 = 25,
    scoring_approach = "dp"
  ),
  silent = TRUE
)
if (inherits(result, "try-error") || length(result) == 0) {
  stop("MetDNA2 exact DP runtime smoke test failed")
}
score <- as.numeric(result@info$scoreReverse)
if (length(score) != 1L || !is.finite(score) || score < 0 || score > 1) {
  stop("MetDNA2 exact DP runtime returned an invalid scoreReverse")
}
cat("[kgmn-dp-preflight] PASS scoreReverse=", score, "\n", sep = "")
