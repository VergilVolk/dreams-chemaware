"""M4 -- Posterior calibration of annotation scores to P(correct).

A raw cosine score is not a probability. Calibration maps score -> P(correct)
using a labelled set of retrieval outcomes, then applies the mapping to real
(unknown) annotations. Two standard calibrators are supported:

  * Platt scaling (Platt, Advances in Large Margin Classifiers 1999)
  * isotonic regression (Zadrozny & Elkan, KDD 2002)

The labelled set is built by *leave-one-spectrum-out* retrieval of the library
against itself: a hit is "correct" when it recovers a spectrum of the *same
compound* (same 14-character InChIKey). This mirrors the high-confidence
annotation validation in Hoffmann et al., Nat Biotechnol 2022
(DOI 10.1038/s41587-021-01045-9).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .params import Params, source
from .retrieve import chunked_topk


def library_self_scores(
    library: np.ndarray,
    lib_manifest: pd.DataFrame,
    topk: int = 5,
    chunk: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-out library self-retrieval -> (scores, correct_labels).

    For each library spectrum, take the best hit that is *not itself*, and label
    it correct iff its InChIKey14 equals the query's. Returns arrays of length
    N (one labelled example per library spectrum)."""
    inchikey = lib_manifest["inchikey"].astype(str).str[:14].to_numpy()
    topk_vals, topk_idx = chunked_topk(library, library, topk, chunk)
    n = library.shape[0]
    scores = np.empty(n, dtype=np.float32)
    labels = np.empty(n, dtype=np.int8)
    for i in range(n):
        picked_val, picked_label = 0.0, 0
        for r in range(topk):
            j = int(topk_idx[i, r])
            if j == i:
                continue  # skip self-match
            picked_val = float(topk_vals[i, r])
            picked_label = int(inchikey[i] == inchikey[j])
            break
        scores[i] = picked_val
        labels[i] = picked_label
    return scores, labels


def fit_platt(scores: np.ndarray, labels: np.ndarray):
    """Platt scaling via sklearn LogisticRegression on the single score feature."""
    from sklearn.linear_model import LogisticRegression

    x = scores.reshape(-1, 1)
    clf = LogisticRegression(C=1.0, class_weight="balanced").fit(x, labels)
    return clf


def fit_isotonic(scores: np.ndarray, labels: np.ndarray):
    """Isotonic regression calibration."""
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(scores, labels)
    return iso


def fit_calibrator(scores, labels, method: str):
    if method == "platt":
        return fit_platt(scores, labels)
    if method == "isotonic":
        return fit_isotonic(scores, labels)
    raise ValueError(f"unknown calibration method: {method}")


def apply_calibrator(hits, calibrator, params: Params) -> pd.DataFrame:
    """Add ``calibrated_prob`` = P(correct | cosine) to the annotation table."""
    out = hits.copy()
    x = out["cosine"].to_numpy().reshape(-1, 1)
    if params.calibration_method == "platt":
        prob = calibrator.predict_proba(x)[:, 1]
    elif params.calibration_method == "isotonic":
        prob = calibrator.predict(out["cosine"].to_numpy())
    else:
        prob = out["cosine"].to_numpy()
    out["calibrated_prob"] = prob
    return out


CALIBRATION_CITATIONS = {
    "platt": source("platt"),
    "isotonic": source("isotonic"),
    "labelling": source("hoffmann"),
}
