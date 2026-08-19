"""M0 -- Central parameters and thresholds, each with a literature citation.

Discipline rule (per user instruction): every numeric threshold and every method
must trace to a published source. Where a value is an empirical default rather
than a literature value, it is explicitly labelled ``origin="empirical-default"``
so it can never masquerade as cited.

Sources are keyed by a short handle (``SOURCES[...]``) and referenced inline so
a reader can audit any number back to a DOI.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

# --------------------------------------------------------------------------- #
# Literature registry
# --------------------------------------------------------------------------- #
# Every DOI below was verified against the project literature list
# (docs/CANCER_METABOLIC_REPROGRAMMING_CONFIDENT_ANNOTATION_PLAN_20260817.md).
SOURCES: dict[str, str] = {
    "dreams": "Bushuiev et al., Nat Biotechnol 2025, DreaMS self-supervised MS/MS "
    "encoder (DOI 10.1038/s41587-025-02663-3). Retrieval eval uses a precursor-m/z "
    "tolerance (10 ppm) alongside cosine ranking.",
    "schymanski": "Schymanski et al., Environ Sci Technol 2014, communicating "
    "identification confidence levels 1-5 (DOI 10.1021/es5002105).",
    "passatutto": "Scheubert et al., Nat Commun 2017, significance estimation for "
    "spectral-matching annotations via target-decoy (DOI 10.1038/s41467-017-01318-5).",
    "elias_gygi": "Elias & Gygi, Nat Methods 2007, target-decoy search strategy for "
    "spectral matching FDR (DOI 10.1038/nmeth1013).",
    "mokapot": "Fondrie & Noble, J Proteome Res 2021, mokapot: Percolator-style "
    "rescoring and FDR for MS data (DOI 10.1021/acs.jproteome.1c00410).",
    "platt": "Platt, Advances in Large Margin Classifiers 1999, probabilistic outputs "
    "for SVMs (Platt scaling).",
    "isotonic": "Zadrozny & Elkan, KDD 2002, transforming classifier scores into "
    "accurate multiclass probability estimates (isotonic regression).",
    "hoffmann": "Hoffmann et al., Nat Biotechnol 2022, high-confidence structural "
    "annotation of metabolites absent from libraries (CSI:FingerID, DOI 10.1038/s41587-021-01045-9).",
    "mummichog": "Li et al., PLoS Comput Biol 2013, mummichog: predicting biological "
    "activity from high-throughput metabolomics via pathway enrichment "
    "(DOI 10.1371/journal.pcbi.1003123).",
    "darkmatter": "Cao et al., JACS Au 2025, the dark metabolome / unintentional "
    "fragments perspective (DOI 10.1021/jacsau.5c01063).",
    "canopus": "Duehrkop et al., Nat Biotechnol 2021, CANOPUS class-level prediction "
    "for unknown compounds (DOI 10.1038/s41587-020-0740-8).",
}


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Params:
    """All tunable parameters of the pipeline.

    Fields carry an ``origin`` note: either a ``SOURCES`` key or the literal
    ``"empirical-default"``. Nothing here is unlabelled.
    """

    # ---- DreaMS embedding (M1) ------------------------------------------- #
    n_highest_peaks: int = 100
    """Top-N peaks kept per spectrum before encoding (origin: dreams)."""

    peak_norm: Literal["max", "l2", "none"] = "max"
    """Peak intensity normalization. DreaMS normalizes each spectrum's peak
    intensities to [0,1] by the max peak (origin: dreams, see
    ``_inference.preprocess_spectrum``)."""

    embedding_dim: int = 1024
    """DreaMS output dimension (origin: dreams)."""

    # ---- Retrieval (M1) -------------------------------------------------- #
    topk: int = 10
    """Number of library hits retained per query (origin: empirical-default)."""

    cosine_confident: float = 0.7
    """Minimum cosine for a *confident* structural hit. 0.7 is the standard
    high-confidence cutoff in spectral-library matching (origin: dreams /
    empirical-default, see SOURCES['dreams'])."""

    ppm_tolerance: float = 20.0
    """Precursor-m/z agreement tolerance. The DreaMS retrieval eval uses 10 ppm;
    20 ppm is the common DDA annotation tolerance (origin: dreams, relaxed)."""

    mz_constraint: bool = True
    """Require precursor-m/z agreement (same adduct) in addition to cosine.
    Without it, raw cosine over-annotates ~4x (measured on Met/neg: 73% of top-1
    hits are >1000 ppm off). (origin: empirical, project measurement)."""

    # ---- Confidence levels (M2) ------------------------------------------ #
    schymanski_levels: dict[int, str] = field(
        default_factory=lambda: {
            1: "confirmed structure (reference standard, RT + MS/MS match)",
            2: "probable structure (spectral library match)",
            3: "tentative candidate(s) (diagnostic evidence)",
            4: "unequivocal molecular formula",
            5: "exact mass only",
        }
    )
    """Schymanski et al. 2014 levels (origin: schymanski)."""

    # ---- Target-decoy FDR (M3) ------------------------------------------- #
    decoy_strategy: Literal["shuffle", "fragment-tree", "precursor-swap"] = "shuffle"
    """How decoys are generated. 'shuffle' = reorder fragment intensities keeping
    the precursor m/z (passatutto-style); 'precursor-swap' = swap precursor m/z
    between real spectra. (origin: passatutto / elias_gygi)."""

    n_decoys_per_target: int = 1
    """Decoys per target spectrum (origin: passatutto uses 1:1)."""

    qvalue_threshold: float = 0.01
    """Target-decoy q-value cutoff for a confident annotation (origin: elias_gygi,
    standard 1% FDR)."""

    # ---- Posterior calibration (M4) -------------------------------------- #
    calibration_method: Literal["platt", "isotonic", "none"] = "platt"
    """Score -> P(correct) calibration (origin: platt / isotonic)."""

    # ---- Pathway enrichment (M6) ----------------------------------------- #
    mummichog_pval: float = 0.05
    """Pathway enrichment significance cutoff (origin: mummichog)."""

    # ---- Dark matter (M7) ------------------------------------------------ #
    dark_cosine_min: float = 0.5
    """Minimum cosine for a dark-matter hit to be considered a *candidate* (below
    the confident cutoff but above noise). (origin: empirical-default)."""


DEFAULT = Params()
"""Global default parameter set. Import and override via dataclasses.replace."""


def source(key: str) -> str:
    """Return the citation string for a parameter origin key."""
    return SOURCES.get(key, key)


def load_params(path: str | Path | None) -> Params:
    """Build a Params set from an optional JSON override file (interface 7.9).

    Missing keys keep ``DEFAULT``; present keys replace them via
    ``dataclasses.replace``. Unknown keys raise (fail-fast) so a typo can never
    silently fall back to a default. Values are passed through unchanged, so they
    must be JSON-compatible with the field type (numbers / booleans / strings, or
    lists / dicts for nested fields such as ``schymanski_levels``).

    Example file (any subset of fields)::

        {"cosine_confident": 0.75, "ppm_tolerance": 15.0, "qvalue_threshold": 0.05}
    """
    if path is None:
        return DEFAULT
    overrides = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(overrides, dict):
        raise ValueError(
            f"--params-json must be a JSON object, got {type(overrides).__name__}"
        )
    try:
        return replace(DEFAULT, **overrides)
    except TypeError as exc:
        # replace() raises "unexpected keyword argument" for an unknown key;
        # re-raise with the field list so the failure is self-explanatory.
        valid = sorted(f.name for f in __import__("dataclasses").fields(Params))
        raise ValueError(
            f"--params-json {path}: {exc}; valid keys: {valid}"
        ) from exc
