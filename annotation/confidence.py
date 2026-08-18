"""M2 -- Schymanski identification-confidence levels (Level 1-5).

Maps each annotation to the 5-level confidence scheme of Schymanski et al.,
Environ Sci Technol 2014 (DOI 10.1021/es5002105). Honest scoping: Level 1
("confirmed structure") requires a reference standard measured on the same
instrument with matching RT + MS/MS, which this platform does not consume, so
the platform's ceiling is **Level 2a** (probable structure via spectral-library
match) plus **Level 3** (tentative candidate via diagnostic rule evidence).
Level 4 (unequivocal molecular formula) would require an external formula
predictor (SIRIUS etc.) and is not integrated here; those annotations are left
at Level 5 (exact mass).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .params import Params, source


def assign_schymanski(
    hits: pd.DataFrame,
    params: Params,
    rules_evidence: np.ndarray | None = None,
) -> pd.DataFrame:
    """Add a ``schymanski_level`` column (int) to the annotation table.

    Decision rule (top-1 only, i.e. ``rank == 1`` rows):
      * Level 2a (2) -- library match: cosine >= cosine_confident AND m/z pass.
      * Level 3  (3) -- tentative candidate: precursor m/z passes AND there is a
        structural clue (cosine >= dark_cosine_min, OR positive diagnostic-rule
        evidence).
      * Level 5  (5) -- exact mass only (no structural clue).

    ``rules_evidence`` (optional) is a boolean array of length ``n_query``,
    indexed by ``query_idx``, marking spectra with diagnostic-rule evidence
    (see ``annotation.rule_evidence``). It is a *semantic* Schymanski signal,
    not a correctness prediction. Level 1 and 4 are intentionally never emitted.
    """
    out = hits.copy()
    lvl = pd.Series(5, index=out.index, dtype=int)

    top1 = out["rank"] == 1
    lib_match = top1 & (out["cosine"] >= params.cosine_confident) & out["mz_pass"]
    lvl[lib_match] = 2

    has_rule = np.zeros(len(out), dtype=bool)
    if rules_evidence is not None:
        has_rule = np.asarray(rules_evidence, dtype=bool)[out["query_idx"].to_numpy()]
    structural_clue = (out["cosine"] >= params.dark_cosine_min) | has_rule
    tentative = top1 & ~lib_match & out["mz_pass"] & structural_clue
    lvl[tentative] = 3

    out["schymanski_level"] = lvl
    out["schymanski_label"] = lvl.map(params.schymanski_levels)
    return out


def level_summary(hits: pd.DataFrame) -> pd.DataFrame:
    """Count of top-1 annotations per Schymanski level."""
    top1 = hits[hits["rank"] == 1]
    return top1["schymanski_level"].value_counts().sort_index().rename("count").reset_index()


LEVEL_CITATIONS = {
    2: source("schymanski") + " (Level 2a: spectral-library match)",
    3: source("schymanski") + " (Level 3: diagnostic evidence)",
    5: source("schymanski") + " (Level 5: exact mass only)",
}
