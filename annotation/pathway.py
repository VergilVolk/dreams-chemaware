"""M6 -- Pathway enrichment.

Two enrichment modes, both with a literature basis:

1. **Annotated-compound enrichment** -- classic over-representation analysis
   (hypergeometric / Fisher exact test) of a set of annotated compounds against
   a compound->pathway mapping.

2. **mummichog-style m/z enrichment** (Li et al., PLoS Comput Biol 2013,
   DOI 10.1371/journal.pcbi.1003123) -- for *unannotated* features where only the
   precursor m/z is known: each m/z is matched to every database metabolite
   within a ppm window, and pathways are scored by how many query m/z features
   map into them.

Honest dependency: neither mode ships a compound->pathway database. Such a table
(HMDB / KEGG / Reactome) is an external resource and must be supplied as a
DataFrame -- see :func:`enrich_annotated` / :func:`enrich_mz`. The algorithms are
database-agnostic and reference-standard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .params import Params, source


def _bh_correct(p: np.ndarray) -> np.ndarray:
    n = len(p)
    ranked = np.argsort(p)
    bh = p[ranked] * n / (np.arange(n) + 1)
    q_ranked = np.minimum.accumulate(bh[::-1])[::-1]
    out = np.empty(n)
    out[ranked] = q_ranked
    return out


def enrich_annotated(
    query_compounds: list[str],
    mapping: pd.DataFrame,
    params: Params,
    compound_col: str = "compound",
    pathway_col: str = "pathway",
) -> pd.DataFrame:
    """Over-representation of annotated compounds via hypergeometric test.

    ``mapping`` maps compounds to pathways (two columns). ``query_compounds`` are
    the compound IDs (e.g. InChIKey14) whose pathway membership we test. The
    background is the full set of compounds present in ``mapping``."""
    from scipy.stats import hypergeom

    mapping = mapping.dropna(subset=[compound_col, pathway_col]).drop_duplicates()
    background = set(mapping[compound_col].astype(str))
    query = set(query_compounds) & background

    N = len(background)
    n = len(query)
    rows = []
    for pathway, members in mapping.groupby(pathway_col)[compound_col].agg(set).items():
        K = len(members & background)
        k = len(members & query)
        if K == 0:
            continue
        p = float(hypergeom.sf(k - 1, N, K, n))
        rows.append({
            "pathway": pathway,
            "n_pathway_background": K,
            "n_query_in_pathway": k,
            "n_query_total": n,
            "p_value": p,
            "fold_enrichment": (k / n) / (K / N) if K > 0 and n > 0 else 0.0,
        })
    res = pd.DataFrame(rows)
    if len(res):
        res["q_value"] = _bh_correct(res["p_value"].to_numpy())
    return res.sort_values("q_value").reset_index(drop=True)


def enrich_mz(
    query_mz: np.ndarray,
    metabolites: pd.DataFrame,
    params: Params,
    mz_col: str = "mz",
    compound_col: str = "compound",
    pathway_col: str = "pathway",
) -> pd.DataFrame:
    """mummichog-style enrichment of unannotated m/z features.

    ``metabolites`` maps each database metabolite to its monoisotopic m/z and its
    pathway(s). Every query m/z is assigned to all metabolites within
    ``params.ppm_tolerance``; a pathway is hit when any of its metabolites is
    within tolerance of at least one query m/z. Pathways are ranked by number of
    distinct query m/z they explain, with a permutation-free hypergeometric p-value
    against the number of matched m/z features."""
    from scipy.stats import hypergeom

    met = metabolites.dropna(subset=[mz_col, pathway_col])
    mz_arr = met[mz_col].to_numpy(dtype=np.float64)

    explained = np.zeros(len(query_mz), dtype=bool)
    pathway_hits: dict[str, set] = {}
    for i, q in enumerate(query_mz):
        dppm = np.abs(q - mz_arr) / np.maximum(np.abs(mz_arr), 1e-9) * 1e6
        within = dppm <= params.ppm_tolerance
        if within.any():
            explained[i] = True
            for pw in met.loc[within, pathway_col].dropna().unique():
                pathway_hits.setdefault(pw, set()).add(i)

    total_matched = int(explained.sum())
    rows = []
    for pw, hits in pathway_hits.items():
        k = len(hits)
        K = int(met[met[pathway_col] == pw].shape[0])
        p = float(hypergeom.sf(k - 1, len(met), K, total_matched)) if total_matched else 1.0
        rows.append({"pathway": pw, "n_query_explained": k, "p_value": p})
    res = pd.DataFrame(rows)
    if len(res):
        res["q_value"] = _bh_correct(res["p_value"].to_numpy())
    return res.sort_values("q_value").reset_index(drop=True)


PATHWAY_CITATIONS = {
    "ora": "hypergeometric / Fisher over-representation analysis (standard)",
    "mummichog": source("mummichog"),
    "bh": "Benjamini & Hochberg, J R Stat Soc B 1995",
    "database": "HMDB / KEGG / Reactome compound->pathway mapping is an external "
    "resource and must be supplied; not shipped (avoids fabricated mappings).",
}
