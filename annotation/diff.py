"""M5 -- Differential abundance between two groups (e.g. tumor vs control).

For DDA data the available semi-quantitative proxy is the **MS2 spectral count**
of an annotated compound per sample group (number of MS2 spectra whose
confident top-1 hit is that compound). This is a standard, if semi-quantitative,
DDA abundance proxy; it is less precise than MS1 peak-area or DIA
quantification, and that caveat is reported explicitly.

Group comparison uses Fisher's exact test on the 2x2 count table
(compound-hit spectra vs. other-hit spectra in each group), with
Benjamini-Hochberg FDR correction across compounds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .params import Params, source


def confident_top1(hits: pd.DataFrame, params: Params, fdr_pass: bool = False) -> pd.DataFrame:
    """Top-1 hits that are structurally confident: cosine >= cutoff AND m/z pass
    (AND q-value pass when fdr_pass=True)."""
    conf = hits[hits["rank"] == 1]
    conf = conf[(conf["cosine"] >= params.cosine_confident) & conf["mz_pass"]]
    if fdr_pass and "fdr_pass" in conf.columns:
        conf = conf[conf["fdr_pass"]]
    return conf


def group_counts(conf: pd.DataFrame, group_col: str, inchikey_col: str = "lib_inchikey") -> pd.DataFrame:
    """Spectral counts per (compound, group) over confident top-1 hits."""
    g = conf.groupby([inchikey_col, group_col]).size().rename("n").reset_index()
    return g


def differential(
    conf: pd.DataFrame,
    group_col: str,
    group_a: str,
    group_b: str,
    params: Params,
) -> pd.DataFrame:
    """Fisher-exact differential test of confident top-1 spectral counts between
    group_a and group_b. Returns one row per compound with counts, p-value and
    BH-corrected q-value."""
    from scipy.stats import fisher_exact

    counts = group_counts(conf, group_col)
    total_a = int(conf[conf[group_col] == group_a].shape[0])
    total_b = int(conf[conf[group_col] == group_b].shape[0])

    rows = []
    for inchikey, sub in counts.groupby("lib_inchikey"):
        a = int(sub.loc[sub[group_col] == group_a, "n"].sum())
        b = int(sub.loc[sub[group_col] == group_b, "n"].sum())
        table = [[a, total_a - a], [b, total_b - b]]
        odds, p = fisher_exact(table)
        rows.append({
            "lib_inchikey": inchikey,
            f"n_{group_a}": a,
            f"n_{group_b}": b,
            "odds_ratio": float(odds),
            "p_value": float(p),
        })
    res = pd.DataFrame(rows)
    if len(res):
        # Benjamini-Hochberg FDR (Benjamini & Hochberg, J R Stat Soc B 1995).
        # q_i = min_{j >= i} (p_(j) * n / j) for p-values sorted ascending.
        p = res["p_value"].to_numpy()
        n = len(p)
        ranked = np.argsort(p)
        bh = p[ranked] * n / (np.arange(n) + 1)
        q_ranked = np.minimum.accumulate(bh[::-1])[::-1]
        qvals = np.empty(n)
        qvals[ranked] = q_ranked
        res["q_value"] = qvals
    return res.sort_values("q_value").reset_index(drop=True)


DIFF_CITATIONS = {
    "fisher": "Fisher, J R Stat Soc 1925 (exact test)",
    "bh": "Benjamini & Hochberg, J R Stat Soc B 1995 (FDR correction)",
    "dda_counts": "Spectral counting is a semi-quantitative DDA abundance proxy "
    "(less precise than MS1 peak area / DIA); reported as a caveat.",
}
