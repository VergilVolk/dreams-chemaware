"""M8 -- Ablation harness.

Quantifies each module's contribution by toggling it on/off and reporting the
annotation coverage and the false-positive proxy at each step. The canonical
ladder (raw DreaMS -> full pipeline) is:

    raw cosine            -> cosine only (native DreaMS)
    + m/z constraint      -> require precursor-m/z agreement
    + target-decoy FDR    -> require q-value <= threshold
    + calibration         -> report P(correct)

Each step's output is measured on the *same* query set, so the deltas are
directly comparable. No result here promises improvement; it reports what the
data shows.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from .params import Params, DEFAULT


@dataclasses.dataclass(frozen=True)
class AblationStep:
    name: str
    params: Params
    require_fdr: bool = False


def ablation_steps() -> list[AblationStep]:
    """Ordered configurations from native DreaMS to the full pipeline."""
    raw = dataclasses.replace(DEFAULT, mz_constraint=False)
    mz = dataclasses.replace(DEFAULT, mz_constraint=True)
    fdr = dataclasses.replace(DEFAULT, mz_constraint=True)
    return [
        AblationStep("raw_cosine", raw, require_fdr=False),
        AblationStep("+mz_constraint", mz, require_fdr=False),
        AblationStep("+decoy_fdr", fdr, require_fdr=True),
    ]


def metrics(
    hits: pd.DataFrame,
    step: AblationStep,
    n_query: int | None = None,
) -> dict:
    """Annotation coverage + false-positive proxy for one ablation step.

    ``n_query`` is the total number of query spectra (for a rate denominator);
    if omitted, inferred from ``query_idx`` max. The false-positive proxy is the
    fraction of accepted hits whose precursor m/z disagrees (dppm > tolerance) --
    meaningful for the raw-cosine step where no m/z constraint is applied."""
    top1 = hits[hits["rank"] == 1].copy()
    accept = top1["cosine"] >= step.params.cosine_confident
    if step.params.mz_constraint:
        accept &= top1["mz_pass"]
    if step.require_fdr and "fdr_pass" in top1.columns:
        accept &= top1["fdr_pass"]

    accepted = top1[accept]
    n_total = n_query if n_query is not None else int(hits["query_idx"].max() + 1)
    fp_proxy = float((~accepted["mz_pass"]).mean()) if len(accepted) else float("nan")
    return {
        "step": step.name,
        "n_annotated": int(len(accepted)),
        "annotation_rate": float(len(accepted) / n_total) if n_total else 0.0,
        "fp_proxy_mz_mismatch": fp_proxy,
        "mean_cosine": float(accepted["cosine"].mean()) if len(accepted) else float("nan"),
        "unique_compounds": int(accepted["lib_inchikey"].nunique()) if len(accepted) else 0,
    }


def run_ablation(hits: pd.DataFrame, n_query: int | None = None) -> pd.DataFrame:
    """Run every ablation step on the same annotation table and return a table."""
    rows = [metrics(hits, step, n_query) for step in ablation_steps()]
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> str:
    """Human-readable ablation summary."""
    lines = ["Ablation ladder (native DreaMS -> full pipeline):", ""]
    for _, r in df.iterrows():
        lines.append(
            f"  {r['step']:<18} rate={r['annotation_rate']:.3f}  "
            f"annotated={r['n_annotated']}  fp_proxy(m/z mismatch)={r['fp_proxy_mz_mismatch']:.3f}"
        )
    return "\n".join(lines)
