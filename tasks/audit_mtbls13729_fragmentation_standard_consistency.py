"""Audit recurrent candidate fragments against published authentic-standard transitions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mtbls13729/fragmentation_standard_consistency_v1"
CONSENSUS = ROOT / "data/mtbls13729/frozen_candidate_ms2_consensus_v1/candidate_recurrent_fragments.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nearest(frame: pd.DataFrame, feature: int, target: float) -> pd.Series:
    subset = frame[frame.feature_id == feature].copy()
    subset["abs_delta_da"] = (subset.fragment_mz - target).abs()
    return subset.sort_values("abs_delta_da").iloc[0]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fragments = pd.read_csv(CONSENSUS)
    references = [
        {
            "feature_id": 1597,
            "candidate_family": "methylguanosine isomers",
            "reference_precursor_mz": 298.1,
            "reference_product_mz": 166.0,
            "reference_scope": "authentic-standard transition shared by m1G, m2G and m7G",
            "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9140458/",
            "identity_ceiling": "family-level; transition cannot distinguish positional isomers",
        },
        {
            "feature_id": 3019,
            "candidate_family": "dimethylguanosine isomers",
            "reference_precursor_mz": 312.1,
            "reference_product_mz": 180.0,
            "reference_scope": "authentic-standard transition shared by m2,7G and m2,2G",
            "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9140458/",
            "identity_ceiling": "family-level; transition cannot distinguish positional isomers",
        },
        {
            "feature_id": 1717,
            "candidate_family": "N1,N8-diacetylspermidine-like",
            "reference_precursor_mz": 230.2,
            "reference_product_mz": 100.0,
            "reference_scope": "authentic-standard MRM transition for N1,N8-diacetylspermidine",
            "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5877617/",
            "identity_ceiling": "strong diagnostic-transition consistency; no same-method RT/full-spectrum standard match",
        },
        {
            "feature_id": 3222,
            "candidate_family": "long-chain acylcarnitine-like",
            "reference_precursor_mz": 448.3395,
            "reference_product_mz": 85.028,
            "reference_scope": "acylcarnitine-class diagnostic fragment used as class evidence",
            "source": "local c20_4_anchor_ms2_audit_v1; authentic C20:4 standard still required",
            "identity_ceiling": "class-level; acyl-chain position, double bonds and chromatographic identity unresolved",
        },
    ]
    rows = []
    for ref in references:
        hit = nearest(fragments, ref["feature_id"], ref["reference_product_mz"])
        rows.append(
            {
                **ref,
                "observed_fragment_mz": float(hit.fragment_mz),
                "nominal_transition_match": bool(round(float(hit.fragment_mz), 0) == round(float(ref["reference_product_mz"]), 0)),
                "support_spectra": int(hit.support_spectra),
                "support_fraction": float(hit.support_fraction),
                "support_samples": int(hit.support_samples),
                "median_relative_intensity": float(hit.median_relative_intensity),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "fragmentation_standard_consistency.csv", index=False)
    report = {
        "status": "mtbls13729_fragmentation_standard_consistency_complete",
        "formal": False,
        "candidates": len(out),
        "results": out.to_dict(orient="records"),
        "key_clarification": "Peak-resolved MS2 acquisition and recurrent diagnostic ions are real evidence, but they are not equivalent to an accepted library bridge or authentic-standard identity. Literature MRM product masses reported to one decimal place are treated as nominal transitions; no high-resolution mass error is computed against them.",
        "claim_limit": "Diagnostic-transition consistency supports a chemical family or candidate. Positional isomers and MSI Level 1 require same-method retention time, full-spectrum comparison, and preferably sample spike-in.",
        "provenance": {"consensus_sha256": sha256(CONSENSUS), "script_sha256": sha256(Path(__file__))},
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
