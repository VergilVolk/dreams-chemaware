from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CURATED = {
    1597: {
        "defensible_identity": "methylguanosine isomer family ([M+H]+)",
        "identity_grade": "MSI level 3-like class/isomer-family evidence",
        "orthogonal_evidence": "ribose loss in 30/42 spectra; cross-adduct neutral-loss consistency with feature 7489",
        "manuscript_role": "modified-guanosine family member",
        "claim_boundary": "positional methylguanosine isomer unresolved; authentic standards required",
    },
    1717: {
        "defensible_identity": "N1,N8-diacetylspermidine-like / acetylated-polyamine ion family",
        "identity_grade": "strong putative candidate; below MSI level 2 without standard/library-spectrum match",
        "orthogonal_evidence": "m/z 100.0759 is the base peak in 73/73 spectra; same-source HILIC HMDB0041947 is rank-1 correlated across samples and paired deltas",
        "manuscript_role": "strongest novel orthogonal candidate axis",
        "claim_boundary": "exact positional identity requires authentic-standard RT, MS2, and spike-in coelution",
    },
    3019: {
        "defensible_identity": "dimethylguanosine isomer family ([M+H]+)",
        "identity_grade": "MSI level 3-like class/isomer-family evidence",
        "orthogonal_evidence": "ribose loss in 32/32 spectra; cross-adduct neutral-loss consistency with feature 8481",
        "manuscript_role": "modified-guanosine family anchor",
        "claim_boundary": "1,7- versus N2,N2-dimethylguanosine and other positional isomers unresolved",
    },
    3180: {
        "defensible_identity": "unknown chlorinated/exogenous-like feature",
        "identity_grade": "unknown structure / biological-plausibility control",
        "orthogonal_evidence": "reproducible DDA spectrum but no defensible endogenous identity",
        "manuscript_role": "negative biological-plausibility control",
        "claim_boundary": "must not be used as an endogenous mechanism metabolite",
    },
    3222: {
        "defensible_identity": "long-chain acylcarnitine; C20:4-acylcarnitine-like",
        "identity_grade": "MSI level 3-like lipid-class evidence",
        "orthogonal_evidence": "59 precursor/RT-matched spectra; strong carnitine motif in 25 samples; recurrent m/z 85.0281 and 60.0808",
        "manuscript_role": "acylcarnitine/FAO-utilization anchor",
        "claim_boundary": "chain double-bond position, stereochemistry, and exact chromatographic identity unresolved",
    },
    4966: {
        "defensible_identity": "C7H9N5O nitrogenous heterocycle / purine-like isomer family",
        "identity_grade": "formula plus recurrent-fragment family evidence; MSI level 4/3 boundary",
        "orthogonal_evidence": "23 peak-resolved spectra; recurrent m/z 110.0347, 153.0404, 137.0817, and 135.0298",
        "manuscript_role": "purine/nitrogenous-heterocycle abundance axis",
        "claim_boundary": "same-formula isomers prevent preQ1 or another exact name assignment",
    },
    7489: {
        "defensible_identity": "methylguanosine isomer family ([M+Na]+)",
        "identity_grade": "supporting adduct/class evidence; sparse MSI level 3-like evidence",
        "orthogonal_evidence": "ribose loss in 3/3 spectra; precursor/adduct consistency with feature 1597",
        "manuscript_role": "supporting modified-guanosine adduct",
        "claim_boundary": "only three peak-resolved spectra; exact positional isomer unresolved",
    },
    16425: {
        "defensible_identity": "unknown reproducible lipid-like feature; legacy LPE-like label unconfirmed",
        "identity_grade": "unknown structure with reproducible MS2",
        "orthogonal_evidence": "25 peak-resolved spectra with highly recurrent fragments, but positive-mode evidence is not diagnostic for LPE",
        "manuscript_role": "exploratory lipid feature only",
        "claim_boundary": "do not claim LPE subclass or acyl chain until compatible reference/standard evidence exists",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--abundance",
        type=Path,
        default=Path("data/mtbls13729/biology_closure_analysis_v1/candidate_identity_and_abundance.csv"),
    )
    parser.add_argument(
        "--coverage",
        type=Path,
        default=Path("data/mtbls13729/frozen_candidate_ms2_coverage_v1/candidate_ms2_coverage.csv"),
    )
    parser.add_argument(
        "--consensus",
        type=Path,
        default=Path("data/mtbls13729/frozen_candidate_ms2_consensus_v1/report.json"),
    )
    parser.add_argument(
        "--crosschrom",
        type=Path,
        default=Path("data/mtbls13729/polyamine_crosschrom_audit_v1/summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mtbls13729/candidate_evidence_ledger_v1"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    abundance = pd.read_csv(args.abundance)
    coverage = pd.read_csv(args.coverage)
    consensus_report = json.loads(args.consensus.read_text(encoding="utf-8"))
    crosschrom = json.loads(args.crosschrom.read_text(encoding="utf-8"))

    consensus_rows = []
    for row in consensus_report["summary"]:
        fragments = row["top_fragments"][:4]
        consensus_rows.append(
            {
                "feature_id": int(row["feature_id"]),
                "recurrent_ms2_clusters": int(row["recurrent_clusters"]),
                "top_recurrent_fragments": "; ".join(
                    f"{f['mz']:.4f} ({100*f['support_fraction']:.1f}%)" for f in fragments
                ),
            }
        )
    consensus = pd.DataFrame(consensus_rows)

    keep = [
        "feature_id",
        "biology_label",
        "candidate_formula",
        "mz",
        "rt_sec",
        "n_samples_detected",
        "global_prevalence",
        "rmu_n",
        "rmu_mean_log2fc",
        "rmu_fold_change",
        "rmu_exact_signflip_p",
        "rmu_positive_fraction",
        "rmu_loo_direction_stable",
        "interaction_log2fc",
        "interaction_exact_permutation_p",
    ]
    ledger = abundance[keep].merge(coverage, on=["feature_id", "biology_label", "mz", "rt_sec"], how="left")
    ledger = ledger.merge(consensus, on="feature_id", how="left")
    curated = pd.DataFrame([{"feature_id": feature_id, **fields} for feature_id, fields in CURATED.items()])
    ledger = ledger.merge(curated, on="feature_id", how="left")

    if set(ledger["feature_id"]) != set(CURATED):
        raise RuntimeError("ledger does not contain exactly the eight frozen candidate features")
    if ledger.isna().any().any():
        missing = ledger.columns[ledger.isna().any()].tolist()
        raise RuntimeError(f"candidate evidence ledger contains missing values: {missing}")

    ledger["legacy_identity_tier_is_not_msi_level"] = True
    ledger["crosschrom_support"] = "not audited"
    mask = ledger["feature_id"] == int(crosschrom["rp_feature_id"])
    ledger.loc[mask, "crosschrom_support"] = (
        f"Spearman rho={crosschrom['cross_sample_concordance']['spearman_rho']:.3f} across "
        f"{crosschrom['common_samples']} samples; paired-delta rho="
        f"{crosschrom['paired_tumor_normal_concordance']['spearman_rho']:.3f}; "
        f"rank {crosschrom['source_hilic_specificity']['target_rank_by_spearman']}/"
        f"{crosschrom['source_hilic_specificity']['eligible_hilic_features']} HILIC annotations"
    )

    ledger = ledger.sort_values("feature_id").reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "candidate_evidence_ledger.csv"
    json_path = args.output_dir / "report.json"
    ledger.to_csv(csv_path, index=False)

    report = {
        "status": "mtbls13729_candidate_evidence_ledger_complete",
        "formal": True,
        "candidates": int(len(ledger)),
        "candidates_with_peak_resolved_ms2": int((ledger["peak_resolved_ms2_spectra"] > 0).sum()),
        "total_peak_resolved_ms2_spectra": int(ledger["peak_resolved_ms2_spectra"].sum()),
        "strongest_novel_candidate": 1717,
        "class_anchor": 3222,
        "identity_contract": (
            "Internal legacy analysis tiers are not MSI identification levels. Exact metabolite names require "
            "compatible experimental reference spectra and/or authentic-standard RT and spike-in evidence."
        ),
        "statistical_contract": (
            "Rmu paired effects are discovery evidence in ten or fewer pairs and were evaluated after candidate "
            "selection; they are not a global feature-space FDR claim."
        ),
        "outputs": {"ledger": str(csv_path.resolve())},
        "records": ledger.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in list(report)[:8]}, indent=2, ensure_ascii=False))
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
