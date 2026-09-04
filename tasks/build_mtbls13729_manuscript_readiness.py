from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data/mtbls13729/candidate_evidence_ledger_v1/candidate_evidence_ledger.csv"
DELTA = ROOT / "data/mtbls13729/original_vs_dreams_biology_delta_v1/candidate_original_paper_delta.csv"
OUT = ROOT / "data/mtbls13729/manuscript_readiness_v1"


CURATION = {
    1597: {
        "paper_role": "primary_modified_guanosine_module",
        "novelty_class": "new ion-family annotation absent from author table",
        "external_evidence": "independent Level-1 CRC tissue data show context-dependent, not universal, behavior; pooled mucinous proteomics and paired epithelial RNA support purine/nucleoside context",
        "allowed_claim": "methylguanosine isomer-family ion elevated in the Rmu discovery subgroup",
        "forbidden_claim": "specific methylguanosine positional isomer or METTL1-derived flux",
        "minimum_validation": "m7G, m2G and Gm standards with same-method RT, MS2 and spike-in",
        "priority": "P0",
    },
    1717: {
        "paper_role": "primary_orthogonal_acetylated_polyamine_anchor",
        "novelty_class": "new chromatographic/subgroup signal; compound name existed in author HILIC table",
        "external_evidence": "rank-1 cross-chromatography concordance; SAT1-acetylspermidine literature supplies a testable mechanism but not identity or causality",
        "allowed_claim": "acetylated-polyamine/N1,N8-diacetylspermidine-like ion elevated in the Rmu discovery subgroup",
        "forbidden_claim": "new discovery of the compound name, SAT1 causality, or neutrophil recruitment in this cohort",
        "minimum_validation": "N1,N8-diacetylspermidine standard with RT, multi-energy MS2 and spike-in; distinguish N1-acetylspermidine and other acetylated polyamines",
        "priority": "P0",
    },
    3019: {
        "paper_role": "primary_modified_guanosine_module_anchor",
        "novelty_class": "new ion-family annotation absent from author table",
        "external_evidence": "independent Level-1 CRC tissue data support context dependence; pooled mucinous proteomics supports purine/nucleoside-processing context",
        "allowed_claim": "dimethylguanosine isomer-family ion elevated in the Rmu discovery subgroup",
        "forbidden_claim": "N2,N2-dimethylguanosine identity without standard or writer/turnover causality",
        "minimum_validation": "N2,N2-dimethylguanosine plus alternative positional-isomer standards with same-method RT, MS2 and spike-in",
        "priority": "P0",
    },
    3180: {
        "paper_role": "negative_biological_plausibility_control",
        "novelty_class": "unidentified reproducible feature",
        "external_evidence": "none",
        "allowed_claim": "reproducible but biologically uninterpretable feature used as a negative control",
        "forbidden_claim": "endogenous metabolite or mechanism member",
        "minimum_validation": "structure elucidation before any biological use",
        "priority": "exclude_from_mechanism",
    },
    3222: {
        "paper_role": "secondary_long_chain_acylcarnitine_anchor",
        "novelty_class": "additional long-chain/C20:4-like anchor within an author-established carnitine program",
        "external_evidence": "independent mucinous proteomics and paired epithelial RNA show lower FAO-axis abundance; literature shows acylcarnitine accumulation is compatible with either increased entry or incomplete utilization",
        "allowed_claim": "long-chain acylcarnitine/C20:4-like ion accumulation supports an FAO-utilization bottleneck hypothesis",
        "forbidden_claim": "activated FAO, reduced FAO flux, or exact C20:4 positional/stereochemical identity",
        "minimum_validation": "C20:4 acylcarnitine standard and C16:0/C18:0/C18:1 class panel with isotope internal standards if feasible",
        "priority": "P1",
    },
    4966: {
        "paper_role": "secondary_purine_companion_axis",
        "novelty_class": "new purine-like isomer-family signal within author-covered purine context",
        "external_evidence": "strong patient-level correlation with modified-guanosine module; pooled mucinous proteomics and paired epithelial RNA support purine synthesis/salvage context",
        "allowed_claim": "C7H9N5O nitrogenous-heterocycle/purine-like companion axis",
        "forbidden_claim": "preQ1 or another exact same-formula identity",
        "minimum_validation": "candidate same-formula standards or orthogonal structure elucidation",
        "priority": "P1",
    },
    7489: {
        "paper_role": "supporting_adduct_for_feature_1597",
        "novelty_class": "supporting sodium adduct, not an independent biological discovery",
        "external_evidence": "cross-adduct exact-mass and neutral-loss consistency",
        "allowed_claim": "supporting adduct evidence for the methylguanosine-like ion family",
        "forbidden_claim": "independent metabolite or independent statistical discovery",
        "minimum_validation": "validated together with feature 1597 standards",
        "priority": "support_only",
    },
    16425: {
        "paper_role": "exploratory_unidentified_lipid_like_feature",
        "novelty_class": "legacy LPE-like label not confirmed",
        "external_evidence": "none sufficient",
        "allowed_claim": "exploratory reproducible lipid-like feature",
        "forbidden_claim": "LPE identity or mechanism member",
        "minimum_validation": "lipid-class diagnostic fragments and authentic standard before biological interpretation",
        "priority": "exclude_from_primary_mechanism",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ledger = pd.read_csv(LEDGER)
    delta = pd.read_csv(DELTA)
    assert set(ledger["feature_id"].astype(int)) == set(CURATION)
    assert set(delta["feature_id"].astype(int)) == set(CURATION)

    merged = ledger.merge(
        delta[["feature_id", "author_exact_name_present", "author_context_entries", "strict_mz_rt_match_count"]],
        on="feature_id",
        how="left",
        validate="one_to_one",
    )
    rows = []
    for row in merged.itertuples(index=False):
        fid = int(row.feature_id)
        curated = CURATION[fid]
        rows.append(
            {
                "feature_id": fid,
                "paper_role": curated["paper_role"],
                "priority": curated["priority"],
                "defensible_identity": row.defensible_identity,
                "identity_grade": row.identity_grade,
                "novelty_class": curated["novelty_class"],
                "author_exact_name_present": bool(row.author_exact_name_present),
                "author_context_entries": int(row.author_context_entries),
                "strict_author_mz_rt_matches": int(row.strict_mz_rt_match_count),
                "rmu_pairs": int(row.rmu_n),
                "rmu_mean_log2fc": float(row.rmu_mean_log2fc),
                "rmu_exact_signflip_p": float(row.rmu_exact_signflip_p),
                "rmu_positive_fraction": float(row.rmu_positive_fraction),
                "peak_resolved_ms2_spectra": int(row.peak_resolved_ms2_spectra),
                "peak_resolved_samples": int(row.peak_resolved_samples),
                "orthogonal_evidence": row.orthogonal_evidence,
                "external_evidence": curated["external_evidence"],
                "allowed_claim": curated["allowed_claim"],
                "forbidden_claim": curated["forbidden_claim"],
                "minimum_validation": curated["minimum_validation"],
            }
        )
    readiness = pd.DataFrame(rows).sort_values(["priority", "feature_id"])
    readiness.to_csv(OUT / "candidate_manuscript_readiness.csv", index=False)

    primary = readiness["priority"].eq("P0")
    report = {
        "status": "mtbls13729_manuscript_readiness_complete",
        "candidates": int(len(readiness)),
        "primary_candidates": readiness.loc[primary, "feature_id"].astype(int).tolist(),
        "secondary_candidates": readiness.loc[readiness["priority"].eq("P1"), "feature_id"].astype(int).tolist(),
        "controls_or_excluded": readiness.loc[
            readiness["priority"].str.contains("exclude"), "feature_id"
        ].astype(int).tolist(),
        "core_biology_structure": {
            "coherent_axis": "modified-guanosine plus purine companion axis",
            "parallel_axis_1": "acetylated-polyamine abundance",
            "parallel_axis_2": "long-chain acylcarnitine accumulation with an FAO-utilization bottleneck hypothesis",
            "causal_chain_claimed": False,
        },
        "provenance": {
            "candidate_ledger_sha256": sha256(LEDGER),
            "original_delta_sha256": sha256(DELTA),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": "Discovery candidates from ten or fewer Rmu pairs; exact identities and flux require standards and perturbation/tracing.",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
