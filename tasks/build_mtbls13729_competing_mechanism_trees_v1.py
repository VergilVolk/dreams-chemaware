"""Freeze competing biological mechanisms and their decisive tests for MTBLS13729.

This is an evidence-calibration artifact.  It deliberately represents mutually
compatible causal generators for each observed abundance program so that a
static LC-MS/MS signal is not silently promoted to flux or enzyme causality.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data/mtbls13729/manuscript_evidence_matrix_v2/candidate_manuscript_evidence_matrix_v2.csv"
COORDINATION = ROOT / "data/mtbls13729/module_coordination_v2/report.json"
LINEAGE = ROOT / "data/external/TCGA_COADREAD_Xena_20260830/proline_sialic_lineage_sensitivity_v1/report.json"
OUT = ROOT / "data/mtbls13729/competing_mechanism_trees_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def row(
    axis: str,
    hypothesis_id: str,
    generator: str,
    support: str,
    contradiction: str,
    discriminator: str,
    priority: str,
    claim_now: str,
    forbidden: str,
    sources: str,
) -> dict[str, str]:
    return {
        "axis": axis,
        "hypothesis_id": hypothesis_id,
        "causal_generator": generator,
        "current_support": support,
        "current_counterevidence_or_ambiguity": contradiction,
        "decisive_discriminator": discriminator,
        "validation_priority": priority,
        "claim_now": claim_now,
        "forbidden_claim": forbidden,
        "primary_sources": sources,
    }


def main() -> None:
    for path in (LEDGER, COORDINATION, LINEAGE):
        if not path.is_file():
            raise FileNotFoundError(path)

    ledger = pd.read_csv(LEDGER)
    required_features = {345, 374, 703, 1597, 1717, 3019, 3222}
    present_features = set(pd.to_numeric(ledger["feature_id"], errors="raise").astype(int))
    missing = sorted(required_features - present_features)
    if missing:
        raise RuntimeError(f"required manuscript anchors missing: {missing}")

    rows = [
        row(
            "modified_guanosine",
            "MG-1",
            "Increased METTL1/WDR4-dependent RNA m7G deposition enlarges the modified-RNA pool and secondarily the free nucleoside pool.",
            "CRC literature establishes oncogenic METTL1-dependent m7G biology; MTBLS13729 contains recurrent methyl/dimethylguanosine-family ions and a purine companion.",
            "Free-tissue ions are unresolved positional isomers; no RNA-class-specific modification measurement or METTL1 perturbation exists; abundance and writer activity are not equivalent.",
            "First resolve m7G/m2G/Gm/m2,2G by same-method standards; then quantify matched free nucleosides and enzymatically digested tRNA/rRNA/mRNA with isotope-dilution LC-MS before and after METTL1 catalytic perturbation.",
            "P0 identity, P2 causality",
            "Modified-guanosine positional-isomer families are elevated in the Rmu discovery subgroup and motivate an RNA-modification hypothesis.",
            "METTL1-driven modified-guanosine flux or m7G causality.",
            "https://pubmed.ncbi.nlm.nih.gov/41627602/; https://pmc.ncbi.nlm.nih.gov/articles/PMC8567201/",
        ),
        row(
            "modified_guanosine",
            "MG-2",
            "Accelerated turnover or degradation of pre-existing methylated RNA releases modified ribonucleosides without increased writer activity.",
            "Free modified ribonucleosides are established readouts of RNA turnover, and isotope-labeling studies show that abundance alone cannot separate deposition from decay.",
            "No RNA integrity, RNA half-life, intracellular/extracellular partition or pulse-chase measurement is available in MTBLS13729.",
            "Measure RNA integrity and RNA-class modification abundance together with free intra/extracellular nucleosides; perform 13C-methyl-methionine pulse-chase or NAIL-MS turnover analysis.",
            "P1 after identity",
            "RNA turnover is an equally plausible source of the modified-guanosine pool.",
            "Elevated free methylguanosine proves increased epitranscriptomic writing.",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC8567201/; https://www.nature.com/articles/s41467-020-20576-4",
        ),
        row(
            "modified_guanosine",
            "MG-3",
            "Cell death, secretion/export, renal-like clearance differences or tissue composition change the recoverable free-nucleoside pool.",
            "The current material is bulk static tissue; external modified-nucleoside directions are heterogeneous rather than universal.",
            "No matched extracellular fluid, necrosis score, cell-type-resolved metabolite measurement or clearance measurement exists.",
            "Test intracellular-to-extracellular ratios, LDH/cell-death and RNA-integrity covariates, pathology tumour content and cell-type composition; require direction to persist after these checks.",
            "P2",
            "The biological source of the free modified-guanosine signal remains unresolved.",
            "The signal is tumour-cell-autonomous or subtype-specific.",
            "https://pubmed.ncbi.nlm.nih.gov/7508341/; https://pmc.ncbi.nlm.nih.gov/articles/PMC8671418/",
        ),
        row(
            "acetylated_polyamine",
            "PA-1",
            "Host SAT1/SSAT-mediated polyamine acetylation and export raises an acetylated-polyamine pool.",
            "CRC tissue studies report elevated acetylated polyamines and SAT1-linked biology; feature 1717 has recurrent m/z 100.0759 fragmentation and cross-panel abundance agreement.",
            "The local ion is only N1,N8-diacetylspermidine-like; CRC tissue literature most strongly establishes N1,N12-diacetylspermine and N1-acetylspermidine, not this exact identity.",
            "Run an isomer panel containing N1,N8-diacetylspermidine, N1,N12-diacetylspermine and monoacetylspermidines with RT, MS2 and spike-in; then assay SAT1 activity/expression and paired intra/extracellular pools.",
            "P0 identity, P1 source",
            "An acetylated-polyamine family is elevated and is compatible with enhanced polyamine acetylation/export.",
            "Feature 1717 is confirmed N1,N8-diacetylspermidine or proves SAT1 activity.",
            "https://pubmed.ncbi.nlm.nih.gov/23443255/; https://pubmed.ncbi.nlm.nih.gov/6692383/",
        ),
        row(
            "acetylated_polyamine",
            "PA-2",
            "A different positional acetyl-polyamine isomer, especially N1,N12-diacetylspermine, explains the signal.",
            "N1,N12-diacetylspermine is elevated in paired CRC tissue and has a larger CRC evidence base than N1,N8-diacetylspermidine.",
            "Nominal mass and a common polyamine fragment cannot resolve positional isomers.",
            "Require chromatographic separation and same-method co-injection of all plausible acetylated polyamine standards; compare diagnostic product-ion ratios across collision energies.",
            "P0",
            "The exact acetylated-polyamine identity is open.",
            "A database name or one shared fragment establishes positional identity.",
            "https://pubmed.ncbi.nlm.nih.gov/23443255/; https://pubmed.ncbi.nlm.nih.gov/34603448/",
        ),
        row(
            "acetylated_polyamine",
            "PA-3",
            "Bacterial biofilm metabolism contributes to the tissue acetylated-polyamine pool.",
            "Primary CRC work showed that antibiotic removal of colon biofilms lowered tissue N1,N12-diacetylspermine, implicating both host tumour and microbial structures.",
            "MTBLS13729 has no biofilm status, microbiome profile or spatial source measurement.",
            "Stratify by biofilm FISH/16S/metagenomics and repeat paired tissue quantification; use spatial co-localization or ex vivo antibiotic/biofilm perturbation when feasible.",
            "P2",
            "A microbial contribution is a credible competing source.",
            "The acetylated-polyamine pool is necessarily produced by cancer cells.",
            "https://pubmed.ncbi.nlm.nih.gov/25959674/",
        ),
        row(
            "long_chain_acylcarnitine",
            "AC-1",
            "Increased CPT1A-dependent fatty-acid import increases long-chain acylcarnitine formation and oxidation capacity.",
            "CRC mechanistic studies support CPT1A-dependent growth in specific cellular and dietary contexts; MTBLS13729 recovers a recurrent long-chain/C20:4-acylcarnitine-like ion.",
            "Static accumulation cannot show oxidation completion, and patient-level acylcarnitine coordination with other local modules is absent.",
            "Confirm the C20:4/C16:0/C18:0/C18:1 panel with isotope standards; measure palmitate-derived isotopologues, oxygen consumption and CPT1A perturbation with rescue.",
            "P0 identity, P2 flux",
            "Long-chain acylcarnitine accumulation is compatible with increased mitochondrial fatty-acid entry.",
            "FAO is activated.",
            "https://www.nature.com/articles/s41388-026-03835-4; https://www.nature.com/articles/s41419-020-02936-6",
        ),
        row(
            "long_chain_acylcarnitine",
            "AC-2",
            "Incomplete beta-oxidation or a CACT/CPT2/downstream bottleneck causes long-chain acylcarnitine accumulation despite impaired utilization.",
            "External omics context includes lower FAO-related programs, while the local signal is an accumulated pool rather than a measured flux.",
            "No acyl-CoA, short/medium-chain product pattern, isotope fate, OCR or enzyme-activity measurement exists.",
            "Combine chain-length-resolved acylcarnitines with acyl-CoAs, TCA isotopologues and acid-soluble metabolites after U-13C16-palmitate; perturb CPT2/CACT and downstream beta-oxidation separately.",
            "P1 panel, P2 flux",
            "A utilization bottleneck is an equally plausible interpretation.",
            "Acylcarnitine abundance gives the direction of FAO flux.",
            "https://www.nature.com/articles/s41467-025-63243-2",
        ),
        row(
            "long_chain_acylcarnitine",
            "AC-3",
            "Substrate availability, necrosis or stromal/immune composition changes the long-chain lipid pool without a tumour-cell-autonomous shuttle change.",
            "Independent human data reproduce long-chain lipid remodeling but not the exact C20:4 acylcarnitine identity; free C20:4 itself is not consistently altered.",
            "Bulk tissue cannot assign the source compartment.",
            "Adjust for pathology tumour content and lineage composition; use spatial lipidomics or sorted-cell/organoid validation and paired plasma/tissue substrate measurements.",
            "P2",
            "The signal sits in a broader CRC long-chain-lipid remodeling context.",
            "The exact metabolite and mechanism independently replicate in another cohort.",
            "https://www.nature.com/articles/s41467-025-63243-2",
        ),
        row(
            "neu5ac_mucin",
            "SA-1",
            "Increased GNE/NANS synthesis and SLC35A1 transport expand the sialic precursor/donor pool in mucinous tumours.",
            "Feature 703 orthogonally recovers source-Level-1 Neu5Ac; TCGA lineage-sensitive analysis strongly supports mucinous-relative synthesis/transport and secretory-mucin programs.",
            "Free Neu5Ac does not report CMP-Neu5Ac, glycan incorporation, linkage or cell-surface abundance.",
            "Quantify Neu5Ac, CMP-Neu5Ac and ManNAc with standards; perform labeled ManNAc/Neu5Ac incorporation and glycoproteomic/glycomic readout.",
            "P0 identity/bridge, P1 destination",
            "Neu5Ac abundance is embedded in a mucinous-relative precursor/transport and secretory-mucin context.",
            "Global hypersialylation or a specific glycan linkage is established.",
            "https://www.nature.com/articles/s41598-024-79893-z; https://www.nature.com/articles/s41598-022-26521-3",
        ),
        row(
            "neu5ac_mucin",
            "SA-2",
            "Selective sialyltransferase activity changes glycan linkage and carrier proteins without a proportional change in the free Neu5Ac pool.",
            "CRC glycoproteomics identifies extensive sialylated proteins, but the local mucin-sialylation transcript axis attenuates after broad-lineage adjustment.",
            "No linkage-aware glycomics, glycopeptide site occupancy, lectin signal or transferase activity was measured locally.",
            "Use linkage-aware N/O-glycomics, intact glycopeptides, MAL-II/SNA or equivalent orthogonal lectins, and transferase perturbation; relate these measurements to free Neu5Ac within the same samples.",
            "P1",
            "Selective glycan remodeling remains plausible but unmeasured.",
            "Free Neu5Ac is a direct surrogate of tissue hypersialylation.",
            "https://www.nature.com/articles/s41598-024-79893-z",
        ),
        row(
            "neu5ac_mucin",
            "SA-3",
            "Sialidase-mediated degradation or lysosomal release raises free Neu5Ac while surface/glycoprotein sialylation is unchanged or reduced.",
            "The local free-pool increase coexists with heterogeneous external sialylation directions, which is compatible with turnover rather than uniform synthesis.",
            "NEU1-4 activity, lysosomal source and released glycan products were not measured.",
            "Measure sialidase activity, free versus conjugated Neu5Ac and released glycan products; perturb sialidases and repeat linkage-aware glycomics.",
            "P2",
            "Increased turnover/release is a live competing explanation.",
            "Free Neu5Ac increase means net glycan synthesis increased.",
            "https://www.nature.com/articles/s41598-024-79893-z",
        ),
        row(
            "proline_glutamate",
            "PG-1",
            "General CRC proline/P5C synthesis and redox adaptation increase proline and glutamate pools.",
            "Features 345/374 recover source-Level-1 identities; paired TCGA and external proteomics support a general CRC proline-synthesis program.",
            "Mucinous-versus-conventional lineage-adjusted proline synthesis is borderline lower, not higher; static pools do not reveal synthesis rate.",
            "Trace U-13C5-glutamine and U-13C5-proline into P5C/proline/glutamate and collagen; perturb ALDH18A1/PYCR1 and rescue with proline.",
            "P1 if wet work becomes possible",
            "Proline/glutamate abundance is consistent with a general CRC metabolic program, not a proven mucinous-specific one.",
            "Mucinous CRC specifically activates proline synthesis or flux.",
            "https://pubmed.ncbi.nlm.nih.gov/35130302/; https://www.nature.com/articles/s42255-024-01118-4",
        ),
        row(
            "proline_glutamate",
            "PG-2",
            "Matrix/stromal composition and collagen turnover contribute to the tissue proline pool.",
            "Collagen/proline context attenuates strongly after lineage-proxy adjustment and the spatial context contains a CAF/collagen compartment.",
            "Broad-lineage scores are proxies rather than measured fractions; the remaining effect can still be biological niche signal.",
            "Use pathology and spatial transcript/proteomic collagen mapping, hydroxyproline/collagen turnover assays and cell-type-resolved tracing.",
            "P2",
            "Matrix composition is a plausible contributor to the bulk proline signal.",
            "The proline pool is tumour-cell autonomous.",
            "https://www.nature.com/articles/s42255-024-01118-4",
        ),
    ]

    frame = pd.DataFrame(rows)
    if frame["hypothesis_id"].duplicated().any():
        raise RuntimeError("hypothesis IDs must be unique")
    if frame.isna().any().any() or (frame.astype(str).apply(lambda col: col.str.strip() == "")).any().any():
        raise RuntimeError("mechanism tree contains empty fields")

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "competing_mechanism_hypotheses_v1.csv"
    frame.to_csv(csv_path, index=False)

    md = [
        "# MTBLS13729 competing mechanism hypotheses v1",
        "",
        "Static abundance is the observation; every row below is a distinct causal generator that remains compatible with it.",
        "Passing an annotation gate does not select among these mechanisms.",
        "",
    ]
    for axis, group in frame.groupby("axis", sort=False):
        md.extend([f"## {axis}", ""])
        for item in group.itertuples(index=False):
            md.extend(
                [
                    f"### {item.hypothesis_id}: {item.causal_generator}",
                    "",
                    f"- Current support: {item.current_support}",
                    f"- Ambiguity: {item.current_counterevidence_or_ambiguity}",
                    f"- Decisive discriminator: {item.decisive_discriminator}",
                    f"- Claim now: {item.claim_now}",
                    f"- Forbidden: {item.forbidden_claim}",
                    f"- Primary sources: {item.primary_sources}",
                    "",
                ]
            )
    (OUT / "competing_mechanism_hypotheses_v1.md").write_text("\n".join(md), encoding="utf-8")

    report = {
        "status": "mtbls13729_competing_mechanism_trees_v1_complete",
        "formal": False,
        "axes": int(frame["axis"].nunique()),
        "hypotheses": int(len(frame)),
        "hypotheses_per_axis": {str(k): int(v) for k, v in frame.groupby("axis").size().items()},
        "highest_priority_identity_panels": [
            "m7G/m2G/Gm/m2,2G",
            "N1,N8-diacetylspermidine/N1,N12-diacetylspermine/monoacetylspermidines",
            "C20:4/C16:0/C18:0/C18:1 acylcarnitines",
        ],
        "minimum_causal_upgrades": {
            "modified_guanosine": "RNA-class-resolved isotope-dilution LC-MS plus turnover or METTL1 catalytic perturbation",
            "acetylated_polyamine": "positional-isomer standards plus host-versus-biofilm source test",
            "long_chain_acylcarnitine": "chain-resolved standards plus fatty-acid isotope fate and respiratory readout",
            "neu5ac_mucin": "free/donor-pool quantification plus linkage-aware glycomics",
            "proline_glutamate": "isotope tracing plus cell-type/matrix source resolution",
        },
        "decision": (
            "No abundance axis currently identifies a unique causal generator. The manuscript may present testable "
            "competing mechanisms, but exact identities and flux/enzyme claims require the listed discriminators."
        ),
        "provenance": {
            "ledger_sha256": sha256(LEDGER),
            "coordination_sha256": sha256(COORDINATION),
            "lineage_sensitivity_sha256": sha256(LINEAGE),
            "output_csv_sha256": sha256(csv_path),
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
