"""Freeze a claim-aware external literature context ledger for MTBLS13729.

The ledger separates direct metabolite replication, disease-context support,
fragmentation standards, glycan destination evidence and adversarial evidence.
It never promotes a local candidate identity from literature context alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mtbls13729/external_biology_context_v2"


ROWS = [
    {
        "axis": "acetylated_polyamine",
        "citation": "Umemori et al., Clin Chim Acta 2010, PMID 20655890",
        "url": "https://pubmed.ncbi.nlm.nih.gov/20655890/",
        "design": "urine; 33 colorectal-cancer patients; ELISA",
        "external_result": "urinary N1,N8-diacetylspermidine sensitivity 36.3% in CRC",
        "relationship_to_local": "disease-context support",
        "direction": "compatible",
        "identity_value": "none for the local ion",
        "hard_limit": "different biospecimen and assay; no tissue, mucinous-subtype or positional-identity replication",
    },
    {
        "axis": "acetylated_polyamine",
        "citation": "Park et al., Cancers 2018, PMCID PMC5877617",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5877617/",
        "design": "targeted urinary polyamine LC-MS/MS panel",
        "external_result": "authentic-standard transition N1,N8-diacetylspermidine 230.2 to 100.0",
        "relationship_to_local": "fragmentation-standard consistency",
        "direction": "not applicable",
        "identity_value": "strong diagnostic-transition support",
        "hard_limit": "no same-method retention time, full-spectrum match or spike-in coelution",
    },
    {
        "axis": "modified_guanosine",
        "citation": "Bever et al., JNCI 2024, 116:1126-1137",
        "url": "https://academic.oup.com/jnci/article/116/7/1126/7617725",
        "design": "684 incident CRC cases and 684 matched controls in NHS/HPFS; plasma LC-MS/MS",
        "external_result": "1-methylguanosine and N2,N2-dimethylguanosine associated with prevalent and incident CRC, including cases diagnosed more than 5 years later",
        "relationship_to_local": "large-cohort disease-context support",
        "direction": "compatible at CRC-context level",
        "identity_value": "none for local positional isomers",
        "hard_limit": "plasma risk study, sex heterogeneity and no mucinous tissue comparison",
    },
    {
        "axis": "modified_guanosine",
        "citation": "Targeted nucleoside study 2023, PMCID PMC10334214",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10334214/",
        "design": "serum; 51 controls, 37 adenoma and 55 CRC; targeted LC-MS/MS",
        "external_result": "N2-methylguanosine and 2-O-methylguanosine were lower in CRC",
        "relationship_to_local": "adversarial disease-context evidence",
        "direction": "opposite/identity-dependent",
        "identity_value": "supports the need to resolve positional isomers",
        "hard_limit": "serum differs from paired tissue and the local ions are unresolved isomer families",
    },
    {
        "axis": "modified_guanosine",
        "citation": "Targeted serum progression study 2015, PMID 26342311",
        "url": "https://pubmed.ncbi.nlm.nih.gov/26342311/",
        "design": "serial serum samples from 20 CRC patients; targeted LC-MS/MS",
        "external_result": "1-methylguanosine and N2,N2-dimethylguanosine contributed to a five-metabolite progression-monitoring panel",
        "relationship_to_local": "disease-monitoring context",
        "direction": "context compatible",
        "identity_value": "none for local positional isomers",
        "hard_limit": "small monitoring cohort; model-level result rather than isolated metabolite replication",
    },
    {
        "axis": "neu5ac_sialylation",
        "citation": "The Colorectal Cancer Glycocode 2026, PMID 42117844",
        "url": "https://pubmed.ncbi.nlm.nih.gov/42117844/",
        "design": "integrated transcriptomics across TCGA, Sidra-LUMC and CPTAC-2; n=988",
        "external_result": "high sialylome activity enriched for mucinous histology, MSI and BRAF-mutant CRC",
        "relationship_to_local": "independent subtype-context support",
        "direction": "compatible",
        "identity_value": "none for free Neu5Ac abundance",
        "hard_limit": "transcriptional score is not glycan structure, free-metabolite abundance or biochemical flux",
    },
    {
        "axis": "neu5ac_sialylation",
        "citation": "Reid et al., Histopathology 2002, PMID 11903571",
        "url": "https://pubmed.ncbi.nlm.nih.gov/11903571/",
        "design": "histochemistry in 64 adenocarcinomas including 23 colorectal tumours",
        "external_result": "all mucinous adenocarcinomas contained C9-O-acylated sialic-acid forms",
        "relationship_to_local": "glycan-destination context",
        "direction": "compatible with mucin remodeling",
        "identity_value": "none for free Neu5Ac abundance",
        "hard_limit": "histochemical glycan-bound sialic acid does not replicate the free-pool measurement",
    },
    {
        "axis": "neu5ac_sialylation",
        "citation": "Corfield et al., Glycoconj J 1999, PMID 10579699",
        "url": "https://pubmed.ncbi.nlm.nih.gov/10579699/",
        "design": "human colonic mucins and adenoma-carcinoma cell-line sequence",
        "external_result": "reduced sialic-acid O-acetylation in colorectal cancer and malignant mucinous cell lines",
        "relationship_to_local": "adversarial/decoupling context",
        "direction": "supports destination remodeling rather than uniform hypersialylation",
        "identity_value": "none for free Neu5Ac abundance",
        "hard_limit": "older biochemical context; no matched free Neu5Ac measurement",
    },
    {
        "axis": "neu5ac_sialylation",
        "citation": "MUC2 glycosite mapping and MSI imaging 2026, PMCID PMC13357551",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC13357551/",
        "design": "on-slide mucinase digestion, LC-MS and mass-spectrometry imaging",
        "external_result": "colon cancers showed low mono- and di-O-acetylated Neu5Ac-containing glycans relative to healthy colon",
        "relationship_to_local": "glycan-destination and acetylation context",
        "direction": "supports pool-to-destination decoupling",
        "identity_value": "none for free Neu5Ac abundance",
        "hard_limit": "small tissue exploration and not subtype-resolved free-metabolite replication",
    },
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ledger = pd.DataFrame(ROWS)
    if ledger.url.duplicated().any() or ledger.isna().any().any():
        raise RuntimeError("external evidence ledger has duplicate URLs or missing values")
    ledger.to_csv(OUT / "external_biology_context_ledger.csv", index=False)

    report = {
        "status": "mtbls13729_external_biology_context_v2_complete",
        "formal": False,
        "sources": int(len(ledger)),
        "axes": int(ledger.axis.nunique()),
        "axis_counts": {str(k): int(v) for k, v in ledger.axis.value_counts().items()},
        "decisions": {
            "neu5ac": (
                "External transcriptomic and glycan studies strengthen the mucinous-sialylation-remodeling context. "
                "They do not replicate the local free-Neu5Ac abundance result; the defensible novelty remains pool-to-destination decoupling."
            ),
            "modified_guanosine": (
                "Large prospective and targeted CRC studies establish modified guanosines as a real CRC-relevant axis, "
                "while opposing serum directions show that positional identity and tissue context matter. The local result is a family module, not a new CRC biomarker."
            ),
            "acetylated_polyamine": (
                "N1,N8-diacetylspermidine is already a reported CRC-associated urinary analyte. Local novelty can only be "
                "the paired tissue/family context and algorithm-enabled recovery, not the existence of a cancer association."
            ),
        },
        "manuscript_effect": (
            "The literature search upgrades external biological context for all three axes but does not upgrade any local structural identity. "
            "It strengthens a multi-axis CRC remodeling narrative and narrows novelty claims."
        ),
        "claim_limit": "Literature context is not an independent reanalysis, exact-identity validation, free-metabolite replication, subtype replication, flux or causality.",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    text = """# MTBLS13729 external biology context v2

## What changed

The external literature materially strengthens biological relevance but **does not validate local identities**.

- The acetylated-polyamine signal is not a previously unknown cancer metabolite: urinary N1,N8-diacetylspermidine has already been studied in CRC. Our defensible contribution is algorithm-enabled recovery of a strong paired-tissue family signal with raw-MS2 and cross-chromatography support.
- Modified guanosines have unusually strong external CRC evidence, including a 684-case prospective plasma study. However, a targeted serum study reported some methylguanosines in the opposite direction. That conflict supports our family- and context-specific interpretation rather than an exact universal biomarker claim.
- The Neu5Ac result now has independent mucinous/sialylome and glycan-O-acetylation context. The external data distinguish glycan-bound destination remodeling from the local free pool and therefore strengthen the **pool-to-destination decoupling** framing, not a blanket hypersialylation claim.

## Remaining decisive validation

1. Same-method standard/spike-in and linkage-aware glycan readout for Neu5Ac.
2. N1,N8-diacetylspermidine positional-isomer standards.
3. Methyl- and dimethylguanosine positional-isomer standards.
4. Independent subtype-resolved tissue metabolomics; none of the cited studies supplies it.
"""
    (OUT / "REPORT.md").write_text(text, encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
