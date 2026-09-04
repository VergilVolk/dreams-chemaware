"""Audit what the MTBLS13729 source article actually claims for each biology axis.

This is deliberately a narrative audit, not an annotation benchmark.  It
separates a named metabolite in the article body from a pathway-only mention,
and separates both from identities that appear only in the supplementary
annotation table.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_HTML = ROOT / "data/mtbls13729/source_paper_supplements/fulltext.html"
DELTA = ROOT / "data/mtbls13729/original_paper_delta_v2/candidate_original_paper_delta_v2.csv"
OUT = ROOT / "data/mtbls13729/source_narrative_audit_v1"


AXES = [
    {
        "axis": "modified_guanosine_purine",
        "named_terms": [r"methylguanosine", r"dimethylguanosine"],
        "context_terms": [r"\bguanosine\b", r"\bguanine\b", r"purine metabolism"],
        "project_increment": (
            "source-table-absent modified-guanosine ion families with raw MS2, "
            "cross-adduct evidence and paired abundance"
        ),
        "claim_boundary": "modified-guanosine family, not a resolved positional isomer or RNA source",
    },
    {
        "axis": "acetylated_polyamine",
        "named_terms": [r"diacetylspermidine", r"acetylspermidine", r"acetylspermine"],
        "context_terms": [r"polyamine", r"spermidine", r"spermine"],
        "project_increment": (
            "a distinct positive-RP acetylated-polyamine feature with recurrent MS2 and "
            "cross-chromatography concordance"
        ),
        "claim_boundary": "acetylated-polyamine family pending positional-isomer standards",
    },
    {
        "axis": "long_chain_acylcarnitine",
        "named_terms": [r"oleoyl[^<]{0,40}carnitine", r"stearyl[^<]{0,40}carnitine"],
        "context_terms": [r"carnitine shuttle", r"carnitine metabolism", r"carnitine metabolites"],
        "project_increment": (
            "a C20:4-like long-chain acylcarnitine anchor plus a correction of the "
            "source article's activated-shuttle interpretation to competing flux hypotheses"
        ),
        "claim_boundary": "carnitine-shuttle imbalance; no flux direction from pool size",
    },
    {
        "axis": "proline_p5c",
        "named_terms": [r"\bproline\b"],
        "context_terms": [r"arginine and proline metabolism"],
        "project_increment": (
            "orthogonal positive-RP recovery of source Level-1 proline and cross-cohort "
            "placement in a general CRC proline/P5C-matrix program"
        ),
        "claim_boundary": "general CRC abundance/context program, not mucinous-specific flux",
    },
    {
        "axis": "glutamate",
        "named_terms": [r"glutamic acid", r"\bglutamate\b"],
        "context_terms": [r"alanine, aspartate and glutamate metabolism"],
        "project_increment": "orthogonal positive-RP recovery of source Level-1 glutamate",
        "claim_boundary": "same-cohort technical recovery, not independent replication or flux",
    },
    {
        "axis": "neu5ac_mucin_glycan",
        "named_terms": [r"N[\s-]*Acetylneuraminic acid", r"sialic acid"],
        "context_terms": [r"glycosphingolipid", r"mucin-rich", r"sphingolipid metabolism"],
        "project_increment": (
            "orthogonal positive-RP recovery plus cross-cohort evidence that separates a "
            "free Neu5Ac pool from selective mucin-glycan remodeling"
        ),
        "claim_boundary": "not global hypersialylation, glycan linkage, source, or flux",
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_article_body(raw: str) -> str:
    article = raw.split('<section id="ref-list1"', 1)[0]
    article = re.sub(r"<script\b.*?</script>", " ", article, flags=re.I | re.S)
    article = re.sub(r"<style\b.*?</style>", " ", article, flags=re.I | re.S)
    article = re.sub(r"<[^>]+>", " ", article)
    article = html.unescape(article)
    return re.sub(r"\s+", " ", article).strip()


def matches(text: str, patterns: list[str]) -> list[str]:
    found: list[str] = []
    for pattern in patterns:
        found.extend(match.group(0) for match in re.finditer(pattern, text, flags=re.I))
    return found


def remove_patterns(text: str, patterns: list[str]) -> str:
    """Remove broader pathway phrases before testing named-metabolite mentions."""
    for pattern in patterns:
        text = re.sub(pattern, " ", text, flags=re.I)
    return text


def main() -> None:
    if not SOURCE_HTML.is_file() or not DELTA.is_file():
        raise FileNotFoundError("source article or original-paper delta v2 is missing")
    OUT.mkdir(parents=True, exist_ok=True)
    text = clean_article_body(SOURCE_HTML.read_text(encoding="utf-8", errors="replace"))
    delta = pd.read_csv(DELTA)

    rows = []
    for spec in AXES:
        context = matches(text, spec["context_terms"])
        named = matches(remove_patterns(text, spec["context_terms"]), spec["named_terms"])
        if named:
            status = "explicitly_named_in_article_body"
        elif context:
            status = "pathway_or_family_context_only"
        else:
            status = "not_discussed_in_article_body"
        rows.append(
            {
                "axis": spec["axis"],
                "source_narrative_status": status,
                "named_term_hits": len(named),
                "context_term_hits": len(context),
                "named_terms_observed": " | ".join(sorted(set(named), key=str.lower)),
                "context_terms_observed": " | ".join(sorted(set(context), key=str.lower)),
                "project_increment": spec["project_increment"],
                "claim_boundary": spec["claim_boundary"],
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "source_narrative_axis_audit.csv", index=False)

    counts = {str(k): int(v) for k, v in frame.source_narrative_status.value_counts().items()}
    report = {
        "status": "mtbls13729_source_narrative_audit_v1_complete",
        "formal": False,
        "article_title": (
            "Integrated Spatial and Bulk Untargeted Metabolomics Characterize "
            "Location- and Histology-Associated Metabolic Heterogeneity in Colorectal Cancer"
        ),
        "doi": "10.1021/acs.jproteome.5c01260",
        "axes": int(len(frame)),
        "source_narrative_status_counts": counts,
        "source_article_key_overclaims": [
            "static abundance described as altered sphingolipid metabolic flux",
            "Neu5Ac/sialic acid linked directly to glycosphingolipid formation without linkage-resolved glycomics",
            "carnitine accumulation interpreted as activated shuttle and beta-oxidation without tracing",
        ],
        "source_article_design_boundary": (
            "The article reports six representative samples plus pooled QC, whereas the public deposition "
            "used here contains 60 paired biological samples and no deposited pooled-QC/blank files. "
            "Therefore the reanalysis cannot reproduce the paper's QC drift workflow."
        ),
        "selected_candidate_delta_counts": {
            str(k): int(v) for k, v in delta.delta_type.value_counts().items()
        },
        "claim_limit": (
            "A body-text absence is narrative novelty, not proof that a molecule is absent from supplementary "
            "tables. A body-text presence does not validate the article's flux or causal interpretation."
        ),
        "provenance": {
            "source_html_sha256": sha256(SOURCE_HTML),
            "delta_v2_sha256": sha256(DELTA),
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# MTBLS13729 source narrative audit v1",
        "",
        "| Axis | Source narrative | Named hits | Context hits | Auditable project increment |",
        "|---|---|---:|---:|---|",
    ]
    for item in frame.itertuples(index=False):
        lines.append(
            f"| {item.axis} | {item.source_narrative_status} | {item.named_term_hits} | "
            f"{item.context_term_hits} | {item.project_increment} |"
        )
    lines.extend(
        [
            "",
            "The audit distinguishes narrative novelty from chemical novelty. Neu5Ac and carnitine are already "
            "central source-paper claims; our increment is evidence calibration and mechanism correction. "
            "Modified-guanosine and acetylated-polyamine families are stronger narrative increments, but exact "
            "positional identities still require standards.",
        ]
    )
    (OUT / "source_narrative_axis_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
