from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUPPLEMENT = ROOT / "data/mtbls13729/source_paper_supplements/pr5c01260_si_005.xlsx"
ARTICLE_HTML = ROOT / "data/mtbls13729/source_paper_supplements/fulltext.html"
CANDIDATE_LEDGER = (
    ROOT
    / "data/mtbls13729/candidate_evidence_ledger_v1/candidate_evidence_ledger.csv"
)
OUT = ROOT / "data/mtbls13729/original_vs_dreams_biology_delta_v1"


DIFFERENTIAL_SHEETS = {
    "ltu_vs_rtu": "Ltu vs Rtu(p<0.05)",
    "rtu_vs_rmu": "Rtu vs Rmu(p<0.05)",
    "ltu_vs_normal": "Ltu vs N-Ltu(p<0.05)",
    "rtu_vs_normal": "Rtu vs N-Rtu(p<0.05)",
    "rmu_vs_normal": "Rmu vs N-Rmu(p<0.05)",
    "cancer_vs_normal": "cancer\xa0vs normal(p<0.05)",
}


FAMILY_RULES = {
    1597: {
        "author_identity_regex": r"methylguanosine",
        "author_context_regex": r"guanosine|methylguanine|purine",
        "novelty_class": "new modified-guanosine ion family; not a new purine pathway",
    },
    1717: {
        "author_identity_regex": r"N1,N8-Diacetylspermidine",
        "author_context_regex": r"spermidine|polyamine",
        "novelty_class": (
            "author name present in a different HILIC feature; new positive-RP "
            "chromatographic/isomer-family Rmu signal"
        ),
    },
    3019: {
        "author_identity_regex": r"dimethylguanosine",
        "author_context_regex": r"guanosine|methylguanine|purine",
        "novelty_class": "new dimethylguanosine ion family; positional identity unresolved",
    },
    3180: {
        "author_identity_regex": r"chlorinated/exogenous-like feature",
        "author_context_regex": r"chloral|chlor",
        "novelty_class": "negative biological-plausibility control, not a mechanism result",
    },
    3222: {
        "author_identity_regex": r"arachidonoylcarnitine|C20:4.*carnitine",
        "author_context_regex": r"carnitine",
        "novelty_class": (
            "new long-chain/C20:4-like class anchor within the author's carnitine program"
        ),
    },
    4966: {
        "author_identity_regex": r"preQ1|queuosine|C7H9N5O",
        "author_context_regex": r"purine|guanine|guanosine",
        "novelty_class": "new unresolved C7H9N5O purine-like ion family",
    },
    7489: {
        "author_identity_regex": r"methylguanosine",
        "author_context_regex": r"guanosine|methylguanine|purine",
        "novelty_class": "supporting sodium adduct of the new methylguanosine ion family",
    },
    16425: {
        "author_identity_regex": r"feature 16425",
        "author_context_regex": r"LysoPE|lysophosphatidylethanolamine",
        "novelty_class": (
            "new reproducible but unresolved lipid feature within an author-covered LysoPE context"
        ),
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_sheet(sheet: str) -> pd.DataFrame:
    frame = pd.read_excel(SUPPLEMENT, sheet_name=sheet, header=1)
    frame["m/z"] = pd.to_numeric(frame["m/z"], errors="coerce")
    frame["RT [min]"] = pd.to_numeric(frame["RT [min]"], errors="coerce")
    return frame.loc[frame["m/z"].notna()].copy()


def matching_rows(frame: pd.DataFrame, regex: str) -> pd.DataFrame:
    return frame.loc[
        frame["metabolites"].astype(str).str.contains(regex, case=False, regex=True, na=False)
    ].copy()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(CANDIDATE_LEDGER)
    master = read_sheet("metabolites")
    differential = {key: read_sheet(sheet) for key, sheet in DIFFERENTIAL_SHEETS.items()}
    article_text = ARTICLE_HTML.read_text(encoding="utf-8", errors="ignore")
    article_text = re.sub(r"<[^>]+>", " ", article_text)
    article_text = re.sub(r"\s+", " ", article_text)

    rows: list[dict[str, object]] = []
    author_identity_rows: list[pd.DataFrame] = []
    for candidate in candidates.itertuples(index=False):
        feature_id = int(candidate.feature_id)
        rule = FAMILY_RULES[feature_id]
        identity_hits = matching_rows(master, rule["author_identity_regex"])
        context_hits = matching_rows(master, rule["author_context_regex"])

        ppm = (master["m/z"] - float(candidate.mz)).abs() / float(candidate.mz) * 1e6
        drt = (master["RT [min]"] - float(candidate.rt_sec) / 60.0).abs()
        strict_hits = master.loc[(ppm <= 10.0) & (drt <= 0.25)].copy()
        liberal_hits = master.loc[(ppm <= 15.0) & (drt <= 1.5)].copy()

        if not identity_hits.empty:
            annotated = identity_hits.copy()
            annotated.insert(0, "candidate_feature_id", feature_id)
            author_identity_rows.append(annotated)

        record: dict[str, object] = {
            "feature_id": feature_id,
            "dreams_identity": candidate.defensible_identity,
            "dreams_mz": float(candidate.mz),
            "dreams_rt_min": float(candidate.rt_sec) / 60.0,
            "author_exact_name_present": not identity_hits.empty,
            "author_identity_names": " | ".join(sorted(identity_hits["metabolites"].astype(str).unique())),
            "author_identity_types": " | ".join(sorted(identity_hits["Type"].astype(str).unique())),
            "author_identity_levels": " | ".join(
                sorted(identity_hits["MSI(Metabolomics Standards Initiative)"].astype(str).unique())
            ),
            "author_context_entries": int(len(context_hits)),
            "strict_mz_rt_match_count": int(len(strict_hits)),
            "liberal_mz_rt_match_count": int(len(liberal_hits)),
            "closest_author_ppm": float(ppm.min()),
            "closest_author_rt_min": float(master.loc[ppm.idxmin(), "RT [min]"]),
            "closest_author_name": str(master.loc[ppm.idxmin(), "metabolites"]),
            "main_text_exact_identity_mentioned": bool(
                re.search(rule["author_identity_regex"], article_text, flags=re.IGNORECASE)
            ),
            "main_text_context_mentioned": bool(
                re.search(rule["author_context_regex"], article_text, flags=re.IGNORECASE)
            ),
            "novelty_class": rule["novelty_class"],
            "claim_boundary": candidate.claim_boundary,
        }

        for comparison, sheet in differential.items():
            if identity_hits.empty:
                significant = False
            else:
                significant = False
                for _, author_row in identity_hits.iterrows():
                    same_name = sheet["metabolites"].astype(str).str.casefold().eq(
                        str(author_row["metabolites"]).casefold()
                    )
                    same_mz = (sheet["m/z"] - float(author_row["m/z"])).abs() <= 1e-5
                    same_rt = (
                        sheet["RT [min]"] - float(author_row["RT [min]"])
                    ).abs() <= 1e-5
                    if bool((same_name & same_mz & same_rt).any()):
                        significant = True
                        break
            record[f"author_identity_significant__{comparison}"] = significant
        rows.append(record)

    delta = pd.DataFrame(rows)
    delta.to_csv(OUT / "candidate_original_paper_delta.csv", index=False)

    if author_identity_rows:
        pd.concat(author_identity_rows, ignore_index=True).to_csv(
            OUT / "author_exact_identity_rows.csv", index=False
        )

    context_patterns = {
        "carnitine": r"carnitine",
        "purine_nucleoside": r"guanosine|guanine|purine|adenosine|inosine",
        "polyamine": r"spermidine|spermine|putrescine",
        "lysope": r"LysoPE|lysophosphatidylethanolamine",
    }
    context_tables: list[pd.DataFrame] = []
    for family, pattern in context_patterns.items():
        table = matching_rows(master, pattern)
        table.insert(0, "author_context_family", family)
        for comparison, sheet in differential.items():
            composite = set(
                zip(
                    sheet["metabolites"].astype(str).str.casefold(),
                    sheet["m/z"].round(5),
                    sheet["RT [min]"].round(5),
                )
            )
            table[f"significant__{comparison}"] = [
                (str(name).casefold(), round(float(mz), 5), round(float(rt), 5)) in composite
                for name, mz, rt in zip(table["metabolites"], table["m/z"], table["RT [min]"])
            ]
        context_tables.append(table)
    author_context = pd.concat(context_tables, ignore_index=True)
    author_context.to_csv(OUT / "author_pathway_context_rows.csv", index=False)

    comparison_rows: list[pd.DataFrame] = []
    for comparison, sheet in differential.items():
        table = sheet.loc[
            sheet["metabolites"].astype(str).str.contains(
                "carnitine|guanosine|guanine|spermidine|spermine|putrescine",
                case=False,
                regex=True,
                na=False,
            )
        ].copy()
        table.insert(0, "comparison", comparison)
        comparison_rows.append(table)
    pd.concat(comparison_rows, ignore_index=True).to_csv(
        OUT / "author_relevant_differential_rows.csv", index=False
    )

    level_counts = (
        master["MSI(Metabolomics Standards Initiative)"].value_counts(dropna=False).to_dict()
    )
    report = {
        "status": "mtbls13729_original_vs_dreams_biology_delta_complete",
        "author_uhplc_annotations": int(len(master)),
        "author_annotation_levels": {str(k): int(v) for k, v in level_counts.items()},
        "author_differential_counts": {
            key: int(len(frame)) for key, frame in differential.items()
        },
        "dreams_candidates": int(len(delta)),
        "author_context_counts": {
            family: int((author_context["author_context_family"] == family).sum())
            for family in context_patterns
        },
        "strict_mz_rt_matches": int((delta["strict_mz_rt_match_count"] > 0).sum()),
        "author_exact_names_present": int(delta["author_exact_name_present"].sum()),
        "main_text_exact_identities_mentioned": int(
            delta["main_text_exact_identity_mentioned"].sum()
        ),
        "key_findings": [
            (
                "The author table already contains N1,N8-diacetylspermidine as a Level 2 HILIC "
                "annotation, but the DreaMS feature 1717 is a distinct positive-RP chromatographic "
                "feature and the author identity is not significant in the author's Rmu-vs-normal sheet."
            ),
            (
                "The author table covers guanosine, methylguanine and purine metabolism, but not the "
                "methylguanosine/dimethylguanosine ion families recovered by DreaMS."
            ),
            (
                "The author paper already establishes a broad carnitine program. Feature 3222 can only "
                "be claimed as an additional long-chain/C20:4-like anchor and a refinement of pathway "
                "interpretation, not discovery of carnitine metabolism itself."
            ),
        ],
        "claim_limit": (
            "This audit establishes annotation and discussion overlap. It does not upgrade DreaMS "
            "candidate identities or prove that a chromatographically distinct feature is a new molecule."
        ),
        "provenance": {
            "supplement": str(SUPPLEMENT.relative_to(ROOT)).replace("\\", "/"),
            "supplement_sha256": sha256(SUPPLEMENT),
            "candidate_ledger": str(CANDIDATE_LEDGER.relative_to(ROOT)).replace("\\", "/"),
            "candidate_ledger_sha256": sha256(CANDIDATE_LEDGER),
            "article_html": str(ARTICLE_HTML.relative_to(ROOT)).replace("\\", "/"),
            "article_html_sha256": sha256(ARTICLE_HTML),
        },
    }
    (OUT / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
