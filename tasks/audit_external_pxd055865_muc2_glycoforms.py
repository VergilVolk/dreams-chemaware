from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


SAMPLE_SHEETS = {
    "Colon1a": ["Colon1a Tumor"],
    "Colon1b": ["Colon1b Tumor"],
    "Colon2": ["Colon2 Tumor1", "Colon2 Tumor2"],
    "HealthyColon": ["Healthy Colon"],
}

COLON_PATIENT = {
    "Colon1a": "Patient1",
    "Colon1b": "Patient1",
    "Colon2": "Patient2",
    "HealthyColon": "HealthyDonor",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def glycan_tokens(sequence: str) -> list[str]:
    return re.findall(r"\[([^\]]+)\]", sequence)


def monosaccharide_count(tokens: list[str], symbol: str) -> int:
    total = 0
    for token in tokens:
        for match in re.finditer(rf"{re.escape(symbol)}(\d+)", token):
            total += int(match.group(1))
    return total


def classify_sequence(sequence: str) -> dict[str, object]:
    tokens = glycan_tokens(sequence)
    counts = {symbol: monosaccharide_count(tokens, symbol) for symbol in "HNAF"}
    token_text = "|".join(tokens)
    o_acetyl_neu5ac = bool(re.search(r"Ac(?:\d+)?A\d+", token_text))
    o_acetyl_galnac = "AcN" in token_text
    return {
        "glycan_tokens": token_text,
        "n_tokens": len(tokens),
        "hex": counts["H"],
        "hexnac": counts["N"],
        "neu5ac": counts["A"],
        "fucose": counts["F"],
        "sialylated": counts["A"] > 0,
        "fucosylated": counts["F"] > 0,
        "o_acetyl_neu5ac": o_acetyl_neu5ac,
        "o_acetyl_galnac": o_acetyl_galnac,
        "tn_only": bool(tokens)
        and counts["N"] > 0
        and counts["H"] == 0
        and counts["A"] == 0
        and counts["F"] == 0,
        "t_or_extended_unsialylated": counts["H"] > 0 and counts["A"] == 0,
        "unparsed_token": bool(re.search(r"[^HNAF0-9.+-]", token_text)),
    }


def load_sample(workbook: Path, sheets: list[str]) -> pd.DataFrame:
    frames = []
    for sheet in sheets:
        frame = pd.read_excel(workbook, sheet_name=sheet)
        frame["source_sheet"] = sheet
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    required = {"Sequence", "Protein Name", "Base Sequence", "Area"}
    missing = required - set(combined.columns)
    if missing:
        raise RuntimeError(f"missing required columns in {sheets}: {sorted(missing)}")
    combined = combined[combined["Sequence"].notna()].copy()
    combined["Sequence"] = combined["Sequence"].astype(str)
    combined["Protein Name"] = combined["Protein Name"].fillna("").astype(str)
    combined["Base Sequence"] = combined["Base Sequence"].fillna("").astype(str)
    combined["Area"] = pd.to_numeric(combined["Area"], errors="coerce")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glycopeptides",
        type=Path,
        default=Path(
            "data/external/PXD055865_2026_MUC2/"
            "41467_2026_72853_MOESM4_ESM.xlsx"
        ),
    )
    parser.add_argument(
        "--source-spectra",
        type=Path,
        default=Path(
            "data/external/PXD055865_2026_MUC2/"
            "41467_2026_72853_MOESM7_ESM.xlsx"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/external/PXD055865_2026_MUC2/audit_v1"),
    )
    args = parser.parse_args()

    for path in (args.glycopeptides, args.source_spectra):
        if not path.exists():
            raise FileNotFoundError(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    specimen_rows: list[dict[str, object]] = []
    presence_frames: list[pd.DataFrame] = []
    token_vocabulary: Counter[str] = Counter()

    for specimen, sheets in SAMPLE_SHEETS.items():
        frame = load_sample(args.glycopeptides, sheets)
        frame["specimen"] = specimen
        frame["patient"] = COLON_PATIENT[specimen]
        frame["muc2"] = frame["Protein Name"].str.lower().eq("mucin-2")

        classified = frame["Sequence"].map(classify_sequence).apply(pd.Series)
        frame = pd.concat([frame, classified], axis=1)
        for sequence in frame["Sequence"]:
            token_vocabulary.update(glycan_tokens(sequence))

        # A sequence/protein pair is the most conservative identification unit available
        # in Supplementary Data 2. Charge states and duplicate regions are collapsed.
        unique = frame.drop_duplicates(["Sequence", "Protein Name"]).copy()
        muc2 = unique[unique["muc2"]].copy()
        specimen_rows.append(
            {
                "specimen": specimen,
                "patient": COLON_PATIENT[specimen],
                "source_sheets": ";".join(sheets),
                "glycopeptide_rows": int(len(frame)),
                "unique_glycopeptide_protein_pairs": int(len(unique)),
                "unique_muc2_glycopeptides": int(len(muc2)),
                "unique_sialylated_muc2": int(muc2["sialylated"].sum()),
                "unique_fucosylated_muc2": int(muc2["fucosylated"].sum()),
                "unique_o_acetyl_neu5ac_muc2": int(
                    muc2["o_acetyl_neu5ac"].sum()
                ),
                "unique_o_acetyl_galnac_muc2": int(
                    muc2["o_acetyl_galnac"].sum()
                ),
                "unique_tn_only_muc2": int(muc2["tn_only"].sum()),
                "unique_unsialylated_hex_muc2": int(
                    muc2["t_or_extended_unsialylated"].sum()
                ),
                "sialylated_muc2_fraction": (
                    float(muc2["sialylated"].mean()) if len(muc2) else None
                ),
            }
        )
        keep = [
            "specimen",
            "patient",
            "Sequence",
            "Base Sequence",
            "Protein Name",
            "glycan_tokens",
            "hex",
            "hexnac",
            "neu5ac",
            "fucose",
            "sialylated",
            "fucosylated",
            "o_acetyl_neu5ac",
            "o_acetyl_galnac",
            "tn_only",
            "t_or_extended_unsialylated",
        ]
        presence_frames.append(unique[keep])

    specimen_summary = pd.DataFrame(specimen_rows)
    presence = pd.concat(presence_frames, ignore_index=True)
    muc2_presence = presence[presence["Protein Name"].str.lower().eq("mucin-2")].copy()

    pivot = (
        muc2_presence.assign(present=1)
        .pivot_table(
            index=[
                "Sequence",
                "Base Sequence",
                "glycan_tokens",
                "sialylated",
                "o_acetyl_neu5ac",
                "o_acetyl_galnac",
            ],
            columns="specimen",
            values="present",
            aggfunc="max",
            fill_value=0,
        )
        .reset_index()
    )
    for specimen in SAMPLE_SHEETS:
        if specimen not in pivot:
            pivot[specimen] = 0
    pivot["present_any_tumour"] = (
        pivot[["Colon1a", "Colon1b", "Colon2"]].max(axis=1).astype(int)
    )
    pivot["tumour_only_vs_healthy"] = (
        (pivot["present_any_tumour"] == 1) & (pivot["HealthyColon"] == 0)
    )
    pivot["healthy_only"] = (
        (pivot["present_any_tumour"] == 0) & (pivot["HealthyColon"] == 1)
    )

    source_book = pd.ExcelFile(args.source_spectra)
    source_sheet_names = source_book.sheet_names
    source_oacetyl = [name for name in source_sheet_names if name in {"Figure4A+S26A", "FigureS26B"}]

    tumour_unique = set(
        pivot.loc[pivot["present_any_tumour"].eq(1), "Sequence"].astype(str)
    )
    healthy_unique = set(
        pivot.loc[pivot["HealthyColon"].eq(1), "Sequence"].astype(str)
    )
    tumour_sialylated = set(
        pivot.loc[
            pivot["present_any_tumour"].eq(1) & pivot["sialylated"].eq(True),
            "Sequence",
        ].astype(str)
    )
    healthy_sialylated = set(
        pivot.loc[
            pivot["HealthyColon"].eq(1) & pivot["sialylated"].eq(True),
            "Sequence",
        ].astype(str)
    )
    tumour_oac_neu5ac = set(
        pivot.loc[
            pivot["present_any_tumour"].eq(1)
            & pivot["o_acetyl_neu5ac"].eq(True),
            "Sequence",
        ].astype(str)
    )
    healthy_oac_neu5ac = set(
        pivot.loc[
            pivot["HealthyColon"].eq(1)
            & pivot["o_acetyl_neu5ac"].eq(True),
            "Sequence",
        ].astype(str)
    )
    tumour_oac_galnac = set(
        pivot.loc[
            pivot["present_any_tumour"].eq(1)
            & pivot["o_acetyl_galnac"].eq(True),
            "Sequence",
        ].astype(str)
    )
    healthy_oac_galnac = set(
        pivot.loc[
            pivot["HealthyColon"].eq(1)
            & pivot["o_acetyl_galnac"].eq(True),
            "Sequence",
        ].astype(str)
    )

    report = {
        "status": "pxd055865_muc2_glycoform_audit_complete",
        "formal": False,
        "dataset": {
            "accession": "PXD055865",
            "paper_doi": "10.1038/s41467-026-72853-3",
            "paper_url": "https://www.nature.com/articles/s41467-026-72853-3",
            "proteomexchange_url": (
                "https://proteomecentral.proteomexchange.org/cgi/GetDataset?ID=PXD055865"
            ),
            "design": (
                "three colorectal mucinous carcinoma specimens from two patients; "
                "Colon1a and Colon1b are from one patient; one independent healthy colon"
            ),
        },
        "specimens": specimen_summary.to_dict(orient="records"),
        "collapsed_presence": {
            "unique_muc2_glycopeptides_any_tumour": len(tumour_unique),
            "unique_muc2_glycopeptides_healthy": len(healthy_unique),
            "shared_tumour_and_healthy": len(tumour_unique & healthy_unique),
            "tumour_only": len(tumour_unique - healthy_unique),
            "healthy_only": len(healthy_unique - tumour_unique),
            "sialylated_any_tumour": len(tumour_sialylated),
            "sialylated_healthy": len(healthy_sialylated),
            "sialylated_shared": len(tumour_sialylated & healthy_sialylated),
            "sialylated_tumour_only": len(tumour_sialylated - healthy_sialylated),
            "sialylated_healthy_only": len(healthy_sialylated - tumour_sialylated),
            "o_acetyl_neu5ac_any_tumour": len(tumour_oac_neu5ac),
            "o_acetyl_neu5ac_healthy": len(healthy_oac_neu5ac),
            "o_acetyl_galnac_any_tumour": len(tumour_oac_galnac),
            "o_acetyl_galnac_healthy": len(healthy_oac_galnac),
        },
        "source_spectrum_support": {
            "di_o_acetyl_neu5ac_sheets": source_oacetyl,
            "o_acetyl_galnac_sheet_present": "Figure4B+S27A" in source_sheet_names,
            "source_sheet_count": len(source_sheet_names),
        },
        "token_vocabulary": dict(token_vocabulary.most_common()),
        "interpretation": (
            "The public dataset supplies manually reviewed, carrier-resolved MUC2 "
            "glycopeptide evidence and spatial context. It supports structural destination "
            "heterogeneity but is not an abundance replication of free Neu5Ac in MTBLS13729."
        ),
        "claim_limit": (
            "Only two mucinous colorectal cancer patients and one independent healthy colon "
            "are represented. Identification counts are discovery coverage, not unbiased "
            "abundance. Colon1a and Colon1b are not independent patients. No statistical "
            "tumour-versus-normal population claim is permitted."
        ),
        "provenance": {
            "glycopeptides_sha256": sha256(args.glycopeptides),
            "source_spectra_sha256": sha256(args.source_spectra),
        },
    }

    specimen_summary.to_csv(args.output_dir / "specimen_summary.csv", index=False)
    pivot.to_csv(args.output_dir / "muc2_glycopeptide_presence.csv", index=False)
    with (args.output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)

    readme = f"""# PXD055865 MUC2 glycoform audit

## Scope

This audit reads the authors' Supplementary Data 2 and source-spectrum workbook
for the 2026 MUC2 spatial glycopeptide study. It collapses duplicate charge states
to unique sequence/protein pairs and keeps Colon1a and Colon1b assigned to the
same patient.

## Key counts

- Unique MUC2 glycopeptides in any mucinous colorectal tumour: {len(tumour_unique)}
- Unique MUC2 glycopeptides in the independent healthy colon: {len(healthy_unique)}
- Shared: {len(tumour_unique & healthy_unique)}
- Tumour-only: {len(tumour_unique - healthy_unique)}
- Healthy-only: {len(healthy_unique - tumour_unique)}
- Sialylated tumour / healthy unique sequences: {len(tumour_sialylated)} / {len(healthy_sialylated)}
- O-acetyl-Neu5Ac tumour / healthy unique sequences: {len(tumour_oac_neu5ac)} / {len(healthy_oac_neu5ac)}
- O-acetyl-GalNAc tumour / healthy unique sequences: {len(tumour_oac_galnac)} / {len(healthy_oac_galnac)}
- Source-spectrum sheets for di-O-acetyl-Neu5Ac: {', '.join(source_oacetyl)}

## Permitted use

This is strong external structural and spatial context for the destination arm of
the MTBLS13729 free-Neu5Ac finding. It is not an independent patient-level
metabolite abundance replication and does not establish flux.

## Critical limitation

There are two independent mucinous colorectal cancer patients, because Colon1a
and Colon1b are two resections from the same patient, plus one independent healthy
colon. Identification coverage and signal-dependent MS/MS selection prevent a
population-level tumour-versus-normal statistical claim.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
