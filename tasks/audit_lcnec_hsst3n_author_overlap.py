"""Match phenotype-blind qualified LCNEC HSST3n MS2 families to Table S2.

The output separates author-reported chemistry from acquisition-qualified
families not represented in the published 1,052-metabolite atlas.  Unmatched
does not mean a novel metabolite; it only defines the dark-feature headroom for
annotation and paired phenotype testing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--families",
        type=Path,
        default=Path("data/validation/lcnec_hsst3n_qc_headroom_gate/precursor_family_ledger.csv"),
    )
    parser.add_argument(
        "--supplement",
        type=Path,
        default=Path("data/validation/lcnec_zenodo19005638_preflight/article_mmc7.xlsx"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/validation/lcnec_hsst3n_author_overlap_gate")
    )
    parser.add_argument("--ppm", type=float, default=5.0)
    parser.add_argument("--rt-sec", type=float, default=15.0)
    args = parser.parse_args()

    with args.families.open("r", encoding="utf-8", newline="") as handle:
        families = [row for row in csv.DictReader(handle) if row["passes_all"].lower() == "true"]
    published_all = pd.read_excel(args.supplement, sheet_name="Table S2", header=3).iloc[:, 1:]
    published = published_all.copy()
    required = {"Metabolite", "RT (min)", "m/z", "Platform", "MSI Level"}
    if not required.issubset(published.columns):
        raise RuntimeError(f"unexpected Table S2 columns: {published.columns.tolist()}")
    published = published[published["Platform"].astype(str).eq("HSST3n")].copy()
    published["m/z"] = pd.to_numeric(published["m/z"], errors="coerce")
    published["rt_sec"] = pd.to_numeric(published["RT (min)"], errors="coerce") * 60.0
    published = published[published["m/z"].notna() & published["rt_sec"].notna()]

    output_rows: list[dict[str, object]] = []
    for family in families:
        mz = float(family["mz_median"])
        rt = float(family["rt_median_sec"])
        candidates: list[tuple[float, float, pd.Series]] = []
        for _, author in published.iterrows():
            ppm = abs(float(author["m/z"]) - mz) / mz * 1e6
            rt_delta = abs(float(author["rt_sec"]) - rt)
            if ppm <= args.ppm and rt_delta <= args.rt_sec:
                candidates.append((ppm, rt_delta, author))
        candidates.sort(key=lambda value: (value[0], value[1]))
        best = candidates[0] if candidates else None
        output_rows.append(
            {
                **family,
                "author_matched": best is not None,
                "author_match_count": len(candidates),
                "author_metabolite": best[2]["Metabolite"] if best else "",
                "author_msi_level": best[2]["MSI Level"] if best else "",
                "author_mz": best[2]["m/z"] if best else "",
                "author_rt_sec": best[2]["rt_sec"] if best else "",
                "match_ppm": best[0] if best else "",
                "match_rt_delta_sec": best[1] if best else "",
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = args.output_dir / "qualified_family_author_overlap.csv"
    with ledger_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    matched = [row for row in output_rows if row["author_matched"]]
    unmatched = [row for row in output_rows if not row["author_matched"]]
    report = {
        "status": "lcnec_hsst3n_author_overlap_complete",
        "formal": True,
        "published_table_valid_rows_all_platforms": int(
            (published_all["Metabolite"].notna() & published_all["Platform"].notna()).sum()
        ),
        "paper_declared_metabolites": 1052,
        "published_hsst3n_rows": int(len(published)),
        "qualified_qc_families": len(output_rows),
        "matched_to_published_hsst3n": len(matched),
        "unmatched_acquisition_qualified_families": len(unmatched),
        "unmatched_fraction": len(unmatched) / len(output_rows) if output_rows else math.nan,
        "gates": {
            "unmatched_families_ge_100": len(unmatched) >= 100,
            "unmatched_families_ge_150": len(unmatched) >= 150,
        },
        "pass_to_phenotype_blind_annotation_and_paired_screen": len(unmatched) >= 100,
        "parameters": {"ppm": args.ppm, "rt_sec": args.rt_sec},
        "provenance": {"families_sha256": sha256(args.families), "supplement_sha256": sha256(args.supplement)},
        "claim_limit": "Unmatched means absent from the published HSST3n m/z-RT ledger, not a novel metabolite or disease association.",
    }
    (args.output_dir / "author_overlap_gate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
