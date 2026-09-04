from __future__ import annotations

"""Audit broad annotated-feature candidates against the source-paper supplement.

This is deliberately separate from the eight-candidate frozen ledger.  It asks
whether a discovery-matrix signal is already represented by the authors' 345
UHPLC annotations and by their Rmu-vs-normal differential table.  It does not
upgrade an annotation or validate abundance.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUPPLEMENT = ROOT / "data/mtbls13729/source_paper_supplements/pr5c01260_si_005.xlsx"
PRIORITY = ROOT / "data/mtbls13729/full_annotated_feature_audit_v1/all_priority.csv"
CONSENSUS = ROOT / "data/mtbls13729/ms1_consensus"
OUT = ROOT / "data/mtbls13729/broad_candidate_novelty_audit_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_sheet(name: str) -> pd.DataFrame:
    frame = pd.read_excel(SUPPLEMENT, sheet_name=name, header=1)
    frame["m/z"] = pd.to_numeric(frame["m/z"], errors="coerce")
    frame["RT [min]"] = pd.to_numeric(frame["RT [min]"], errors="coerce")
    return frame.loc[frame["m/z"].notna()].copy()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(PRIORITY)
    master = read_sheet("metabolites")
    rmu = read_sheet("Rmu vs N-Rmu(p<0.05)")

    coordinate_parts: list[pd.DataFrame] = []
    for panel in ("neg_rp", "pos_rp"):
        targets = pd.read_csv(CONSENSUS / f"{panel}__requantification_targets.csv.gz")
        targets.insert(0, "panel", panel)
        coordinate_parts.append(targets[["panel", "feature_id", "mz", "rt_sec"]])
    coordinates = pd.concat(coordinate_parts, ignore_index=True)
    candidates = candidates.merge(
        coordinates,
        on=["panel", "feature_id"],
        how="left",
        validate="one_to_one",
    )
    if candidates[["mz", "rt_sec"]].isna().any().any():
        raise RuntimeError("candidate coordinates are incomplete")

    records: list[dict[str, object]] = []
    matched_rows: list[pd.DataFrame] = []
    for row in candidates.itertuples(index=False):
        mz = float(row.mz)
        rt_min = float(row.rt_sec) / 60.0
        ppm = (master["m/z"] - mz).abs() / mz * 1e6
        drt = (master["RT [min]"] - rt_min).abs()
        strict = master.loc[(ppm <= 10.0) & (drt <= 0.25)].copy()
        liberal = master.loc[(ppm <= 15.0) & (drt <= 1.5)].copy()
        name = str(row.best_name).strip().casefold()
        exact_name = master.loc[master["metabolites"].astype(str).str.strip().str.casefold().eq(name)].copy()
        exact_ik = master.loc[
            master["InChIKey"].astype(str).str.slice(0, 14).eq(str(row.best_ik14))
        ].copy()

        rmu_names = set(rmu["metabolites"].astype(str).str.strip().str.casefold())
        author_identity_present = not exact_name.empty or not exact_ik.empty
        author_identity_rmu_significant = bool(
            name in rmu_names
            or any(str(v).strip().casefold() in rmu_names for v in exact_ik["metabolites"])
        )

        if not strict.empty:
            table = strict.copy()
            table.insert(0, "candidate_panel", row.panel)
            table.insert(1, "candidate_feature_id", int(row.feature_id))
            table.insert(2, "candidate_name", str(row.best_name))
            matched_rows.append(table)

        closest_index = ppm.idxmin()
        records.append(
            {
                "panel": row.panel,
                "feature_id": int(row.feature_id),
                "dreams_name": str(row.best_name),
                "dreams_ik14": str(row.best_ik14),
                "mz": mz,
                "rt_min": rt_min,
                "screen_fdr10": bool(row.screen_fdr10),
                "author_exact_name_present": not exact_name.empty,
                "author_exact_ik14_present": not exact_ik.empty,
                "author_identity_present": author_identity_present,
                "author_identity_rmu_significant": author_identity_rmu_significant,
                "strict_mz_rt_match_count": int(len(strict)),
                "strict_mz_rt_names": " | ".join(sorted(strict["metabolites"].astype(str).unique())),
                "liberal_mz_rt_match_count": int(len(liberal)),
                "closest_author_ppm": float(ppm.loc[closest_index]),
                "closest_author_rt_delta_min": float(drt.loc[closest_index]),
                "closest_author_name": str(master.loc[closest_index, "metabolites"]),
                "novel_identity_candidate": bool(not author_identity_present and strict.empty),
                "novel_rmu_association_candidate": bool(not author_identity_rmu_significant),
            }
        )

    result = pd.DataFrame(records).sort_values(["screen_fdr10", "panel", "feature_id"], ascending=[False, True, True])
    result.to_csv(OUT / "candidate_original_paper_overlap.csv", index=False)
    master.to_csv(OUT / "author_uhplc_annotations_normalized.csv.gz", index=False, compression="gzip")
    if matched_rows:
        pd.concat(matched_rows, ignore_index=True).to_csv(
            OUT / "strict_coordinate_matches.csv", index=False
        )

    report = {
        "status": "mtbls13729_broad_candidate_novelty_audit_complete",
        "candidates": int(len(result)),
        "screen_fdr10": int(result["screen_fdr10"].sum()),
        "author_identity_present": int(result["author_identity_present"].sum()),
        "author_identity_rmu_significant": int(result["author_identity_rmu_significant"].sum()),
        "novel_identity_candidates": int(result["novel_identity_candidate"].sum()),
        "novel_rmu_association_candidates": int(result["novel_rmu_association_candidate"].sum()),
        "claim_limit": (
            "Overlap with the authors' table does not prove identity equivalence across chromatography. "
            "Absence from the table does not prove a novel metabolite. Abundance remains a discovery-matrix screen."
        ),
        "provenance": {
            "supplement_sha256": sha256(SUPPLEMENT),
            "priority_sha256": sha256(PRIORITY),
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(result.to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
