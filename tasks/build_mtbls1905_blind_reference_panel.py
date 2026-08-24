"""Build an auditable molecular-connectivity truth panel for MTBLS1905.

Published names are mapped to MassBank InChIKeys only when normalised names
yield one molecular connectivity (InChIKey14).  Ambiguous names are explicitly
excluded rather than guessed.  A target is evaluable only if that connectivity
also occurs in the fixed, mass-constrained reference candidate set.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def normalise(value: object) -> str:
    value = str(value).lower()
    value = value.replace("(l-)", "").replace("(l)", "")
    return re.sub(r"[^a-z0-9]", "", value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-matches", type=Path, default=Path("data/external/MTBLS1905/qc_ms2/audit/published_target_qc_ms2_matches.tsv"))
    parser.add_argument("--massbank", type=Path, default=Path("data/massbank/massbank_202406_msms.csv"))
    parser.add_argument("--candidate-manifest", type=Path, default=Path("data/external/MTBLS1905/reference/hnscc_positive_reference_dreams/manifest.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/external/MTBLS1905/reference/blind_connectivity_panel.tsv"))
    args = parser.parse_args()
    targets = pd.read_csv(args.target_matches, sep="\t")
    targets = targets[targets["n_qc_ms2_hits"] > 0].copy()
    target_names = targets["metabolite"].drop_duplicates().tolist()
    massbank = pd.read_csv(args.massbank, usecols=["Name", "InChIKey"])
    massbank = massbank.dropna(subset=["Name", "InChIKey"])
    massbank["norm_name"] = massbank["Name"].map(normalise)
    candidate = pd.read_csv(args.candidate_manifest)
    available = set(candidate["inchikey"].dropna().str.split("-").str[0])
    records: list[dict] = []
    for name in target_names:
        matches = massbank[massbank["norm_name"].eq(normalise(name))]
        ik14 = sorted(set(matches["InChIKey"].str.split("-").str[0]))
        record = {
            "metabolite": name,
            "n_massbank_name_records": int(len(matches)),
            "candidate_ik14": ";".join(ik14),
            "n_connectivities": len(ik14),
        }
        if len(ik14) == 1 and ik14[0] in available:
            record["panel_status"] = "evaluable"
            record["truth_ik14"] = ik14[0]
        elif len(ik14) == 0:
            record["panel_status"] = "unresolved_name"
            record["truth_ik14"] = ""
        elif len(ik14) > 1:
            record["panel_status"] = "ambiguous_connectivity"
            record["truth_ik14"] = ""
        else:
            record["panel_status"] = "truth_absent_from_reference"
            record["truth_ik14"] = ik14[0]
        records.append(record)
    panel = pd.DataFrame(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.out, sep="\t", index=False)
    report = {
        "direct_qc_ms2_published_targets": len(records),
        "evaluable_connectivity_targets": int((panel.panel_status == "evaluable").sum()),
        "excluded": panel.panel_status.value_counts().to_dict(),
        "truth_definition": "unique MassBank InChIKey14 from exact normalised published name; must exist in fixed mass-constrained candidate reference",
    }
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
