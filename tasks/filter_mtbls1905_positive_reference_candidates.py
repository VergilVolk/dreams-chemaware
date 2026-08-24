"""Create a mass-constrained positive reference set for the fixed HNSCC panel.

No target structure is injected into the model.  This merely mirrors a
realistic precursor-mass candidate restriction before DreaMS spectral ranking.
The original source blocks are copied verbatim and a mapping file records which
published target m/z windows each reference spectrum belongs to.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, default=Path("data/external/MTBLS1905/qc_ms2/audit/published_target_qc_ms2_matches.tsv"))
    parser.add_argument("--library", type=Path, default=Path("data/reference/unified_v2/unified_pos.mgf"))
    parser.add_argument("--out-mgf", type=Path, default=Path("data/external/MTBLS1905/reference/hnscc_positive_mass_candidates.mgf"))
    parser.add_argument("--out-map", type=Path, default=Path("data/external/MTBLS1905/reference/hnscc_positive_mass_candidate_map.tsv"))
    parser.add_argument("--ppm", type=float, default=10.0)
    args = parser.parse_args()
    hits = pd.read_csv(args.targets, sep="\t")
    targets = hits[hits["n_qc_ms2_hits"] > 0][["metabolite", "target_mz"]].drop_duplicates()
    if targets.empty:
        raise ValueError("No published targets with direct QC-MS2")
    args.out_mgf.parent.mkdir(parents=True, exist_ok=True)
    args.out_map.parent.mkdir(parents=True, exist_ok=True)
    selected = 0
    mappings: list[dict] = []
    block: list[str] = []
    precursor: float | None = None
    in_block = False
    with args.library.open(encoding="utf-8", errors="replace") as reader, args.out_mgf.open("w", encoding="utf-8") as writer:
        for line in reader:
            stripped = line.strip()
            if stripped == "BEGIN IONS":
                in_block, block, precursor = True, [line], None
                continue
            if not in_block:
                continue
            block.append(line)
            if stripped.startswith("PEPMASS="):
                try:
                    precursor = float(stripped.split("=", 1)[1].split()[0])
                except ValueError:
                    precursor = None
            if stripped != "END IONS":
                continue
            in_block = False
            if precursor is None:
                continue
            delta = (targets["target_mz"] - precursor).abs() / targets["target_mz"] * 1e6
            matched = targets[delta <= args.ppm]
            if matched.empty:
                continue
            writer.writelines(block)
            writer.write("\n")
            for name in matched["metabolite"]:
                mappings.append({"library_record_index": selected, "library_precursor_mz": precursor, "target_metabolite": name})
            selected += 1
    with args.out_map.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["library_record_index", "library_precursor_mz", "target_metabolite"], delimiter="\t")
        writer.writeheader(); writer.writerows(mappings)
    print(f"Selected {selected} reference spectra across {len(targets)} targets; wrote {args.out_mgf}")


if __name__ == "__main__":
    main()
