#!/usr/bin/env python
"""Freeze the small MTBLS13729 biology-closure target panel for local EIC work.

The target list is copied from the already constructed outcome-blind MS1
consensus table.  No phenotype statistic is read or used for target selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


CANDIDATES = (
    (4966, "purine_like_C7H9N5O", "C7H9N5O", "C"),
    (3019, "dimethylguanosine_isomer_family", "C12H17N5O5", "A"),
    (1597, "methylguanosine_isomer_family_MH", "C11H15N5O5", "D"),
    (7489, "methylguanosine_isomer_family_MNa", "C11H15N5O5", "C"),
    (1717, "N1_N8_diacetylspermidine_like", "C11H23N3O2", "A"),
    (3222, "C20_4_acylcarnitine_like", "C27H45NO4", "A"),
    (3180, "chlorinated_or_exogenous_like", "unknown", "C"),
    (16425, "LPE_like", "unknown", "A"),
)

FAMILY_SUPPORT = (
    # Phenotype-blind global peak graph: accepted, split-replicated Na/H edge
    # to feature 3019.  This row validates an ion family and is never counted
    # as an independent biological discovery.
    (8481, "dimethylguanosine_isomer_family_MNa_support", "C12H17N5O5", "family_support"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--consensus-dir",
        type=Path,
        default=Path("data/mtbls13729/ms1_consensus"),
    )
    parser.add_argument("--include-family-support", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mtbls13729/biology_closure_targets_v1"),
    )
    args = parser.parse_args()

    source_targets = args.consensus_dir / "pos_rp__requantification_targets.csv.gz"
    source_samples = args.consensus_dir / "pos_rp__samples.csv"
    if not source_targets.exists() or not source_samples.exists():
        raise FileNotFoundError("MTBLS13729 pos_rp consensus targets/samples are missing")

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty frozen target directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    targets = pd.read_csv(source_targets)
    selected_definitions = CANDIDATES + (FAMILY_SUPPORT if args.include_family_support else ())
    wanted = pd.DataFrame(
        selected_definitions,
        columns=["feature_id", "biology_label", "candidate_formula", "identity_tier"],
    )
    selected = wanted.merge(targets, on="feature_id", how="left", validate="one_to_one")
    if selected["mz"].isna().any() or selected["rt_sec"].isna().any():
        missing = selected.loc[selected["mz"].isna(), "feature_id"].tolist()
        raise RuntimeError(f"frozen biology targets absent from consensus table: {missing}")
    if selected["feature_id"].duplicated().any() or len(selected) != len(selected_definitions):
        raise RuntimeError("biology target selection is not one-to-one")

    target_columns = list(targets.columns)
    selected[target_columns].to_csv(
        output / "pos_rp__requantification_targets.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.read_csv(source_samples).to_csv(output / "pos_rp__samples.csv", index=False)
    selected.to_csv(output / "biology_candidate_ledger.csv", index=False)

    report = {
        "status": "mtbls13729_biology_closure_targets_frozen",
        "formal": False,
        "selection": (
            "eight candidates frozen before local EIC reproduction plus one phenotype-blind ion-family support peak"
            if args.include_family_support else
            "eight candidates frozen before this local EIC reproduction; no phenotype column read"
        ),
        "panel": "pos_rp",
        "targets": int(len(selected)),
        "samples": int(len(pd.read_csv(source_samples))),
        "feature_ids": selected["feature_id"].astype(int).tolist(),
        "independent_discoveries": 8,
        "family_support_peaks": int(args.include_family_support),
        "provenance": {
            "source_targets_sha256": sha256(source_targets),
            "source_samples_sha256": sha256(source_samples),
        },
        "claim_limit": (
            "This panel reproduces targeted EICs for already frozen discoveries. "
            "It is not a new discovery screen and does not validate molecular identity."
        ),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
