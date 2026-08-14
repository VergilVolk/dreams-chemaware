"""Join adapter effects with lipid edit candidates and summarize by molecule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from classify_lipid_structural_edits import classify, describe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--external-pilot", type=Path, default=Path("data/validation/external_ring_balanced_pilot"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.results)
    official = frame.loc[frame["seed"] == -1].copy()
    adapted = frame.loc[frame["seed"] >= 0].copy()
    cache = {}
    def get(smiles):
        if smiles not in cache: cache[smiles] = describe(smiles)
        return cache[smiles]

    labels = {}
    for split in ("discovery", "confirmation"):
        units = json.loads((args.external_pilot / f"{split}_manifest.json").read_text(encoding="utf-8"))["units"]
        by_id = {int(unit["pair_id"]): unit for unit in units}
        by_ik = {unit["ik14"]: unit for unit in units}
        for record in official[(official["split"] == split) & official["phospholipid_like"]].to_dict(orient="records"):
            query, negative = by_ik[record["ik14"]], by_id[int(record["best_negative_pair_id"])]
            label, reason = classify(get(query["smiles"]), get(negative["smiles"]))
            labels[(split, record["ik14"], int(record["view"]))] = (label, reason)

    rows = []
    for record in adapted[adapted["phospholipid_like"]].to_dict(orient="records"):
        baseline = official[(official["split"] == record["split"]) & (official["ik14"] == record["ik14"]) & (official["view"] == record["view"])].iloc[0]
        label, reason = labels[(record["split"], record["ik14"], int(record["view"]))]
        rows.append({
            "seed": int(record["seed"]), "split": record["split"], "ik14": record["ik14"],
            "view": int(record["view"]), "edit_candidate": label, "reason": reason,
            "official_top1": bool(baseline["top1"]), "adapted_top1": bool(record["top1"]),
            "top1_delta": int(bool(record["top1"])) - int(bool(baseline["top1"])),
            "margin_delta": float(record["margin"] - baseline["margin"]),
            "pairwise_delta": float(record["pairwise"] - baseline["pairwise"]),
        })
    output = pd.DataFrame(rows); output.to_csv(args.output_dir / "view_effects.csv", index=False)
    molecule = output.groupby(["seed", "split", "edit_candidate", "ik14"]).agg(
        top1_delta=("top1_delta", "mean"), margin_delta=("margin_delta", "mean"),
        pairwise_delta=("pairwise_delta", "mean"),
    ).reset_index()
    summary = molecule.groupby(["seed", "split", "edit_candidate"]).agg(
        molecules=("ik14", "nunique"), top1_delta=("top1_delta", "mean"),
        margin_delta=("margin_delta", "mean"), pairwise_delta=("pairwise_delta", "mean"),
        improved_margin=("margin_delta", lambda x: int((x > 0).sum())),
        worsened_margin=("margin_delta", lambda x: int((x < 0).sum())),
    ).reset_index()
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    report = {
        "status": "adapter_effect_by_lipid_edit_candidate", "summary": summary.to_dict(orient="records"),
        "claim_limit": "Small automated candidate classes; descriptive only, with no multiplicity-adjusted inferential claim.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
