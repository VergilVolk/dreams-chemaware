"""Audit whether clean E0 failures are enriched in chemically defined query domains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact

from classify_external_e0_failures import info
from rdkit.Chem import rdFingerprintGenerator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot-dir",
        type=Path,
        default=Path("data/validation/external_ring_balanced_pilot"),
    )
    parser.add_argument(
        "--query-results",
        type=Path,
        default=Path("data/validation/external_ring_balanced_e0/query_results.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/validation/external_failure_domain_enrichment"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(args.query_results)
    results = results.loc[results["candidate_protocol"] == "same_formula_negative_pair_ids"].copy()
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=128)
    rows = []
    for split in ("discovery", "confirmation"):
        units = json.loads((args.pilot_dir / f"{split}_manifest.json").read_text(encoding="utf-8"))["units"]
        eligible = {
            unit["ik14"]: unit
            for unit in units
            if unit.get("is_query_anchor") and unit.get("same_formula_negative_pair_ids")
        }
        split_results = results.loc[results["split"] == split]
        for ik14, unit in eligible.items():
            query_rows = split_results.loc[split_results["ik14"] == ik14]
            rows.append({
                "split": split,
                "ik14": ik14,
                "ring_class": unit["ring_class"],
                "domain": info(unit["smiles"], fpgen)["domain"],
                "formula": unit["formula"],
                "precursor_mz": unit["precursor_mz"],
                "failed_any_view": bool((~query_rows["top1_correct"]).any()),
                "failed_both_views": bool((~query_rows["top1_correct"]).all()),
                "mean_margin": float(query_rows["margin"].mean()),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "eligible_queries.csv", index=False)

    summary = frame.groupby(["split", "domain"], sort=False).agg(
        eligible_molecules=("ik14", "nunique"),
        failed_any=("failed_any_view", "sum"),
        failed_both=("failed_both_views", "sum"),
        median_precursor_mz=("precursor_mz", "median"),
        mean_margin=("mean_margin", "mean"),
    ).reset_index()
    summary["failure_rate_any"] = summary["failed_any"] / summary["eligible_molecules"]
    summary["failure_rate_both"] = summary["failed_both"] / summary["eligible_molecules"]
    summary.to_csv(args.output_dir / "domain_failure_rates.csv", index=False)

    contrasts = []
    for split in ("discovery", "confirmation"):
        part = frame.loc[frame["split"] == split]
        phospho = part["domain"] == "acyclic_phospholipid_like"
        for comparator_name, comparator in {
            "all_non_phospholipid": ~phospho,
            "other_acyclic": part["domain"] == "other_acyclic",
        }.items():
            if phospho.sum() == 0 or comparator.sum() == 0:
                continue
            for failure_col in ("failed_any_view", "failed_both_views"):
                table = [
                    [int((part.loc[phospho, failure_col]).sum()), int((~part.loc[phospho, failure_col]).sum())],
                    [int((part.loc[comparator, failure_col]).sum()), int((~part.loc[comparator, failure_col]).sum())],
                ]
                odds, pvalue = fisher_exact(table)
                contrasts.append({
                    "split": split,
                    "failure_definition": failure_col,
                    "contrast": f"phospholipid_like_vs_{comparator_name}",
                    "phospholipid_n": int(phospho.sum()),
                    "comparator_n": int(comparator.sum()),
                    "phospholipid_failure_rate": table[0][0] / sum(table[0]),
                    "comparator_failure_rate": table[1][0] / sum(table[1]),
                    "odds_ratio": float(odds),
                    "fisher_p": float(pvalue),
                })
    contrast_frame = pd.DataFrame(contrasts)
    contrast_frame.to_csv(args.output_dir / "phospholipid_contrasts.csv", index=False)

    report = {
        "status": "external_failure_domain_enrichment",
        "protocol": "same-formula 10 ppm; molecule is failed if either or both clean query views rank a wrong molecule first",
        "n_eligible": frame.groupby("split")["ik14"].nunique().to_dict(),
        "domain_failure_rates": summary.to_dict(orient="records"),
        "phospholipid_contrasts": contrast_frame.to_dict(orient="records"),
        "claim_limit": "Exploratory chemical-domain labels; confirmation split is independent of discovery within this external cohort, but annotated01 may overlap DreaMS SSL pretraining.",
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(contrast_frame.to_string(index=False))


if __name__ == "__main__":
    main()
