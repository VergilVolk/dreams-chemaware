"""Summarize module-2 causal evidence records without inventing a confidence score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "data/validation/module2_evidence_records"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def summarize(records: list[dict], split: str) -> pd.DataFrame:
    rows = []
    for record in records:
        peaks = record["removed_peak_evidence"]
        rows.append({
            "split": split,
            "mechanism": record["mechanism"],
            "pair_type": record["pair_type"],
            "clean_similarity": record["clean_similarity"],
            "targeted_similarity_change": record["targeted_similarity_change"],
            "matched_random_similarity_change": record["matched_random_similarity_change"],
            "directional_support": record["directional_support"],
            "removed_peak_count": len(peaks),
            "has_factor_match": any(peak["candidate_factors"] for peak in peaks),
            "has_rule_match": any(peak["matched_rules"] for peak in peaks),
        })
    frame = pd.DataFrame(rows)
    output = []
    for (mechanism, pair_type), group in frame.groupby(["mechanism", "pair_type"], sort=True):
        support = group["directional_support"].to_numpy(float)
        output.append({
            "split": split,
            "mechanism": mechanism,
            "pair_type": pair_type,
            "directed_records": int(len(group)),
            "clean_similarity_median": float(group["clean_similarity"].median()),
            "targeted_change_median": float(group["targeted_similarity_change"].median()),
            "matched_random_change_median": float(group["matched_random_similarity_change"].median()),
            "directional_support_median": float(np.median(support)),
            "directional_support_positive_fraction": float(np.mean(support > 0)),
            "factor_match_fraction": float(group["has_factor_match"].mean()),
            "rule_match_fraction": float(group["has_rule_match"].mean()),
            "removed_peak_count_median": float(group["removed_peak_count"].median()),
        })
    return pd.DataFrame(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    summaries = []
    for split in ("discovery", "confirmation"):
        path = args.input_dir / f"{split}_evidence_records.jsonl"
        summaries.append(summarize(load_jsonl(path), split))
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(args.input_dir / "mechanism_evidence_summary.csv", index=False)
    payload = {
        "status": "module2_evidence_summary_complete",
        "unit": "directed spectrum-pair intervention record",
        "directional_support_definition": (
            "targeted peak deletion effect minus intensity/mass-matched random deletion effect; "
            "positive values support specificity"
        ),
        "claim_boundary": (
            "mass matches provide candidate annotations; no unique fragment identity or "
            "bond-breaking mechanism is asserted"
        ),
        "rows": summary.to_dict(orient="records"),
    }
    (args.input_dir / "mechanism_evidence_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
