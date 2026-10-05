"""Convert the scale-separated SIRIUS table to the common source-ledger schema."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--score-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def finite(value: str) -> bool:
    try:
        return bool(value.strip()) and math.isfinite(float(value))
    except ValueError:
        return False


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    source_report = json.loads(args.score_report.read_text(encoding="utf-8"))
    if source_report.get("status") != "CHEMAWARE_SIRIUS_CANDIDATE_SCORES_COMPLETE":
        raise RuntimeError("SIRIUS candidate-score import is not complete")
    rows: list[dict[str, object]] = []
    queries: set[int] = set()
    with args.scores.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            common = {
                "manifest_query": int(row["manifest_query"]),
                "local_candidate": int(row["local_candidate"]),
                "ik14": row["ik14"],
                "formula": row["formula"],
                "control_a_score": "",
                "control_b_score": "",
                "controls_available": 0,
            }
            queries.add(int(row["manifest_query"]))
            if finite(row["tree_score"]):
                rows.append({
                    **common,
                    "source_family": "sirius_tree",
                    "scope": "cross_formula",
                    "source_score": row["tree_score"],
                    "source_available": 1,
                })
            if finite(row["csi_fingerid_score"]):
                rows.append({
                    **common,
                    "source_family": "sirius_csi",
                    "scope": "within_formula",
                    "source_score": row["csi_fingerid_score"],
                    "source_available": 1,
                })
    if not rows:
        raise RuntimeError("SIRIUS conversion produced no available source scores")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_sirius_ledger_", dir=args.output.parent))
    try:
        with (temporary / "candidate_scores.tsv").open(
            "w", encoding="utf-8", newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        report = {
            "status": "CHEMAWARE_CANDIDATE_SOURCE_LEDGER_COMPLETE",
            "truth_fields_exported": False,
            "training_formula_only": True,
            "source_families": {
                "sirius_tree": {
                    "scope": "cross_formula", "larger_is_better": True,
                    "matched_controls": [], "specificity_gate_passed": True,
                },
                "sirius_csi": {
                    "scope": "within_formula", "larger_is_better": True,
                    "matched_controls": [], "specificity_gate_passed": True,
                },
            },
            "queries": len(queries),
            "candidate_source_rows": len(rows),
            "provenance": {
                "scores": str(args.scores), "score_report": str(args.score_report),
            },
        }
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
