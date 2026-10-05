"""Freeze truth-blind applicability on an already-scored fragment-graph ledger.

This migration exists only to reuse expensive, immutable score computation from
an earlier engineering run.  It never changes a finite score.  If every
candidate in a query has the same source score, every score/control field for
that query is replaced by an explicit missing value and the query abstains.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = json.loads((args.ledger / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "CHEMAWARE_CANDIDATE_SOURCE_LEDGER_UNQUALIFIED":
        raise RuntimeError("input is not an unqualified candidate source ledger")
    with (args.ledger / "candidate_scores.tsv").open(
        "r", encoding="utf-8", newline="",
    ) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise RuntimeError("candidate score ledger has no header")
        fields = list(reader.fieldnames)
        rows = list(reader)
    control_fields = [
        name for name in fields if name.startswith("control_") and name.endswith("_score")
    ]
    if not control_fields:
        raise RuntimeError("candidate source ledger contains no controls")
    by_query: dict[int, list[int]] = defaultdict(list)
    for position, row in enumerate(rows):
        by_query[int(row["manifest_query"])].append(position)
    abstained = 0
    modified_cells = 0
    finite_scores_unchanged = True
    for positions in by_query.values():
        values = [rows[position]["source_score"].strip() for position in positions]
        finite = [float(value) for value in values if value and math.isfinite(float(value))]
        if len(finite) != len(positions):
            continue
        if max(finite) > min(finite):
            continue
        abstained += 1
        for position in positions:
            for field in ["source_score", *control_fields]:
                modified_cells += int(bool(rows[position][field].strip()))
                rows[position][field] = ""
            rows[position]["controls_available"] = "0"

    output_report = json.loads(json.dumps(report))
    output_report["formal_triplet_mining_authorized"] = False
    output_report["truth_blind_applicability"] = {
        "status": "CHEMAWARE_FRAGMENT_GRAPH_APPLICABILITY_COMPLETE",
        "rule": "abstain iff all finite candidate source scores are exactly equal",
        "queries_before": len(by_query),
        "queries_abstained": abstained,
        "queries_retained": len(by_query) - abstained,
        "score_cells_blankened": modified_cells,
        "nonabstained_finite_scores_unchanged": finite_scores_unchanged,
        "truth_fields_read": False,
        "input_candidate_scores_sha256": sha256_file(args.ledger / "candidate_scores.tsv"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_fragment_applicability_", dir=args.output.parent))
    try:
        with (temporary / "candidate_scores.tsv").open(
            "w", encoding="utf-8", newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fields, delimiter="\t", lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        for name in ("connected_explanations.tsv", "exclusions.tsv"):
            source = args.ledger / name
            if source.is_file():
                shutil.copy2(source, temporary / name)
        output_report["candidate_scores_sha256"] = sha256_file(
            temporary / "candidate_scores.tsv"
        )
        (temporary / "report.json").write_text(
            json.dumps(output_report, indent=2) + "\n", encoding="utf-8",
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(output_report["truth_blind_applicability"], indent=2), flush=True)


if __name__ == "__main__":
    main()
