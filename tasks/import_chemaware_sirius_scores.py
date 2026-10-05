"""Normalize exhaustive SIRIUS 6 results onto the ChemAware candidate graph.

The source panel contains one fixed-formula replica for every candidate formula
of every role-0/1 query.  This importer joins raw fragmentation-tree and
CSI:FingerID scores back to the exact graph without reading candidate labels or
combining the two score families with a fitted weight.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", type=Path, required=True)
    parser.add_argument(
        "--formula-summary", type=Path, action="append", required=True,
        help="SIRIUS formula-candidate TSV/CSV; repeat for sharded exports",
    )
    parser.add_argument(
        "--structure-summary", type=Path, action="append", required=True,
        help="SIRIUS structure-candidate TSV/CSV; repeat for sharded exports",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def normalized_name(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def resolve_column(fieldnames: Iterable[str], aliases: Iterable[str]) -> str:
    mapping = {normalized_name(name): name for name in fieldnames}
    for alias in aliases:
        key = normalized_name(alias)
        if key in mapping:
            return mapping[key]
    raise KeyError(
        f"none of columns {tuple(aliases)} found in {tuple(fieldnames)}"
    )


def table_rows(path: Path) -> list[dict[str, str]]:
    delimiter = "," if path.suffix.lower() == ".csv" else "\t"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise RuntimeError(f"SIRIUS table has no header: {path}")
        return [dict(row) for row in reader]


def expand_tables(paths: list[Path]) -> list[Path]:
    output: list[Path] = []
    for path in paths:
        if path.is_dir():
            output.extend(sorted(
                candidate for candidate in path.rglob("*")
                if candidate.is_file() and candidate.suffix.lower() in {".tsv", ".csv"}
            ))
        elif path.is_file():
            output.append(path)
        else:
            raise FileNotFoundError(path)
    return output


def finite_float(value: str, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid numeric SIRIUS value for {context}: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"non-finite SIRIUS value for {context}: {value!r}")
    return result


def ik14(value: str) -> str:
    cleaned = value.strip()
    if len(cleaned) < 14:
        raise ValueError(f"invalid SIRIUS InChIKey: {value!r}")
    return cleaned[:14]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def formula_scores(paths: list[Path]) -> dict[tuple[str, str], float]:
    output: dict[tuple[str, str], float] = {}
    for path in expand_tables(paths):
        rows = table_rows(path)
        if not rows:
            continue
        try:
            feature = resolve_column(rows[0], (
                "mappingFeatureId", "featureId", "compoundName", "name",
            ))
            formula = resolve_column(rows[0], (
                "molecularFormula", "formula", "neutralFormula",
            ))
            tree = resolve_column(rows[0], ("TreeScore", "treeScore"))
        except KeyError:
            continue
        for row in rows:
            key = (row[feature].strip(), row[formula].strip())
            value = finite_float(row[tree], f"{path}:{key}:TreeScore")
            output[key] = max(value, output.get(key, -math.inf))
    return output


def structure_scores(
    paths: list[Path],
) -> dict[tuple[str, str, str], float]:
    output: dict[tuple[str, str, str], float] = {}
    for path in expand_tables(paths):
        rows = table_rows(path)
        if not rows:
            continue
        try:
            feature = resolve_column(rows[0], (
                "mappingFeatureId", "featureId", "compoundName", "name",
            ))
            formula = resolve_column(rows[0], (
                "molecularFormula", "formula", "neutralFormula",
            ))
            inchikey = resolve_column(rows[0], (
                "InChIkey2D", "InChIKey2D", "InChIKey", "inchikey",
                "structureId", "candidateId", "id",
            ))
            csi = resolve_column(rows[0], (
                "CSI:FingerIDScore", "CSIFingerIDScore", "csiScore",
            ))
        except KeyError:
            continue
        for row in rows:
            key = (
                row[feature].strip(), row[formula].strip(), ik14(row[inchikey]),
            )
            value = finite_float(row[csi], f"{path}:{key}:CSI:FingerIDScore")
            output[key] = max(value, output.get(key, -math.inf))
    return output


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    panel_report = json.loads(
        (args.source_panel / "report.json").read_text(encoding="utf-8")
    )
    if panel_report.get("status") != "CHEMAWARE_SIRIUS_SOURCE_PANEL_COMPLETE":
        raise RuntimeError("source panel is not a completed ChemAware SIRIUS panel")
    registry = read_tsv(args.source_panel / "formula_feature_registry.tsv")
    candidates = read_tsv(args.source_panel / "candidate_ledger.tsv")
    formula_feature: dict[tuple[int, str], str] = {}
    for row in registry:
        key = (int(row["manifest_query"]), row["candidate_formula"])
        if key in formula_feature:
            raise RuntimeError(f"duplicate formula feature registry key: {key}")
        formula_feature[key] = row["formula_feature_id"]

    trees = formula_scores(args.formula_summary)
    structures = structure_scores(args.structure_summary)
    if not trees:
        raise RuntimeError("no SIRIUS formula table with TreeScore was found")
    if not structures:
        raise RuntimeError("no SIRIUS structure table with CSI:FingerIDScore was found")
    output_rows: list[dict[str, object]] = []
    missing_tree = missing_structure = 0
    features_with_formula_drift: set[str] = set()
    for row in candidates:
        query = int(row["manifest_query"])
        formula = row["formula"]
        feature = formula_feature[(query, formula)]
        tree_key = (feature, formula)
        tree = trees.get(tree_key, math.nan)
        csi = structures.get((feature, formula, row["ik14"]), math.nan)
        missing_tree += int(not math.isfinite(tree))
        missing_structure += int(not math.isfinite(csi))
        output_rows.append({
            "manifest_query": query,
            "local_candidate": int(row["local_candidate"]),
            "ik14": row["ik14"],
            "formula": formula,
            "formula_feature_id": feature,
            "tree_score": "" if not math.isfinite(tree) else f"{tree:.17g}",
            "csi_fingerid_score": "" if not math.isfinite(csi) else f"{csi:.17g}",
            "tree_available": int(math.isfinite(tree)),
            "csi_available": int(math.isfinite(csi)),
        })

    known_features = {row["formula_feature_id"] for row in registry}
    registered_feature_formulas = {
        (row["formula_feature_id"], row["candidate_formula"])
        for row in registry
    }
    for feature, formula in trees:
        if feature in known_features and (feature, formula) not in registered_feature_formulas:
            features_with_formula_drift.add(feature)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_sirius_scores_", dir=args.output.parent))
    try:
        with (temporary / "candidate_scores.tsv").open(
            "w", encoding="utf-8", newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(output_rows[0]), delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(output_rows)
        report = {
            "status": "CHEMAWARE_SIRIUS_CANDIDATE_SCORES_COMPLETE",
            "candidate_rows": len(output_rows),
            "formula_features": len(registry),
            "tree_scores_available": len(output_rows) - missing_tree,
            "tree_scores_missing": missing_tree,
            "csi_scores_available": len(output_rows) - missing_structure,
            "csi_scores_missing": missing_structure,
            "formula_drift_features": len(features_with_formula_drift),
            "scientific_contract": {
                "score_fusion": "none; TreeScore and CSI:FingerIDScore remain separate",
                "truth_fields_read": False,
                "formula_layer": "raw TreeScore across all fixed-formula replicas",
                "structure_layer": "CSI:FingerIDScore only within the same formula",
                "missingness": "explicitly retained; downstream triplet mining may not impute",
            },
            "gates": {
                "candidate_cardinality_matches_source": (
                    len(output_rows) == int(panel_report["candidate_rows"])
                ),
                "formula_registry_cardinality_matches_source": (
                    len(registry) == int(panel_report["query_formula_features"])
                ),
                "no_formula_drift": not features_with_formula_drift,
            },
        }
        if not all(report["gates"].values()):
            raise RuntimeError(f"SIRIUS score import gates failed: {report['gates']}")
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8", newline="\n",
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
