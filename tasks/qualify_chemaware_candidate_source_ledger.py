"""Formula-disjoint specificity qualification for a truth-blind source ledger."""
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

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold-seed", type=int, default=20260928)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--confirmation-folds", type=int, nargs="+", default=(3, 4))
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--minimum-applicable-queries", type=int, default=100)
    return parser.parse_args()


def formula_fold(value: str, seed: int, folds: int) -> int:
    digest = hashlib.sha256(f"{seed}|{value}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % folds


def cluster_ci(values: np.ndarray, formulas: np.ndarray, draws: int, seed: int) -> list[float]:
    unique = np.unique(formulas)
    means = np.asarray([np.mean(values[formulas == formula]) for formula in unique])
    rng = np.random.default_rng(seed)
    sampled = rng.choice(means, size=(draws, len(means)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(sampled, [0.025, 0.975])]


def optional_float(value: str) -> float:
    """Treat an absent external score as inapplicable, never as zero."""
    if value is None or not str(value).strip():
        return math.nan
    parsed = float(value)
    return parsed if math.isfinite(parsed) else math.nan


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.bootstrap_draws < 1000 or args.minimum_applicable_queries < 20:
        raise ValueError("specificity thresholds were weakened")
    source_report = json.loads((args.ledger / "report.json").read_text(encoding="utf-8"))
    if source_report.get("status") != "CHEMAWARE_CANDIDATE_SOURCE_LEDGER_UNQUALIFIED":
        raise RuntimeError("source ledger is not awaiting qualification")
    scores: dict[
        str, dict[int, dict[int, tuple[float, dict[str, float], str]]]
    ] = defaultdict(
        lambda: defaultdict(dict),
    )
    raw_rows: list[dict[str, str]] = []
    with (args.ledger / "candidate_scores.tsv").open(
        "r", encoding="utf-8", newline="",
    ) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise RuntimeError("candidate source ledger has no header")
        control_columns = sorted(
            name for name in reader.fieldnames
            if name.startswith("control_") and name.endswith("_score")
        )
        if not control_columns:
            raise RuntimeError("candidate source ledger has no matched controls")
        for family, contract in source_report["source_families"].items():
            if len(contract.get("matched_controls", [])) != len(control_columns):
                raise RuntimeError(
                    f"matched-control cardinality drift for {family}: "
                    f"contract={len(contract.get('matched_controls', []))} "
                    f"table={len(control_columns)}"
                )
        for row in reader:
            raw_rows.append(row)
            scores[row["source_family"]][int(row["manifest_query"])][int(row["local_candidate"])] = (
                optional_float(row["source_score"]),
                {name: optional_float(row[name]) for name in control_columns},
                row["formula"],
            )
    with np.load(args.manifest, allow_pickle=False) as loaded:
        manifest = {key: np.asarray(loaded[key]) for key in loaded.files}

    family_results = {}
    admitted = []
    for family, query_scores in scores.items():
        scope = source_report["source_families"][family]["scope"]
        rows = []
        for query, candidate_scores in sorted(query_scores.items()):
            formula = str(manifest["query_formula"][query])
            if formula_fold(formula, args.fold_seed, args.folds) not in args.confirmation_folds:
                continue
            left, right = map(int, manifest["query_ptr"][query:query + 2])
            labels = np.asarray(manifest["molecule_label"][left:right], dtype=bool)
            true_values = np.flatnonzero(labels)
            if len(true_values) != 1:
                raise RuntimeError(f"query {query} does not have one true candidate")
            truth = int(true_values[0])
            if truth not in candidate_scores:
                continue
            truth_source, truth_controls, true_formula = candidate_scores[truth]
            if not math.isfinite(truth_source) or not all(
                math.isfinite(value) for value in truth_controls.values()
            ):
                continue
            applicable = [
                candidate for candidate, body in candidate_scores.items()
                if candidate != truth and (
                    (scope == "cross_formula" and body[2] != true_formula)
                    or (scope == "within_formula" and body[2] == true_formula)
                    or scope == "all_candidates"
                )
                and math.isfinite(body[0])
                and all(math.isfinite(value) for value in body[1].values())
            ]
            if not applicable:
                continue
            values = np.asarray([candidate_scores[truth][0]] + [candidate_scores[c][0] for c in applicable])
            controls = {
                name: np.asarray(
                    [candidate_scores[truth][1][name]]
                    + [candidate_scores[c][1][name] for c in applicable]
                )
                for name in control_columns
            }
            rows.append({
                "query": query, "formula": formula,
                "correct_hit1": float(np.all(values[0] > values[1:])),
                "correct_margin": float(values[0] - np.max(values[1:])),
                "control_hit1": {
                    name: float(np.all(values_for_control[0] > values_for_control[1:]))
                    for name, values_for_control in controls.items()
                },
                "control_margin": {
                    name: float(values_for_control[0] - np.max(values_for_control[1:]))
                    for name, values_for_control in controls.items()
                },
            })
        if rows:
            formulas = np.asarray([row["formula"] for row in rows])
            correct_hit = np.asarray([row["correct_hit1"] for row in rows])
            correct_margin = np.asarray([row["correct_margin"] for row in rows])
            control_results = {}
            for control_index, name in enumerate(control_columns):
                control_hit = np.asarray([row["control_hit1"][name] for row in rows])
                control_margin = np.asarray([row["control_margin"][name] for row in rows])
                ci_hit = cluster_ci(
                    correct_hit - control_hit, formulas, args.bootstrap_draws,
                    args.fold_seed + 101 + 10 * control_index,
                )
                ci_margin = cluster_ci(
                    correct_margin - control_margin, formulas, args.bootstrap_draws,
                    args.fold_seed + 107 + 10 * control_index,
                )
                control_results[name] = {
                    "hit1": float(np.mean(control_hit)),
                    "correct_minus_control_hit1_ci95": ci_hit,
                    "correct_minus_control_margin_ci95": ci_margin,
                    "passed": ci_hit[0] > 0 and ci_margin[0] > 0,
                }
            passed = (
                len(rows) >= args.minimum_applicable_queries
                and all(body["passed"] for body in control_results.values())
            )
            result = {
                "applicable_queries": len(rows),
                "formula_clusters": len(np.unique(formulas)),
                "correct_hit1": float(np.mean(correct_hit)),
                "controls": control_results,
                "specificity_gate_passed": bool(passed),
            }
        else:
            result = {"applicable_queries": 0, "formula_clusters": 0, "specificity_gate_passed": False}
        family_results[family] = result
        if result["specificity_gate_passed"]:
            admitted.append(family)

    qualification = {
        "status": "CHEMAWARE_SOURCE_SPECIFICITY_QUALIFICATION_COMPLETE",
        "admitted_families": admitted,
        "family_results": family_results,
        "fold_contract": {
            "folds": args.folds, "fold_seed": args.fold_seed,
            "confirmation_folds": list(args.confirmation_folds),
            "formula_disjoint": True,
            "roles_2_3_4_untouched": True,
        },
        "bootstrap_draws": args.bootstrap_draws,
        "control_columns": control_columns,
    }
    qualified_report = json.loads(json.dumps(source_report))
    qualified_report["status"] = "CHEMAWARE_CANDIDATE_SOURCE_LEDGER_COMPLETE"
    qualified_report["formal_triplet_mining_authorized"] = bool(admitted)
    qualified_report["specificity_qualification"] = qualification
    # Keep only rejected families in this field.  Retaining every original
    # family here made an admitted family appear simultaneously qualified and
    # unqualified, which is scientifically harmless but provenance-ambiguous.
    qualified_report["unqualified_source_families"] = {
        family: {
            **source_report["source_families"][family],
            "specificity_gate_passed": False,
        }
        for family in source_report["source_families"]
        if family not in admitted
    }
    qualified_report["source_families"] = {
        family: {
            **source_report["source_families"][family],
            "specificity_gate_passed": True,
        }
        for family in admitted
    }
    qualified_rows = [row for row in raw_rows if row["source_family"] in admitted]
    qualified_report["candidate_source_rows"] = len(qualified_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_qualified_source_", dir=args.output.parent))
    try:
        with (temporary / "candidate_scores.tsv").open(
            "w", encoding="utf-8", newline="",
        ) as handle:
            if not raw_rows:
                raise RuntimeError("source ledger contains no candidate rows")
            writer = csv.DictWriter(
                handle, fieldnames=list(raw_rows[0]), delimiter="\t", lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(qualified_rows)
        (temporary / "report.json").write_text(
            json.dumps(qualified_report, indent=2) + "\n", encoding="utf-8",
        )
        (temporary / "qualification_report.json").write_text(
            json.dumps(qualification, indent=2) + "\n", encoding="utf-8",
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(qualification, indent=2), flush=True)


if __name__ == "__main__":
    main()
