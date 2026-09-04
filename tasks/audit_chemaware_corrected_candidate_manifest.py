"""Audit scale, imbalance, and structural strata of the corrected manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(values: np.ndarray) -> dict:
    return {
        "min": int(values.min()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, .9)),
        "p99": float(np.quantile(values, .99)),
        "max": int(values.max()),
        "mean": float(values.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz",
    )
    parser.add_argument(
        "--mces", type=Path, default=ROOT / "tasks/massspecgym_isomers/pairs.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/audit.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    with np.load(args.manifest) as source:
        arrays = {name: source[name] for name in source.files}
    query_ptr = arrays["query_ptr"].astype(np.int64)
    molecule_ptr = arrays["molecule_ptr"].astype(np.int64)
    query_ik = arrays["query_ik14"].astype(str)
    query_formula = arrays["query_formula"].astype(str)
    query_adduct = arrays["query_adduct"].astype(str)
    molecule_ik = arrays["molecule_ik14"].astype(str)
    molecule_formula = arrays["molecule_formula"].astype(str)
    label = arrays["molecule_label"].astype(np.int8)
    pair_rows = arrays["pair_candidate_row"].astype(np.int64)
    if query_ptr[0] != 0 or query_ptr[-1] != len(label) or np.any(np.diff(query_ptr) < 2):
        raise RuntimeError("invalid query pointer contract")
    if molecule_ptr[0] != 0 or molecule_ptr[-1] != len(pair_rows) or np.any(np.diff(molecule_ptr) < 1):
        raise RuntimeError("invalid molecule pointer contract")

    mces_body = json.loads(args.mces.read_text(encoding="utf-8"))
    grade = {}
    for name in ("near", "mid", "far"):
        for row in mces_body[name]:
            key = tuple(sorted((str(row["ik_a"]), str(row["ik_b"]))))
            if key in grade and grade[key] != name:
                raise RuntimeError(f"conflicting MCES label: {key}")
            grade[key] = name

    same_formula_negative = 0
    total_negative = 0
    queries_with_same_formula_negative = 0
    queries_with_near = 0
    grade_edges = Counter()
    unique_pairs = set()
    unique_same_formula_pairs = set()
    for query, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
        left, right = int(left), int(right)
        labels = label[left:right]
        if labels[0] != 1 or labels.sum() != 1:
            raise RuntimeError(f"query {query} lacks one first-position positive")
        negatives = np.arange(left + 1, right)
        total_negative += len(negatives)
        same = molecule_formula[negatives] == query_formula[query]
        same_formula_negative += int(same.sum())
        queries_with_same_formula_negative += int(np.any(same))
        near = False
        for molecule, is_same_formula in zip(negatives, same):
            key = tuple(sorted((str(query_ik[query]), str(molecule_ik[molecule]))))
            unique_pairs.add(key)
            if is_same_formula:
                unique_same_formula_pairs.add(key)
            name = grade.get(key, "unlabelled")
            grade_edges[name] += 1
            near |= name == "near"
        queries_with_near += int(near)

    identity_counts = np.asarray(list(Counter(query_ik).values()), dtype=np.int64)
    formula_counts = np.asarray(list(Counter(query_formula).values()), dtype=np.int64)
    report = {
        "status": "chemaware_corrected_candidate_manifest_audited",
        "formal_training_authorized": False,
        "counts": {
            "queries": int(len(query_ik)),
            "query_identities": int(len(identity_counts)),
            "query_formulas": int(len(formula_counts)),
            "candidate_molecules": int(len(label)),
            "candidate_spectrum_edges": int(len(pair_rows)),
            "negative_molecule_edges": int(total_negative),
            "unique_query_negative_identity_pairs": int(len(unique_pairs)),
        },
        "same_formula_stratum": {
            "negative_molecule_edges": int(same_formula_negative),
            "fraction_of_negative_molecule_edges": float(same_formula_negative / total_negative),
            "unique_identity_pairs": int(len(unique_same_formula_pairs)),
            "queries_with_at_least_one": int(queries_with_same_formula_negative),
            "query_fraction": float(queries_with_same_formula_negative / len(query_ik)),
        },
        "mces_strata_on_negative_molecule_edges": dict(sorted(grade_edges.items())),
        "queries_with_mces_near_negative": int(queries_with_near),
        "queries_with_mces_near_negative_fraction": float(queries_with_near / len(query_ik)),
        "imbalance": {
            "queries_per_identity": describe(identity_counts),
            "queries_per_formula": describe(formula_counts),
            "identity_equal_weighting_required": True,
            "formula_clustered_inference_required": True,
        },
        "adduct_query_counts": {
            str(value): int(np.sum(query_adduct == value)) for value in np.unique(query_adduct)
        },
        "claim_limit": (
            "MCES coverage is a partial structural stratum, not a complete label for all "
            "candidate pairs. Candidate counts establish problem scale, not model performance."
        ),
        "provenance": {
            "manifest_sha256": sha256(args.manifest),
            "mces_sha256": sha256(args.mces),
            "script_sha256": sha256(Path(__file__)),
        },
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
