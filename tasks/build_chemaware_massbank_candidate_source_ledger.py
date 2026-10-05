"""Create truth-blind candidate scores from recurrent MassBank annotations.

The source corpus supplies recurrent product-ion and neutral-loss formulae plus
parent-structure consensus predicates.  The observed query spectrum activates
rules; a candidate receives support only when its protonated composition and
Morgan environments satisfy the rule.  Exact MassBank source identities are
suppressed.

Two controls preserve capacity: cyclic rule masses and cyclic formula/structure
predicates.  No candidate truth is read; formula-disjoint qualification is a
separate mandatory step.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from build_chemaware_layered_rule_source_ledger import (
    contains_formula,
    parse_formula,
    read_candidates,
)
from build_chemaware_motifdb_candidate_source_ledger import (
    candidate_fingerprint,
    polarity,
    read_query_registry,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path,
        default=(
            ROOT / "data/validation/chemaware_massbank_annotated_fragment_corpus_v1_20260928"
            / "massbank_fragment_corpus.json"
        ),
    )
    parser.add_argument(
        "--candidate-ledger", type=Path,
        default=(
            ROOT / "data/validation/chemaware_sirius_source_panel_v9_profiled_collision_20260928"
            / "candidate_ledger.tsv"
        ),
    )
    parser.add_argument(
        "--query-registry", type=Path,
        default=(
            ROOT / "data/validation/chemaware_sirius_source_panel_v9_profiled_collision_20260928"
            / "query_registry.tsv"
        ),
    )
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-peaks", type=int, default=64)
    parser.add_argument("--intensity-power", type=float, default=0.5)
    parser.add_argument("--ppm", type=float, default=20.0)
    parser.add_argument("--floor-da", type=float, default=0.01)
    parser.add_argument("--minimum-structure-coverage", type=float, default=0.5)
    parser.add_argument(
        "--scope", choices=("cross_formula", "within_formula", "all_candidates"),
        default="all_candidates",
    )
    return parser.parse_args()


def cyclic_values(records: list[dict[str, Any]], key: str) -> list[Any]:
    positions = sorted(
        range(len(records)),
        key=lambda index: (records[index]["kind"], len(records[index]["consensus_morgan_bits"]), index),
    )
    grouped: dict[str, list[int]] = defaultdict(list)
    for position in positions:
        grouped[records[position]["kind"]].append(position)
    output: list[Any] = [None] * len(records)
    for family_positions in grouped.values():
        shift = max(1, len(family_positions) // 3)
        for offset, position in enumerate(family_positions):
            output[position] = records[family_positions[(offset + shift) % len(family_positions)]][key]
    if any(value is None for value in output):
        raise RuntimeError("failed to create MassBank cyclic control")
    return output


def matched_observations(
    values: np.ndarray,
    intensities: np.ndarray,
    targets: np.ndarray,
    ppm: float,
    floor_da: float,
) -> np.ndarray:
    output = np.zeros(len(targets), dtype=np.float64)
    if not len(targets) or not len(values):
        return output
    order = np.argsort(targets, kind="stable")
    sorted_targets = targets[order]
    maximum_tolerance = max(floor_da, float(np.max(np.abs(targets))) * ppm * 1e-6)
    for value, intensity in zip(values, intensities):
        left = int(np.searchsorted(sorted_targets, value - maximum_tolerance, side="left"))
        right = int(np.searchsorted(sorted_targets, value + maximum_tolerance, side="right"))
        for sorted_position in range(left, right):
            original = order[sorted_position]
            tolerance = max(floor_da, abs(targets[original]) * ppm * 1e-6)
            if abs(value - targets[original]) <= tolerance:
                output[original] = max(output[original], float(intensity))
    return output


def protonated_parent_formula(value: str) -> dict[str, int]:
    parsed = parse_formula(value)
    if parsed is None:
        raise ValueError(f"invalid candidate formula: {value}")
    output = dict(parsed)
    output["H"] = output.get("H", 0) + 1
    return output


def charged_species_formula(value: str) -> bool:
    """Net-charge notation marks an inherently ionic species.

    Such a candidate has no neutral protonatable parent, so MassBank
    neutral-fragment rules do not apply to it.  It is skipped (no evidence
    emitted, native edges retained downstream) rather than crashing the
    full-graph run or being scored against a fabricated parent formula.
    """
    return value.endswith(("+", "-"))


def score_rules(
    observations: np.ndarray,
    candidates_formula: list[dict[str, int]],
    candidates_bits: list[frozenset[int]],
    candidates_ik14: list[str],
    rule_formula: list[dict[str, int]],
    rule_bits: list[frozenset[int]],
    rule_source_identities: list[set[str]],
    rule_weight: np.ndarray,
    minimum_coverage: float,
    suppress_source_identity: bool,
) -> tuple[np.ndarray, int, int]:
    candidate_count = len(candidates_formula)
    numerator = np.zeros(candidate_count, dtype=np.float64)
    denominator = 0.0
    exact_suppressed = 0
    active_discriminative = 0
    for rule_index in np.flatnonzero(observations > 0):
        predicate = rule_bits[int(rule_index)]
        if not predicate:
            continue
        compatible = np.zeros(candidate_count, dtype=bool)
        coverage = np.zeros(candidate_count, dtype=np.float64)
        for candidate in range(candidate_count):
            if not contains_formula(candidates_formula[candidate], rule_formula[int(rule_index)]):
                continue
            candidate_coverage = (
                len(candidates_bits[candidate].intersection(predicate)) / len(predicate)
                if candidates_bits[candidate] else 0.0
            )
            if suppress_source_identity and candidates_ik14[candidate] in rule_source_identities[int(rule_index)]:
                if candidate_coverage >= minimum_coverage:
                    exact_suppressed += 1
                continue
            coverage[candidate] = candidate_coverage
            compatible[candidate] = candidate_coverage >= minimum_coverage
        count = int(np.sum(compatible))
        if count == 0 or count == candidate_count:
            continue
        idf = math.log((candidate_count + 1.0) / (count + 1.0))
        weight = float(observations[rule_index] * rule_weight[rule_index] * idf)
        numerator += weight * coverage
        denominator += weight
        active_discriminative += 1
    if denominator <= 0:
        return np.zeros(candidate_count, dtype=np.float64), exact_suppressed, 0
    return numerator / denominator, exact_suppressed, active_discriminative


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if (
        args.top_peaks < 16
        or not 0 < args.intensity_power <= 1
        or not 0.4 <= args.minimum_structure_coverage <= 0.9
    ):
        raise ValueError("MassBank source thresholds were weakened")
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    if corpus.get("schema") != "chemaware.massbank-annotated-fragment-evidence.v1":
        raise RuntimeError("MassBank annotated fragment corpus schema is not recognized")
    records = corpus["records"]
    radius = int(corpus["aggregation"]["morgan_radius"])
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius)
    rows, by_query = read_candidates(args.candidate_ledger)
    query_registry = read_query_registry(args.query_registry)
    if set(query_registry) != set(by_query):
        raise RuntimeError("candidate ledger and query registry cover different queries")

    record_mass = np.asarray([record["exact_mass"] for record in records], dtype=np.float64)
    mass_control = np.asarray(cyclic_values(records, "exact_mass"), dtype=np.float64)
    record_formula = [parse_formula(record["formula"]) for record in records]
    if any(value is None for value in record_formula):
        raise RuntimeError("MassBank corpus has an invalid formula")
    record_formula = [value for value in record_formula if value is not None]
    record_bits = [frozenset(int(bit) for bit in record["consensus_morgan_bits"]) for record in records]
    predicate_control_formula = [
        parse_formula(str(value)) for value in cyclic_values(records, "formula")
    ]
    if any(value is None for value in predicate_control_formula):
        raise RuntimeError("MassBank predicate control has an invalid formula")
    predicate_control_formula = [value for value in predicate_control_formula if value is not None]
    predicate_control_bits = [
        frozenset(int(bit) for bit in values)
        for values in cyclic_values(records, "consensus_morgan_bits")
    ]
    record_source = [set(record["source_ik14"]) for record in records]
    empty_source = [set() for _ in records]
    record_weight = np.asarray(
        [math.log1p(float(record["source_identity_count"])) for record in records],
        dtype=np.float64,
    )
    family_indices = {
        kind: np.asarray([index for index, record in enumerate(records) if record["kind"] == kind])
        for kind in ("product_ion", "neutral_loss")
    }
    fingerprint_by_identity: dict[str, frozenset[int]] = {}
    for row in rows:
        fingerprint_by_identity.setdefault(
            row["ik14"], candidate_fingerprint(row["smiles"], generator)
        )

    output_rows = []
    charged_skipped = 0
    active_queries = Counter()
    discriminative_counts = Counter()
    exact_suppressed = Counter()
    query_rows = sorted(by_query)
    with h5py.File(args.data, "r") as handle:
        for query in query_rows:
            registry = query_registry[query]
            if polarity(registry["adduct"]) != 1:
                continue
            spectrum_row = registry["spectrum_row"]
            spectrum = np.asarray(handle["spectrum"][spectrum_row], dtype=np.float64)
            precursor = float(handle["precursor_mz"][spectrum_row])
            valid = (spectrum[0] > 0) & np.isfinite(spectrum[0]) & (spectrum[1] > 0)
            mz = spectrum[0, valid]
            intensity = spectrum[1, valid]
            order = np.argsort(-intensity, kind="stable")[:args.top_peaks]
            mz = mz[order]
            intensity = intensity[order]
            intensity = intensity / max(float(np.max(intensity)), 1e-12)
            intensity = np.power(intensity, args.intensity_power)
            positions = by_query[query]
            scoreable = [
                pos for pos in positions
                if not charged_species_formula(rows[pos]["formula"])
            ]
            charged_skipped += len(positions) - len(scoreable)
            positions = scoreable
            if not positions:
                continue
            candidates_formula = [protonated_parent_formula(rows[pos]["formula"]) for pos in positions]
            candidates_bits = [fingerprint_by_identity[rows[pos]["ik14"]] for pos in positions]
            candidates_ik14 = [rows[pos]["ik14"] for pos in positions]
            for kind, indices in family_indices.items():
                if not len(indices):
                    continue
                values = precursor - mz if kind == "neutral_loss" else mz
                observed = matched_observations(
                    values, intensity, record_mass[indices], args.ppm, args.floor_da,
                )
                shifted = matched_observations(
                    values, intensity, mass_control[indices], args.ppm, args.floor_da,
                )
                scores, suppressed, active = score_rules(
                    observed,
                    candidates_formula,
                    candidates_bits,
                    candidates_ik14,
                    [record_formula[i] for i in indices],
                    [record_bits[i] for i in indices],
                    [record_source[i] for i in indices],
                    record_weight[indices],
                    args.minimum_structure_coverage,
                    True,
                )
                mass_scores, _, _ = score_rules(
                    shifted,
                    candidates_formula,
                    candidates_bits,
                    candidates_ik14,
                    [record_formula[i] for i in indices],
                    [record_bits[i] for i in indices],
                    empty_source[:len(indices)],
                    record_weight[indices],
                    args.minimum_structure_coverage,
                    False,
                )
                predicate_scores, _, _ = score_rules(
                    observed,
                    candidates_formula,
                    candidates_bits,
                    candidates_ik14,
                    [predicate_control_formula[i] for i in indices],
                    [predicate_control_bits[i] for i in indices],
                    empty_source[:len(indices)],
                    record_weight[indices],
                    args.minimum_structure_coverage,
                    False,
                )
                family = f"massbank_recurrent_{kind}_{args.scope}"
                if np.ptp(scores) > 0:
                    active_queries[family] += 1
                discriminative_counts[family] += active
                exact_suppressed[family] += suppressed
                for local, position in enumerate(positions):
                    row = rows[position]
                    output_rows.append({
                        "manifest_query": query,
                        "local_candidate": int(row["local_candidate"]),
                        "ik14": row["ik14"],
                        "formula": row["formula"],
                        "source_family": family,
                        "scope": args.scope,
                        "source_score": f"{scores[local]:.17g}",
                        "control_a_score": f"{mass_scores[local]:.17g}",
                        "control_b_score": f"{predicate_scores[local]:.17g}",
                        "controls_available": 1,
                    })

    families = sorted({row["source_family"] for row in output_rows})
    report = {
        "status": "CHEMAWARE_CANDIDATE_SOURCE_LEDGER_UNQUALIFIED",
        "truth_fields_exported": False,
        "source_families": {
            family: {
                "scope": args.scope,
                "larger_is_better": True,
                "confidence_tier": "C_calibration_required",
                "matched_controls": ["rule_mass_cyclic", "formula_structure_predicate_cyclic"],
                "specificity_gate_passed": False,
            }
            for family in families
        },
        "queries": len(query_rows),
        "candidates": len(rows),
        "charged_species_candidates_skipped": charged_skipped,
        "candidate_source_rows": len(output_rows),
        "rules": len(records),
        "rules_by_kind": {kind: int(len(indices)) for kind, indices in family_indices.items()},
        "active_queries": dict(active_queries),
        "mean_discriminative_rules_per_query": {
            family: discriminative_counts[family] / max(len(query_rows), 1)
            for family in families
        },
        "exact_source_identity_matches_suppressed": dict(exact_suppressed),
        "scope": args.scope,
        "formal_triplet_mining_authorized": False,
        "provenance": {
            "corpus": str(args.corpus),
            "corpus_sha256": sha256_file(args.corpus),
            "candidate_ledger": str(args.candidate_ledger),
            "candidate_ledger_sha256": sha256_file(args.candidate_ledger),
            "query_registry": str(args.query_registry),
            "query_registry_sha256": sha256_file(args.query_registry),
            "data": str(args.data),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_massbank_source_", dir=args.output.parent))
    try:
        if not output_rows:
            raise RuntimeError("no scoreable candidates produced a source row")
        with (temporary / "candidate_scores.tsv").open(
            "w", encoding="utf-8", newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(output_rows[0]), delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(output_rows)
        report["candidate_scores_sha256"] = sha256_file(temporary / "candidate_scores.tsv")
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
