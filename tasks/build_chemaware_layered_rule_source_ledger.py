"""Create truth-blind candidate scores from qualified layered chemical rules.

Only curated, formula-specified NL/fragment observations and independently
confirmed structure-conditioned observations are executable here.  The
observed query spectrum is never altered and no peak is synthesized.  Two
equal-capacity nulls are emitted with every score:

* mass-shifted: rule masses are cyclically reassigned within rule type;
* formula/predicate-swapped: chemical prerequisites are cyclically reassigned.

This program does not inspect candidate truth.  A separate formula-disjoint
qualification step must pass before the ledger may mine native triplets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


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
        default=ROOT / "data/validation/chemaware_layered_rule_corpus_v5_20260928/rule_corpus.json",
    )
    parser.add_argument(
        "--candidate-ledger", type=Path,
        default=(ROOT / "data/validation/chemaware_sirius_source_panel_v9_profiled_collision_20260928"
                 / "candidate_ledger.tsv"),
    )
    parser.add_argument(
        "--query-registry", type=Path,
        default=(ROOT / "data/validation/chemaware_sirius_source_panel_v9_profiled_collision_20260928"
                 / "query_registry.tsv"),
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
    return parser.parse_args()


def parse_formula(value: str) -> dict[str, int] | None:
    value = str(value).strip()
    if not value or any(character in value for character in "[]+-."):
        return None
    tokens = list(FORMULA_TOKEN.finditer(value))
    if not tokens or "".join(match.group(0) for match in tokens) != value:
        return None
    result: dict[str, int] = {}
    for match in tokens:
        element = match.group(1)
        count = int(match.group(2) or 1)
        result[element] = result.get(element, 0) + count
    return result


def contains_formula(parent: dict[str, int], child: dict[str, int]) -> bool:
    return all(parent.get(element, 0) >= count for element, count in child.items())


def decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def read_candidates(path: Path) -> tuple[list[dict[str, Any]], dict[int, list[int]]]:
    rows: list[dict[str, Any]] = []
    by_query: dict[int, list[int]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            parsed = {
                "manifest_query": int(row["manifest_query"]),
                "local_candidate": int(row["local_candidate"]),
                "ik14": row["ik14"],
                "formula": row["formula"],
                "smiles": row["smiles"],
                "feature_id": row["feature_id"],
            }
            by_query[parsed["manifest_query"]].append(len(rows))
            rows.append(parsed)
    if not rows:
        raise RuntimeError("candidate ledger is empty")
    for query, positions in by_query.items():
        local = [rows[position]["local_candidate"] for position in positions]
        if local != list(range(len(local))):
            raise RuntimeError(f"candidate ledger is not contiguous for query {query}")
    return rows, dict(by_query)


def read_query_rows(path: Path) -> dict[int, int]:
    output: dict[int, int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            query = int(row["manifest_query"])
            spectrum_row = int(row["spectrum_row"])
            if query in output:
                raise RuntimeError(f"query registry repeats manifest query {query}")
            output[query] = spectrum_row
    if not output:
        raise RuntimeError("query registry is empty")
    return output


def executable_rules(
    corpus: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    formula_families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    structure_rules = []
    for record in corpus["records"]:
        tier = record["confidence_tier"]
        if (
            record["record_kind"] != "curated_multi_record_observation"
            and tier not in {"A_physical_or_confirmed", "B_curated_conditional"}
        ):
            continue
        if (
            record["record_kind"] == "curated_multi_record_observation"
            and tier not in {"B_curated_conditional", "C_calibration_required"}
        ):
            continue
        if record["record_kind"] == "fixed_mass_observation":
            payload = record["payload"]
            formula = parse_formula(str(payload.get("formula", "")))
            if formula is None or payload.get("match_type") not in {"mass_diff", "peak_mz"}:
                continue
            mode = str(payload.get("mode", ""))
            if mode not in {"pos", "pos+neg", "positive"}:
                continue
            formula_families["layered_curated_formula_observations"].append({
                "record_id": record["record_id"],
                "kind": payload["match_type"],
                "mass": float(payload["value"]),
                "formula": formula,
                "weight": float(record["admission_weight"]),
            })
        elif record["record_kind"] == "curated_multi_record_observation":
            payload = record["payload"]
            if payload.get("ion_mode") != "positive":
                continue
            formula = parse_formula(str(payload.get("formula", "")))
            if formula is None:
                continue
            family = (
                "msfinder_recurrent_formula_observations"
                if record["confidence_tier"] == "B_curated_conditional"
                else "msfinder_longtail_formula_observations"
            )
            formula_families[family].append({
                "record_id": record["record_id"],
                "kind": (
                    "mass_diff" if payload["kind"] == "neutral_loss" else "peak_mz"
                ),
                "mass": float(payload["exact_mass"]),
                "formula": formula,
                "weight": (
                    float(record["admission_weight"])
                    * math.log1p(float(payload["empirical_frequency"]))
                ),
            })
        elif record["record_kind"] == "structure_conditioned_association":
            payload = record["payload"]
            observation = payload["observation"]
            queries = [Chem.MolFromSmarts(value) for value in payload["parent_predicate"]["smarts_any"]]
            if not queries or any(value is None for value in queries):
                raise RuntimeError(f"invalid admitted SMARTS: {record['record_id']}")
            structure_rules.append({
                "record_id": record["record_id"],
                "kind": "mass_diff" if observation["kind"] == "neutral_loss" else "peak_mz",
                "mass": float(observation["exact_mass_da"]),
                "tolerance": float(observation["tolerance_da"]),
                "queries": queries,
                "weight": float(record["admission_weight"]),
            })
    if not formula_families or any(not rules for rules in formula_families.values()):
        raise RuntimeError("layered corpus contains no executable formula observation")
    return dict(formula_families), structure_rules


def cyclic_controls(rules: list[dict[str, Any]], key: str) -> list[Any]:
    output: list[Any] = [None] * len(rules)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, rule in enumerate(rules):
        grouped[str(rule["kind"])].append(index)
    for positions in grouped.values():
        shift = max(1, len(positions) // 3)
        for offset, position in enumerate(positions):
            output[position] = rules[positions[(offset + shift) % len(positions)]][key]
    if any(value is None for value in output):
        raise RuntimeError("failed to construct equal-capacity cyclic control")
    return output


def observation_strength(
    mz: np.ndarray, intensity: np.ndarray, precursor: float,
    targets: np.ndarray, neutral: np.ndarray, ppm: float, floor_da: float,
) -> np.ndarray:
    output = np.zeros(len(targets), dtype=np.float64)
    for is_neutral in (False, True):
        positions = np.flatnonzero(neutral == is_neutral)
        if not len(positions):
            continue
        values = precursor - mz if is_neutral else mz
        selected_targets = targets[positions]
        tolerance = np.maximum(floor_da, np.abs(selected_targets) * ppm * 1e-6)
        distance = np.abs(values[:, None] - selected_targets[None, :])
        matched = distance <= tolerance[None, :]
        output[positions] = np.max(
            np.where(matched, intensity[:, None], 0.0), axis=0,
        )
    return output


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.top_peaks < 8 or not 0 < args.intensity_power <= 1:
        raise ValueError("invalid peak-selection settings")
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    if corpus.get("schema") != "chemaware.layered-chemical-evidence.v4":
        raise RuntimeError("unexpected layered corpus schema")
    formula_families, structure_rules = executable_rules(corpus)
    rows, by_query = read_candidates(args.candidate_ledger)
    query_spectrum_row = read_query_rows(args.query_registry)
    if set(query_spectrum_row) != set(by_query):
        raise RuntimeError("candidate ledger and query registry cover different queries")
    formula_runtime = {}
    for family, rules in formula_families.items():
        formula_runtime[family] = {
            "rules": rules,
            "mass": np.asarray([rule["mass"] for rule in rules], dtype=np.float64),
            "neutral": np.asarray([rule["kind"] == "mass_diff" for rule in rules]),
            "weight": np.asarray([rule["weight"] for rule in rules], dtype=np.float64),
            "control_mass": np.asarray(cyclic_controls(rules, "mass"), dtype=np.float64),
            "control_formula": cyclic_controls(rules, "formula"),
        }

    output_rows: list[dict[str, Any]] = []
    query_rows = sorted(by_query)
    with h5py.File(args.data, "r") as handle:
        for query in query_rows:
            spectrum_row = query_spectrum_row[query]
            if spectrum_row < 0 or spectrum_row >= len(handle["spectrum"]):
                raise RuntimeError(f"query {query} spectrum row is out of range")
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
            structure_observation = []
            for rule in structure_rules:
                target = np.asarray([rule["mass"]], dtype=np.float64)
                neutral = np.asarray([rule["kind"] == "mass_diff"])
                structure_observation.append(float(observation_strength(
                    mz, intensity, precursor, target, neutral,
                    args.ppm, max(args.floor_da, rule["tolerance"]),
                )[0]))

            candidate_molecules = [
                Chem.MolFromSmiles(str(rows[position]["smiles"]))
                for position in by_query[query]
            ]
            parent_formulas = []
            for position in by_query[query]:
                parent = parse_formula(str(rows[position]["formula"]))
                if parent is None:
                    raise RuntimeError(f"invalid candidate formula at query {query}")
                parent_formulas.append(parent)
            candidate_count = len(parent_formulas)
            family_scores = {}
            for family, runtime in formula_runtime.items():
                rules = runtime["rules"]
                formula_presence_matrix = np.asarray([
                    [contains_formula(parent, rule["formula"]) for rule in rules]
                    for parent in parent_formulas
                ], dtype=np.float64)
                swapped_presence_matrix = np.asarray([
                    [contains_formula(parent, child) for child in runtime["control_formula"]]
                    for parent in parent_formulas
                ], dtype=np.float64)
                correct_observation = observation_strength(
                    mz, intensity, precursor, runtime["mass"], runtime["neutral"],
                    args.ppm, args.floor_da,
                )
                shifted_observation = observation_strength(
                    mz, intensity, precursor, runtime["control_mass"], runtime["neutral"],
                    args.ppm, args.floor_da,
                )
                frequency = formula_presence_matrix.sum(axis=0)
                idf = np.log((candidate_count + 1.0) / (frequency + 1.0))
                control_frequency = swapped_presence_matrix.sum(axis=0)
                control_idf = np.log((candidate_count + 1.0) / (control_frequency + 1.0))
                weighted = correct_observation * runtime["weight"]
                shifted_weighted = shifted_observation * runtime["weight"]
                denominator = max(float(np.sum(weighted)), 1e-12)
                family_scores[family] = (
                    (formula_presence_matrix @ (weighted * idf)) / denominator,
                    (formula_presence_matrix @ (shifted_weighted * idf))
                    / max(float(np.sum(shifted_weighted)), 1e-12),
                    (swapped_presence_matrix @ (weighted * control_idf)) / denominator,
                )

            if structure_rules:
                structure_correct = np.zeros(candidate_count, dtype=np.float64)
                structure_swapped = np.zeros(candidate_count, dtype=np.float64)
                shift = max(1, len(structure_rules) // 2)
                for rule_index, rule in enumerate(structure_rules):
                    value = structure_observation[rule_index] * rule["weight"]
                    swapped_rule = structure_rules[(rule_index + shift) % len(structure_rules)]
                    for candidate, molecule in enumerate(candidate_molecules):
                        if molecule is not None and any(
                            molecule.HasSubstructMatch(pattern) for pattern in rule["queries"]
                        ):
                            structure_correct[candidate] += value
                        if molecule is not None and any(
                            molecule.HasSubstructMatch(pattern) for pattern in swapped_rule["queries"]
                        ):
                            structure_swapped[candidate] += value
            else:
                structure_correct = np.zeros(candidate_count, dtype=np.float64)
                structure_swapped = np.zeros(candidate_count, dtype=np.float64)

            for local, position in enumerate(by_query[query]):
                common = rows[position]
                for family, scores_for_family in family_scores.items():
                    correct_score, mass_control_score, formula_control_score = scores_for_family
                    output_rows.append({
                        "manifest_query": query,
                        "local_candidate": local,
                        "ik14": common["ik14"],
                        "formula": common["formula"],
                        "source_family": family,
                        "scope": "cross_formula",
                        "source_score": f"{correct_score[local]:.17g}",
                        "control_a_score": f"{mass_control_score[local]:.17g}",
                        "control_b_score": f"{formula_control_score[local]:.17g}",
                        "controls_available": 1,
                    })
                output_rows.append({
                        "manifest_query": query,
                        "local_candidate": local,
                        "ik14": common["ik14"],
                        "formula": common["formula"],
                        "source_family": "layered_confirmed_structure_observations",
                        "scope": "within_formula",
                        "source_score": f"{structure_correct[local]:.17g}",
                        "control_a_score": "0",
                        "control_b_score": f"{structure_swapped[local]:.17g}",
                        "controls_available": 1,
                    })

    report = {
        "status": "CHEMAWARE_CANDIDATE_SOURCE_LEDGER_UNQUALIFIED",
        "truth_fields_exported": False,
        "source_families": {
            "layered_curated_formula_observations": {
                "scope": "cross_formula", "larger_is_better": True,
                "confidence_tier": "B_curated_conditional",
                "matched_controls": ["mass_cyclic", "formula_cyclic"],
                "specificity_gate_passed": False,
            },
            "msfinder_recurrent_formula_observations": {
                "scope": "cross_formula", "larger_is_better": True,
                "confidence_tier": "B_curated_conditional",
                "matched_controls": ["within_tier_mass_cyclic", "within_tier_formula_cyclic"],
                "specificity_gate_passed": False,
            },
            "msfinder_longtail_formula_observations": {
                "scope": "cross_formula", "larger_is_better": True,
                "confidence_tier": "C_calibration_required",
                "matched_controls": ["within_tier_mass_cyclic", "within_tier_formula_cyclic"],
                "specificity_gate_passed": False,
            },
            "layered_confirmed_structure_observations": {
                "scope": "within_formula", "larger_is_better": True,
                "confidence_tier": "A_physical_or_confirmed",
                "matched_controls": ["zero_information", "predicate_cyclic"],
                "specificity_gate_passed": False,
            },
        },
        "queries": len(query_rows),
        "candidates": len(rows),
        "candidate_source_rows": len(output_rows),
        "executable_formula_rules": {
            family: len(rules) for family, rules in formula_families.items()
        },
        "confirmed_structure_rules": len(structure_rules),
        "top_peaks": args.top_peaks,
        "intensity_power": args.intensity_power,
        "formal_triplet_mining_authorized": False,
        "provenance": {
            "corpus": str(args.corpus), "corpus_sha256": sha256_file(args.corpus),
            "candidate_ledger": str(args.candidate_ledger),
            "candidate_ledger_sha256": sha256_file(args.candidate_ledger),
            "query_registry": str(args.query_registry),
            "query_registry_sha256": sha256_file(args.query_registry),
            "data": str(args.data),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_layered_source_", dir=args.output.parent))
    try:
        with (temporary / "candidate_scores.tsv").open(
            "w", encoding="utf-8", newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(output_rows[0]), delimiter="\t", lineterminator="\n",
            )
            writer.writeheader(); writer.writerows(output_rows)
        report["candidate_scores_sha256"] = sha256_file(temporary / "candidate_scores.tsv")
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
