"""Score ChemAware candidates with truth-blind MS2LDA MotifDB evidence.

For every query, a motif contributes only when at least two of its curated
fragment/neutral-loss features are observed.  Candidate compatibility is the
coverage of the motif's consensus Morgan environments.  Exact structures used
to auto-annotate a motif are suppressed for that candidate, preventing a
direct library identity lookup.

Two capacity-matched controls are emitted:
* candidate-role cyclic: candidate scores are reassigned within formula where
  possible, then among singleton-formula candidates;
* motif-association cyclic: spectral motifs retain their peaks/losses but use
  another motif's consensus structural predicate of similar complexity.

No truth label is read.  A separate formula-disjoint qualification step decides
whether this source can mine native DreaMS triplets.
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
from typing import Any

import h5py
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from build_chemaware_layered_rule_source_ledger import read_candidates


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
        "--motif-corpus",
        type=Path,
        default=(
            ROOT / "data/validation/chemaware_motifdb_evidence_corpus_v1_20260928"
            / "motif_corpus.json"
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
    parser.add_argument("--minimum-feature-matches", type=int, default=2)
    parser.add_argument("--minimum-structure-coverage", type=float, default=0.5)
    parser.add_argument(
        "--scope",
        choices=("cross_formula", "within_formula", "all_candidates"),
        default="cross_formula",
        help=(
            "Frozen deployment domain. cross_formula is the v1 route selected on "
            "development folds 0-2 before its one confirmation-fold evaluation."
        ),
    )
    return parser.parse_args()


def read_query_registry(path: Path) -> dict[int, dict[str, Any]]:
    output = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            query = int(row["manifest_query"])
            if query in output:
                raise RuntimeError(f"query registry repeats manifest query {query}")
            output[query] = {
                "spectrum_row": int(row["spectrum_row"]),
                "adduct": row["adduct"],
                "precursor_mz": float(row["precursor_mz"]),
            }
    if not output:
        raise RuntimeError("query registry is empty")
    return output


def polarity(adduct: str) -> int:
    value = str(adduct).strip()
    if value.endswith("+"):
        return 1
    if value.endswith("-"):
        return -1
    return 0


def candidate_fingerprint(smiles: str, generator: Any) -> frozenset[int]:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return frozenset()
    return frozenset(int(bit) for bit in generator.GetSparseFingerprint(molecule).GetOnBits())


def feature_match(
    observed_mz: np.ndarray,
    observed_intensity: np.ndarray,
    precursor_mz: float,
    features: list[dict[str, float]],
    neutral_loss: bool,
    motif_accuracy: float,
    ppm: float,
    floor_da: float,
) -> tuple[float, int]:
    if not features:
        return 0.0, 0
    values = precursor_mz - observed_mz if neutral_loss else observed_mz
    target = np.asarray([feature["mz"] for feature in features], dtype=np.float64)
    weight = np.asarray([feature["weight"] for feature in features], dtype=np.float64)
    tolerance = np.maximum.reduce([
        np.full(len(target), floor_da),
        np.full(len(target), motif_accuracy),
        np.abs(target) * ppm * 1e-6,
    ])
    distance = np.abs(values[:, None] - target[None, :])
    matched = distance <= tolerance[None, :]
    best = np.max(np.where(matched, observed_intensity[:, None], 0.0), axis=0)
    active = best > 0
    denominator = max(float(np.sum(weight)), 1e-12)
    return float(np.sum(weight * best) / denominator), int(np.sum(active))


def cyclic_predicates(records: list[dict[str, Any]]) -> list[frozenset[int]]:
    """Rotate predicates within polarity and nearby bit-count complexity."""
    output: list[frozenset[int] | None] = [None] * len(records)
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[int(record["charge"])].append(index)
    for positions in grouped.values():
        positions.sort(key=lambda pos: (len(records[pos]["consensus_morgan_bits"]), pos))
        shift = max(1, len(positions) // 3)
        for offset, position in enumerate(positions):
            donor = positions[(offset + shift) % len(positions)]
            output[position] = frozenset(
                int(value) for value in records[donor]["consensus_morgan_bits"]
            )
    if any(value is None for value in output):
        raise RuntimeError("failed to construct motif-association control")
    return [value for value in output if value is not None]


def cyclic_candidate_scores(scores: np.ndarray, formulas: list[str]) -> np.ndarray:
    output = scores.copy()
    groups: dict[str, list[int]] = defaultdict(list)
    for index, formula in enumerate(formulas):
        groups[formula].append(index)
    singletons = []
    for positions in groups.values():
        if len(positions) == 1:
            singletons.extend(positions)
            continue
        shift = max(1, len(positions) // 2)
        for offset, position in enumerate(positions):
            output[position] = scores[positions[(offset + shift) % len(positions)]]
    if len(singletons) > 1:
        shift = max(1, len(singletons) // 2)
        for offset, position in enumerate(singletons):
            output[position] = scores[singletons[(offset + shift) % len(singletons)]]
    elif len(singletons) == 1 and len(scores) > 1:
        position = singletons[0]
        output[position] = scores[(position + 1) % len(scores)]
    return output


def score_candidates(
    candidate_bits: list[frozenset[int]],
    candidate_ik14: list[str],
    records: list[dict[str, Any]],
    motif_observation: np.ndarray,
    predicates: list[frozenset[int]],
    minimum_coverage: float,
    suppress_source_identity: bool,
) -> tuple[np.ndarray, int, int]:
    candidate_count = len(candidate_bits)
    numerator = np.zeros(candidate_count, dtype=np.float64)
    denominator = 0.0
    exact_overlap_suppressed = 0
    discriminative_motifs = 0
    for motif_index, record in enumerate(records):
        observation = float(motif_observation[motif_index])
        if observation <= 0:
            continue
        predicate = predicates[motif_index]
        if not predicate:
            continue
        coverage = np.asarray([
            len(bits.intersection(predicate)) / len(predicate) if bits else 0.0
            for bits in candidate_bits
        ], dtype=np.float64)
        if suppress_source_identity:
            source_identities = set(record["source_structure_ik14"])
            for candidate, ik14 in enumerate(candidate_ik14):
                if ik14 in source_identities:
                    if coverage[candidate] > 0:
                        exact_overlap_suppressed += 1
                    coverage[candidate] = 0.0
        compatible = coverage >= minimum_coverage
        compatible_count = int(np.sum(compatible))
        if compatible_count == 0 or compatible_count == candidate_count:
            continue
        idf = math.log((candidate_count + 1.0) / (compatible_count + 1.0))
        weight = observation * idf
        numerator += weight * coverage
        denominator += weight
        discriminative_motifs += 1
    if denominator <= 0:
        return np.zeros(candidate_count, dtype=np.float64), exact_overlap_suppressed, 0
    return numerator / denominator, exact_overlap_suppressed, discriminative_motifs


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if (
        args.top_peaks < 16
        or not 0 < args.intensity_power <= 1
        or args.minimum_feature_matches < 2
        or not 0.4 <= args.minimum_structure_coverage <= 0.9
    ):
        raise ValueError("MotifDB candidate-source thresholds were weakened")
    corpus = json.loads(args.motif_corpus.read_text(encoding="utf-8"))
    if corpus.get("schema") != "chemaware.ms2lda-motif-evidence.v1":
        raise RuntimeError("MotifDB evidence corpus schema is not recognized")
    records = corpus["records"]
    radius = int(corpus["fingerprint"]["radius"])
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius)
    predicates = [
        frozenset(int(value) for value in record["consensus_morgan_bits"])
        for record in records
    ]
    control_predicates = cyclic_predicates(records)
    rows, by_query = read_candidates(args.candidate_ledger)
    query_registry = read_query_registry(args.query_registry)
    if set(query_registry) != set(by_query):
        raise RuntimeError("candidate ledger and query registry cover different queries")

    fingerprint_by_identity: dict[str, frozenset[int]] = {}
    for row in rows:
        fingerprint_by_identity.setdefault(
            row["ik14"], candidate_fingerprint(row["smiles"], generator)
        )
    family_name = f"ms2lda_motifdb_consensus_substructure_{args.scope}"
    output_rows = []
    active_queries = 0
    total_discriminative_motifs = 0
    exact_overlap_suppressed = 0
    query_rows = sorted(by_query)
    with h5py.File(args.data, "r") as handle:
        for query in query_rows:
            registry = query_registry[query]
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
            charge = polarity(registry["adduct"])
            motif_observation = np.zeros(len(records), dtype=np.float64)
            for index, record in enumerate(records):
                if charge and int(record["charge"]) != charge:
                    continue
                fragment_score, fragment_matches = feature_match(
                    mz, intensity, precursor, record["fragments"], False,
                    float(record["ms2accuracy"]), args.ppm, args.floor_da,
                )
                loss_score, loss_matches = feature_match(
                    mz, intensity, precursor, record["neutral_losses"], True,
                    float(record["ms2accuracy"]), args.ppm, args.floor_da,
                )
                matches = fragment_matches + loss_matches
                if matches < args.minimum_feature_matches:
                    continue
                feature_count = len(record["fragments"]) + len(record["neutral_losses"])
                motif_observation[index] = (
                    (fragment_score * len(record["fragments"]) +
                     loss_score * len(record["neutral_losses"]))
                    / max(feature_count, 1)
                )

            positions = by_query[query]
            candidate_bits = [fingerprint_by_identity[rows[pos]["ik14"]] for pos in positions]
            candidate_ik14 = [rows[pos]["ik14"] for pos in positions]
            candidate_formulas = [rows[pos]["formula"] for pos in positions]
            scores, suppressed, discriminative = score_candidates(
                candidate_bits, candidate_ik14, records, motif_observation,
                predicates, args.minimum_structure_coverage, True,
            )
            association_control, _, _ = score_candidates(
                candidate_bits, candidate_ik14, records, motif_observation,
                control_predicates, args.minimum_structure_coverage, False,
            )
            candidate_control = cyclic_candidate_scores(scores, candidate_formulas)
            if np.ptp(scores) > 0:
                active_queries += 1
            exact_overlap_suppressed += suppressed
            total_discriminative_motifs += discriminative
            for local, position in enumerate(positions):
                row = rows[position]
                output_rows.append({
                    "manifest_query": query,
                    "local_candidate": local,
                    "ik14": row["ik14"],
                    "formula": row["formula"],
                    "source_family": family_name,
                    "scope": args.scope,
                    "source_score": f"{scores[local]:.17g}",
                    "control_a_score": f"{candidate_control[local]:.17g}",
                    "control_b_score": f"{association_control[local]:.17g}",
                    "controls_available": 1,
                })

    report = {
        "status": "CHEMAWARE_CANDIDATE_SOURCE_LEDGER_UNQUALIFIED",
        "truth_fields_exported": False,
        "source_families": {
            family_name: {
                "scope": args.scope,
                "larger_is_better": True,
                "confidence_tier": "C_calibration_required",
                "matched_controls": [
                    "candidate_role_cyclic_within_formula",
                    "motif_structure_association_cyclic",
                ],
                "specificity_gate_passed": False,
            }
        },
        "queries": len(query_rows),
        "candidates": len(rows),
        "candidate_source_rows": len(output_rows),
        "motifs": len(records),
        "active_queries": active_queries,
        "mean_discriminative_motifs_per_query": (
            total_discriminative_motifs / max(len(query_rows), 1)
        ),
        "exact_source_identity_matches_suppressed": exact_overlap_suppressed,
        "top_peaks": args.top_peaks,
        "intensity_power": args.intensity_power,
        "minimum_feature_matches": args.minimum_feature_matches,
        "minimum_structure_coverage": args.minimum_structure_coverage,
        "scope": args.scope,
        "scope_selection": (
            "cross_formula selected using formula folds 0-2 only; folds 3-4 are reserved "
            "for a single source-qualification evaluation"
            if args.scope == "cross_formula"
            else "non-primary diagnostic scope"
        ),
        "formal_triplet_mining_authorized": False,
        "provenance": {
            "motif_corpus": str(args.motif_corpus),
            "motif_corpus_sha256": sha256_file(args.motif_corpus),
            "candidate_ledger": str(args.candidate_ledger),
            "candidate_ledger_sha256": sha256_file(args.candidate_ledger),
            "query_registry": str(args.query_registry),
            "query_registry_sha256": sha256_file(args.query_registry),
            "data": str(args.data),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_motifdb_source_", dir=args.output.parent))
    try:
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
