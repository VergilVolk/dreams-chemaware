"""Formula-OOF audit of candidate-differential motif x mass-event rules.

The previous one/two-cut audits require a brittle hard fragment assignment.
This preflight instead represents a chemical rule as a training-derived
interaction between:

* a local Morgan environment present in the true structure but absent from the
  current adversary (or vice versa); and
* an observed fragment-m/z or neutral-loss bin at one query peak.

The interactions are feature-hashed with fixed, outcome-independent hashes and
fed only to the training-time A4 action selector.  Three matched controls test
whether any gain comes from the correct chemical pairing: zero-padded base,
same-formula structure-pair permutation, and within-query peak permutation.
No DreaMS parameter is updated and the sealed P3 set is not read.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import GroupKFold

from audit_chemaware_candidate_differential_rules import decode, stable_u64
from audit_chemaware_rules_for_safe_actions import formula_bootstrap_difference, run_arm
from train_noise_v3_a4_nonlinear_action_teacher import build_variant_table


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--a4-dir", type=Path,
        default=ROOT / "data/validation/g8r_noise_v3_a4_exact_peak_scan",
    )
    parser.add_argument(
        "--hdf5", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-queries", type=int, default=500)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[260903])
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--risk-penalty", type=float, default=2.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--max-exclusive-bits", type=int, default=48)
    parser.add_argument("--event-bin-da", type=float, default=0.05)
    parser.add_argument(
        "--event-source", choices=("mass_bins", "rule_catalog"), default="mass_bins",
    )
    parser.add_argument(
        "--rule-catalog", type=Path,
        default=ROOT / "dreams/models/chem_aware/chem_rules_data.json",
    )
    parser.add_argument("--hash-dim", type=int, default=256)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=260903)
    return parser.parse_args()


def sparse_bits(smiles: str, generator) -> set[int]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return set()
    return set(map(int, generator.GetSparseCountFingerprint(molecule).GetNonzeroElements()))


def bounded(values: set[int], maximum: int, key: int) -> tuple[int, ...]:
    return tuple(sorted(values, key=lambda value: stable_u64(value, seed=key))[:maximum])


def hashed_rule_vector(
    true_only: tuple[int, ...],
    adversary_only: tuple[int, ...],
    mz: float,
    neutral_loss: float,
    dimension: int,
    bin_da: float,
    seed: int,
    catalog_events: tuple[tuple[str, int], ...] | None = None,
) -> np.ndarray:
    output = np.zeros(dimension, dtype=np.float32)
    fragment_bin = int(round(mz / bin_da))
    loss_bin = int(round(neutral_loss / bin_da))
    events = []
    for side, bits in (("T", true_only), ("A", adversary_only)):
        for bit in bits:
            if catalog_events is None:
                events.append((side, "F", bit, fragment_bin))
                if neutral_loss > 0:
                    events.append((side, "N", bit, loss_bin))
            else:
                events.extend((side, category, bit, rule) for category, rule in catalog_events)
    if not events:
        return output
    for event in events:
        raw = stable_u64(*event, seed=seed)
        index = raw % dimension
        sign = 1.0 if (raw >> 63) == 0 else -1.0
        output[index] += sign
    output /= math.sqrt(len(events))
    return output


def build_features(table, query_path: Path, hdf5_path: Path, args):
    queries = pd.read_csv(query_path).set_index("scan_position")
    positions = np.unique(table.query_position)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=args.radius)
    catalog = None
    if args.event_source == "rule_catalog":
        source = json.loads(args.rule_catalog.read_text(encoding="utf-8"))["rules"]
        catalog = {}
        for category in ("CF", "NL"):
            values = sorted({
                round(float(rule["value"]), 6)
                for rule in source
                if rule.get("category") == category
                and rule.get("match_type") in {"peak_mz", "mass_diff"}
                and rule.get("mode", "pos+neg") in {"pos", "positive", "pos+neg"}
            })
            catalog[category] = np.asarray(values, dtype=np.float64)

    def matched_catalog_events(mz: float, loss: float) -> tuple[tuple[str, int], ...]:
        if catalog is None:
            return ()
        output = []
        for category, observed in (("CF", mz), ("NL", loss)):
            values = catalog[category]
            if observed <= 0 or not len(values):
                continue
            tolerance = max(0.01, abs(observed) * 20e-6)
            left = bisect.bisect_left(values, observed - tolerance)
            right = bisect.bisect_right(values, observed + tolerance)
            output.extend((category, int(index)) for index in range(left, right))
        return tuple(output)
    structure: dict[int, dict] = {}
    with h5py.File(hdf5_path, "r") as handle:
        for position in positions:
            row = queries.loc[int(position)]
            query_row = int(row["query_row"])
            adversary_row = int(row["baseline_adversarial_pair_row"])
            truth = sparse_bits(decode(handle["smiles"][query_row]), generator)
            adversary = sparse_bits(decode(handle["smiles"][adversary_row]), generator)
            structure[int(position)] = {
                "formula": str(row["query_formula"]),
                "query_index": int(row["query_index"]),
                "precursor": float(handle["precursor_mz"][query_row]),
                "true_only": bounded(truth - adversary, args.max_exclusive_bits, args.seed),
                "adversary_only": bounded(adversary - truth, args.max_exclusive_bits, args.seed + 1),
            }

    # Same-formula cyclic structure donor. Singleton formulas remain fixed and
    # are reported; the separate within-query peak permutation has no fixed
    # points when a spectrum has at least two eligible peaks.
    donor: dict[int, int] = {}
    groups: dict[str, list[int]] = {}
    for position in positions:
        groups.setdefault(structure[int(position)]["formula"], []).append(int(position))
    for formula, values in groups.items():
        values.sort(key=lambda p: stable_u64(formula, structure[p]["query_index"], seed=args.seed + 2))
        for index, position in enumerate(values):
            donor[position] = values[(index + 1) % len(values)]

    correct_by_key: dict[tuple[int, int], np.ndarray] = {}
    structure_control_by_key: dict[tuple[int, int], np.ndarray] = {}
    peak_control_by_key: dict[tuple[int, int], np.ndarray] = {}
    for position in positions:
        indices = np.flatnonzero(table.query_position == position)
        tokens = np.unique(table.token[indices])
        token_vectors = []
        donor_position = donor[int(position)]
        current, shuffled = structure[int(position)], structure[donor_position]
        for token in sorted(tokens):
            local = indices[table.token[indices] == token][0]
            mz = float(table.x[local, 2] * 1000.0)
            loss = current["precursor"] - mz
            events = matched_catalog_events(mz, loss) if catalog is not None else None
            correct_by_key[(int(position), int(token))] = hashed_rule_vector(
                current["true_only"], current["adversary_only"], mz, loss,
                args.hash_dim, args.event_bin_da, args.seed + 3, events,
            )
            structure_control_by_key[(int(position), int(token))] = hashed_rule_vector(
                shuffled["true_only"], shuffled["adversary_only"], mz, loss,
                args.hash_dim, args.event_bin_da, args.seed + 3, events,
            )
            token_vectors.append(correct_by_key[(int(position), int(token))])
        ordered = sorted(tokens)
        if len(ordered) > 1:
            shift = 1 + stable_u64(current["query_index"], seed=args.seed + 4) % (len(ordered) - 1)
            rotated = np.roll(np.asarray(token_vectors), int(shift), axis=0)
        else:
            rotated = np.asarray(token_vectors)
        for index, token in enumerate(ordered):
            peak_control_by_key[(int(position), int(token))] = rotated[index]

    def expand(source: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
        return np.vstack([
            source[(int(position), int(token))]
            for position, token in zip(table.query_position, table.token)
        ]).astype(np.float32)
    coverage = {
        "queries": int(len(positions)),
        "same_formula_structure_permutation_fixed_fraction": float(np.mean([
            donor[int(position)] == int(position) for position in positions
        ])),
        "any_true_exclusive_motif_fraction": float(np.mean([
            len(structure[int(position)]["true_only"]) > 0 for position in positions
        ])),
        "any_adversary_exclusive_motif_fraction": float(np.mean([
            len(structure[int(position)]["adversary_only"]) > 0 for position in positions
        ])),
        "median_true_exclusive_motifs": float(np.median([
            len(structure[int(position)]["true_only"]) for position in positions
        ])),
        "median_adversary_exclusive_motifs": float(np.median([
            len(structure[int(position)]["adversary_only"]) for position in positions
        ])),
        "nonzero_rule_feature_variant_fraction": float(np.mean([
            np.any(correct_by_key[(int(position), int(token))])
            for position, token in zip(table.query_position, table.token)
        ])),
    }
    return expand(correct_by_key), expand(structure_control_by_key), expand(peak_control_by_key), coverage


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.hash_dim < 32 or args.event_bin_da <= 0:
        raise ValueError("invalid hash/event-bin settings")
    RDLogger.DisableLog("rdApp.*")
    table = build_variant_table(args.a4_dir, args.max_queries)
    correct, structure_control, peak_control, coverage = build_features(
        table, args.a4_dir / "scan_queries.csv.gz", args.hdf5, args,
    )
    query_meta = pd.DataFrame({
        "position": table.query_position, "formula": table.formula,
    }).drop_duplicates("position").sort_values("position")
    query_fold = np.full(int(query_meta.position.max()) + 1, -1, dtype=np.int8)
    positions, groups = query_meta.position.to_numpy(np.int64), query_meta.formula.to_numpy(str)
    for fold, (_, test) in enumerate(GroupKFold(args.folds).split(positions, groups=groups)):
        query_fold[positions[test]] = fold
    variant_fold = query_fold[table.query_position]
    names = [f"motif_event_hash_{index:03d}" for index in range(args.hash_dim)]
    arms = {
        "base_matched_capacity": run_arm(table, np.zeros_like(correct), names, variant_fold, args),
        "correct_motif_event_rules": run_arm(table, correct, names, variant_fold, args),
        "same_formula_structure_permuted": run_arm(table, structure_control, names, variant_fold, args),
        "within_query_peak_permuted": run_arm(table, peak_control, names, variant_fold, args),
    }
    comparisons = {}
    for index, control in enumerate((
        "base_matched_capacity", "same_formula_structure_permuted", "within_query_peak_permuted",
    )):
        comparisons[f"correct_minus_{control}"] = formula_bootstrap_difference(
            arms["correct_motif_event_rules"]["selected"], arms[control]["selected"],
            args.bootstrap_resamples, args.seed + index,
        )
    pass_gate = bool(
        all(value["ci_low"] > 0 for value in comparisons.values())
        and coverage["any_true_exclusive_motif_fraction"] >= 0.50
        and coverage["any_adversary_exclusive_motif_fraction"] >= 0.50
    )
    report = {
        "status": "candidate_differential_motif_event_rule_preflight",
        "formal_training_authorized": False,
        "dreaMS_parameters_updated": False,
        "queries": int(len(np.unique(table.query_position))),
        "variants": int(len(table.x)),
        "formulas": int(len(np.unique(table.formula))),
        "formula_fold_overlap": 0,
        "rule_representation": {
            "local_environment": f"unfolded Morgan radius {args.radius}",
            "candidate_relation": "true-only versus current-adversary-only environment",
            "spectral_event": (
                f"fragment and neutral-loss bins at {args.event_bin_da} Da"
                if args.event_source == "mass_bins"
                else "positive-mode CF/NL events from the curated 335-rule catalog"
            ),
            "event_source": args.event_source,
            "feature_hash_dimension": args.hash_dim,
            "maximum_exclusive_bits_per_side": args.max_exclusive_bits,
        },
        "coverage": coverage,
        "arms": {
            name: {key: value for key, value in result.items() if key != "selected"}
            for name, result in arms.items()
        },
        "paired_20pct_policy_differences": comparisons,
        "pass_to_counterfactual_training_mechanism": pass_gate,
        "gate": (
            "Correct motif-event pairing must beat equal-capacity base, same-formula structure "
            "permutation, and within-query peak permutation with positive formula-cluster CI lower "
            "bounds before any shared-encoder training arm is authorized."
        ),
        "claim_limit": (
            "Hashed interactions test whether a learnable chemical rule family exists. They are not "
            "yet human-readable mechanisms and do not constitute a DreaMS performance result."
        ),
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, result in arms.items():
        result["selected"].to_csv(args.output_dir / f"{name}_selected.csv.gz", index=False, compression="gzip")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
