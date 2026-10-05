"""Build the large-coverage native-DreaMS Noise Stage-6 triplet corpus.

Stage-6 is an exact continuation from the Stage-1 targeted champion.  It
preserves every effective Stage-1 targeted triplet exactly, adds one clean
identity-preservation event for every eligible outer-train query, and appends
exactly two calibrated, fully noisy triplets per qualifying query.  Anchor and
positive are independently perturbed views of the same identity; the negative
is an independently perturbed source-local view of a different identity and is
retained only when it becomes a harder false match.  Only triplet content
changes; the training runtime remains native.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from audit_noise_dreams_native_official_replay import encode_actions, load_npz
from build_noise_dreams_native_triplets import (
    Registry,
    fixed_unicode,
    query_molecules,
    sha256_file,
    stable_fold,
)
from build_noise_dreams_native_residual_stage2 import (
    assert_exact_checkpoint_reconstruction,
)
from dreams.utils.data import SpectrumPreprocessor
from dreams.utils.dformats import DataFormatA
from evaluate_noise_dreams_native import encode_rows
from noise_dreams_native_spectrum import native_action_model_input
from noise_final_core import CandidateGraph
from train_e1_identity import load_base_model
from train_noise_dreams_native import (
    make_hdf5_spectrum,
    native_query_disjoint_one_pass_batches,
)


STAGE6_BUILDER_VERSION = "noise_native_incremental_two_sided_hard_triplets_v2"
STAGE6_STATUS = "NOISE_DREAMS_NATIVE_SCALE_STAGE6_COMPLETE"
STAGE6_MINIMUM_BROAD_EVENTS = 20_000
STAGE6_MINIMUM_BROAD_QUERIES = 10_000
STAGE6_MAXIMUM_OPTIMIZER_STEPS = 60_000
SEVERITIES = {
    "easy": (0.10, 0.15, 2),
    "medium": (0.20, 0.30, 5),
    "hard": (0.30, 0.40, 8),
}
NEGATIVE_VIEW_REPLICATES = 3
MINIMUM_HARD_DELTA = 0.01
REGISTERED_STAGE1_SOURCES = {
    "A4_exact",
    "E10B",
    "E11",
    "E12B",
    "N_mature",
    "P_guided_original",
    "V4_gradient_path",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-run", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--maximum-positive-rows", type=int, default=8)
    parser.add_argument("--maximum-negative-molecules", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def deterministic_seed(seed: int, query: int, label: str) -> int:
    payload = f"{seed}|{query}|{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def role_formula_is_train_safe(
    formula: str,
    *,
    validation_formulas: set[str],
    outer_fold: int,
    formula_fold_seed: int,
) -> bool:
    formula = str(formula)
    return bool(
        formula not in validation_formulas
        and stable_fold(formula, 5, formula_fold_seed) != outer_fold
    )


def identity_preserving_noise_view(
    clean: np.ndarray, *, source_key: int, severity: str, seed: int,
    replicate: int = 0,
) -> np.ndarray:
    """Apply only acquisition-like nuisance transforms to a native tensor."""
    if severity not in SEVERITIES:
        raise ValueError(severity)
    clean = np.asarray(clean, dtype=np.float32)
    if clean.shape != (101, 2) or abs(float(clean[0, 1]) - 1.1) > 1e-6:
        raise RuntimeError("Stage-6 received a non-native clean tensor")
    fragments = clean[1:][(clean[1:, 0] > 0) & (clean[1:, 1] > 0)].copy()
    if len(fragments) < 8:
        raise RuntimeError("Stage-6 cannot perturb a spectrum with fewer than eight peaks")
    drop_fraction, jitter, added = SEVERITIES[severity]
    # Nuisance generation is source-local and label-blind: the seed contains no
    # query role, candidate rank, action family or correction outcome.  Pairing
    # and hardness are assessed later and cannot alter a generated view.
    rng = np.random.default_rng(deterministic_seed(
        seed, source_key, f"{severity}|replicate={replicate}",
    ))

    # Random peak loss is the central hard-positive nuisance.  Preserve the
    # base peak and at least eight fragments so the view cannot erase identity.
    base = int(np.argmax(fragments[:, 1]))
    removable = np.asarray([i for i in range(len(fragments)) if i != base])
    drop_n = min(int(round(drop_fraction * len(fragments))), len(fragments) - 8)
    if drop_n > 0:
        # Acquisition dropout is biased toward weak fragments.  Removing the
        # strongest identity-bearing peaks preferentially would manufacture an
        # adversarial view rather than a realistic missing-fragment nuisance.
        weights = 1.0 / np.sqrt(np.maximum(fragments[removable, 1], 1e-4))
        weights /= weights.sum()
        dropped = rng.choice(removable, size=drop_n, replace=False, p=weights)
        keep = np.ones(len(fragments), dtype=bool)
        keep[dropped] = False
        fragments = fragments[keep]

    fragments[:, 1] *= rng.uniform(1.0 - jitter, 1.0 + jitter, len(fragments))
    fragments[:, 1] = np.maximum(fragments[:, 1], 1e-5)

    # Tiny background peaks model detector/background clutter.  They are not
    # borrowed from any positive or candidate molecule and therefore cannot
    # leak a target identity or a candidate answer.
    occupied = fragments[:, 0].tolist()
    lower = max(5.0, float(np.min(fragments[:, 0])))
    upper = max(lower + 1.0, min(float(clean[0, 0]), float(np.max(fragments[:, 0])) + 25.0))
    background: list[tuple[float, float]] = []
    attempts = 0
    while len(background) < added and attempts < 200:
        attempts += 1
        mz = float(rng.uniform(lower, upper))
        if min(abs(mz - existing) for existing in occupied) <= 0.02:
            continue
        occupied.append(mz)
        background.append((mz, float(rng.uniform(0.005, 0.03))))
    if len(background) != added:
        raise RuntimeError("Stage-6 could not place deterministic background peaks")
    fragments = np.concatenate(
        (fragments, np.asarray(background, dtype=np.float32)), axis=0,
    )
    fragments = fragments[np.argsort(fragments[:, 0], kind="stable")]
    if len(fragments) > 100:
        keep = np.argsort(fragments[:, 1], kind="stable")[-100:]
        fragments = fragments[np.sort(keep)]
    fragments[:, 1] /= float(np.max(fragments[:, 1]))

    output = np.zeros((101, 2), dtype=np.float32)
    output[0] = clean[0]
    output[1:1 + len(fragments)] = fragments
    return output


def exact_native_clean_tensors(
    data: Path, rows: np.ndarray, preprocessor: SpectrumPreprocessor,
) -> np.ndarray:
    output = np.empty((len(rows), 101, 2), dtype=np.float32)
    with h5py.File(data, "r") as handle:
        for index, row in enumerate(np.asarray(rows, dtype=np.int64)):
            spectrum = make_hdf5_spectrum(
                handle["spectrum"][int(row)], float(handle["precursor_mz"][int(row)]),
            )
            output[index] = preprocessor(
                spectrum.get_peak_list(), prec_mz=spectrum.get_precursor_mz(),
                high_form=False, augment=False,
            )
    return output


def event_members(pool: dict[str, np.ndarray], prefix: str, event: int) -> np.ndarray:
    left, right = map(int, pool[f"{prefix}_ptr"][event:event + 2])
    return np.asarray(pool[f"{prefix}_idx"][left:right], dtype=np.int64)


class PoolAppender:
    """Append full-library events while preserving the Stage-1 pool prefix."""

    def __init__(self, source: dict[str, np.ndarray]) -> None:
        self.registry_kind = list(map(int, source["registry_kind"]))
        self.registry_source = list(map(int, source["registry_source_index"]))
        self.positions: dict[tuple[int, int], int] = {}
        for position, key in enumerate(zip(self.registry_kind, self.registry_source)):
            self.positions.setdefault((int(key[0]), int(key[1])), position)
        self.anchor = list(map(int, source["anchor_idx"]))
        self.positive = list(map(int, source["positive_idx"]))
        self.negative = list(map(int, source["negative_idx"]))
        self.positive_ptr = list(map(int, source["positive_ptr"]))
        self.negative_ptr = list(map(int, source["negative_ptr"]))
        self.kind = list(map(int, source["event_kind"]))
        self.query = list(map(int, source["event_query"]))
        self.action = list(map(int, source["event_action_index"]))
        self.formula = list(map(str, source["event_formula"]))

    def registry(self, kind: int, source: int) -> int:
        key = (int(kind), int(source))
        if key not in self.positions:
            self.positions[key] = len(self.registry_kind)
            self.registry_kind.append(key[0])
            self.registry_source.append(key[1])
        return self.positions[key]

    def append(
        self, anchor: int, positives: list[int], negatives: list[int], *,
        kind: int, query: int, action: int, formula: str,
    ) -> None:
        if not positives or not negatives:
            raise RuntimeError("Stage-6 tried to append an empty triplet pool")
        if anchor in positives or anchor in negatives or set(positives) & set(negatives):
            raise RuntimeError("Stage-6 triplet roles overlap")
        self.anchor.append(int(anchor))
        self.positive.extend(map(int, positives))
        self.negative.extend(map(int, negatives))
        self.positive_ptr.append(len(self.positive))
        self.negative_ptr.append(len(self.negative))
        self.kind.append(int(kind))
        self.query.append(int(query))
        self.action.append(int(action))
        self.formula.append(str(formula))

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "registry_kind": np.asarray(self.registry_kind, dtype=np.int8),
            "registry_source_index": np.asarray(self.registry_source, dtype=np.int64),
            "anchor_idx": np.asarray(self.anchor, dtype=np.int64),
            "positive_ptr": np.asarray(self.positive_ptr, dtype=np.int64),
            "positive_idx": np.asarray(self.positive, dtype=np.int64),
            "negative_ptr": np.asarray(self.negative_ptr, dtype=np.int64),
            "negative_idx": np.asarray(self.negative, dtype=np.int64),
            "event_kind": np.asarray(self.kind, dtype=np.int8),
            "event_query": np.asarray(self.query, dtype=np.int64),
            "event_action_index": np.asarray(self.action, dtype=np.int64),
            "event_formula": fixed_unicode(self.formula),
        }


def spectrum_rows_for_query(graph: CandidateGraph, query: int) -> tuple[list[int], list[np.ndarray]]:
    _, rows, pointers, _ = graph.query_block(query)
    positive = list(map(int, rows[int(pointers[0]):int(pointers[1])]))
    negative = [
        np.asarray(rows[int(left):int(right)], dtype=np.int64)
        for left, right in zip(pointers[1:-1], pointers[2:])
    ]
    positive = [row for row in positive if row != int(graph.query_row[query])]
    if not positive or not negative:
        raise RuntimeError(f"query {query} lacks a positive or negative molecule")
    return positive, negative


def pool_prefix_equal(full: dict[str, np.ndarray], prefix: dict[str, np.ndarray]) -> bool:
    event_count = len(prefix["anchor_idx"])
    registry_count = len(prefix["registry_kind"])
    positive_count = len(prefix["positive_idx"])
    negative_count = len(prefix["negative_idx"])
    checks = [
        np.array_equal(full["registry_kind"][:registry_count], prefix["registry_kind"]),
        np.array_equal(full["registry_source_index"][:registry_count], prefix["registry_source_index"]),
        np.array_equal(full["anchor_idx"][:event_count], prefix["anchor_idx"]),
        np.array_equal(full["positive_ptr"][:event_count + 1], prefix["positive_ptr"]),
        np.array_equal(full["positive_idx"][:positive_count], prefix["positive_idx"]),
        np.array_equal(full["negative_ptr"][:event_count + 1], prefix["negative_ptr"]),
        np.array_equal(full["negative_idx"][:negative_count], prefix["negative_idx"]),
        np.array_equal(full["event_kind"][:event_count], prefix["event_kind"]),
        np.array_equal(full["event_query"][:event_count], prefix["event_query"]),
        np.array_equal(full["event_action_index"][:event_count], prefix["event_action_index"]),
        np.array_equal(full["event_formula"][:event_count].astype(str), prefix["event_formula"].astype(str)),
    ]
    return bool(all(checks))


def pool_arrays_equal(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> bool:
    return bool(
        set(left) == set(right)
        and all(np.array_equal(left[key], right[key]) for key in left)
    )


def two_sided_hard_triplet_qualifies(
    *,
    anchor_identity_similarity: float,
    positive_identity_similarity: float,
    negative_identity_similarity: float,
    anchor_positive_drop: float,
    positive_view_drop: float,
    joint_positive_drop: float,
    negative_target_gain: float,
    negative_control_gain: float,
    target_margin: float,
    control_margin: float,
    identity_floor: float,
    margin_floor: float,
    margin: float,
    minimum_hard_delta: float,
) -> bool:
    """One auditable predicate for every event counted as two-sided hard."""
    return bool(
        anchor_identity_similarity >= identity_floor
        and positive_identity_similarity >= identity_floor
        and negative_identity_similarity >= identity_floor
        and anchor_positive_drop >= minimum_hard_delta
        and positive_view_drop >= minimum_hard_delta
        and joint_positive_drop >= minimum_hard_delta
        and negative_target_gain >= minimum_hard_delta
        and negative_control_gain >= minimum_hard_delta
        and margin_floor <= target_margin < margin - 1e-7
        and margin_floor <= control_margin < margin - 1e-7
    )


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-6 mining requires an allocated GPU")
    if (
        args.outer_fold != 0
        or args.formula_fold_seed != 20260825
        or abs(args.margin - 0.1) > 1e-12
        or args.maximum_positive_rows != 8
        or args.maximum_negative_molecules != 3
    ):
        raise RuntimeError("Stage-6 registered mining settings drifted")

    stage1_triplets = args.stage1_run / "triplets"
    stage1_checkpoint = args.stage1_run / "checkpoint" / "targeted_final.ckpt"
    stage1_evaluation = args.stage1_run / "evaluation" / "targeted" / "report.json"
    required = [
        stage1_checkpoint,
        stage1_evaluation,
        stage1_triplets / "train_pool.npz",
        stage1_triplets / "validation_pool.npz",
        stage1_triplets / "action_spectra.npz",
        stage1_triplets / "selected_actions.csv.gz",
        stage1_triplets / "qualified_actions.csv.gz",
        stage1_triplets / "action_aliases.csv.gz",
        args.graph,
        args.embedding_cache,
        args.data,
        args.architecture_checkpoint,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    evaluated = json.loads(stage1_evaluation.read_text(encoding="utf-8"))
    if (
        evaluated.get("provenance", {}).get("checkpoint_sha256")
        != sha256_file(stage1_checkpoint)
    ):
        raise RuntimeError("Stage-6 warm start is not the evaluated Stage-1 champion")

    graph = CandidateGraph(args.graph)
    stage1_train = load_npz(stage1_triplets / "train_pool.npz")
    stage1_validation = load_npz(stage1_triplets / "validation_pool.npz")
    stage1_bank = load_npz(stage1_triplets / "action_spectra.npz")
    stage1_selected = pd.read_csv(
        stage1_triplets / "selected_actions.csv.gz", low_memory=False,
    )
    stage1_targeted = np.asarray(stage1_bank["targeted_action_spectra"], dtype=np.float32)
    stage1_control = np.asarray(stage1_bank["control_action_spectra"], dtype=np.float32)
    if stage1_targeted.shape != stage1_control.shape or stage1_targeted.shape[1:] != (101, 2):
        raise RuntimeError("Stage-1 action bank is not aligned")

    validation_formulas = set(stage1_validation["event_formula"].astype(str))
    outer_train = np.asarray([
        stable_fold(str(formula), 5, args.formula_fold_seed) != args.outer_fold
        for formula in graph.query_formula
    ], dtype=bool)
    eligible_queries = np.asarray([
        query for query in np.flatnonzero(outer_train)
        if str(graph.query_formula[query]) not in validation_formulas
    ], dtype=np.int64)
    if set(graph.query_formula[eligible_queries].astype(str)) & validation_formulas:
        raise RuntimeError("Stage-6 train and validation formulas overlap")

    preprocessor = SpectrumPreprocessor(
        dformat=DataFormatA(), prec_intens=1.1, n_highest_peaks=100,
        spec_entropy_cleaning=False, precision=32,
        mz_shift_aug_p=0, mz_shift_aug_max=0,
    )
    device = torch.device("cuda")
    model, checkpoint_kind = load_base_model(
        stage1_checkpoint, args.architecture_checkpoint, device, 100,
    )
    assert_exact_checkpoint_reconstruction(model, stage1_checkpoint)
    model.eval()

    with np.load(args.embedding_cache, allow_pickle=False) as body:
        measured_rows = np.asarray(body["rows"], dtype=np.int64)
    measured_embeddings = encode_rows(
        model, measured_rows, args.data, preprocessor,
        batch_size=args.batch_size, device=device, label="noise-stage6-stage1-measured",
    )
    row_position = {int(row): index for index, row in enumerate(measured_rows)}
    clean_rows = graph.query_row[eligible_queries].astype(np.int64)
    clean_tensors = exact_native_clean_tensors(args.data, clean_rows, preprocessor)
    clean_by_query = {
        int(query): clean_tensors[position]
        for position, query in enumerate(eligible_queries)
    }

    # First freeze row-level memberships from the Stage-1 geometry.  Nothing in
    # the nuisance generator receives a candidate rank, identity label, query
    # spectrum, or action outcome.  Hardness is only a later mining decision.
    memberships: dict[int, dict[str, object]] = {}
    identity_calibration_by_formula: dict[str, list[float]] = defaultdict(list)
    margin_calibration_by_formula: dict[str, list[float]] = defaultdict(list)
    for query in eligible_queries:
        query = int(query)
        _, rows, pointers, molecule_left = graph.query_block(query)
        clean_vector = measured_embeddings[row_position[int(graph.query_row[query])]]
        positive_rows = [
            int(row) for row in rows[int(pointers[0]):int(pointers[1])]
            if int(row) != int(graph.query_row[query]) and int(row) in row_position
        ]
        positives = sorted(
            [
                (float(measured_embeddings[row_position[row]] @ clean_vector), row)
                for row in positive_rows
            ],
            key=lambda body: (body[0], body[1]),
        )
        positive_formula = str(graph.molecule_formula[int(molecule_left)])
        query_formula = str(graph.query_formula[query])
        if (
            positive_formula != query_formula
            or not role_formula_is_train_safe(
                positive_formula, validation_formulas=validation_formulas,
                outer_fold=args.outer_fold,
                formula_fold_seed=args.formula_fold_seed,
            )
        ):
            continue
        negatives: list[tuple[float, int, str, str]] = []
        for local_molecule, (left, right) in enumerate(
            zip(pointers[1:-1], pointers[2:])
        ):
            available = [
                int(row) for row in rows[int(left):int(right)]
                if int(row) in row_position
            ]
            if not available:
                continue
            scores = np.asarray([
                float(measured_embeddings[row_position[row]] @ clean_vector)
                for row in available
            ])
            at = int(np.argmax(scores))
            molecule = int(molecule_left + 1 + local_molecule)
            negative_formula = str(graph.molecule_formula[molecule])
            if not role_formula_is_train_safe(
                negative_formula, validation_formulas=validation_formulas,
                outer_fold=args.outer_fold,
                formula_fold_seed=args.formula_fold_seed,
            ):
                continue
            negatives.append((
                float(scores[at]), int(available[at]),
                str(graph.molecule_ik14[molecule]), negative_formula,
            ))
        negatives.sort(key=lambda body: (-body[0], body[1], body[2], body[3]))
        if not positives or len(negatives) < 2:
            continue
        identity_calibration_by_formula[query_formula].append(float(np.median([
            score for score, _ in positives
        ])))
        query_margins = np.asarray([
            positive_score - negative_score
            for positive_score, _ in positives
            for negative_score, _, _, _ in negatives[:3]
        ], dtype=np.float64)
        margin_calibration_by_formula[query_formula].append(float(
            np.quantile(query_margins, 0.01)
        ))
        memberships[query] = {
            "positives": positives,
            "positive_formula": positive_formula,
            "negatives": negatives,
        }

    if not identity_calibration_by_formula or not margin_calibration_by_formula:
        raise RuntimeError("Stage-6 found no measured triplet calibration population")
    formula_identity_statistics = np.asarray([
        np.median(values) for values in identity_calibration_by_formula.values()
    ], dtype=np.float64)
    formula_margin_statistics = np.asarray([
        np.median(values) for values in margin_calibration_by_formula.values()
    ], dtype=np.float64)
    raw_identity_q05 = float(np.quantile(formula_identity_statistics, 0.05))
    raw_margin_q01 = float(np.quantile(formula_margin_statistics, 0.01))
    # Empirical floors are primary. Absolute safety bounds prevent a damaged
    # tail in the source ledger from licensing an unrecognizable positive or
    # a nearly inverted triplet as "hard".
    identity_floor = max(0.50, raw_identity_q05)
    margin_floor = max(-0.25, raw_margin_q01)

    candidate_specs: list[dict[str, object]] = []
    role_rows: set[int] = set()
    for query in eligible_queries:
        query = int(query)
        membership = memberships.get(query)
        if membership is None:
            continue
        safe_positives = [
            body for body in membership["positives"]
            if float(body[0]) >= identity_floor
        ][:args.maximum_positive_rows]
        hard_negatives = list(
            membership["negatives"][:args.maximum_negative_molecules]
        )
        if not safe_positives or len(hard_negatives) < 2:
            continue
        # Shuffle source choices once per query, independently of the severity
        # order. This preserves hard-source mining without coupling easy noise
        # to the hardest positive/negative rank and hard noise to the easiest.
        pairing_rng = np.random.default_rng(deterministic_seed(
            args.seed, query, "stage6-source-pairing",
        ))
        positives = [
            safe_positives[index]
            for index in pairing_rng.permutation(len(safe_positives))
        ]
        negatives = [
            hard_negatives[index]
            for index in pairing_rng.permutation(len(hard_negatives))
        ]
        for severity_index, severity in enumerate(SEVERITIES):
            positive_score, positive_row = positives[severity_index % len(positives)]
            negative_score, negative_row, negative_ik14, negative_formula = negatives[
                severity_index % len(negatives)
            ]
            role_rows.update((int(positive_row), int(negative_row)))
            candidate_specs.append({
                "query_index": query,
                "query_row": int(graph.query_row[query]),
                "query_formula": str(graph.query_formula[query]),
                "query_ik14": str(graph.query_ik14[query]),
                "positive_formula": str(membership["positive_formula"]),
                "severity": severity,
                "positive_row": int(positive_row),
                "negative_row": int(negative_row),
                "negative_ik14": str(negative_ik14),
                "negative_formula": str(negative_formula),
                "clean_positive_similarity": float(positive_score),
                "clean_negative_similarity": float(negative_score),
            })

    role_rows_array = np.asarray(sorted(role_rows), dtype=np.int64)
    role_tensors_array = exact_native_clean_tensors(
        args.data, role_rows_array, preprocessor,
    )
    clean_by_row = {
        int(row): role_tensors_array[position]
        for position, row in enumerate(role_rows_array)
    }
    anchor_candidates: list[np.ndarray] = []
    positive_candidates: list[np.ndarray] = []
    negative_candidates: list[np.ndarray] = []
    skipped_short = 0
    retained_specs: list[dict[str, object]] = []
    for spec in candidate_specs:
        query = int(spec["query_index"])
        severity = str(spec["severity"])
        positive_row = int(spec["positive_row"])
        negative_row = int(spec["negative_row"])
        try:
            anchor_view = identity_preserving_noise_view(
                clean_by_query[query], source_key=int(spec["query_row"]),
                severity=severity, seed=args.seed,
            )
            positive_view = identity_preserving_noise_view(
                clean_by_row[positive_row], source_key=positive_row,
                severity=severity, seed=args.seed,
            )
            negative_views = [
                identity_preserving_noise_view(
                    clean_by_row[negative_row], source_key=negative_row,
                    severity=severity, seed=args.seed,
                    replicate=replicate,
                )
                for replicate in range(NEGATIVE_VIEW_REPLICATES)
            ]
        except RuntimeError as error:
            if "fewer than eight peaks" not in str(error):
                raise
            skipped_short += 1
            continue
        for view in (anchor_view, positive_view, *negative_views):
            replay = native_action_model_input(view, preprocessor)
            if not np.array_equal(view, replay):
                raise RuntimeError("Stage-6 noisy role does not replay exactly")
        retained_specs.append(spec)
        anchor_candidates.append(anchor_view)
        positive_candidates.append(positive_view)
        negative_candidates.extend(negative_views)

    candidate_specs = retained_specs
    anchor_array = np.asarray(anchor_candidates, dtype=np.float32)
    positive_array = np.asarray(positive_candidates, dtype=np.float32)
    negative_array = np.asarray(negative_candidates, dtype=np.float32)
    if (
        anchor_array.shape != positive_array.shape
        or len(negative_array) != NEGATIVE_VIEW_REPLICATES * len(anchor_array)
    ):
        raise RuntimeError("Stage-6 noisy-role proposal arrays are not aligned")

    # Encode in bounded chunks: only scalar selection evidence survives.  This
    # avoids retaining nearly a million 1024-D embeddings in host memory.
    by_query: dict[int, list[dict[str, object]]] = {}
    mining_chunk = 4096
    for start in range(0, len(candidate_specs), mining_chunk):
        stop = min(start + mining_chunk, len(candidate_specs))
        count = stop - start
        role_block = np.concatenate((
            anchor_array[start:stop], positive_array[start:stop],
            negative_array[
                start * NEGATIVE_VIEW_REPLICATES:stop * NEGATIVE_VIEW_REPLICATES
            ],
        ), axis=0)
        encoded = encode_actions(
            model, role_block, batch_size=args.batch_size, device=device,
            label=f"noise-stage6-two-sided-{start:06d}", preprocessor=preprocessor,
        )
        anchor_embeddings = encoded[:count]
        positive_embeddings = encoded[count:2 * count]
        negative_embeddings = encoded[2 * count:].reshape(
            count, NEGATIVE_VIEW_REPLICATES, -1,
        )
        for local, spec in enumerate(candidate_specs[start:stop]):
            query = int(spec["query_index"])
            positive_row = int(spec["positive_row"])
            negative_row = int(spec["negative_row"])
            clean_anchor_embedding = measured_embeddings[
                row_position[int(spec["query_row"])]
            ]
            clean_positive_embedding = measured_embeddings[row_position[positive_row]]
            clean_negative_embedding = measured_embeddings[row_position[negative_row]]
            anchor_embedding = anchor_embeddings[local]
            positive_embedding = positive_embeddings[local]
            clean_positive_similarity = float(spec["clean_positive_similarity"])
            anchor_identity_similarity = float(anchor_embedding @ clean_anchor_embedding)
            positive_identity_similarity = float(positive_embedding @ clean_positive_embedding)
            anchor_only_similarity = float(anchor_embedding @ clean_positive_embedding)
            positive_only_similarity = float(clean_anchor_embedding @ positive_embedding)
            noisy_positive_similarity = float(anchor_embedding @ positive_embedding)
            anchor_drop = clean_positive_similarity - anchor_only_similarity
            positive_drop = clean_positive_similarity - positive_only_similarity
            joint_positive_drop = clean_positive_similarity - noisy_positive_similarity
            clean_negative_similarity = float(spec["clean_negative_similarity"])
            negative_options: list[dict[str, float | int]] = []
            for replicate in range(NEGATIVE_VIEW_REPLICATES):
                negative_embedding = negative_embeddings[local, replicate]
                negative_identity_similarity = float(
                    negative_embedding @ clean_negative_embedding
                )
                noisy_negative_target_similarity = float(
                    anchor_embedding @ negative_embedding
                )
                noisy_negative_control_similarity = float(
                    clean_anchor_embedding @ negative_embedding
                )
                target_gain = noisy_negative_target_similarity - float(
                    anchor_embedding @ clean_negative_embedding
                )
                control_gain = (
                    noisy_negative_control_similarity - clean_negative_similarity
                )
                target_margin = (
                    noisy_positive_similarity - noisy_negative_target_similarity
                )
                control_margin = (
                    positive_only_similarity - noisy_negative_control_similarity
                )
                if two_sided_hard_triplet_qualifies(
                    anchor_identity_similarity=anchor_identity_similarity,
                    positive_identity_similarity=positive_identity_similarity,
                    negative_identity_similarity=negative_identity_similarity,
                    anchor_positive_drop=anchor_drop,
                    positive_view_drop=positive_drop,
                    joint_positive_drop=joint_positive_drop,
                    negative_target_gain=target_gain,
                    negative_control_gain=control_gain,
                    target_margin=target_margin,
                    control_margin=control_margin,
                    identity_floor=identity_floor,
                    margin_floor=margin_floor,
                    margin=args.margin,
                    minimum_hard_delta=MINIMUM_HARD_DELTA,
                ):
                    negative_options.append({
                        "replicate": replicate,
                        "negative_identity_similarity": negative_identity_similarity,
                        "negative_target_gain": target_gain,
                        "negative_control_gain": control_gain,
                        "noisy_negative_target_similarity": noisy_negative_target_similarity,
                        "noisy_negative_control_similarity": noisy_negative_control_similarity,
                        "target_margin": target_margin,
                        "control_margin": control_margin,
                    })
            if not negative_options:
                continue
            # The hardest valid source-local negative is retained; invalid or
            # merely numerous proposals never count as hard events.
            negative_choice = min(
                negative_options,
                key=lambda body: (
                    float(body["target_margin"]), -float(body["negative_target_gain"]),
                    int(body["replicate"]),
                ),
            )
            candidate_index = start + local
            by_query.setdefault(query, []).append({
                **spec,
                "candidate_index": candidate_index,
                "negative_candidate_index": (
                    candidate_index * NEGATIVE_VIEW_REPLICATES
                    + int(negative_choice["replicate"])
                ),
                "anchor_identity_similarity": anchor_identity_similarity,
                "positive_identity_similarity": positive_identity_similarity,
                "anchor_only_similarity": anchor_only_similarity,
                "positive_only_similarity": positive_only_similarity,
                "noisy_positive_similarity": noisy_positive_similarity,
                "anchor_positive_drop": anchor_drop,
                "positive_view_drop": positive_drop,
                "joint_positive_drop": joint_positive_drop,
                **negative_choice,
            })
    del model
    torch.cuda.empty_cache()

    chosen_records: list[dict[str, object]] = []
    chosen_targeted_roles: list[np.ndarray] = []
    chosen_control_roles: list[np.ndarray] = []
    for query in eligible_queries:
        query = int(query)
        candidates = sorted(
            by_query.get(query, []),
            key=lambda body: (float(body["target_margin"]), str(body["severity"])),
        )
        # Two nominal events are accepted only when both the noisy anchor and
        # the negative molecule differ.  A repeated pair cannot inflate scale.
        pair: list[dict[str, object]] = []
        for candidate in candidates:
            if pair and (
                str(candidate["severity"]) == str(pair[0]["severity"])
                or str(candidate["negative_ik14"]) == str(pair[0]["negative_ik14"])
            ):
                continue
            pair.append(candidate)
            if len(pair) == 2:
                break
        if len(pair) != 2:
            continue
        for selected in pair:
            candidate_index = int(selected["candidate_index"])
            negative_index = int(selected["negative_candidate_index"])
            positive_row = int(selected["positive_row"])
            negative_row = int(selected["negative_row"])
            # Target/control share the noisy positive and noisy negative.  The
            # sole arm difference is the independently noised versus clean
            # query anchor, so the causal contrast remains one-factor clean.
            chosen_targeted_roles.extend((
                anchor_array[candidate_index], positive_array[candidate_index],
                negative_array[negative_index],
            ))
            chosen_control_roles.extend((
                clean_by_query[query], positive_array[candidate_index],
                negative_array[negative_index],
            ))
            chosen_records.append({
                **selected,
                "realized_severity": str(selected["severity"]),
                "positive_rows": [positive_row],
                "negative_rows": [negative_row],
            })

    # Difficulty labels are measured, not inferred from the requested nuisance
    # severity.  Stable global thirds make the three strata nonempty and
    # strictly ordered without changing inclusion or training dose.
    difficulty_order = np.argsort(np.asarray([
        float(record["target_margin"]) for record in chosen_records
    ]), kind="stable")
    for rank, record_index in enumerate(difficulty_order):
        fraction = rank / max(1, len(difficulty_order))
        tier = "hard" if fraction < 1 / 3 else "medium" if fraction < 2 / 3 else "easy"
        chosen_records[int(record_index)]["difficulty_tier"] = tier

    broad_targeted = np.asarray(chosen_targeted_roles, dtype=np.float32)
    broad_control = np.asarray(chosen_control_roles, dtype=np.float32)
    if len(chosen_records) < STAGE6_MINIMUM_BROAD_EVENTS:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        audit_path = args.output.parent / f"{args.output.name}_mining_audit.json"
        audit_path.write_text(json.dumps({
            "status": "NOISE_DREAMS_NATIVE_SCALE_STAGE6_INSUFFICIENT_REAL_HARD_EVENTS",
            "candidate_specs_after_native_replay": int(len(candidate_specs)),
            "queries_with_at_least_one_qualified_relation": int(len(by_query)),
            "qualified_events_after_two_per_query_distinct_negative_gate": int(
                len(chosen_records)
            ),
            "qualified_queries": int(len({
                int(record["query_index"]) for record in chosen_records
            })),
            "required_events": STAGE6_MINIMUM_BROAD_EVENTS,
            "quality_thresholds_were_not_relaxed": True,
        }, indent=2), encoding="utf-8")
        raise RuntimeError(
            f"Stage-6 two-sided hard triplets {len(chosen_records)} < "
            f"{STAGE6_MINIMUM_BROAD_EVENTS}; audit={audit_path}"
        )

    appender = PoolAppender(stage1_train)
    existing_clean = set(map(int, stage1_train["event_query"][stage1_train["event_kind"] == 0]))
    safe_eligible_queries = np.asarray(sorted(
        query for query, membership in memberships.items()
        if any(float(score) >= identity_floor for score, _ in membership["positives"])
    ), dtype=np.int64)
    for query in safe_eligible_queries:
        query = int(query)
        if query in existing_clean:
            continue
        membership = memberships[query]
        positive_rows = [
            int(row) for score, row in membership["positives"]
            if float(score) >= identity_floor
        ][:args.maximum_positive_rows]
        ranked = list(membership["negatives"])
        anchor = appender.registry(Registry.HDF5, int(graph.query_row[query]))
        positives = [appender.registry(Registry.HDF5, row) for row in positive_rows]
        negatives = [
            appender.registry(Registry.HDF5, row)
            for _, row, _, _ in ranked[:args.maximum_negative_molecules]
        ]
        appender.append(
            anchor, positives, negatives, kind=0, query=query, action=-1,
            formula=str(graph.query_formula[query]),
        )

    stage1_action_count = len(stage1_targeted)
    for broad_index, record in enumerate(chosen_records):
        query = int(record["query_index"])
        semantic_action_index = stage1_action_count + broad_index
        anchor_action_index = stage1_action_count + 3 * broad_index
        positive_action_index = anchor_action_index + 1
        negative_action_index = anchor_action_index + 2
        anchor = appender.registry(Registry.ACTION, anchor_action_index)
        positives = [appender.registry(Registry.ACTION, positive_action_index)]
        negatives = [appender.registry(Registry.ACTION, negative_action_index)]
        appender.append(
            anchor, positives, negatives, kind=2, query=query,
            action=semantic_action_index, formula=str(record["query_formula"]),
        )
        record["semantic_action_index"] = semantic_action_index
        record["anchor_action_index"] = anchor_action_index
        record["positive_action_index"] = positive_action_index
        record["negative_action_index"] = negative_action_index
    train_pool = appender.arrays()
    _, schedule_filler, schedule_audit = native_query_disjoint_one_pass_batches(
        train_pool,
        np.arange(len(train_pool["anchor_idx"]), dtype=np.int64),
        batch_size=4,
        seed=3407,
    )
    targeted_bank = np.concatenate((stage1_targeted, broad_targeted), axis=0)
    # Both Stage-6 arms start from the Stage-1 targeted champion and replay the
    # same Stage-1 targeted triplets.  Only the new broad anchor stream differs;
    # otherwise targeted-vs-control would change two factors at once.
    control_bank = np.concatenate((stage1_targeted, broad_control), axis=0)
    stage1_representable = np.asarray(
        stage1_bank["native_action_view_representable"], dtype=bool,
    )
    if stage1_representable.shape != (stage1_action_count,):
        raise RuntimeError("Stage-1 representability mask is not aligned")
    representable = np.concatenate((
        stage1_representable, np.ones(len(broad_targeted), dtype=bool),
    ))

    prefix_actions = np.asarray(stage1_train["event_action_index"], dtype=np.int64)
    prefix_actions = prefix_actions[prefix_actions >= 0]
    full_actions = np.asarray(train_pool["event_action_index"], dtype=np.int64)
    full_actions = full_actions[full_actions >= 0]
    tier_counts = Counter(record["difficulty_tier"] for record in chosen_records)
    broad_query_counts = Counter(int(record["query_index"]) for record in chosen_records)
    anchor_identity_counts = Counter(
        str(record["query_ik14"]) for record in chosen_records
    )
    anchor_formula_counts = Counter(
        str(record["query_formula"]) for record in chosen_records
    )
    available_anchor_identities = {
        str(graph.query_ik14[int(query)]) for query in safe_eligible_queries
    }
    available_anchor_formulas = {
        str(graph.query_formula[int(query)]) for query in safe_eligible_queries
    }
    formula_count_values = np.asarray(
        list(anchor_formula_counts.values()), dtype=np.float64,
    )
    formula_effective_sample_size = float(
        formula_count_values.sum() ** 2
        / np.maximum(np.sum(formula_count_values ** 2), 1.0)
    )
    broad_margins = np.asarray([record["target_margin"] for record in chosen_records])
    tier_margin_medians = {
        tier: float(np.median([
            record["target_margin"] for record in chosen_records
            if record["difficulty_tier"] == tier
        ]))
        for tier in SEVERITIES
    }
    multi_candidate_fraction = float(np.mean([
        len(by_query.get(int(record["query_index"]), [])) >= 2
        for record in chosen_records
    ]))
    negative_identity_counts = Counter(
        str(record["negative_ik14"]) for record in chosen_records
    )
    negative_row_counts = Counter(
        int(record["negative_row"]) for record in chosen_records
    )
    available_negative_identities = {
        str(body[2])
        for membership in memberships.values()
        for body in membership["negatives"]
    }
    available_negative_rows = {
        int(body[1])
        for membership in memberships.values()
        for body in membership["negatives"]
    }
    minimum_unique_negative_identities = max(
        1_000, int(np.ceil(0.50 * len(available_negative_identities))),
    )
    minimum_unique_negative_rows = max(
        5_000, min(25_000, int(np.ceil(0.25 * len(available_negative_rows)))),
    )
    broad_action_indices = np.asarray([
        int(record["anchor_action_index"]) for record in chosen_records
    ], dtype=np.int64)
    expected_difference = np.zeros(len(targeted_bank), dtype=bool)
    expected_difference[broad_action_indices] = True
    actual_difference = np.any(
        np.abs(targeted_bank - control_bank) > 0, axis=(1, 2),
    )
    every_role_has_source_base_peak = True
    for broad_index, record in enumerate(chosen_records):
        sources = (
            clean_by_query[int(record["query_index"])],
            clean_by_row[int(record["positive_row"])],
            clean_by_row[int(record["negative_row"])],
        )
        for role_offset, source in enumerate(sources):
            fragments = source[1:][source[1:, 0] > 0]
            base_mz = float(fragments[int(np.argmax(fragments[:, 1])), 0])
            view = broad_targeted[3 * broad_index + role_offset]
            if not np.any(np.abs(view[1:, 0] - base_mz) <= 1e-7):
                every_role_has_source_base_peak = False
                break
        if not every_role_has_source_base_peak:
            break
    gates = {
        "stage1_train_pool_is_exact_prefix": pool_prefix_equal(train_pool, stage1_train),
        "every_stage1_action_event_is_retained_exactly_once": bool(
            np.array_equal(full_actions[:len(prefix_actions)], prefix_actions)
        ),
        "all_role_safe_queries_have_clean_preservation": bool(
            set(map(int, safe_eligible_queries)).issubset(set(map(int, train_pool["event_query"][train_pool["event_kind"] == 0])))
        ),
        "at_least_twenty_thousand_new_two_sided_hard_events": len(chosen_records) >= STAGE6_MINIMUM_BROAD_EVENTS,
        "hard_event_count_is_recomputed_from_final_pool_not_candidates": bool(
            int(np.sum(train_pool["event_kind"] == 2))
            - int(np.sum(stage1_train["event_kind"] == 2))
            == len(chosen_records)
            and len(broad_targeted) == 3 * len(chosen_records)
        ),
        "at_least_ten_thousand_independent_broad_queries": len(broad_query_counts) >= STAGE6_MINIMUM_BROAD_QUERIES,
        "every_broad_query_has_exactly_two_exposures": bool(
            broad_query_counts and set(broad_query_counts.values()) == {2}
        ),
        "all_broad_target_and_control_triplets_are_native_hinge_active": bool(
            len(broad_margins)
            and all(
                margin_floor <= float(record["target_margin"]) < args.margin - 1e-7
                and margin_floor <= float(record["control_margin"]) < args.margin - 1e-7
                for record in chosen_records
            )
        ),
        "anchor_and_positive_each_make_every_saved_pair_harder": all(
            float(record["anchor_positive_drop"]) >= MINIMUM_HARD_DELTA
            and float(record["positive_view_drop"]) >= MINIMUM_HARD_DELTA
            and float(record["joint_positive_drop"]) >= MINIMUM_HARD_DELTA
            for record in chosen_records
        ),
        "noisy_negative_is_harder_for_target_and_control_in_every_event": all(
            float(record["negative_target_gain"]) >= MINIMUM_HARD_DELTA
            and float(record["negative_control_gain"]) >= MINIMUM_HARD_DELTA
            for record in chosen_records
        ),
        "all_three_noisy_roles_preserve_source_identity": all(
            float(record["anchor_identity_similarity"]) >= identity_floor
            and float(record["positive_identity_similarity"]) >= identity_floor
            and float(record["negative_identity_similarity"]) >= identity_floor
            for record in chosen_records
        ),
        "all_three_noisy_roles_retain_their_source_base_peak": every_role_has_source_base_peak,
        "two_events_per_query_use_distinct_negative_identities": all(
            len({
                str(record["negative_ik14"]) for record in chosen_records
                if int(record["query_index"]) == query
            }) == 2
            for query in broad_query_counts
        ),
        "every_noisy_negative_remains_a_different_identity": all(
            str(record["negative_ik14"]) != str(record["query_ik14"])
            for record in chosen_records
        ),
        "targeted_and_control_share_every_membership": bool(
            targeted_bank.shape == control_bank.shape
            and len(train_pool["event_action_index"])
            == len(train_pool["anchor_idx"])
            and np.max(train_pool["registry_source_index"][
                train_pool["registry_kind"] == Registry.ACTION
            ]) < len(targeted_bank)
        ),
        "targeted_and_control_differ_only_in_anchor_view": bool(
            targeted_bank.shape == control_bank.shape
            and np.array_equal(actual_difference, expected_difference)
            and np.array_equal(broad_targeted[1::3], broad_control[1::3])
            and np.array_equal(broad_targeted[2::3], broad_control[2::3])
        ),
        "all_registered_stage1_actions_and_sources_are_retained": bool(
            len(prefix_actions) and len(np.unique(prefix_actions)) == stage1_action_count
            and np.array_equal(targeted_bank[:stage1_action_count], stage1_targeted)
            and np.array_equal(control_bank[:stage1_action_count], stage1_targeted)
            and set(stage1_selected["source"].astype(str))
            == REGISTERED_STAGE1_SOURCES
        ),
        "training_event_query_formulas_are_validation_disjoint": not bool(
            set(train_pool["event_formula"].astype(str))
            & set(stage1_validation["event_formula"].astype(str))
        ),
        "training_event_query_formulas_are_outer_disjoint": all(
            stable_fold(str(formula), 5, args.formula_fold_seed) != args.outer_fold
            for formula in set(train_pool["event_formula"].astype(str))
        ),
        "new_stage6_positive_and_negative_role_formulas_are_disjoint": all(
            str(record["positive_formula"]) == str(record["query_formula"])
            and role_formula_is_train_safe(
                str(record["positive_formula"]),
                validation_formulas=validation_formulas,
                outer_fold=args.outer_fold,
                formula_fold_seed=args.formula_fold_seed,
            )
            and role_formula_is_train_safe(
                str(record["negative_formula"]),
                validation_formulas=validation_formulas,
                outer_fold=args.outer_fold,
                formula_fold_seed=args.formula_fold_seed,
            )
            for record in chosen_records
        ),
        "query_disjoint_one_pass_schedule_is_realizable": bool(
            schedule_audit["every_base_event_exposed_exactly_once"]
            and schedule_audit["same_query_events_never_share_an_optimizer_batch"]
            and schedule_audit["query_balanced_oversampling"] is False
            and schedule_audit["synthetic_query_equalization_events"] == 0
            and schedule_audit["padding_is_final_batch_only"]
            and len(schedule_filler) < 4
            and schedule_audit["batches_per_epoch"] <= STAGE6_MAXIMUM_OPTIMIZER_STEPS
        ),
    }
    if not all(gates.values()):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        audit_path = args.output.parent / f"{args.output.name}_gate_audit.json"
        audit_path.write_text(json.dumps({
            "status": "NOISE_DREAMS_NATIVE_SCALE_STAGE6_GATE_FAIL",
            "gates": gates,
            "events": int(len(chosen_records)),
            "queries": int(len(broad_query_counts)),
            "identity_floor": identity_floor,
            "margin_floor": margin_floor,
        }, indent=2), encoding="utf-8")
        raise RuntimeError(
            f"Stage-6 scale triplet gates failed: {gates}; audit={audit_path}"
        )

    broad_frame = pd.DataFrame(chosen_records)
    broad_frame["positive_rows"] = broad_frame["positive_rows"].map(
        lambda values: ";".join(map(str, values))
    )
    broad_frame["negative_rows"] = broad_frame["negative_rows"].map(
        lambda values: ";".join(map(str, values))
    )
    report = {
        "status": STAGE6_STATUS,
        "builder_version": STAGE6_BUILDER_VERSION,
        "checkpoint_kind_for_mining": checkpoint_kind,
        "scientific_contract": (
            "Exact Stage-1 targeted checkpoint and Adam continuation on every "
            "Stage-1 targeted triplet, one clean event per eligible outer-train "
            "query, and exactly two frozen-Stage-1-calibrated two-sided noisy "
            "triplets per qualifying query. Noisy anchor and noisy positive "
            "independently make the same-identity relation harder; a source-local "
            "noisy negative becomes harder for both arms. Both arms share the "
            "Stage-1 stream, noisy positives, noisy negatives and memberships; "
            "only the new broad anchor view differs."
        ),
        "training_initialization": "stage1_targeted_champion_exact_continuation",
        "library_mode": "exact_parent_prefix_plus_append_only_stage6_events",
        "mining_model": "frozen_stage1_targeted_champion",
        "stage1_events_retained": int(len(stage1_train["anchor_idx"])),
        "stage1_action_spectra_retained": int(stage1_action_count),
        "eligible_outer_train_queries": int(len(eligible_queries)),
        "eligible_queries_after_new_role_formula_filter": int(len(safe_eligible_queries)),
        "clean_preservation_events": int(np.sum(train_pool["event_kind"] == 0)),
        "broad_two_sided_hard_events": int(len(chosen_records)),
        "broad_two_sided_hard_queries": int(len(broad_query_counts)),
        "total_triplet_events": int(len(train_pool["anchor_idx"])),
        "planned_optimizer_steps": int(np.ceil(len(train_pool["anchor_idx"]) / 4)),
        "skipped_short_spectra": int(skipped_short),
        "measured_difficulty_counts": dict(sorted(tier_counts.items())),
        "measured_difficulty_margin_medians": tier_margin_medians,
        "difficulty_labels_are_descriptive_post_selection": True,
        "multiple_valid_difficulty_choice_fraction": multi_candidate_fraction,
        "empirical_calibration": {
            "weighting": "query_summary_then_formula_equal",
            "calibration_formulas": int(len(formula_identity_statistics)),
            "raw_same_identity_q05": raw_identity_q05,
            "applied_identity_floor": identity_floor,
            "raw_outer_train_clean_margin_q01": raw_margin_q01,
            "applied_margin_floor": margin_floor,
            "minimum_hard_delta": MINIMUM_HARD_DELTA,
            "negative_source_local_replicates": NEGATIVE_VIEW_REPLICATES,
        },
        "negative_diversity": {
            "available_role_safe_negative_identities": int(
                len(available_negative_identities)
            ),
            "available_role_safe_negative_rows": int(len(available_negative_rows)),
            "minimum_unique_negative_identities": int(
                minimum_unique_negative_identities
            ),
            "minimum_unique_negative_rows": int(minimum_unique_negative_rows),
            "unique_negative_identities": int(len(negative_identity_counts)),
            "unique_negative_rows": int(len(negative_row_counts)),
            "maximum_events_per_negative_identity": int(max(
                negative_identity_counts.values(), default=0,
            )),
            "maximum_events_per_negative_row": int(max(
                negative_row_counts.values(), default=0,
            )),
        },
        "anchor_diversity": {
            "available_role_safe_query_identities": int(
                len(available_anchor_identities)
            ),
            "selected_query_identities": int(len(anchor_identity_counts)),
            "available_role_safe_query_formulas": int(len(available_anchor_formulas)),
            "selected_query_formulas": int(len(anchor_formula_counts)),
            "maximum_events_per_query_formula": int(max(
                anchor_formula_counts.values(), default=0,
            )),
            "formula_effective_sample_size": formula_effective_sample_size,
        },
        "native_one_pass_schedule": schedule_audit,
        "realized_severity_counts": dict(sorted(Counter(
            record["realized_severity"] for record in chosen_records
        ).items())),
        "margin_summary": {
            "minimum": float(np.min(broad_margins)),
            "median": float(np.median(broad_margins)),
            "p90": float(np.quantile(broad_margins, 0.9)),
            "maximum": float(np.max(broad_margins)),
        },
        "noise_operators": {
            tier: {
                "peak_dropout_fraction": values[0],
                "intensity_jitter_fraction": values[1],
                "tiny_background_peaks": values[2],
            }
            for tier, values in SEVERITIES.items()
        },
        "gates": gates,
        "provenance": {
            "stage1_checkpoint_sha256": sha256_file(stage1_checkpoint),
            "stage1_evaluation_sha256": sha256_file(stage1_evaluation),
            "stage1_train_pool_sha256": sha256_file(stage1_triplets / "train_pool.npz"),
            "stage1_action_spectra_sha256": sha256_file(stage1_triplets / "action_spectra.npz"),
            "graph_sha256": sha256_file(args.graph),
            "embedding_registry_sha256": sha256_file(args.embedding_cache),
        },
        "legacy_scope": {
            "stage1_library_is_replayed_exactly": True,
            "stage1_candidate_role_formulas_reaudited_here": False,
            "new_stage6_increment_is_role_formula_disjoint": True,
            "absolute_formula_blind_training_claim": False,
        },
        "outer_performance_claimed": False,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        np.savez_compressed(staging / "train_pool.npz", **train_pool)
        np.savez_compressed(staging / "validation_pool.npz", **stage1_validation)
        if not pool_arrays_equal(
            stage1_validation, load_npz(staging / "validation_pool.npz")
        ):
            raise RuntimeError("written Stage-6 validation pool differs from Stage-1")
        np.savez_compressed(
            staging / "action_spectra.npz",
            targeted_action_spectra=targeted_bank,
            control_action_spectra=control_bank,
            native_action_view_representable=representable,
            stage1_action_count=np.asarray([stage1_action_count], dtype=np.int64),
            stage6_role_code=np.concatenate((
                np.full(stage1_action_count, -1, dtype=np.int8),
                np.tile(np.asarray([0, 1, 2], dtype=np.int8), len(chosen_records)),
            )),
            stage6_source_row=np.concatenate((
                np.full(stage1_action_count, -1, dtype=np.int64),
                np.asarray([
                    row
                    for record in chosen_records
                    for row in (
                        int(record["query_row"]), int(record["positive_row"]),
                        int(record["negative_row"]),
                    )
                ], dtype=np.int64),
            )),
            stage6_source_formula=fixed_unicode(
                [""] * stage1_action_count
                + [
                    formula
                    for record in chosen_records
                    for formula in (
                        str(record["query_formula"]),
                        str(record["positive_formula"]),
                        str(record["negative_formula"]),
                    )
                ]
            ),
            # A later stage can reuse this archive and train pool as an exact
            # prefix instead of reconstructing the triplet library.
            appendable_library_version=fixed_unicode([
                "noise_native_incremental_triplet_library_v1"
            ]),
        )
        broad_frame.to_csv(staging / "broad_events.csv.gz", index=False, compression="gzip")
        for name in (
            "selected_actions.csv.gz", "qualified_actions.csv.gz",
            "action_aliases.csv.gz",
        ):
            shutil.copy2(stage1_triplets / name, staging / name)
        shutil.copy2(stage1_triplets / "report.json", staging / "stage1_triplet_report.json")
        report["output_artifacts"] = {
            "train_pool_sha256": sha256_file(staging / "train_pool.npz"),
            "validation_pool_sha256": sha256_file(staging / "validation_pool.npz"),
            "action_spectra_sha256": sha256_file(staging / "action_spectra.npz"),
            "broad_events_sha256": sha256_file(staging / "broad_events.csv.gz"),
            "selected_actions_sha256": sha256_file(staging / "selected_actions.csv.gz"),
            "qualified_actions_sha256": sha256_file(staging / "qualified_actions.csv.gz"),
            "action_aliases_sha256": sha256_file(staging / "action_aliases.csv.gz"),
            "stage1_triplet_report_sha256": sha256_file(
                staging / "stage1_triplet_report.json"
            ),
        }
        report["validation_pool_exact_reuse_verified_after_write"] = True
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
