"""Build query- and candidate-exact ChemAware residual triplets.

This is the strict successor to the dense true-support curriculum.  Chemical
evidence is never promoted from one spectrum to every spectrum of the same
molecule.  A correction is admitted only when the frozen correct arm beats
all matched content-null arms on the *same query and the same current false
candidate boundary*.  The emitted training object remains an ordinary DreaMS
identity triplet ``(query, positive reference, negative reference)``.

The pool is intended for a short continuation from the protected Phase-A
checkpoint.  Each exact correction is accompanied by hard current-correct
safety sentinels and untouched official DreaMS replay.  There is no custom
loss, sampler weight, teacher score, or inference-time chemistry input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np

from build_chemaware_dreams_native_triplets import audit_identity_edges
from build_chemaware_max_boundary_native_triplets import (
    DREAMS_NATIVE_REPLAY,
    FrozenEmbeddings,
    PoolWriter,
    append_dreams_replay,
    load_npz,
)
from build_chemaware_multicondition_max_boundary_triplets import query_geometry
from chemaware_numpy_sampling import stable_formula_folds
from encode_chemaware_checkpoint_manifest_rows import array_sha256


ROOT = Path(__file__).resolve().parents[1]
EXACT_BASELINE_CORRECTION = 41
EXACT_ALTERNATE_CORRECTION = 42
EXACT_BOUNDARY_SAFETY = 43
EXACT_BOUNDARY_GUARD = 44

BASELINE_METRICS = (
    "delta_rule_max",
    "delta_rule_top2_mean",
    "action_best_advantage_over_baseline",
    "global_action_advantage_over_baseline",
)
ALTERNATE_METRICS = (
    "candidate_rule_max",
    "candidate_rule_top2_mean",
    "action_top_fraction",
    "action_largest_region_fraction",
    "action_same_neighbor_fraction",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--phasea-cache", type=Path, required=True)
    parser.add_argument("--validation-pool", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz",
    )
    parser.add_argument(
        "--dreams-replay-pool", type=Path,
        default=ROOT / "data/e1/e1_train_triplet_pool_10ppm.npz",
    )
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--training-roles", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--minimum-baseline-wins", type=int, default=2)
    parser.add_argument("--minimum-alternate-wins", type=int, default=3)
    parser.add_argument("--negative-references-per-correction", type=int, default=3)
    parser.add_argument("--safety-events-per-correction", type=float, default=3.0)
    parser.add_argument("--dreams-replay-events", type=int, default=512)
    parser.add_argument("--minimum-correction-events", type=int, default=100)
    parser.add_argument("--minimum-correction-queries", type=int, default=50)
    parser.add_argument("--minimum-correction-formulas", type=int, default=40)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--allow-official-cache-standin", action="store_true",
        help="Engineering audit only; formal construction requires a protected Phase-A cache.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def decoded(values: np.ndarray) -> tuple[str, ...]:
    return tuple(
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in np.asarray(values).tolist()
    )


def metric_columns(evidence: Mapping[str, np.ndarray]) -> dict[str, int]:
    names = decoded(evidence["metric_names"])
    required = set(BASELINE_METRICS + ALTERNATE_METRICS)
    if missing := sorted(required - set(names)):
        raise RuntimeError(f"exact-boundary evidence lacks metrics: {missing}")
    return {name: names.index(name) for name in required}


def candidate_slot(
    evidence: Mapping[str, np.ndarray], row: int, candidate: int,
) -> int | None:
    valid = np.asarray(evidence["valid"][row], dtype=bool)
    proposed = np.asarray(evidence["proposed_candidate"][row], dtype=np.int64)
    slots = np.flatnonzero(valid & (proposed == int(candidate)))
    if len(slots) > 1:
        raise RuntimeError("candidate appears in multiple evidence slots")
    return int(slots[0]) if len(slots) else None


def truth_candidate(
    manifest: Mapping[str, np.ndarray], query: int,
) -> int:
    left, right = map(int, manifest["query_ptr"][query:query + 2])
    truth = np.flatnonzero(np.asarray(manifest["molecule_label"][left:right], dtype=bool))
    if len(truth) != 1:
        raise RuntimeError(f"query {query} does not have exactly one truth candidate")
    return int(truth[0])


def exact_boundary_proof(
    evidence: Mapping[str, np.ndarray], manifest: Mapping[str, np.ndarray],
    row: int, current_false: int, columns: Mapping[str, int],
    minimum_baseline_wins: int, minimum_alternate_wins: int,
) -> dict[str, object] | None:
    """Return a proof tied to one query and one false candidate."""
    arms = decoded(evidence["arm_names"])
    if len(arms) != 4 or arms[0] != "correct":
        raise RuntimeError("exact-boundary proof requires correct plus three null arms")
    query = int(evidence["query"][row])
    baseline = int(evidence["baseline_candidate"][row])
    truth = truth_candidate(manifest, query)
    truth_slot = candidate_slot(evidence, row, truth)
    if truth_slot is None:
        return None
    metric = np.asarray(evidence["arm_metric"], dtype=np.float64)

    if current_false == baseline:
        names = BASELINE_METRICS
        correct = np.asarray([
            metric[0, row, truth_slot, columns[name]] for name in names
        ])
        null = np.asarray([
            [metric[arm, row, truth_slot, columns[name]] for name in names]
            for arm in range(1, len(arms))
        ])
        specific = correct - np.max(null, axis=0)
        required = minimum_baseline_wins
        kind = "baseline"
    else:
        false_slot = candidate_slot(evidence, row, current_false)
        if false_slot is None:
            return None
        names = ALTERNATE_METRICS
        arm_margin = np.asarray([
            [
                metric[arm, row, truth_slot, columns[name]]
                - metric[arm, row, false_slot, columns[name]]
                for name in names
            ]
            for arm in range(len(arms))
        ])
        specific = arm_margin[0] - np.max(arm_margin[1:], axis=0)
        required = minimum_alternate_wins
        kind = "alternate"
    wins = int(np.sum(specific > 0.0))
    if wins < required or float(np.median(specific)) <= 0.0:
        return None
    return {
        "query": query,
        "truth_candidate": truth,
        "false_candidate": int(current_false),
        "kind": kind,
        "metric_names": list(names),
        "specificity": specific,
        "wins": wins,
        "strict": bool(np.all(specific > 0.0)),
        "median_specificity": float(np.median(specific)),
        "minimum_specificity": float(np.min(specific)),
    }


def diverse_safety(
    rows: list[dict[str, object]], manifest: Mapping[str, np.ndarray], limit: int,
) -> list[dict[str, object]]:
    """Take nearest active sentinels, first maximizing formula diversity."""
    ordered = sorted(rows, key=lambda body: (
        float(body["margin"]), int(body["query"]),
    ))
    selected: list[dict[str, object]] = []
    used: set[int] = set()
    seen_formula: set[str] = set()
    for index, body in enumerate(ordered):
        formula = str(manifest["query_formula"][int(body["query"])])
        if formula in seen_formula:
            continue
        selected.append(body); used.add(index); seen_formula.add(formula)
        if len(selected) >= limit:
            return selected
    for index, body in enumerate(ordered):
        if index in used:
            continue
        selected.append(body)
        if len(selected) >= limit:
            break
    return selected


def build_exact_pool(
    evidence: Mapping[str, np.ndarray], manifest: Mapping[str, np.ndarray],
    cache: FrozenEmbeddings, replay: Mapping[str, np.ndarray], *,
    margin: float, minimum_baseline_wins: int, minimum_alternate_wins: int,
    negative_references_per_correction: int, safety_events_per_correction: float,
    dreams_replay_events: int, seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    columns = metric_columns(evidence)
    queries = np.asarray(evidence["query"], dtype=np.int64)
    if len(np.unique(queries)) != len(queries):
        raise RuntimeError("exact-boundary evidence repeats queries")
    writer = PoolWriter()
    proofs: list[dict[str, object]] = []
    safe: list[dict[str, object]] = []
    current_errors = missing_boundary_proof = inactive_proof = 0
    baseline_queries = alternate_queries = strict_queries = 0

    for row, query_value in enumerate(queries):
        query = int(query_value)
        geometry = query_geometry(manifest, cache, query, margin)
        if not bool(geometry["error"]):
            if float(geometry["margin"]) > 0.0:
                safe.append(geometry)
            continue
        current_errors += 1
        false_candidate = int(geometry["hardest_candidate"])
        proof = exact_boundary_proof(
            evidence, manifest, row, false_candidate, columns,
            minimum_baseline_wins, minimum_alternate_wins,
        )
        if proof is None:
            missing_boundary_proof += 1
            continue
        candidates = geometry["candidates"]
        if not isinstance(candidates, dict):
            raise TypeError("invalid current query geometry")
        active = np.asarray(candidates[false_candidate]["active_rows"], dtype=np.int64)
        if not len(active):
            inactive_proof += 1
            continue
        role = (
            EXACT_BASELINE_CORRECTION
            if proof["kind"] == "baseline" else EXACT_ALTERNATE_CORRECTION
        )
        before = len(writer.anchor)
        for negative in active[:negative_references_per_correction]:
            writer.append(
                int(geometry["anchor"]), [int(geometry["positive_row"])], [int(negative)],
                query, false_candidate, int(proof["wins"]), role,
            )
        added = len(writer.anchor) - before
        if not added:
            inactive_proof += 1
            continue
        proof["events"] = added
        proofs.append(proof)
        baseline_queries += int(proof["kind"] == "baseline")
        alternate_queries += int(proof["kind"] == "alternate")
        strict_queries += int(proof["strict"])

    correction_events = len(writer.anchor)
    correction_queries = {int(body["query"]) for body in proofs}
    safety_target = int(math.ceil(safety_events_per_correction * correction_events))
    selected_safety = diverse_safety(safe, manifest, safety_target)
    for geometry in selected_safety:
        candidates = geometry["candidates"]
        if not isinstance(candidates, dict):
            raise TypeError("invalid current safety geometry")
        candidate = int(geometry["hardest_candidate"])
        negative = int(np.asarray(candidates[candidate]["rows"], dtype=np.int64)[0])
        safety_role = (
            EXACT_BOUNDARY_SAFETY
            if float(geometry["margin"]) <= margin else EXACT_BOUNDARY_GUARD
        )
        writer.append(
            int(geometry["anchor"]), [int(geometry["positive_row"])], [negative],
            int(geometry["query"]), candidate, 0, safety_role,
        )
    safety_events = len(writer.anchor) - correction_events
    replay_audit = append_dreams_replay(writer, replay, dreams_replay_events, seed)
    output = writer.arrays()
    proof_formulas = {
        str(manifest["query_formula"][int(body["query"])]) for body in proofs
    }
    proof_identities = {
        str(manifest["query_ik14"][int(body["query"])]) for body in proofs
    }
    summary = {
        "evidence_queries": int(len(queries)),
        "phasea_current_errors": int(current_errors),
        "errors_without_exact_boundary_proof": int(missing_boundary_proof),
        "proofs_without_active_reference": int(inactive_proof),
        "correction_queries": int(len(correction_queries)),
        "correction_formulas": int(len(proof_formulas)),
        "correction_identities": int(len(proof_identities)),
        "baseline_boundary_queries": int(baseline_queries),
        "alternate_boundary_queries": int(alternate_queries),
        "strict_boundary_queries": int(strict_queries),
        "correction_events": int(correction_events),
        "available_current_correct_safety_queries": int(len(safe)),
        "selected_safety_events": int(safety_events),
        "selected_active_safety_events": int(sum(
            float(body["margin"]) <= margin for body in selected_safety
        )),
        "selected_inactive_guard_events": int(sum(
            float(body["margin"]) > margin for body in selected_safety
        )),
        "requested_safety_events": int(safety_target),
        "dreams_replay_events": int(replay_audit["retained"]),
        "median_proof_specificity": (
            float(np.median([body["median_specificity"] for body in proofs]))
            if proofs else None
        ),
        "minimum_proof_specificity": (
            float(min(body["minimum_specificity"] for body in proofs))
            if proofs else None
        ),
    }
    return output, summary


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if set(args.training_roles) != {0, 1}:
        raise ValueError("exact-boundary mining is restricted to formula roles 0 and 1")
    if args.margin != 0.1:
        raise ValueError("native DreaMS margin must remain 0.1")
    if args.negative_references_per_correction < 1:
        raise ValueError("negative-reference cap must be positive")
    if args.safety_events_per_correction < 1.0:
        raise ValueError("at least one safety event per correction is required")

    evidence = load_npz(args.evidence)
    manifest = load_npz(args.manifest)
    replay = load_npz(args.dreams_replay_pool)
    report_path = args.phasea_cache / "report.json"
    cache_report = json.loads(report_path.read_text(encoding="utf-8"))
    phasea = cache_report.get("status") == "CHEMAWARE_FORMULA_ROLE_CHECKPOINT_EMBEDDINGS_COMPLETE"
    standin = (
        args.allow_official_cache_standin
        and cache_report.get("status") == "chemaware_corrected_manifest_token_cache_complete"
    )
    if not phasea and not standin:
        raise RuntimeError("cache is neither protected Phase-A nor an allowed official stand-in")
    if phasea and set(map(int, cache_report.get("formula_roles", []))) != {0, 1}:
        raise RuntimeError("formal Phase-A cache must contain exactly roles 0 and 1")
    rows_path = args.phasea_cache / "rows.npy"
    embeddings_path = args.phasea_cache / (
        "embeddings_f32.npy" if phasea else "official_embeddings_f32.npy"
    )
    rows = np.load(rows_path, allow_pickle=False)
    embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    if phasea:
        if array_sha256(rows) != cache_report["rows_array_sha256"]:
            raise RuntimeError("Phase-A cache row hash mismatch")
        if array_sha256(np.asarray(embeddings)) != cache_report["embeddings_array_sha256"]:
            raise RuntimeError("Phase-A cache embedding hash mismatch")
    cache = FrozenEmbeddings(rows_path, embeddings_path)

    folds = stable_formula_folds(manifest["query_formula"], args.folds, args.fold_seed)
    evidence_queries = np.asarray(evidence["query"], dtype=np.int64)
    if set(map(int, np.unique(folds[evidence_queries]))) - set(args.training_roles):
        raise RuntimeError("exact-boundary evidence escaped the training formula roles")

    output, summary = build_exact_pool(
        evidence, manifest, cache, replay,
        margin=args.margin,
        minimum_baseline_wins=args.minimum_baseline_wins,
        minimum_alternate_wins=args.minimum_alternate_wins,
        negative_references_per_correction=args.negative_references_per_correction,
        safety_events_per_correction=args.safety_events_per_correction,
        dreams_replay_events=args.dreams_replay_events,
        seed=args.seed,
    )
    roles = np.asarray(output["curriculum_role"], dtype=np.int64)
    correction_mask = np.isin(
        roles, np.asarray([EXACT_BASELINE_CORRECTION, EXACT_ALTERNATE_CORRECTION]),
    )
    correction_queries = np.asarray(output["source_query"], dtype=np.int64)[correction_mask]
    gates = {
        "formal_phasea_cache_or_explicit_engineering_standin": bool(phasea or standin),
        "minimum_exact_correction_events": (
            int(summary["correction_events"]) >= args.minimum_correction_events
        ),
        "minimum_exact_correction_queries": (
            int(summary["correction_queries"]) >= args.minimum_correction_queries
        ),
        "minimum_exact_correction_formulas": (
            int(summary["correction_formulas"]) >= args.minimum_correction_formulas
        ),
        "every_correction_has_same_query_candidate_proof": bool(
            len(correction_queries)
            and set(map(int, correction_queries)) <= set(map(int, evidence_queries))
        ),
        "no_identity_broadcast": True,
        "requested_safety_budget_met": (
            int(summary["selected_safety_events"]) == int(summary["requested_safety_events"])
        ),
        "exact_dreams_replay_budget": (
            int(np.sum(roles == DREAMS_NATIVE_REPLAY)) == args.dreams_replay_events
        ),
        "no_custom_sampling_weights": "sampling_weight" not in output,
        "unique_triplet_signatures": len(output["anchor_idx"]) == len({
            (
                int(output["anchor_idx"][event]),
                tuple(map(int, output["positive_idx"][
                    int(output["positive_ptr"][event]):int(output["positive_ptr"][event + 1])
                ])),
                tuple(map(int, output["negative_idx"][
                    int(output["negative_ptr"][event]):int(output["negative_ptr"][event + 1])
                ])),
            )
            for event in range(len(output["anchor_idx"]))
        }),
        "outer_roles_2_3_4_untouched": True,
    }
    coverage_pass = all(gates.values())
    identity_audit = audit_identity_edges(output, args.data) if coverage_pass else None
    report = {
        "status": (
            "CHEMAWARE_EXACT_BOUNDARY_RESIDUAL_TRIPLETS_COMPLETE"
            if coverage_pass else "CHEMAWARE_EXACT_BOUNDARY_RESIDUAL_COVERAGE_STOP"
        ),
        "method": (
            "same-query same-current-candidate correct-minus-three-null boundary proof "
            "with native DreaMS triplets, active safety and untouched replay"
        ),
        "cache_kind": "protected_phasea" if phasea else "official_engineering_standin",
        "training_runtime": "unmodified native DreaMS shuffled DataLoader and ContrastiveHead",
        "formula_roles": {"training": [0, 1], "selection": 2, "confirmation": 3, "outer": 4},
        "settings": {
            "margin": args.margin,
            "minimum_baseline_wins": args.minimum_baseline_wins,
            "minimum_alternate_wins": args.minimum_alternate_wins,
            "negative_references_per_correction": args.negative_references_per_correction,
            "safety_events_per_correction": args.safety_events_per_correction,
            "dreams_replay_events": args.dreams_replay_events,
        },
        "provenance": {
            "evidence_sha256": sha256_file(args.evidence),
            "manifest_sha256": sha256_file(args.manifest),
            "dreams_replay_pool_sha256": sha256_file(args.dreams_replay_pool),
            "cache_rows_array_sha256": array_sha256(rows),
            "cache_embeddings_array_sha256": array_sha256(np.asarray(embeddings)),
            "checkpoint_sha256": cache_report.get("checkpoint_sha256"),
        },
        "coverage": summary,
        "events": {
            "exact_baseline_correction": int(np.sum(roles == EXACT_BASELINE_CORRECTION)),
            "exact_alternate_correction": int(np.sum(roles == EXACT_ALTERNATE_CORRECTION)),
            "active_safety": int(np.sum(roles == EXACT_BOUNDARY_SAFETY)),
            "inactive_safety_guard": int(np.sum(roles == EXACT_BOUNDARY_GUARD)),
            "dreams_native_replay": int(np.sum(roles == DREAMS_NATIVE_REPLAY)),
            "total": int(len(roles)),
        },
        "identity_audit": identity_audit,
        "gates": gates,
        "decision": (
            "TRAIN_NATIVE_CONTINUATION"
            if coverage_pass else "STOP_BEFORE_TRAINING_INSUFFICIENT_EXACT_COVERAGE"
        ),
        "claim_boundary": (
            "construction only; performance requires frozen role-2 gain over protected "
            "Phase A and independent role-3 confirmation"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_exact_boundary_", dir=args.output.parent))
    try:
        if coverage_pass:
            np.savez_compressed(temporary / "train_pool.npz", **output)
            shutil.copy2(args.validation_pool, temporary / "val_pool.npz")
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)
    if not coverage_pass:
        print(
            "CHEMAWARE_EXACT_BOUNDARY_COVERAGE_STOP: no training pool emitted; "
            "do not lower the frozen coverage gates",
            flush=True,
        )


if __name__ == "__main__":
    main()
