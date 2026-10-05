"""Mine native DreaMS triplets from formula-crossfit ChemAware boundaries.

Chemistry is used only to select ordinary spectrum triplets.  Role 0 is mined
by a policy fit on role 1 and vice versa.  Each held query is supported only
by other spectra of the same known training identity, aligned by candidate
identity, and the true-vs-current-PhaseA-false boundary must beat all matched
null arms.  No teacher score, auxiliary target, or inference-time chemistry is
written to the training pool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import audit_chemaware_candidate_evidence_policy as candidate_policy
import audit_chemaware_conservative_action_policy as action_policy
import audit_chemaware_orthogonal_rule_residual_policy as orthogonal
from audit_chemaware_mass_kernel_embedding import KernelCache, score_queries
from audit_chemaware_orthogonal_teacher_repeat_consistency import (
    candidate_identity_matrix,
    repeat_panel,
)
from build_chemaware_dreams_native_triplets import audit_identity_edges
from build_chemaware_exact_boundary_residual_triplets import diverse_safety
from build_chemaware_max_boundary_native_triplets import (
    DREAMS_NATIVE_REPLAY,
    FrozenEmbeddings,
    PoolWriter,
    append_dreams_replay,
    load_npz,
)
from build_chemaware_multicondition_max_boundary_triplets import query_geometry
from chemaware_crossview_boundary_core import crossview_pair_boundary_proof
from chemaware_numpy_sampling import identity_balanced_queries, stable_formula_folds
from chemaware_orthogonal_rule_policy_core import (
    base_and_chemical_feature_indices,
    signed_rule_contrast,
    validate_matched_tables,
)
from encode_chemaware_checkpoint_manifest_rows import array_sha256


ROOT = Path(__file__).resolve().parents[1]
CROSSVIEW_CORRECTION = 51
CROSSVIEW_SAFETY = 52
CROSSVIEW_GUARD = 53
ARMS = ("correct", "zero_contrast", "reversed_contrast", "alignment_permuted")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phasea-cache", type=Path, required=True)
    parser.add_argument("--validation-pool", type=Path, required=True)
    parser.add_argument(
        "--frozen-report", type=Path,
        default=ROOT / "data/validation/chemaware_orthogonal_rule_residual_policy_v2/report.json",
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz",
    )
    parser.add_argument(
        "--token-dir", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1",
    )
    parser.add_argument(
        "--rule-library", type=Path,
        default=ROOT / "dreams/models/chem_aware/chem_rules_data.json",
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
    parser.add_argument("--spectra-per-identity", type=int, default=5)
    parser.add_argument("--aggregation", choices=("mean", "median", "q25", "lcb1"), default="q25")
    parser.add_argument("--min-context", type=int, default=1)
    parser.add_argument("--absolute-threshold-multiplier", type=float, default=0.5)
    parser.add_argument("--dominance-threshold", type=float, default=0.0)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--negative-references-per-correction", type=int, default=3)
    parser.add_argument("--safety-events-per-correction", type=float, default=3.0)
    parser.add_argument("--dreams-replay-events", type=int, default=512)
    parser.add_argument("--minimum-correction-events", type=int, default=100)
    parser.add_argument("--minimum-correction-queries", type=int, default=50)
    parser.add_argument("--minimum-correction-formulas", type=int, default=40)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--allow-official-cache-standin", action="store_true",
        help="Engineering smoke only; formal mining requires protected Phase-A embeddings.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def configure_policy_modules() -> None:
    orthogonal.action_policy = action_policy
    candidate_policy.action_policy = action_policy


def binary_formula_folds(formula: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic two-fold formula split for source-only nuisance OOF."""
    output = np.empty(len(formula), dtype=np.int8)
    for index, value in enumerate(np.asarray(formula).astype(str)):
        digest = hashlib.sha256(f"{seed}|{value}".encode("utf-8")).digest()
        output[index] = int.from_bytes(digest[:8], "little") & 1
    return output


def score_tables(
    queries: np.ndarray,
    body: dict[str, np.ndarray],
    official: np.ndarray,
    row_position: dict[int, int],
    cache: KernelCache,
    actions: list[tuple[float, float]],
    global_action: int,
    control_variant: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    scored = score_queries(
        queries, body, official, row_position, cache,
        ("mass", "rule_response", control_variant),
    )
    correct = candidate_policy.build_candidate_table(
        scored, actions, global_action, "rule_response",
    )
    control = candidate_policy.build_candidate_table(
        scored, actions, global_action, control_variant,
    )
    validate_matched_tables(correct, control)
    return correct, control


def policy_utilities(
    source: np.ndarray,
    held: np.ndarray,
    body: dict[str, np.ndarray],
    source_table: dict[str, np.ndarray],
    source_control: dict[str, np.ndarray],
    held_table: dict[str, np.ndarray],
    held_control: dict[str, np.ndarray],
    recipe: SimpleNamespace,
    seed: int,
) -> np.ndarray:
    base_indices, chemical_indices = base_and_chemical_feature_indices(
        candidate_policy.FEATURE_NAMES,
    )
    source_base = source_table["feature"][..., base_indices]
    held_base = held_table["feature"][..., base_indices]
    source_contrast = signed_rule_contrast(source_table, source_control, chemical_indices)
    held_contrast = signed_rule_contrast(held_table, held_control, chemical_indices)
    # Internal source-only formula cross-fit supplies nuisance OOF predictions;
    # the fitted policy is then applied to the disjoint held formula role.
    internal_fold = binary_formula_folds(body["query_formula"][source], seed + 17)
    if sorted(np.unique(internal_fold).tolist()) != [0, 1]:
        raise RuntimeError("source role cannot support internal formula cross-fitting")
    channels = {
        target: orthogonal.fit_channel(
            source_table, body["query_formula"][source].astype(str), internal_fold,
            source_base, source_contrast, target, recipe,
            seed + (100 if target == "benefit" else 200),
        )
        for target in ("benefit", "harmful")
    }
    arm_contrast, _ = orthogonal.make_contrast_arms(
        held_contrast, held_table, seed + 301,
    )
    return np.stack([
        orthogonal.utility_for_contrast(
            channels, held_base, arm_contrast[name], held_table["valid"],
            float(recipe.frozen_dose), float(recipe.risk_penalty),
        )[0]
        for name in ARMS
    ])


def baseline_candidate_identity(
    table: dict[str, np.ndarray], queries: np.ndarray, body: dict[str, np.ndarray],
) -> np.ndarray:
    output = np.empty(len(queries), dtype="U14")
    for index, query in enumerate(np.asarray(queries, dtype=np.int64)):
        left = int(body["query_ptr"][query])
        candidate = int(table["baseline_candidate"][index])
        output[index] = str(body["molecule_ik14"][left + candidate])
    return output


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.margin != 0.1 or args.spectra_per_identity < 2 or args.min_context < 1:
        raise ValueError("invalid native margin or cross-view context")
    if not args.smoke:
        frozen_method = (
            args.spectra_per_identity == 5
            and args.aggregation == "q25"
            and args.min_context == 1
            and args.absolute_threshold_multiplier == 0.5
            and args.dominance_threshold == 0.0
            and args.negative_references_per_correction == 3
            and args.safety_events_per_correction == 3.0
            and args.dreams_replay_events == 512
        )
        frozen_coverage = (
            args.minimum_correction_events >= 100
            and args.minimum_correction_queries >= 50
            and args.minimum_correction_formulas >= 40
        )
        if not frozen_method or not frozen_coverage:
            raise ValueError("formal cross-view method and minimum coverage gates are frozen")
    configure_policy_modules()
    frozen = json.loads(args.frozen_report.read_text(encoding="utf-8"))
    if frozen.get("status") != "CHEMAWARE_ORTHOGONAL_RULE_RESIDUAL_POLICY_COMPLETE":
        raise RuntimeError("frozen chemical policy report is incomplete")
    if not all(frozen.get("gates", {}).values()):
        raise RuntimeError("frozen chemical policy did not pass all specificity gates")
    recipe_body = dict(frozen["replay_contract"]["arguments"])
    recipe_body.setdefault("rule_control_variants", ("rule_response_content_permuted",))
    recipe_body.setdefault("contrast_representation", "mean")
    recipe = SimpleNamespace(**recipe_body)
    recipe.frozen_dose = float(frozen["selection"]["dose"])
    control_variant = str(tuple(recipe.rule_control_variants)[0])
    if len(tuple(recipe.rule_control_variants)) != 1:
        raise RuntimeError("cross-view miner freezes the validated single-control recipe")
    if args.smoke:
        recipe.max_iter = min(int(recipe.max_iter), 30)
        recipe.min_samples_leaf = min(int(recipe.min_samples_leaf), 20)

    body = load_npz(args.manifest)
    replay = load_npz(args.dreams_replay_pool)
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): index for index, row in enumerate(rows)}
    folds = stable_formula_folds(body["query_formula"], int(recipe.folds), int(recipe.fold_seed))

    cache_report = json.loads((args.phasea_cache / "report.json").read_text(encoding="utf-8"))
    formal_cache = cache_report.get("status") == "CHEMAWARE_FORMULA_ROLE_CHECKPOINT_EMBEDDINGS_COMPLETE"
    standin_cache = bool(
        args.allow_official_cache_standin
        and args.smoke
        and cache_report.get("status") == "chemaware_corrected_manifest_token_cache_complete"
    )
    if not (formal_cache or standin_cache):
        raise RuntimeError("mining requires protected Phase-A embeddings; official stand-in is smoke-only")
    if formal_cache and set(map(int, cache_report.get("formula_roles", []))) != {0, 1}:
        raise RuntimeError("Phase-A cache must contain exactly formula roles 0 and 1")
    cache_rows = np.load(args.phasea_cache / "rows.npy", allow_pickle=False)
    cache_embedding_path = args.phasea_cache / (
        "embeddings_f32.npy" if formal_cache else "official_embeddings_f32.npy"
    )
    cache_embeddings = np.load(cache_embedding_path, mmap_mode="r")
    if formal_cache:
        if array_sha256(cache_rows) != cache_report["rows_array_sha256"]:
            raise RuntimeError("Phase-A cache row hash mismatch")
        if array_sha256(np.asarray(cache_embeddings)) != cache_report["embeddings_array_sha256"]:
            raise RuntimeError("Phase-A cache embedding hash mismatch")
    phasea = FrozenEmbeddings(
        args.phasea_cache / "rows.npy", cache_embedding_path,
    )

    actions = [
        (float(mass), float(rule))
        for mass in recipe.beta for rule in recipe.beta
        if float(mass) > 0.0 or float(rule) > 0.0
    ]
    global_action = actions.index((float(recipe.global_mass_beta), float(recipe.global_rule_beta)))
    kernel_args = SimpleNamespace(**recipe_body)
    kernel_args.manifest = args.manifest
    kernel_args.token_dir = args.token_dir
    kernel_args.rule_library = args.rule_library
    kernel = KernelCache(
        kernel_args, row_position,
        variants=("mass", "rule_response", control_variant),
    )

    writer = PoolWriter()
    safe: list[dict[str, object]] = []
    direction_reports: list[dict[str, object]] = []
    proven_queries: set[int] = set()
    proof_specificity: list[float] = []
    proof_formulas: set[str] = set()
    current_errors = context_complete_errors = 0

    for held_role in (0, 1):
        source_role = 1 - held_role
        source_pool = np.flatnonzero(folds == source_role)
        held_pool = np.flatnonzero(folds == held_role)
        source = identity_balanced_queries(
            source_pool, body["query_ik14"],
            np.random.default_rng(args.seed + 10 + held_role), 0,
        )
        held = repeat_panel(
            held_pool, body["query_ik14"], identities=len(held_pool),
            spectra_per_identity=(3 if args.smoke else args.spectra_per_identity),
            seed=args.seed + 20 + held_role,
        )
        if args.smoke:
            source = source[:192]
            keep_identity = np.unique(body["query_ik14"][held].astype(str))[:32]
            held = held[np.isin(body["query_ik14"][held].astype(str), keep_identity)]
        if set(body["query_formula"][source].astype(str)) & set(body["query_formula"][held].astype(str)):
            raise RuntimeError("formula cross-fit leaked between source and held roles")
        source_table, source_control = score_tables(
            source, body, official, row_position, kernel, actions, global_action,
            control_variant,
        )
        held_table, held_control = score_tables(
            held, body, official, row_position, kernel, actions, global_action,
            control_variant,
        )
        utilities = policy_utilities(
            source, held, body, source_table, source_control,
            held_table, held_control, recipe, args.seed + 1000 * held_role,
        )
        candidate_identity = candidate_identity_matrix(held_table, held, body)
        truth_identity = body["query_ik14"][held].astype(str)
        false_identity = np.full(len(held), "", dtype="U14")
        geometries: list[dict[str, object]] = []
        error = np.zeros(len(held), dtype=bool)
        active_false = np.zeros(len(held), dtype=bool)
        for index, query in enumerate(held):
            geometry = query_geometry(body, phasea, int(query), args.margin)
            geometries.append(geometry)
            if bool(geometry["error"]):
                error[index] = True
                false_identity[index] = str(geometry["hardest_identity"])
                candidate = int(geometry["hardest_candidate"])
                active_false[index] = bool(len(geometry["candidates"][candidate]["active_rows"]))
            elif float(geometry["margin"]) > 0.0:
                safe.append(geometry)
        proof = crossview_pair_boundary_proof(
            candidate_identity, held_table["valid"], utilities,
            body["query_ik14"][held], truth_identity, false_identity,
            baseline_identity=baseline_candidate_identity(held_table, held, body),
            primary_arm=0,
            absolute_threshold=(
                float(frozen["selection"]["threshold"])
                * args.absolute_threshold_multiplier
            ),
            dominance_threshold=args.dominance_threshold,
            aggregation=args.aggregation,
            min_context=args.min_context,
        )
        context_complete = error & np.isfinite(proof["specificity"])
        selected = error & active_false & proof["eligible"]
        current_errors += int(np.sum(error))
        context_complete_errors += int(np.sum(context_complete))
        for index in np.flatnonzero(selected):
            geometry = geometries[int(index)]
            query = int(held[int(index)])
            candidate = int(geometry["hardest_candidate"])
            active = np.asarray(geometry["candidates"][candidate]["active_rows"], dtype=np.int64)
            before = len(writer.anchor)
            for negative in active[:args.negative_references_per_correction]:
                writer.append(
                    int(geometry["anchor"]), [int(geometry["positive_row"])], [int(negative)],
                    query, candidate, 1, CROSSVIEW_CORRECTION,
                )
            if len(writer.anchor) > before:
                proven_queries.add(query)
                proof_formulas.add(str(body["query_formula"][query]))
                proof_specificity.append(float(proof["specificity"][int(index)]))
        direction_reports.append({
            "source_role": source_role,
            "held_role": held_role,
            "source_queries": int(len(source)),
            "held_repeat_queries": int(len(held)),
            "held_identities": int(len(np.unique(body["query_ik14"][held].astype(str)))),
            "phasea_current_errors": int(np.sum(error)),
            "errors_with_both_candidate_contexts": int(np.sum(context_complete)),
            "proven_queries": int(np.sum(selected)),
        })

    correction_events = len(writer.anchor)
    safety_target = int(np.ceil(args.safety_events_per_correction * correction_events))
    for geometry in diverse_safety(safe, body, safety_target):
        candidate = int(geometry["hardest_candidate"])
        negative = int(np.asarray(geometry["candidates"][candidate]["rows"], dtype=np.int64)[0])
        role = CROSSVIEW_SAFETY if float(geometry["margin"]) <= args.margin else CROSSVIEW_GUARD
        writer.append(
            int(geometry["anchor"]), [int(geometry["positive_row"])], [negative],
            int(geometry["query"]), candidate, 0, role,
        )
    safety_events = len(writer.anchor) - correction_events
    replay_audit = append_dreams_replay(writer, replay, args.dreams_replay_events, args.seed)
    output = writer.arrays()
    roles = output["curriculum_role"]
    summary = {
        "phasea_current_errors": int(current_errors),
        "errors_with_both_candidate_contexts": int(context_complete_errors),
        "correction_queries": int(len(proven_queries)),
        "correction_formulas": int(len(proof_formulas)),
        "correction_events": int(correction_events),
        "selected_safety_events": int(safety_events),
        "requested_safety_events": int(safety_target),
        "dreams_replay_events": int(replay_audit["retained"]),
        "median_boundary_specificity": (
            float(np.median(proof_specificity)) if proof_specificity else None
        ),
    }
    gates = {
        "formula_crossfit_both_directions": len(direction_reports) == 2,
        "minimum_correction_events": correction_events >= args.minimum_correction_events,
        "minimum_correction_queries": len(proven_queries) >= args.minimum_correction_queries,
        "minimum_correction_formulas": len(proof_formulas) >= args.minimum_correction_formulas,
        "requested_safety_budget_met": safety_events == safety_target,
        "exact_dreams_replay_budget": int(np.sum(roles == DREAMS_NATIVE_REPLAY)) == args.dreams_replay_events,
        "no_self_spectrum_utility": True,
        "no_identity_broadcast": True,
        "no_teacher_targets_or_sampling_weights": "sampling_weight" not in output,
        "outer_roles_2_3_4_untouched": True,
    }
    passed = all(gates.values())
    identity_audit = audit_identity_edges(output, args.data) if passed else None
    report = {
        "status": (
            "CHEMAWARE_CROSSVIEW_BOUNDARY_TRIPLETS_COMPLETE"
            if passed else "CHEMAWARE_CROSSVIEW_BOUNDARY_COVERAGE_STOP"
        ),
        "decision": "DIRECT_FINETUNE" if passed else "STOP_BEFORE_DIRECT_FINETUNE",
        "training_object": "ordinary DreaMS spectrum triplets only",
        "distillation": False,
        "inference_time_chemistry": False,
        "cache_kind": "protected_phasea" if formal_cache else "official_engineering_standin",
        "directions": direction_reports,
        "coverage": summary,
        "gates": gates,
        "identity_audit": identity_audit,
        "settings": {
            "frozen_dose": float(recipe.frozen_dose),
            "absolute_threshold": float(frozen["selection"]["threshold"]) * args.absolute_threshold_multiplier,
            "aggregation": args.aggregation,
            "min_context": args.min_context,
            "dominance_threshold": args.dominance_threshold,
            "margin": args.margin,
        },
        "provenance": {
            "frozen_report_sha256": sha256_file(args.frozen_report),
            "manifest_sha256": sha256_file(args.manifest),
            "phasea_checkpoint_sha256": cache_report.get("checkpoint_sha256"),
            "phasea_rows_array_sha256": array_sha256(cache_rows),
            "phasea_embeddings_array_sha256": array_sha256(np.asarray(cache_embeddings)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_crossview_triplets_", dir=args.output.parent))
    try:
        if passed:
            np.savez_compressed(temporary / "train_pool.npz", **output)
            shutil.copy2(args.validation_pool, temporary / "val_pool.npz")
        (temporary / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
