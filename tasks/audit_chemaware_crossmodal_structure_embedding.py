"""Audit a spectrum-only structural residual appended to frozen DreaMS.

The chemical teacher is a frozen ChemBERTa vector derived from the molecule
SMILES.  It is used only as a training label.  At inference, a ridge map
predicts a structural vector from one official DreaMS spectrum embedding and
the deployable shared embedding is::

    concat(z_official, sqrt(beta) * z_structure_predicted) / sqrt(1 + beta)

Formula fold 0 fits the map, fold 1 selects beta, and fold 2 is a bounded
development confirmation.  Historical folds 3/4 are never read.  A teacher-
identity-permuted ridge map has exactly the same capacity and optimization and
is the required causal control.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz")
    parser.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--teacher-dir", type=Path, default=ROOT / "data/validation/chemaware_chemberta_teacher_v1")
    parser.add_argument("--data-semantics-report", type=Path, default=ROOT / "data/validation/chemaware_data_semantics_gate_v1/report.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--fit-fold", type=int, default=0)
    parser.add_argument("--selection-fold", type=int, default=1)
    parser.add_argument("--confirmation-fold", type=int, default=2)
    parser.add_argument("--inner-fold", type=int, default=3)
    parser.add_argument("--outer-fold", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--spectrum-projection-dim", type=int, default=256)
    parser.add_argument("--structure-projection-dim", type=int, default=128)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--betas", type=float, nargs="+", default=(0.0, 0.02, 0.05, 0.10, 0.20))
    parser.add_argument("--max-fit-identities", type=int, default=0)
    parser.add_argument("--max-selection-identities", type=int, default=0)
    parser.add_argument("--max-confirmation-identities", type=int, default=0)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--screen-ledgers", type=Path, nargs="*", default=())
    parser.add_argument(
        "--confirmation-ledger", type=Path,
        default=ROOT / "data/validation/chemaware_psd_observation_locked_confirmation_v1/metric_and_ranks.npz",
        help="Exact previously locked query rows; prevents identity-level query resampling drift.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ])


def stable_formula_folds(formulas: np.ndarray, folds: int, seed: int) -> np.ndarray:
    if folds < 5:
        raise ValueError("five disjoint roles require at least five folds")
    return np.asarray([
        int.from_bytes(hashlib.sha256(f"{seed}|{value}".encode()).digest()[:8], "little") % folds
        for value in formulas.astype(str)
    ], dtype=np.int16)


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def identity_balanced(indices: np.ndarray, identities: np.ndarray, seed: int, limit: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = np.asarray(indices, dtype=np.int64)[rng.permutation(len(indices))]
    chosen: list[int] = []
    seen: set[str] = set()
    for index in order:
        identity = str(identities[index])
        if identity in seen:
            continue
        seen.add(identity); chosen.append(int(index))
        if limit and len(chosen) >= limit:
            break
    return np.asarray(chosen, dtype=np.int64)


def fit_ridge(x: np.ndarray, y: np.ndarray, formula: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    """Fit formula-balanced multi-output ridge with an explicit intercept."""
    if len(x) != len(y) or len(x) != len(formula):
        raise ValueError("ridge arrays are not aligned")
    _, inverse = np.unique(formula.astype(str), return_inverse=True)
    counts = np.bincount(inverse)
    weight = 1.0 / counts[inverse]
    weight *= len(weight) / weight.sum()
    x_mean = np.average(x, axis=0, weights=weight).astype(np.float32)
    y_mean = np.average(y, axis=0, weights=weight).astype(np.float32)
    xc = np.asarray(x - x_mean, dtype=np.float64)
    yc = np.asarray(y - y_mean, dtype=np.float64)
    root = np.sqrt(weight).astype(np.float64)
    lhs = (xc * root[:, None]).T @ (xc * root[:, None])
    lhs.flat[:: lhs.shape[0] + 1] += float(alpha)
    rhs = (xc * root[:, None]).T @ (yc * root[:, None])
    coefficient = np.linalg.solve(lhs, rhs).astype(np.float32)
    return {"x_mean": x_mean, "y_mean": y_mean, "coefficient": coefficient}


def predict_ridge(x: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    return normalize((x - model["x_mean"]) @ model["coefficient"] + model["y_mean"])


def formula_bootstrap(delta: np.ndarray, formula: np.ndarray, seed: int, draws: int) -> dict:
    unique, inverse = np.unique(formula.astype(str), return_inverse=True)
    macro = np.asarray([np.mean(delta[inverse == i]) for i in range(len(unique))], dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        boot[draw] = np.mean(macro[rng.integers(0, len(macro), len(macro))])
    return {
        "formula_macro_delta": float(np.mean(macro)),
        "formula_cluster_bootstrap_95ci": [float(value) for value in np.quantile(boot, (0.025, 0.975))],
        "formula_clusters": int(len(unique)), "draws": int(draws),
    }


def evaluate(
    queries: np.ndarray, body: dict[str, np.ndarray], official: np.ndarray,
    chemical: np.ndarray, row_position: dict[int, int], beta: float,
) -> np.ndarray:
    rank = np.empty(len(queries), dtype=np.int16)
    scale = 1.0 / (1.0 + beta)
    for position, query in enumerate(map(int, queries)):
        left, right = map(int, body["query_ptr"][query:query + 2])
        labels = body["molecule_label"][left:right].astype(bool)
        qpos = row_position[int(body["query_row"][query])]
        scores = []
        for molecule in range(left, right):
            rleft, rright = map(int, body["molecule_ptr"][molecule:molecule + 2])
            rpos = np.asarray([row_position[int(row)] for row in body["pair_candidate_row"][rleft:rright]], dtype=np.int64)
            combined = (official[rpos] @ official[qpos] + beta * (chemical[rpos] @ chemical[qpos])) * scale
            scores.append(float(np.max(combined)))
        values = np.asarray(scores)
        positive = float(np.max(values[labels]))
        rank[position] = 1 + int(np.sum(values[~labels] >= positive))
    return rank


def compare(left: np.ndarray, right: np.ndarray, formulas: np.ndarray, seed: int, draws: int) -> dict:
    delta = (left == 1).astype(np.float64) - (right == 1).astype(np.float64)
    return {"query_delta": float(np.mean(delta)), **formula_bootstrap(delta, formulas, seed, draws)}


def load_teacher(path: Path, fit_identities: np.ndarray, seed: int, output_dim: int) -> tuple[dict[str, np.ndarray], dict]:
    report = json.loads((path / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "chemaware_chemberta_teacher_complete":
        raise RuntimeError("ChemBERTa teacher is incomplete")
    identities = np.load(path / "identities.npy").astype(str)
    raw = np.load(path / "embeddings_f16.npy", mmap_mode="r").astype(np.float32)
    index = {identity: i for i, identity in enumerate(identities)}
    positions = np.asarray([index[value] for value in fit_identities if value in index], dtype=np.int64)
    if not len(positions):
        raise RuntimeError("no fit identities have ChemBERTa targets")
    center = np.mean(raw[positions], axis=0, dtype=np.float64).astype(np.float32)
    rng = np.random.default_rng(seed + 31)
    projection = rng.normal(0.0, 1.0 / np.sqrt(output_dim), size=(raw.shape[1], output_dim)).astype(np.float32)
    projected = normalize((raw - center) @ projection)
    return {identity: projected[i] for i, identity in enumerate(identities)}, {
        "teacher_center": center, "teacher_projection": projection,
        "teacher_report": report,
    }


def main() -> None:
    args = arguments(); started = time.time()
    if args.output.exists():
        raise FileExistsError(args.output)
    roles = (args.fit_fold, args.selection_fold, args.confirmation_fold, args.inner_fold, args.outer_fold)
    if len(set(roles)) != len(roles):
        raise ValueError("fit/selection/confirmation/inner/outer folds must be distinct")
    required = [args.manifest, args.data, args.token_dir / "rows.npy",
                args.token_dir / "official_embeddings_f32.npy", args.token_dir / "report.json",
                args.teacher_dir / "report.json", args.teacher_dir / "identities.npy",
                args.teacher_dir / "embeddings_f16.npy", args.data_semantics_report,
                args.confirmation_ledger]
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(missing)
    semantics = json.loads(args.data_semantics_report.read_text(encoding="utf-8"))
    token_report = json.loads((args.token_dir / "report.json").read_text(encoding="utf-8"))
    if semantics.get("status") != "CHEMAWARE_DATA_SEMANTICS_PASS":
        raise RuntimeError("data semantics gate did not pass")
    if token_report.get("status") != "chemaware_corrected_manifest_token_cache_complete":
        raise RuntimeError("official token cache is incomplete")
    if args.bootstrap_draws < 10_000:
        raise ValueError("at least 10,000 formula-cluster bootstrap draws are required")
    if not args.screen_ledgers:
        raise ValueError("screen ledgers are required to protect the residual confirmation set")

    with np.load(args.manifest) as loaded:
        body = {key: loaded[key] for key in loaded.files}
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    # Preserve the frozen official vectors bit-for-bit.  Re-normalizing values
    # that are already unit length can change exact ties and therefore changes
    # the canonical beta=0 baseline.
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): i for i, row in enumerate(rows)}
    with h5py.File(args.data, "r") as handle:
        row_identity = np.asarray([value[:14] for value in decode(handle["INCHIKEY"][rows])])
        row_formula = decode(handle["FORMULA"][rows])
    row_fold = stable_formula_folds(row_formula, args.folds, args.fold_seed)

    fit_row = np.flatnonzero(row_fold == args.fit_fold)
    by_identity: dict[str, list[int]] = {}
    for position in fit_row:
        by_identity.setdefault(str(row_identity[position]), []).append(int(position))
    fit_identity_all = np.asarray(sorted(by_identity))
    teacher_map, teacher_state = load_teacher(
        args.teacher_dir, fit_identity_all, args.seed, args.structure_projection_dim,
    )
    fit_identity_all = np.asarray([value for value in fit_identity_all if value in teacher_map])
    if args.max_fit_identities:
        rng = np.random.default_rng(args.seed + 1)
        fit_identity_all = fit_identity_all[rng.permutation(len(fit_identity_all))[:args.max_fit_identities]]
    fit_formula = np.asarray([row_formula[by_identity[value][0]] for value in fit_identity_all])
    fit_official = normalize(np.stack([np.mean(official[by_identity[value]], axis=0) for value in fit_identity_all]))

    rng = np.random.default_rng(args.seed + 7)
    spectrum_projection = rng.normal(
        0.0, 1.0 / np.sqrt(args.spectrum_projection_dim),
        size=(official.shape[1], args.spectrum_projection_dim),
    ).astype(np.float32)
    fit_x = normalize(fit_official @ spectrum_projection)
    fit_y = np.stack([teacher_map[value] for value in fit_identity_all]).astype(np.float32)
    real_model = fit_ridge(fit_x, fit_y, fit_formula, args.ridge_alpha)
    permutation = rng.permutation(len(fit_y))
    shuffled_model = fit_ridge(fit_x, fit_y[permutation], fit_formula, args.ridge_alpha)

    all_x = normalize(official @ spectrum_projection)
    real_chemical = predict_ridge(all_x, real_model)
    shuffled_chemical = predict_ridge(all_x, shuffled_model)
    random_weight = rng.normal(
        0.0, 1.0 / np.sqrt(args.structure_projection_dim),
        size=(args.spectrum_projection_dim, args.structure_projection_dim),
    ).astype(np.float32)
    random_chemical = normalize(all_x @ random_weight)

    query_fold = stable_formula_folds(body["query_formula"].astype(str), args.folds, args.fold_seed)
    selection_pool = np.flatnonzero(query_fold == args.selection_fold)
    confirmation_pool = np.flatnonzero(query_fold == args.confirmation_fold)
    excluded: set[str] = set()
    ledger_hashes = {}
    for ledger_path in args.screen_ledgers:
        if not ledger_path.is_file():
            raise FileNotFoundError(ledger_path)
        with np.load(ledger_path) as ledger:
            if "confirmation_formula" not in ledger.files:
                raise RuntimeError(f"screen ledger lacks confirmation_formula: {ledger_path}")
            excluded.update(ledger["confirmation_formula"].astype(str))
        ledger_hashes[str(ledger_path)] = sha256_file(ledger_path)
    confirmation_pool = confirmation_pool[
        ~np.isin(body["query_formula"][confirmation_pool].astype(str), list(excluded))
    ]
    selection = identity_balanced(selection_pool, body["query_ik14"], args.seed + 11, args.max_selection_identities)
    with np.load(args.confirmation_ledger) as locked:
        if "confirmation_query" not in locked.files or "confirmation_formula" not in locked.files:
            raise RuntimeError("confirmation ledger lacks locked queries or formulas")
        confirmation = locked["confirmation_query"].astype(np.int64)
        locked_formula = locked["confirmation_formula"].astype(str)
    if not np.array_equal(body["query_formula"][confirmation].astype(str), locked_formula):
        raise RuntimeError("locked confirmation query/formula alignment failed")
    if np.any(query_fold[confirmation] != args.confirmation_fold):
        raise RuntimeError("locked queries are outside the confirmation fold")
    if not set(map(int, confirmation)).issubset(set(map(int, confirmation_pool))):
        raise RuntimeError("locked queries violate screen-ledger exclusions")
    if len(np.unique(body["query_ik14"][confirmation].astype(str))) != len(confirmation):
        raise RuntimeError("locked confirmation is not identity-distinct")
    if args.max_confirmation_identities:
        confirmation = confirmation[:args.max_confirmation_identities]
    if not len(selection) or not len(confirmation):
        raise RuntimeError("selection or confirmation set is empty")

    zero = np.zeros_like(real_chemical)
    official_selection = evaluate(selection, body, official, zero, row_position, 0.0)
    selection_results = []
    for beta in sorted(set(args.betas)):
        real_rank = evaluate(selection, body, official, real_chemical, row_position, beta)
        shuffled_rank = evaluate(selection, body, official, shuffled_chemical, row_position, beta)
        random_rank = evaluate(selection, body, official, random_chemical, row_position, beta)
        recall = {"official": float(np.mean(official_selection == 1)),
                  "real": float(np.mean(real_rank == 1)),
                  "shuffled": float(np.mean(shuffled_rank == 1)),
                  "random": float(np.mean(random_rank == 1))}
        advantage = min(recall["real"] - recall[name] for name in ("official", "shuffled", "random"))
        selection_results.append({"beta": float(beta), "recall1": recall, "minimum_causal_advantage": float(advantage)})
    chosen = max(selection_results, key=lambda item: (item["minimum_causal_advantage"], -item["beta"]))
    beta = float(chosen["beta"])

    arms = {"official": (zero, 0.0), "real_teacher": (real_chemical, beta),
            "teacher_identity_permuted": (shuffled_chemical, beta), "untrained_random": (random_chemical, beta)}
    ranks = {name: evaluate(confirmation, body, official, chemical, row_position, arm_beta)
             for name, (chemical, arm_beta) in arms.items()}
    formula = body["query_formula"][confirmation].astype(str)
    comparisons = {
        f"real_minus_{name}": compare(ranks["real_teacher"], ranks[name], formula, args.seed + 101 + i, args.bootstrap_draws)
        for i, name in enumerate(("official", "teacher_identity_permuted", "untrained_random"))
    }
    gates = {
        "selection_beta_nonzero": beta > 0,
        "selection_causal_advantage_positive": chosen["minimum_causal_advantage"] > 0,
        **{f"{name}_point_positive": value["query_delta"] > 0 for name, value in comparisons.items()},
        **{f"{name}_formula_ci_positive": value["formula_cluster_bootstrap_95ci"][0] > 0 for name, value in comparisons.items()},
    }
    passed = bool(all(gates.values()))
    report = {
        "status": "CHEMAWARE_CROSSMODAL_STRUCTURE_SCREEN_PASS" if passed else "CHEMAWARE_CROSSMODAL_STRUCTURE_FAIL",
        "pass_to_gpu_pilot": passed,
        "formal_training_authorized": False,
        "release_eligible": False,
        "shared_embedding": {
            "definition": "concat(official_dreams, sqrt(beta)*predicted_structure)/sqrt(1+beta)",
            "dimension": int(official.shape[1] + args.structure_projection_dim),
            "official_dreams_frozen": True, "one_clean_spectrum_at_inference": True,
            "candidate_or_structure_input_at_inference": False,
        },
        "split": {
            "fit_fold": args.fit_fold, "selection_fold": args.selection_fold,
            "confirmation_fold": args.confirmation_fold, "inner_fold_read": False,
            "outer_fold_read": False, "fit_identities": int(len(fit_identity_all)),
            "fit_formulas": int(len(np.unique(fit_formula))), "selection_queries": int(len(selection)),
            "selection_formulas": int(len(np.unique(body["query_formula"][selection]))),
            "confirmation_queries": int(len(confirmation)), "confirmation_formulas": int(len(np.unique(formula))),
            "screen_excluded_confirmation_formulas": int(len(excluded)),
        },
        "teacher": {
            "model": teacher_state["teacher_report"]["model"], "raw_dimension": int(teacher_state["teacher_report"]["dimension"]),
            "projected_dimension": args.structure_projection_dim, "smiles_training_label_only": True,
            "connectivity_level": True,
        },
        "optimization": {
            "kind": "formula-balanced closed-form ridge", "spectrum_projection_dim": args.spectrum_projection_dim,
            "ridge_alpha": args.ridge_alpha, "betas": list(map(float, args.betas)), "selected_beta": beta,
            "same_capacity_permuted_teacher_control": True,
        },
        "selection": selection_results,
        "confirmation": {name: {"recall1": float(np.mean(rank == 1)),
                                  "corrected_vs_official": int(np.sum((ranks["official"] > 1) & (rank == 1))),
                                  "introduced_vs_official": int(np.sum((ranks["official"] == 1) & (rank > 1)))}
                         for name, rank in ranks.items()},
        "comparisons": comparisons, "gates": gates,
        "provenance": {
            "manifest_sha256": sha256_file(args.manifest), "hdf5_sha256": sha256_file(args.data),
            "token_report_sha256": sha256_file(args.token_dir / "report.json"),
            "teacher_report_sha256": sha256_file(args.teacher_dir / "report.json"),
            "teacher_embeddings_sha256": sha256_file(args.teacher_dir / "embeddings_f16.npy"),
            "data_semantics_report_sha256": sha256_file(args.data_semantics_report),
            "confirmation_ledger_sha256": sha256_file(args.confirmation_ledger),
            "screen_ledgers": ledger_hashes, "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": "Bounded development confirmation only. Outer fold 4 remains required before any release claim.",
        "runtime_seconds": time.time() - started,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.output / "model_and_ranks.npz", selection_query=selection, confirmation_query=confirmation,
        confirmation_formula=formula, spectrum_projection=spectrum_projection,
        real_coefficient=real_model["coefficient"], real_x_mean=real_model["x_mean"], real_y_mean=real_model["y_mean"],
        shuffled_coefficient=shuffled_model["coefficient"], shuffled_x_mean=shuffled_model["x_mean"], shuffled_y_mean=shuffled_model["y_mean"],
        **{f"{name}_rank": rank for name, rank in ranks.items()},
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
