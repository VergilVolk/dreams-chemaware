"""Audit direct clean-boundary injection from confirmed ChemAware rules.

Confirmed domain-conditioned rules select clean query/positive/negative
retrieval edges.  No spectrum is perturbed and no action embedding is used.
The correct arm chooses a same-formula negative lacking the confirmed parent
predicate.  Its independent control uses the same query and positive but a
different predicate-present negative matched as closely as possible in frozen
official score.  Controls are trained in a separate arm and are never
subtracted from, or assigned a negative weight in, the correct objective.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from chemaware_iceberg_direct_core import stable_formula_folds
from chemaware_rule_clean_boundary_core import (
    choose_opposed_boundaries,
    formula_balanced_weights,
    formula_cluster_interval,
    paired_margin_each,
    retrieval_metrics,
    weighted_mean,
)
from dreams.models.chem_aware.global_embedding_adapter import (
    GlobalResidualEmbeddingAdapter,
)
from mine_chemaware_domain_conditioned_action_rules import (
    decode,
    domain_label,
    sha256_file,
    take_hdf5_rows,
)
from noise_final_core import strict_rank


ARMS = ("correct_chemistry", "predicate_present_control")

RULE_BUNDLES = {
    "conditioned_rules.json": {
        "report_status": "CHEMAWARE_DOMAIN_CONDITIONED_RULES_ADMITTED",
        "schema": "chemaware_domain_conditioned_action_rules_v1",
    },
    "stratified_meta_rules.json": {
        "report_status": "CHEMAWARE_STRATIFIED_META_ACTIONS_ADMITTED",
        "schema": "chemaware_stratified_meta_action_rules_v1",
    },
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz",
    )
    parser.add_argument(
        "--token-dir",
        type=Path,
        default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260935)
    parser.add_argument("--embedding-train-folds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--embedding-evaluation-fold", type=int, default=3)
    parser.add_argument("--reserve-fold", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 29, 43))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument(
        "--activation-margin",
        type=float,
        default=0.05,
        help=(
            "Frozen official positive-minus-chemical-negative margin above which "
            "the corrective weight is exactly zero."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--lambda-preserve", type=float, default=20.0)
    parser.add_argument("--maximum-control-hardness-gap", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--max-triplets", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def rule_event_intensity(
    mz: np.ndarray,
    intensity: np.ndarray,
    mass: float,
    tolerance: float,
    minimum_intensity: float = 0.01,
) -> np.ndarray:
    match = (
        np.isfinite(mz)
        & np.isfinite(intensity)
        & (intensity >= minimum_intensity)
        & (np.abs(mz - float(mass)) <= float(tolerance))
    )
    return np.max(np.where(match, intensity, 0.0), axis=1)


def molecule_predicates(
    body: dict[str, np.ndarray],
    handle: h5py.File,
    smarts: str,
) -> np.ndarray:
    query = Chem.MolFromSmarts(smarts)
    if query is None:
        raise RuntimeError(f"invalid confirmed-rule SMARTS: {smarts}")
    identity = body["molecule_ik14"].astype(str)
    unique, first, inverse = np.unique(identity, return_index=True, return_inverse=True)
    representative_row = np.asarray(
        [
            body["pair_candidate_row"][body["molecule_ptr"][int(position)]]
            for position in first
        ],
        dtype=np.int64,
    )
    smiles = decode(take_hdf5_rows(handle["smiles"], representative_row))
    value = np.asarray(
        [
            bool((molecule := Chem.MolFromSmiles(text)) is not None and molecule.HasSubstructMatch(query))
            for text in smiles
        ],
        dtype=bool,
    )
    if len(value) != len(unique):
        raise RuntimeError("candidate predicate registry is misaligned")
    return value[inverse]


def molecule_scores_and_references(
    body: dict[str, np.ndarray],
    molecule_indices: np.ndarray,
    query_embedding: np.ndarray,
    official: np.ndarray,
    row_position: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.empty(len(molecule_indices), dtype=np.float32)
    reference = np.empty(len(molecule_indices), dtype=np.int64)
    for local, molecule in enumerate(molecule_indices):
        left, right = map(int, body["molecule_ptr"][int(molecule) : int(molecule) + 2])
        positions = np.asarray(
            [
                row_position[int(row)]
                for row in body["pair_candidate_row"][left:right]
                if int(row) in row_position
            ],
            dtype=np.int64,
        )
        if not len(positions):
            raise RuntimeError("candidate molecule has no cached clean reference")
        values = np.asarray(official[positions] @ query_embedding, dtype=np.float32)
        best = int(np.argmax(values))
        scores[local] = values[best]
        reference[local] = positions[best]
    return scores, reference


def resolve_rule_bundle(rules_dir: Path) -> tuple[Path, dict, list[dict]]:
    report_path = rules_dir / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"rule report missing: {report_path}")
    matches = [rules_dir / name for name in RULE_BUNDLES if (rules_dir / name).is_file()]
    if len(matches) != 1:
        raise RuntimeError(
            "clean-boundary injection requires exactly one supported admitted rule file; "
            f"found={[path.name for path in matches]}"
        )
    rules_path = matches[0]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    body = json.loads(rules_path.read_text(encoding="utf-8"))
    expected = RULE_BUNDLES[rules_path.name]
    rules = body.get("rules", [])
    if (
        report.get("status") != expected["report_status"]
        or body.get("schema") != expected["schema"]
        or len(rules) != int(report.get("counts", {}).get("confirmed_rules", -1))
        or not rules
    ):
        raise RuntimeError(
            f"clean-boundary injection requires an admitted {expected['schema']} bundle"
        )
    return rules_path, report, rules


def rule_domains(rule: dict) -> set[str]:
    context = rule["context"]
    if "supported_acquisition_domains" in context:
        domains = {str(value) for value in context["supported_acquisition_domains"]}
        if not domains:
            raise RuntimeError("meta-rule has no supported acquisition domain")
        return domains
    return {
        domain_label(
            context["adduct"],
            context["instrument_family"],
            {
                "very_low_le_10": 10.0,
                "low_10_20": 20.0,
                "medium_low_20_30": 30.0,
                "medium_high_30_40": 40.0,
                "high_40_60": 60.0,
                "very_high_gt_60": 61.0,
            }[context["collision_energy_band"]],
        )
    }


def build_cohort(
    args: argparse.Namespace,
    rules: list[dict],
    body: dict[str, np.ndarray],
    rows: np.ndarray,
    official: np.ndarray,
) -> dict[str, np.ndarray]:
    row_position = {int(row): index for index, row in enumerate(rows)}
    query_cache = np.asarray(
        [row_position[int(row)] for row in body["query_row"]], dtype=np.int64
    )
    mz = np.load(args.token_dir / "mz_f32.npy", mmap_mode="r")
    intensity = np.load(args.token_dir / "intensity_f32.npy", mmap_mode="r")
    formula_fold = stable_formula_folds(body["query_formula"], args.folds, args.fold_seed)
    records: list[dict] = []

    with h5py.File(args.data, "r") as handle:
        instrument = decode(take_hdf5_rows(handle["INSTRUMENT_TYPE"], body["query_row"]))
        energy = np.asarray(
            take_hdf5_rows(handle["COLLISION_ENERGY"], body["query_row"]),
            dtype=np.float64,
        )
        query_domain = np.asarray(
            [
                domain_label(adduct, instrument_value, collision)
                for adduct, instrument_value, collision in zip(
                    body["query_adduct"].astype(str), instrument, energy
                )
            ]
        )
        for rule_index, rule in enumerate(rules):
            if (
                rule.get("suggested_action", {}).get("operation")
                != "support_boost_observed_match"
                or rule.get("suggested_action", {}).get("may_add_new_mz") is not False
                or rule.get("suggested_action", {}).get(
                    "negative_or_absence_action_authorized"
                )
                is not False
            ):
                raise RuntimeError("confirmed rule violates the clean support-action contract")
            expected_domains = rule_domains(rule)
            smarts_values = rule["parent_predicate"]["smarts_any"]
            if len(smarts_values) != 1:
                raise RuntimeError("initial clean-boundary audit requires one SMARTS per rule")
            molecule_present = molecule_predicates(body, handle, smarts_values[0])
            observation = rule["observation"]
            event = rule_event_intensity(
                np.asarray(mz[query_cache]),
                np.asarray(intensity[query_cache]),
                observation["exact_mass_da"],
                observation["tolerance_da"],
            )
            candidate_query = np.flatnonzero(
                np.isin(query_domain, tuple(expected_domains)) & (event > 0)
            )
            # One chemistry-selected spectrum per identity; no retrieval outcome is used.
            selected: dict[str, int] = {}
            for query_index in candidate_query:
                identity = str(body["query_ik14"][query_index])
                previous = selected.get(identity)
                if previous is None or (event[query_index], -int(body["query_row"][query_index])) > (
                    event[previous],
                    -int(body["query_row"][previous]),
                ):
                    selected[identity] = int(query_index)
            for query_index in selected.values():
                left, right = map(int, body["query_ptr"][query_index : query_index + 2])
                molecule_indices = np.arange(left, right, dtype=np.int64)
                labels = body["molecule_label"][left:right]
                candidate_predicate = molecule_present[left:right]
                query_embedding = np.asarray(official[query_cache[query_index]])
                scores, references = molecule_scores_and_references(
                    body,
                    molecule_indices,
                    query_embedding,
                    official,
                    row_position,
                )
                choice = choose_opposed_boundaries(scores, labels, candidate_predicate)
                if choice is None:
                    continue
                positive, chemistry, control = choice
                records.append(
                    {
                        "rule_index": rule_index,
                        "query_index": query_index,
                        "query_cache": int(query_cache[query_index]),
                        "positive_cache": int(references[positive]),
                        "chemistry_negative_cache": int(references[chemistry]),
                        "control_negative_cache": int(references[control]),
                        "formula": str(body["query_formula"][query_index]),
                        "identity": str(body["query_ik14"][query_index]),
                        "formula_fold": int(formula_fold[query_index]),
                        "candidate_count": int(right - left),
                        "official_positive_score": float(scores[positive]),
                        "official_chemistry_negative_score": float(scores[chemistry]),
                        "official_control_negative_score": float(scores[control]),
                        "diagnostic_peak_intensity": float(event[query_index]),
                    }
                )

    if not records:
        raise RuntimeError("confirmed rules produced no switchable clean boundaries")
    records.sort(key=lambda item: (item["formula_fold"], item["formula"], item["identity"]))
    if args.max_triplets:
        retained: list[dict] = []
        for fold in range(args.folds):
            values = [item for item in records if item["formula_fold"] == fold]
            retained.extend(values[: args.max_triplets])
        records = retained
    keys = tuple(records[0])
    return {
        key: np.asarray([item[key] for item in records])
        for key in keys
    }


@torch.no_grad()
def project_positions(
    model: torch.nn.Module,
    official: np.ndarray,
    positions: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[int, int]]:
    unique = np.unique(np.asarray(positions, dtype=np.int64))
    output = []
    model.eval()
    for left in range(0, len(unique), batch_size):
        batch = torch.from_numpy(
            np.array(official[unique[left : left + batch_size]], dtype=np.float32, copy=True)
        ).to(device)
        output.append(model(batch).cpu().numpy())
    values = np.concatenate(output)
    return values, {int(position): index for index, position in enumerate(unique)}


def evaluation_rows(
    body: dict[str, np.ndarray],
    queries: np.ndarray,
    row_position: dict[int, int],
) -> np.ndarray:
    positions: list[int] = []
    for query in queries:
        positions.append(row_position[int(body["query_row"][query])])
        left, right = map(int, body["query_ptr"][query : query + 2])
        for molecule in range(left, right):
            rleft, rright = map(int, body["molecule_ptr"][molecule : molecule + 2])
            positions.extend(
                row_position[int(row)]
                for row in body["pair_candidate_row"][rleft:rright]
                if int(row) in row_position
            )
    return np.unique(np.asarray(positions, dtype=np.int64))


def evaluate_full_candidates(
    body: dict[str, np.ndarray],
    queries: np.ndarray,
    official: np.ndarray,
    row_position: dict[int, int],
    adapted: np.ndarray | None = None,
    adapted_position: dict[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ranks, margins, counts = [], [], []
    for query in queries:
        qpos = row_position[int(body["query_row"][query])]
        query_value = (
            np.asarray(official[qpos])
            if adapted is None
            else adapted[adapted_position[qpos]]
        )
        left, right = map(int, body["query_ptr"][query : query + 2])
        score = []
        for molecule in range(left, right):
            rleft, rright = map(int, body["molecule_ptr"][molecule : molecule + 2])
            positions = [
                row_position[int(row)]
                for row in body["pair_candidate_row"][rleft:rright]
                if int(row) in row_position
            ]
            reference = (
                np.asarray(official[positions])
                if adapted is None
                else adapted[[adapted_position[position] for position in positions]]
            )
            score.append(float(np.max(reference @ query_value)))
        values = np.asarray(score)
        ranks.append(strict_rank(values))
        margins.append(float(values[0] - np.max(values[1:])))
        counts.append(len(values))
    return np.asarray(ranks), np.asarray(margins), np.asarray(counts)


def evaluate_pair_loss(
    model: torch.nn.Module,
    cohort: dict[str, np.ndarray],
    positions: np.ndarray,
    negative_key: str,
    official: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    model.eval()
    values = []
    weights = formula_balanced_weights(cohort["formula"][positions])
    with torch.inference_mode():
        for left in range(0, len(positions), args.batch_size):
            selected = positions[left : left + args.batch_size]
            query = model(torch.from_numpy(np.array(official[cohort["query_cache"][selected]], copy=True)).to(device))
            positive = model(torch.from_numpy(np.array(official[cohort["positive_cache"][selected]], copy=True)).to(device))
            negative = model(torch.from_numpy(np.array(official[cohort[negative_key][selected]], copy=True)).to(device))
            each = paired_margin_each(query, positive, negative, args.margin, args.temperature)
            values.extend(each.cpu().numpy().tolist())
    return float(np.sum(np.asarray(values) * weights) / np.sum(weights))


def train_arm(
    arm: str,
    seed: int,
    cohort: dict[str, np.ndarray],
    train: np.ndarray,
    evaluation: np.ndarray,
    body: dict[str, np.ndarray],
    rows: np.ndarray,
    official: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict, dict[str, np.ndarray], dict]:
    if arm not in ARMS:
        raise ValueError(arm)
    negative_key = (
        "chemistry_negative_cache"
        if arm == "correct_chemistry"
        else "control_negative_cache"
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    model = GlobalResidualEmbeddingAdapter(
        int(official.shape[1]), args.hidden_dim, dropout=0.0
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rng = np.random.default_rng(seed)
    epoch_weight = formula_balanced_weights(cohort["formula"][train])
    initial_correct_loss = evaluate_pair_loss(
        model, cohort, train, "chemistry_negative_cache", official, args, device
    )
    initial_control_loss = evaluate_pair_loss(
        model, cohort, train, "control_negative_cache", official, args, device
    )
    history = []
    model.train()
    for epoch in range(args.epochs):
        order = rng.permutation(len(train))
        total_pair = total_preserve = total_loss = 0.0
        batches = 0
        for left in range(0, len(order), args.batch_size):
            local = order[left : left + args.batch_size]
            selected = train[local]
            query_x = torch.from_numpy(
                np.array(official[cohort["query_cache"][selected]], copy=True)
            ).to(device)
            positive_x = torch.from_numpy(
                np.array(official[cohort["positive_cache"][selected]], copy=True)
            ).to(device)
            negative_x = torch.from_numpy(
                np.array(official[cohort[negative_key][selected]], copy=True)
            ).to(device)
            query = model(query_x)
            positive = model(positive_x)
            negative = model(negative_x)
            weight = torch.from_numpy(epoch_weight[local]).to(device)
            pair = weighted_mean(
                paired_margin_each(
                    query, positive, negative, args.margin, args.temperature
                ),
                weight,
            )
            preserve = torch.cat(
                (
                    1.0 - torch.sum(query * query_x, dim=1),
                    1.0 - torch.sum(positive * positive_x, dim=1),
                    1.0 - torch.sum(negative * negative_x, dim=1),
                )
            ).mean()
            loss = pair + args.lambda_preserve * preserve
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_pair += float(pair.detach())
            total_preserve += float(preserve.detach())
            total_loss += float(loss.detach())
            batches += 1
        history.append(
            {
                "epoch": epoch + 1,
                "pair_loss": total_pair / batches,
                "preserve_loss": total_preserve / batches,
                "total_loss": total_loss / batches,
            }
        )

    final_correct_train = evaluate_pair_loss(
        model, cohort, train, "chemistry_negative_cache", official, args, device
    )
    final_control_train = evaluate_pair_loss(
        model, cohort, train, "control_negative_cache", official, args, device
    )
    final_correct_eval = evaluate_pair_loss(
        model, cohort, evaluation, "chemistry_negative_cache", official, args, device
    )
    final_control_eval = evaluate_pair_loss(
        model, cohort, evaluation, "control_negative_cache", official, args, device
    )
    row_position = {int(row): index for index, row in enumerate(rows)}
    eval_queries = cohort["query_index"][evaluation].astype(np.int64)
    needed = evaluation_rows(body, eval_queries, row_position)
    adapted, adapted_position = project_positions(
        model, official, needed, device, args.eval_batch_size
    )
    ranks, margins, counts = evaluate_full_candidates(
        body,
        eval_queries,
        official,
        row_position,
        adapted,
        adapted_position,
    )
    preservation = float(
        np.mean(np.sum(adapted * np.asarray(official[needed]), axis=1))
    )
    report = {
        "arm": arm,
        "seed": seed,
        "training": {
            "initial_correct_pair_loss": initial_correct_loss,
            "initial_control_pair_loss": initial_control_loss,
            "final_correct_pair_loss": final_correct_train,
            "final_control_pair_loss": final_control_train,
            "correct_pair_loss_reduction": initial_correct_loss - final_correct_train,
            "history": history,
        },
        "embedding_evaluation": {
            "correct_pair_loss": final_correct_eval,
            "control_pair_loss": final_control_eval,
            "retrieval": retrieval_metrics(ranks, counts),
            "preservation_cosine": preservation,
        },
    }
    checkpoint = {
        "format": "chemaware_rule_clean_boundary_adapter_v1",
        "arm": arm,
        "seed": seed,
        "adapter_config": {
            "dimension": int(official.shape[1]),
            "hidden_dim": args.hidden_dim,
            "dropout": 0.0,
        },
        "adapter_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "candidate_inputs_at_inference": False,
        "perturbed_spectrum_training": False,
        "control_gradient_subtraction": False,
        "formal": False,
    }
    return report, {"rank": ranks, "margin": margins, "count": counts}, checkpoint


def main() -> None:
    args = arguments()
    started = time.time()
    roles = (
        *args.embedding_train_folds,
        args.embedding_evaluation_fold,
        args.reserve_fold,
    )
    if (
        len(set(roles)) != len(roles)
        or min(roles) < 0
        or max(roles) >= args.folds
        or args.epochs < 1
        or args.batch_size < 1
        or args.hidden_dim < 1
        or args.learning_rate <= 0
        or args.lambda_preserve < 0
        or not 0 <= args.activation_margin <= 0.1
        or not 0 < args.maximum_control_hardness_gap <= 0.1
        or args.bootstrap_draws < 1000
    ):
        raise ValueError("invalid clean-boundary injection protocol")
    if args.output.exists() and not args.preflight_only:
        raise FileExistsError(args.output)
    required = (
        args.rules_dir / "report.json",
        args.manifest,
        args.data,
        args.token_dir / "report.json",
        args.token_dir / "rows.npy",
        args.token_dir / "official_embeddings_f32.npy",
        args.token_dir / "mz_f32.npy",
        args.token_dir / "intensity_f32.npy",
    )
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(f"clean-boundary injection inputs missing: {missing}")
    rules_path, rule_report, rules = resolve_rule_bundle(args.rules_dir)
    with np.load(args.manifest) as source:
        body = {key: source[key] for key in source.files}
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(
        args.token_dir / "official_embeddings_f32.npy", mmap_mode="r"
    )
    if official.shape != (len(rows), 1024):
        raise RuntimeError("official embedding cache has unexpected shape")
    cohort = build_cohort(args, rules, body, rows, official)
    fold = cohort["formula_fold"].astype(np.int16)
    hardness_gap = np.abs(
        cohort["official_chemistry_negative_score"]
        - cohort["official_control_negative_score"]
    )
    caliper_eligible = hardness_gap <= args.maximum_control_hardness_gap
    train_cohort = np.flatnonzero(
        np.isin(fold, args.embedding_train_folds) & caliper_eligible
    )
    evaluation = np.flatnonzero(
        (fold == args.embedding_evaluation_fold) & caliper_eligible
    )
    reserve = np.flatnonzero((fold == args.reserve_fold) & caliper_eligible)
    if len(train_cohort) < 12 or len(evaluation) < 5:
        raise RuntimeError(
            "insufficient switchable boundaries before active-margin gating: "
            f"train={len(train_cohort)} eval={len(evaluation)}"
        )
    chemistry_margin = (
        cohort["official_positive_score"]
        - cohort["official_chemistry_negative_score"]
    )
    train = train_cohort[
        chemistry_margin[train_cohort] <= args.activation_margin
    ]
    active_evaluation = evaluation[
        chemistry_margin[evaluation] <= args.activation_margin
    ]
    active_transfer_authorized = bool(
        len(train) >= 12
        and len(np.unique(cohort["formula"][train])) >= 10
        and len(active_evaluation) >= 5
        and len(np.unique(cohort["formula"][active_evaluation])) >= 5
    )

    def boundary_counts(indices: np.ndarray) -> dict[str, int]:
        values = chemistry_margin[indices]
        return {
            "errors_margin_le_0": int(np.sum(values <= 0.0)),
            "within_0.02": int(np.sum(values <= 0.02)),
            "within_0.05": int(np.sum(values <= 0.05)),
            "within_0.10": int(np.sum(values <= 0.10)),
        }

    preflight = {
        "status": (
            "CHEMAWARE_RULE_ACTIVE_BOUNDARY_PREFLIGHT_PASS"
            if active_transfer_authorized
            else "CHEMAWARE_RULE_ACTIVE_BOUNDARY_ACTION_COVERAGE_FAIL"
        ),
        "rules": [item["rule_id"] for item in rules],
        "raw_switchable_triplets": int(len(fold)),
        "triplets": int(np.sum(caliper_eligible)),
        "train_triplets_before_activation": int(len(train_cohort)),
        "train_triplets": int(len(train)),
        "embedding_evaluation_triplets": int(len(evaluation)),
        "active_embedding_evaluation_triplets": int(len(active_evaluation)),
        "reserve_triplets_untouched": int(len(reserve)),
        "train_formulas": int(len(np.unique(cohort["formula"][train]))),
        "embedding_evaluation_formulas": int(
            len(np.unique(cohort["formula"][evaluation]))
        ),
        "maximum_control_hardness_gap": args.maximum_control_hardness_gap,
        "activation_margin": args.activation_margin,
        "active_boundary_transfer_authorized": active_transfer_authorized,
        "active_boundary_gate": {
            "minimum_train_triplets": 12,
            "minimum_train_formulas": 10,
            "minimum_evaluation_triplets": 5,
            "minimum_evaluation_formulas": 5,
            "nonactive_corrective_weight": 0.0,
        },
        "official_chemistry_boundary_audit": {
            "definition": "official_positive_score_minus_official_chemistry_negative_score",
            "training_before_activation": boundary_counts(train_cohort),
            "training_after_activation": boundary_counts(train),
            "embedding_evaluation": boundary_counts(evaluation),
            "quantiles": {
                str(quantile): float(
                    np.quantile(chemistry_margin[train_cohort], quantile)
                )
                for quantile in (0.1, 0.25, 0.5, 0.75, 0.9)
            },
        },
        "mean_selected_control_hardness_gap": float(
            np.mean(hardness_gap[caliper_eligible])
        ),
        "control_hardness_gap_audit": {
            "quantiles": {
                str(quantile): float(
                    np.quantile(
                        hardness_gap,
                        quantile,
                    )
                )
                for quantile in (0.25, 0.5, 0.75, 0.9, 0.95)
            },
            "counts_within_caliper": {
                str(caliper): int(
                    np.sum(
                        hardness_gap <= caliper
                    )
                )
                for caliper in (0.025, 0.05, 0.1)
            },
            "embedding_evaluation_counts_within_caliper": {
                str(caliper): int(
                    np.sum(
                        (fold == args.embedding_evaluation_fold)
                        & (hardness_gap <= caliper)
                    )
                )
                for caliper in (0.025, 0.05, 0.1)
            },
        },
        "fold_roles": {
            "action_discovery": [0, 1],
            "action_confirmation": 2,
            "embedding_training": list(args.embedding_train_folds),
            "embedding_evaluation": args.embedding_evaluation_fold,
            "reserve_untouched": args.reserve_fold,
        },
        "contracts": {
            "clean_spectra_only": True,
            "same_query_and_positive_across_arms": True,
            "correct_and_control_negative_references_always_different": True,
            "control_score_hardness_matched": True,
            "controls_trained_as_independent_arm_only": True,
            "control_gradient_subtraction": False,
            "nonactive_corrective_weight_exact_zero": True,
            "candidate_input_at_inference": False,
            "reserve_fold_evaluated": False,
        },
        "provenance": {
            "manifest_sha256": sha256_file(args.manifest),
            "rules_report_sha256": sha256_file(args.rules_dir / "report.json"),
            "rules_file": rules_path.name,
            "rules_sha256": sha256_file(rules_path),
            "official_embedding_report_sha256": sha256_file(
                args.token_dir / "report.json"
            ),
        },
    }
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return
    if not active_transfer_authorized:
        raise RuntimeError(
            "active-boundary action coverage gate failed; training is forbidden because "
            f"train={len(train)}/{len(np.unique(cohort['formula'][train]))} formulas, "
            f"eval={len(active_evaluation)}/"
            f"{len(np.unique(cohort['formula'][active_evaluation]))} formulas"
        )

    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "cohort.npz", **cohort)
    row_position = {int(row): index for index, row in enumerate(rows)}
    eval_queries = cohort["query_index"][evaluation].astype(np.int64)
    official_rank, official_margin, candidate_count = evaluate_full_candidates(
        body, eval_queries, official, row_position
    )
    official_metrics = retrieval_metrics(official_rank, candidate_count)
    arm_results: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    per_seed: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for seed in args.seeds:
        per_seed[int(seed)] = {}
        for arm in ARMS:
            arm_report, per_query, checkpoint = train_arm(
                arm,
                int(seed),
                cohort,
                train,
                evaluation,
                body,
                rows,
                official,
                args,
            )
            arm_results[arm].append(arm_report)
            per_seed[int(seed)][arm] = per_query
            arm_dir = args.output / arm / f"seed_{seed}"
            arm_dir.mkdir(parents=True)
            torch.save(checkpoint, arm_dir / "adapter.pt")
            np.savez_compressed(
                arm_dir / "embedding_evaluation_per_query.npz",
                query_index=eval_queries,
                formula=cohort["formula"][evaluation],
                identity=cohort["identity"][evaluation],
                official_rank=official_rank,
                official_margin=official_margin,
                adapted_rank=per_query["rank"],
                adapted_margin=per_query["margin"],
                candidate_count=per_query["count"],
            )
            print(
                f"arm={arm} seed={seed} "
                f"train_pair_reduction={arm_report['training']['correct_pair_loss_reduction']:+.6f} "
                f"eval_recall1={arm_report['embedding_evaluation']['retrieval']['recall1']:.6f}",
                flush=True,
            )

    correct_rank = np.stack(
        [per_seed[int(seed)]["correct_chemistry"]["rank"] for seed in args.seeds]
    )
    control_rank = np.stack(
        [
            per_seed[int(seed)]["predicate_present_control"]["rank"]
            for seed in args.seeds
        ]
    )
    correct_margin = np.mean(
        np.stack(
            [per_seed[int(seed)]["correct_chemistry"]["margin"] for seed in args.seeds]
        ),
        axis=0,
    )
    control_margin = np.mean(
        np.stack(
            [
                per_seed[int(seed)]["predicate_present_control"]["margin"]
                for seed in args.seeds
            ]
        ),
        axis=0,
    )
    formula = cohort["formula"][evaluation]
    comparison = {
        "correct_minus_control_recall1": formula_cluster_interval(
            np.mean(correct_rank == 1, axis=0)
            - np.mean(control_rank == 1, axis=0),
            formula,
            args.fold_seed + 1001,
            args.bootstrap_draws,
        ),
        "correct_minus_control_reciprocal_rank": formula_cluster_interval(
            np.mean(1.0 / correct_rank, axis=0)
            - np.mean(1.0 / control_rank, axis=0),
            formula,
            args.fold_seed + 1002,
            args.bootstrap_draws,
        ),
        "correct_minus_control_margin": formula_cluster_interval(
            correct_margin - control_margin,
            formula,
            args.fold_seed + 1003,
            args.bootstrap_draws,
        ),
    }
    report = {
        "status": "CHEMAWARE_RULE_CLEAN_BOUNDARY_DEVELOPMENT_COMPLETE",
        "formal": False,
        "preflight": preflight,
        "method": {
            "name": "rule_selected_active_clean_boundary_transfer",
            "adapter": "zero_initialized_global_residual_embedding_adapter",
            "objective": "formula_balanced_clean_positive_vs_rule_opposed_negative_margin",
            "chemical_rule_enters_as": "clean_boundary_selection",
            "frozen_active_gate": (
                "official_positive_minus_chemical_negative_margin_le_activation_margin"
            ),
            "nonactive_corrective_weight": 0.0,
            "perturbed_spectra": False,
            "teacher_embedding_distillation": False,
            "candidate_inputs_at_inference": False,
            "control_gradient_subtraction": False,
        },
        "optimization": {
            key: value
            for key, value in vars(args).items()
            if not isinstance(value, Path)
        },
        "official_embedding_evaluation": official_metrics,
        "arms": arm_results,
        "paired_specificity": comparison,
        "reserve_fold_evaluated": False,
        "runtime_seconds": time.time() - started,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "triplets": preflight["triplets"],
                "official_embedding_evaluation": official_metrics,
                "paired_specificity": comparison,
                "output": str(args.output),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
