"""Formula-group OOF nonlinear teacher for exact A4 peak actions.

The teacher is training-time only.  It learns three aligned outcomes from the
exact official-DreaMS intervention scan:

* whether an action corrects an official error;
* whether the same action pattern introduces an error on a matched control;
* the continuous change in positive-vs-hard-negative margin.

Every spectrum, dose and peak belonging to one molecular formula stays in one
outer fold.  Query-equal weights prevent peak-rich spectra from dominating.
The script does not fine-tune DreaMS and does not touch sealed P3 identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_A4 = ROOT / "data/validation/g8r_noise_v3_a4_exact_peak_scan"
DEFAULT_OUTPUT = ROOT / "data/validation/g8r_noise_v3_a4_action_teacher"
DEFAULT_S1C = ROOT / "data/validation/g8r_noise_v3_s1c_topk_matrix"
DEFAULT_S2 = ROOT / "data/validation/g8r_noise_v3_s2_sequential"
DEFAULT_S3A = ROOT / "data/validation/g8r_noise_v3_s3a_extended_matrix"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a4-dir", type=Path, default=DEFAULT_A4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--s1c-dir", type=Path, default=DEFAULT_S1C)
    parser.add_argument("--s2-dir", type=Path, default=DEFAULT_S2)
    parser.add_argument("--s3a-dir", type=Path, default=DEFAULT_S3A)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260825, 20260826, 20260827])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--risk-penalty", type=float, default=2.0)
    parser.add_argument("--minimum-new-corrections", type=int, default=80)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--max-queries", type=int, default=0, help="Smoke-only deterministic prefix")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def previous_recoverable(directory: Path) -> set[int]:
    path = directory / "paired_interventions.csv.gz"
    if not path.is_file():
        return set()
    frame = pd.read_csv(path, usecols=lambda name: name in {
        "query_index", "baseline_rank", "target_rank",
    })
    required = {"query_index", "baseline_rank", "target_rank"}
    if not required.issubset(frame.columns):
        return set()
    return set(map(int, frame.loc[
        (frame["baseline_rank"] > 1) & (frame["target_rank"] == 1), "query_index"
    ]))


@dataclass
class VariantTable:
    x: np.ndarray
    feature_names: list[str]
    query_position: np.ndarray
    query_index: np.ndarray
    formula: np.ndarray
    scan_kind: np.ndarray
    baseline_rank: np.ndarray
    baseline_margin: np.ndarray
    result_rank: np.ndarray
    result_margin: np.ndarray
    corrected: np.ndarray
    introduced: np.ndarray
    margin_change: np.ndarray
    action_index: np.ndarray
    token: np.ndarray
    role: np.ndarray
    dose: np.ndarray
    query_weight: np.ndarray
    candidate_molecules: np.ndarray
    peak_count: np.ndarray
    has_near: np.ndarray
    adversary_grade: np.ndarray
    score_error_family: np.ndarray
    positive_deficit: np.ndarray
    negative_excess: np.ndarray


def build_variant_table(a4_dir: Path, max_queries: int) -> VariantTable:
    query_path = a4_dir / "scan_queries.csv.gz"
    h5_path = a4_dir / "exact_peak_scan.h5"
    if not query_path.is_file() or not h5_path.is_file():
        raise FileNotFoundError("A4 scan artifacts are incomplete")
    queries = pd.read_csv(query_path).sort_values("scan_position").reset_index(drop=True)
    if queries["scan_position"].tolist() != list(range(len(queries))):
        raise RuntimeError("A4 scan positions are not contiguous")
    if max_queries:
        # A4 stores all official errors before matched safety controls.  A
        # prefix smoke would therefore contain no harm examples.  Keep a
        # deterministic, approximately balanced sample from both arms.
        error_positions = queries.loc[
            queries["scan_kind"].eq("official_error"), "scan_position"
        ].to_numpy(np.int64)
        control_positions = queries.loc[
            queries["scan_kind"].eq("safety_control"), "scan_position"
        ].to_numpy(np.int64)
        n_error = min((max_queries + 1) // 2, len(error_positions))
        n_control = min(max_queries - n_error, len(control_positions))
        if n_error == 0 or n_control == 0:
            raise RuntimeError("smoke sample requires both errors and safety controls")
        keep_query = np.sort(np.concatenate((
            error_positions[:n_error], control_positions[:n_control],
        )))
        queries = queries.loc[queries["scan_position"].isin(keep_query)].copy()
    else:
        keep_query = np.arange(len(queries), dtype=np.int64)

    with h5py.File(h5_path, "r") as handle:
        doses = np.asarray(json.loads(handle.attrs["attenuations_json"]), dtype=np.float32)
        action_query_all = handle["action_query"][:].astype(np.int64)
        keep_action = np.isin(action_query_all, keep_query) & handle["action_policy_eligible"][:]
        action_ids = np.flatnonzero(keep_action).astype(np.int64)
        if not len(action_ids):
            raise RuntimeError("no policy-eligible A4 actions")
        action_query = action_query_all[action_ids]
        action_token = handle["action_token"][:][action_ids].astype(np.int16)
        action_role = handle["action_role"][:][action_ids].astype(np.int8)
        action_mz = handle["action_mz"][:][action_ids].astype(np.float32)
        action_intensity = handle["action_intensity"][:][action_ids].astype(np.float32)
        action_gradient = handle["action_gradient"][:][action_ids].astype(np.float32)
        action_rank = handle["action_gradient_rank"][:][action_ids].astype(np.int32)
        n_doses = len(doses)
        result_rank_all = handle["result_rank"][:].reshape(-1, n_doses)
        result_margin_all = handle["result_margin"][:].reshape(-1, n_doses)
        result_rank = result_rank_all[action_ids].reshape(-1).astype(np.int16)
        result_margin = result_margin_all[action_ids].reshape(-1).astype(np.float32)

    variant_action = np.repeat(action_ids, len(doses))
    variant_local_action = np.repeat(np.arange(len(action_ids), dtype=np.int64), len(doses))
    variant_query = np.repeat(action_query, len(doses))
    dose = np.tile(doses, len(action_ids))
    role = np.repeat(action_role, len(doses))
    token = np.repeat(action_token, len(doses))
    mz = np.repeat(action_mz, len(doses))
    intensity = np.repeat(action_intensity, len(doses))
    gradient = np.repeat(action_gradient, len(doses))
    gradient_rank = np.repeat(action_rank, len(doses))
    predicted_gain = -dose * intensity * gradient

    q = queries.set_index("scan_position").loc[variant_query]
    query_index = q["query_index"].to_numpy(np.int64)
    formula = q["query_formula"].fillna("UNKNOWN").astype(str).to_numpy()
    scan_kind = q["scan_kind"].astype(str).to_numpy()
    baseline_rank = q["baseline_rank"].to_numpy(np.int16)
    baseline_margin = q["baseline_margin"].to_numpy(np.float32)
    margin_change = result_margin - baseline_margin
    corrected = ((scan_kind == "official_error") & (result_rank == 1)).astype(np.float32)
    introduced = ((scan_kind == "safety_control") & (result_rank > 1)).astype(np.float32)

    role_confounder = (role == 1).astype(np.float32)
    role_shared = (role == 2).astype(np.float32)
    role_unmatched = (role == 3).astype(np.float32)
    candidate_molecules = q["candidate_molecules"].to_numpy(np.int32)
    peak_count = q["peak_count"].to_numpy(np.int32)
    has_near = q["has_near"].astype(bool).to_numpy()
    grade = q["baseline_adversarial_mces_grade"].fillna(-1).to_numpy(np.int8)
    # No score_error_family, positive_deficit, rules_favor_* or scan_kind is an
    # input.  Those fields encode the true identity/error arm and would leak.
    columns = [
        dose,
        dose * dose,
        mz / 1000.0,
        np.log1p(np.clip(mz, 0, None)) / 8.0,
        intensity,
        np.log(np.clip(intensity, 1e-7, None)),
        gradient,
        np.abs(gradient),
        predicted_gain,
        np.log1p(np.clip(gradient_rank, 0, 100)),
        (gradient_rank > 0).astype(np.float32),
        role_confounder,
        role_shared,
        role_unmatched,
        np.log1p(candidate_molecules.astype(np.float32)),
        np.log1p(peak_count.astype(np.float32)),
        has_near.astype(np.float32),
        (grade == 0).astype(np.float32),
        (grade == 1).astype(np.float32),
        (grade == 2).astype(np.float32),
        (grade < 0).astype(np.float32),
        dose * role_confounder,
        dose * role_shared,
        dose * role_unmatched,
    ]
    feature_names = [
        "attenuation", "attenuation_squared", "mz_scaled", "log_mz_scaled",
        "intensity", "log_intensity", "gradient", "abs_gradient",
        "dose_specific_predicted_gain", "log_gradient_rank", "positive_gradient",
        "role_confounder", "role_shared", "role_unmatched",
        "log_candidate_molecules", "log_peak_count", "has_near",
        "adversary_near", "adversary_mid", "adversary_far", "adversary_unknown",
        "dose_x_confounder", "dose_x_shared", "dose_x_unmatched",
    ]
    x = np.column_stack(columns).astype(np.float32)
    if not np.all(np.isfinite(x)):
        raise RuntimeError("non-finite action features")
    count = np.bincount(variant_query, minlength=len(queries)).astype(np.float32)
    query_weight = 1.0 / count[variant_query]
    query_weight *= len(query_weight) / query_weight.sum()
    return VariantTable(
        x=x, feature_names=feature_names, query_position=variant_query,
        query_index=query_index, formula=formula, scan_kind=scan_kind,
        baseline_rank=baseline_rank, baseline_margin=baseline_margin,
        result_rank=result_rank, result_margin=result_margin,
        corrected=corrected, introduced=introduced, margin_change=margin_change,
        action_index=variant_action, token=token, role=role, dose=dose,
        query_weight=query_weight.astype(np.float32),
        candidate_molecules=candidate_molecules, peak_count=peak_count,
        has_near=has_near, adversary_grade=grade,
        score_error_family=q.get(
            "score_error_family", pd.Series("unknown", index=q.index)
        ).fillna("unknown").astype(str).to_numpy(),
        positive_deficit=q.get(
            "positive_deficit", pd.Series(False, index=q.index)
        ).fillna(False).astype(bool).to_numpy(),
        negative_excess=q.get(
            "negative_excess", pd.Series(False, index=q.index)
        ).fillna(False).astype(bool).to_numpy(),
    )


class ActionTeacher(torch.nn.Module):
    def __init__(self, inputs: int, hidden: int):
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(inputs, hidden), torch.nn.SiLU(),
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden // 2), torch.nn.SiLU(),
        )
        self.benefit = torch.nn.Linear(hidden // 2, 1)
        self.harm = torch.nn.Linear(hidden // 2, 1)
        self.delta = torch.nn.Linear(hidden // 2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.benefit(h).squeeze(1), self.harm(h).squeeze(1), self.delta(h).squeeze(1)


def weighted_mean_std(x: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weight = weights.astype(np.float64)
    weight /= weight.sum()
    mean = np.sum(x.astype(np.float64) * weight[:, None], axis=0)
    var = np.sum((x.astype(np.float64) - mean) ** 2 * weight[:, None], axis=0)
    return mean.astype(np.float32), np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)


def weighted_pos_weight(labels: np.ndarray, weights: np.ndarray) -> float:
    positive = float(weights[labels > 0.5].sum())
    negative = float(weights[labels <= 0.5].sum())
    return float(np.clip(negative / max(positive, 1e-8), 1.0, 100.0))


def train_fold_seed(
    table: VariantTable, train: np.ndarray, test: np.ndarray, args: argparse.Namespace, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    mean, std = weighted_mean_std(table.x[train], table.query_weight[train])
    x_train = torch.as_tensor((table.x[train] - mean) / std, device=device)
    x_test = torch.as_tensor((table.x[test] - mean) / std, device=device)
    weight = torch.as_tensor(table.query_weight[train], device=device)
    benefit = torch.as_tensor(table.corrected[train], device=device)
    harm = torch.as_tensor(table.introduced[train], device=device)
    delta_scale = float(max(np.quantile(np.abs(table.margin_change[train]), 0.90), 1e-3))
    delta = torch.as_tensor(table.margin_change[train] / delta_scale, device=device)
    error_mask = torch.as_tensor(table.scan_kind[train] == "official_error", device=device)
    control_mask = torch.as_tensor(table.scan_kind[train] == "safety_control", device=device)
    benefit_pw = weighted_pos_weight(table.corrected[train][error_mask.cpu().numpy()],
                                     table.query_weight[train][error_mask.cpu().numpy()])
    harm_pw = weighted_pos_weight(table.introduced[train][control_mask.cpu().numpy()],
                                  table.query_weight[train][control_mask.cpu().numpy()])
    model = ActionTeacher(table.x.shape[1], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    bce_benefit = torch.nn.BCEWithLogitsLoss(
        reduction="none", pos_weight=torch.tensor(benefit_pw, device=device),
    )
    bce_harm = torch.nn.BCEWithLogitsLoss(
        reduction="none", pos_weight=torch.tensor(harm_pw, device=device),
    )
    n = len(train)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model.train()
    final_loss = math.nan
    for epoch in range(args.epochs):
        order = torch.randperm(n, generator=generator)
        epoch_loss = 0.0
        epoch_weight = 0
        for left in range(0, n, args.batch_size):
            idx = order[left:left + args.batch_size].to(device)
            logit_b, logit_h, predicted_delta = model(x_train[idx])
            local_weight = weight[idx]
            local_error = error_mask[idx]
            local_control = control_mask[idx]
            losses = []
            if bool(local_error.any()):
                value = bce_benefit(logit_b[local_error], benefit[idx][local_error])
                losses.append((value * local_weight[local_error]).sum() /
                              local_weight[local_error].sum().clamp_min(1e-8))
            if bool(local_control.any()):
                value = bce_harm(logit_h[local_control], harm[idx][local_control])
                losses.append((value * local_weight[local_control]).sum() /
                              local_weight[local_control].sum().clamp_min(1e-8))
            regression = torch.nn.functional.smooth_l1_loss(
                predicted_delta, delta[idx], reduction="none", beta=0.25,
            )
            losses.append(0.50 * (regression * local_weight).sum() /
                          local_weight.sum().clamp_min(1e-8))
            loss = torch.stack(losses).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(idx)
            epoch_weight += len(idx)
        final_loss = epoch_loss / max(epoch_weight, 1)
    model.eval()
    with torch.inference_mode():
        benefit_logit, harm_logit, predicted_delta = model(x_test)
        # Undo the prior shift induced by positive-class weighting.
        p_benefit = torch.sigmoid(benefit_logit - math.log(benefit_pw)).cpu().numpy()
        p_harm = torch.sigmoid(harm_logit - math.log(harm_pw)).cpu().numpy()
        predicted_delta = (predicted_delta * delta_scale).cpu().numpy()
    return p_benefit, p_harm, predicted_delta, {
        "seed": seed, "final_loss": final_loss, "benefit_pos_weight": benefit_pw,
        "harm_pos_weight": harm_pw, "delta_scale": delta_scale,
    }


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    labels = np.asarray(labels, dtype=np.int8)
    if not len(labels):
        raise RuntimeError("action discrimination received an empty outcome arm")
    prevalence = float(labels.mean())
    if len(np.unique(labels)) < 2:
        return {"prevalence": prevalence, "roc_auc": None, "average_precision": None}
    return {
        "prevalence": prevalence,
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def select_query_actions(table: VariantTable, utility: np.ndarray) -> pd.DataFrame:
    records = []
    unique_positions = np.unique(table.query_position)
    for position in unique_positions:
        indices = np.flatnonzero(table.query_position == position)
        order = np.lexsort((
            table.dose[indices], table.token[indices], -table.margin_change[indices],
            -utility[indices],
        ))
        chosen = int(indices[order[0]])
        records.append({
            "scan_position": int(position),
            "query_index": int(table.query_index[chosen]),
            "query_formula": str(table.formula[chosen]),
            "scan_kind": str(table.scan_kind[chosen]),
            "baseline_rank": int(table.baseline_rank[chosen]),
            "baseline_margin": float(table.baseline_margin[chosen]),
            "action_index": int(table.action_index[chosen]),
            "token": int(table.token[chosen]),
            "role": int(table.role[chosen]),
            "attenuation": float(table.dose[chosen]),
            "predicted_utility": float(utility[chosen]),
            "exact_margin_change": float(table.margin_change[chosen]),
            "result_rank": int(table.result_rank[chosen]),
            "corrected_if_applied": bool(table.corrected[chosen]),
            "introduced_if_applied": bool(table.introduced[chosen]),
            "candidate_molecules": int(table.candidate_molecules[chosen]),
            "peak_count": int(table.peak_count[chosen]),
            "has_near": bool(table.has_near[chosen]),
            "adversary_grade": int(table.adversary_grade[chosen]),
            "score_error_family": str(table.score_error_family[chosen]),
            "positive_deficit": bool(table.positive_deficit[chosen]),
            "negative_excess": bool(table.negative_excess[chosen]),
        })
    return pd.DataFrame(records).sort_values("scan_position").reset_index(drop=True)


def cluster_bootstrap(
    frame: pd.DataFrame, contribution: np.ndarray, resamples: int, seed: int,
) -> dict[str, float]:
    work = pd.DataFrame({
        "formula": frame["query_formula"].astype(str), "value": contribution.astype(float),
    })
    grouped = work.groupby("formula", sort=True)["value"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sample = rng.integers(0, len(grouped), len(grouped))
        values[index] = sums[sample].sum() / counts[sample].sum()
    return {
        "mean": float(contribution.mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
    }


def evaluate_policy(
    selected: pd.DataFrame, applied: np.ndarray, risk_penalty: float,
    previous: set[int], bootstrap_resamples: int,
) -> dict:
    applied = np.asarray(applied, dtype=bool)
    if len(applied) != len(selected):
        raise ValueError("policy mask is not query-aligned")
    corrected = applied & selected["corrected_if_applied"].to_numpy(bool)
    introduced = applied & selected["introduced_if_applied"].to_numpy(bool)
    contribution = corrected.astype(float) - risk_penalty * introduced.astype(float)
    new_corrections = set(map(int, selected.loc[corrected, "query_index"])) - previous
    return {
        "queries": int(len(selected)),
        "interventions": int(applied.sum()),
        "intervention_rate": float(applied.mean()),
        "corrected": int(corrected.sum()),
        "introduced": int(introduced.sum()),
        "risk_weighted_net": float(corrected.sum() - risk_penalty * introduced.sum()),
        "new_corrections_beyond_s1c_s2_s3a": int(len(new_corrections)),
        "formula_cluster_risk_weighted_net_per_query": cluster_bootstrap(
            selected, contribution, bootstrap_resamples, 20260825,
        ),
    }


def coverage_mask(selected: pd.DataFrame, coverage: float) -> np.ndarray:
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    count = max(1, int(math.ceil(coverage * len(selected))))
    order = np.lexsort((
        selected["scan_position"].to_numpy(np.int64),
        -selected["predicted_utility"].to_numpy(float),
    ))
    mask = np.zeros(len(selected), dtype=bool)
    mask[order[:count]] = True
    return mask


def profile_introduced(selected: pd.DataFrame, applied: np.ndarray) -> dict:
    applied = np.asarray(applied, dtype=bool)
    introduced = selected.loc[applied & selected["introduced_if_applied"]].copy()
    protected = selected.loc[applied & ~selected["introduced_if_applied"] &
                             selected["scan_kind"].eq("safety_control")].copy()
    def distribution(frame: pd.DataFrame, column: str) -> dict[str, int]:
        return {str(key): int(value) for key, value in frame[column].value_counts().items()}
    corrected = selected.loc[applied & selected["corrected_if_applied"]].copy()
    def numeric_summary(frame: pd.DataFrame, column: str) -> dict[str, float | None]:
        return {
            "median": float(frame[column].median()) if len(frame) else None,
            "p10": float(frame[column].quantile(0.10)) if len(frame) else None,
            "p90": float(frame[column].quantile(0.90)) if len(frame) else None,
        }
    return {
        "introduced_queries": int(len(introduced)),
        "protected_intervened_controls": int(len(protected)),
        "introduced_role": distribution(introduced, "role"),
        "introduced_attenuation": distribution(introduced, "attenuation"),
        "protected_role": distribution(protected, "role"),
        "protected_attenuation": distribution(protected, "attenuation"),
        "introduced_adversary_grade": distribution(introduced, "adversary_grade"),
        "protected_adversary_grade": distribution(protected, "adversary_grade"),
        "introduced_has_near_fraction": float(introduced["has_near"].mean()) if len(introduced) else None,
        "protected_has_near_fraction": float(protected["has_near"].mean()) if len(protected) else None,
        "introduced_baseline_margin": numeric_summary(introduced, "baseline_margin"),
        "protected_baseline_margin": numeric_summary(protected, "baseline_margin"),
        "introduced_candidate_molecules": numeric_summary(introduced, "candidate_molecules"),
        "protected_candidate_molecules": numeric_summary(protected, "candidate_molecules"),
        "introduced_peak_count": numeric_summary(introduced, "peak_count"),
        "protected_peak_count": numeric_summary(protected, "peak_count"),
        "corrected_error_family": distribution(corrected, "score_error_family"),
        "corrected_positive_deficit": int(corrected["positive_deficit"].sum()),
        "corrected_negative_excess": int(corrected["negative_excess"].sum()),
    }


def main() -> None:
    args = parse_args()
    formal = args.max_queries == 0
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.folds < 2 or args.epochs < 1 or not args.seeds:
        raise ValueError("invalid cross-validation/training parameters")
    if args.risk_penalty <= 0:
        raise ValueError("risk penalty must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    source_queries = pd.read_csv(args.a4_dir / "scan_queries.csv.gz")
    source_errors = int(source_queries["scan_kind"].eq("official_error").sum())
    source_controls = int(source_queries["scan_kind"].eq("safety_control").sum())
    if formal and (len(source_queries), source_errors, source_controls) != (4998, 1805, 3193):
        raise RuntimeError(
            "formal A4 source-query cardinality mismatch: "
            f"all={len(source_queries)}, errors={source_errors}, controls={source_controls}"
        )
    table = build_variant_table(args.a4_dir, args.max_queries)
    eligible_error_queries = len(np.unique(
        table.query_index[table.scan_kind == "official_error"]
    ))
    eligible_control_queries = len(np.unique(
        table.query_index[table.scan_kind == "safety_control"]
    ))
    zero_action_errors = source_errors - eligible_error_queries if formal else None
    zero_action_controls = source_controls - eligible_control_queries if formal else None
    if formal and (eligible_error_queries, eligible_control_queries) != (1784, 3132):
        raise RuntimeError(
            "formal policy-eligible query cardinality mismatch: "
            f"errors={eligible_error_queries}, controls={eligible_control_queries}"
        )

    query_meta = pd.DataFrame({
        "position": table.query_position, "formula": table.formula,
    }).drop_duplicates("position").sort_values("position")
    query_fold = np.full(int(query_meta["position"].max()) + 1, -1, dtype=np.int8)
    splitter = GroupKFold(args.folds)
    positions = query_meta["position"].to_numpy(np.int64)
    groups = query_meta["formula"].to_numpy()
    for fold, (_, test_local) in enumerate(splitter.split(positions, groups=groups)):
        query_fold[positions[test_local]] = fold
    variant_fold = query_fold[table.query_position]
    if np.any(variant_fold < 0):
        raise RuntimeError("unassigned formula fold")

    oof_benefit = np.zeros(len(table.x), dtype=np.float32)
    oof_harm = np.zeros(len(table.x), dtype=np.float32)
    oof_delta = np.zeros(len(table.x), dtype=np.float32)
    training_log = []
    for fold in range(args.folds):
        test = np.flatnonzero(variant_fold == fold)
        train = np.flatnonzero(variant_fold != fold)
        formula_overlap = set(table.formula[train]) & set(table.formula[test])
        if formula_overlap:
            raise RuntimeError(f"formula leakage in fold {fold}: {len(formula_overlap)}")
        fold_benefit = np.zeros(len(test), dtype=np.float64)
        fold_harm = np.zeros(len(test), dtype=np.float64)
        fold_delta = np.zeros(len(test), dtype=np.float64)
        for seed in args.seeds:
            benefit, harm, delta, log = train_fold_seed(table, train, test, args, seed)
            fold_benefit += benefit / len(args.seeds)
            fold_harm += harm / len(args.seeds)
            fold_delta += delta / len(args.seeds)
            training_log.append({"fold": fold, **log})
        oof_benefit[test] = fold_benefit.astype(np.float32)
        oof_harm[test] = fold_harm.astype(np.float32)
        oof_delta[test] = fold_delta.astype(np.float32)
        print(f"[A4 teacher] fold {fold + 1}/{args.folds}; variants={len(test):,}", flush=True)

    error = table.scan_kind == "official_error"
    control = table.scan_kind == "safety_control"
    benefit_metrics = safe_auc(table.corrected[error], oof_benefit[error])
    harm_metrics = safe_auc(table.introduced[control], oof_harm[control])
    delta_spearman = float(pd.Series(oof_delta).corr(pd.Series(table.margin_change), method="spearman"))
    utility = oof_benefit - args.risk_penalty * oof_harm
    selected = select_query_actions(table, utility)
    previous = set().union(
        previous_recoverable(args.s1c_dir), previous_recoverable(args.s2_dir),
        previous_recoverable(args.s3a_dir),
    )
    coverage_curve = []
    masks: dict[str, np.ndarray] = {}
    for coverage in (0.05, 0.10, 0.20, 0.40):
        mask = coverage_mask(selected, coverage)
        key = f"{coverage:.2f}"
        masks[key] = mask
        metrics = evaluate_policy(
            selected, mask, args.risk_penalty, previous, args.bootstrap_resamples,
        )
        coverage_curve.append({"requested_coverage": coverage, **metrics})
    safe = [row for row in coverage_curve if (
        row["formula_cluster_risk_weighted_net_per_query"]["ci_low"] > 0
        and row["introduced"] <= 0.5 * max(row["corrected"], 1)
    )]
    if safe:
        policy = max(safe, key=lambda row: (
            row["risk_weighted_net"], -row["requested_coverage"],
        ))
        chosen_key = f"{policy['requested_coverage']:.2f}"
        chosen_mask = masks[chosen_key]
    else:
        # No unsafe threshold relaxation: retain the smallest preregistered
        # coverage as a diagnostic and force the formal gate to fail.
        policy = coverage_curve[0]
        chosen_key = "0.05"
        chosen_mask = masks[chosen_key]
    introduced_profile = profile_introduced(selected, chosen_mask)
    ci = policy["formula_cluster_risk_weighted_net_per_query"]
    benefit_ok = (
        benefit_metrics["average_precision"] is not None
        and benefit_metrics["average_precision"] >= 2 * benefit_metrics["prevalence"]
    )
    harm_ok = (
        harm_metrics["average_precision"] is not None
        and harm_metrics["average_precision"] >= 2 * harm_metrics["prevalence"]
    )
    gates = {
        "benefit_auprc_at_least_twice_prevalence": bool(benefit_ok),
        "harm_auprc_at_least_twice_prevalence": bool(harm_ok),
        "risk_weighted_formula_ci_positive": bool(ci["ci_low"] > 0),
        "introduced_no_more_than_half_corrected": bool(
            policy["introduced"] <= 0.5 * max(policy["corrected"], 1)
        ),
        "new_corrections_ge_minimum": bool(
            policy["new_corrections_beyond_s1c_s2_s3a"] >= args.minimum_new_corrections
        ),
    }
    decision = {
        "status": "noise_v3_a4_nonlinear_action_teacher_complete",
        "formal": formal,
        "integrity": {
            "variants": int(len(table.x)),
            "actions": int(len(np.unique(table.action_index))),
            "source_scan_queries": int(len(source_queries)) if formal else None,
            "source_errors": source_errors if formal else None,
            "source_controls": source_controls if formal else None,
            "policy_eligible_queries": int(len(np.unique(table.query_position))),
            "policy_eligible_error_queries": int(eligible_error_queries),
            "policy_eligible_control_queries": int(eligible_control_queries),
            "zero_policy_action_errors": zero_action_errors,
            "zero_policy_action_controls": zero_action_controls,
            "formulas": int(len(np.unique(table.formula))),
            "formula_fold_overlap": 0,
        },
        "feature_names": table.feature_names,
        "leakage_guard": (
            "scan_kind, baseline_rank/margin, score error family, positive/negative deficit and "
            "rules_favor_* are excluded from model inputs. Baseline margin is used only after "
            "selection for outcome reporting."
        ),
        "oof_action_discrimination": {
            "benefit": benefit_metrics,
            "harm": harm_metrics,
            "margin_change_spearman": delta_spearman,
        },
        "oof_policy": policy,
        "oof_risk_coverage_curve": coverage_curve,
        "introduced_error_profile": introduced_profile,
        "gates": gates,
        "pass_to_counterfactual_training": bool(all(gates.values())),
        "decision": (
            "Only a passing formula-OOF teacher may select A4 counterfactual training variants. "
            "A failure routes to the positive-deficit branch and/or feature redesign; it must not "
            "be compensated by wider intervention coverage."
        ),
        "claim_limit": (
            "These are formula-group OOF exact-action selection results on P3-disjoint training "
            "data. They are not fine-tuned DreaMS performance."
        ),
        "training": {
            "folds": args.folds, "seeds": args.seeds, "epochs": args.epochs,
            "batch_size": args.batch_size, "hidden_dim": args.hidden_dim,
            "risk_penalty": args.risk_penalty, "log": training_log,
        },
        "provenance": {
            "scan_queries_sha256": sha256_file(args.a4_dir / "scan_queries.csv.gz"),
            "exact_scan_sha256": sha256_file(args.a4_dir / "exact_peak_scan.h5"),
            "script_sha256": sha256_file(Path(__file__)),
        },
    }

    staging = Path(tempfile.mkdtemp(prefix="a4_action_teacher_", dir=args.output_dir.parent))
    try:
        selected.to_csv(staging / "oof_selected_actions.csv.gz", index=False, compression="gzip")
        np.savez_compressed(
            staging / "oof_action_scores.npz",
            action_index=table.action_index, query_position=table.query_position,
            dose=table.dose, p_benefit=oof_benefit, p_harm=oof_harm,
            predicted_margin_change=oof_delta, utility=utility,
            corrected=table.corrected, introduced=table.introduced,
            exact_margin_change=table.margin_change, fold=variant_fold,
        )
        (staging / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
