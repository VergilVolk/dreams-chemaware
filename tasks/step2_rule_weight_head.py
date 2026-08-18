"""Step 2 — per-task rule weight heads (FP / FN) with differentiable soft-weighting.

Learn, for FP and FN each, a per-rule weight head

    w = sigmoid( MLP(rule meta-features) )  in [0, 1]

initialized at 0.5 and regularized toward 0.5 (lambda * ||w - 0.5||^2). The heads
are trained so that the *weighted* P1 error-evidence score beats the *uniform*
baseline on the molecule-disjoint error-detection ROC-AUC, and are gated on:

    G2-1  weights stay non-extreme (concentrated near 0.5);
    G2-2  learned soft-weights >= uniform weights on P1's error-detection ROC-AUC
          (held-out molecules only);
    G2-3  high-|dw| rules match the Step-1 audit signature (main-library CF/NL
          informative; MassBank bulk CF + EE/HR/ISO/NR near-universal).

Design rationale (see WEIGHTED_RULE_NOISE_TRAINING_PLAN_20260816.md §2, §3 Step 2):

    * Labels stay IK14 + strict 10 ppm. Rule weights never redefine a label; they
      only reweight the P1 error-evidence score. The score below is the same
      "wrong-only minus true-only, normalized" evidence that P1 validated, with
      per-rule counts replaced by weighted sums — the differentiable surrogate for
      the final triplet soft-weighting that Step 4 will consume.

    * FP head weights the "wrong_only" side: rules the query shares with the
      wrongly-retrieved candidate but not the true one (look-alike fragments that
      drive false positives). FN head weights the "true_only" side: rules the query
      shares with the true candidate but not the wrong one (identity evidence
      DreaMS missed, i.e. false negatives).

    * The two heads are independent MLPs over the same per-rule feature vector, so
      every rule gets a single global scalar per task (w_fp[r], w_fn[r]), matching
      the plan's "每条规则一个权重 0.62 / 0.49".

This step does NOT touch the embedding backbone and does NOT run the triplet loss.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

import pilot_rule_noise_stress as pilot


ROOT = Path(__file__).resolve().parents[1]

CATEGORIES = ["CF", "EE", "HR", "ISO", "NL", "NR"]
PANEL_CLASSES = [
    "corrective_candidate", "conflict_mining_only", "generic_high_coverage",
    "noise_fragile", "insufficient_or_nonspecific",
]


# --------------------------------------------------------------------------- #
# 命令行
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 2: per-task rule weight heads (FP/FN)")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--audit-csv", type=Path, default=ROOT / "data/validation/e0_failure_audit/e0_query_audit.csv")
    parser.add_argument("--rule-audit", type=Path, default=ROOT / "data/validation/compact_rule_panel/rule_level_audit.csv")
    parser.add_argument("--screen-dir", type=Path, default=ROOT / "data/validation/rule_noise_pilot")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/validation/e0_baseline/e0_manifest.json")
    parser.add_argument("--labels-npz", type=Path, default=ROOT / "data/validation/weighted_rule_step1/spectrum_labels.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/weighted_rule_step2")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--frac", type=float, default=0.7, help="train fraction of molecules (stratified)")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--reg", type=float, default=5.0, help="lambda for ||w-0.5||^2 regularizer")
    parser.add_argument("--delta", type=float, default=0.25, help="hard bound |w-0.5| <= delta via tanh")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--cat-only", action="store_true",
                        help="ablation: drop the 5 continuous meta-features, keep 6 category one-hot + is_main")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--rule-tolerance", type=float, default=0.02)
    parser.add_argument("--max-molecules", type=int, default=0, help="cap molecules for a fast smoke run (0 = all)")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def one_per_molecule(rows: list[dict[str, str]], rng: np.random.Generator, limit: int) -> list[dict[str, str]]:
    shuffled = [rows[int(i)] for i in rng.permutation(len(rows))]
    selected, seen = [], set()
    for row in shuffled:
        ik = row["query_ik14"]
        if ik in seen:
            continue
        seen.add(ik)
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


# --------------------------------------------------------------------------- #
# 度量：P1 的 auc / bootstrap_auc（位级复刻）
# --------------------------------------------------------------------------- #
def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    differences = positives[:, None] - negatives[None, :]
    return float((np.count_nonzero(differences > 0) + 0.5 * np.count_nonzero(differences == 0)) / differences.size)


def bootstrap_auc(labels: np.ndarray, scores: np.ndarray, seed: int, n_bootstrap: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    values = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        sampled = np.concatenate([
            rng.choice(pos, size=len(pos), replace=True),
            rng.choice(neg, size=len(neg), replace=True),
        ])
        values[index] = auc(labels[sampled], scores[sampled])
    point = auc(labels, scores)
    return point, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def uniform_scores(wrong: np.ndarray, true: np.ndarray) -> np.ndarray:
    """P1 direction_score with the 'all rules' panel (uniform weight = 1 per rule)."""
    wrong_only = wrong.sum(axis=1).astype(np.float64)
    true_only = true.sum(axis=1).astype(np.float64)
    return (wrong_only - true_only) / np.maximum(1.0, wrong_only + true_only)


# --------------------------------------------------------------------------- #
# 规则元特征
# --------------------------------------------------------------------------- #
def load_meta_features(
    rule_audit: list[dict[str, str]], labels_npz: np.lib.npyio.NpzFile, cat_only: bool = False
) -> dict[str, Any]:
    """Build the per-rule feature matrix, gated on exact alignment with Step-1 labels.

    With cat_only=True, drop the 5 continuous meta-features and keep only the
    category one-hot (6) + is_main (1) — the category-only ablation.
    """
    audit_sorted = sorted(rule_audit, key=lambda r: int(r["rule_index"]))
    rule_name = [str(n) for n in labels_npz["rule_name"]]
    rule_category = [str(c) for c in labels_npz["rule_category"]]

    # G0-style consistency gate: audit rows (sorted by rule_index) must equal Step-1 order.
    if [r["rule_name"] for r in audit_sorted] != rule_name:
        raise RuntimeError("rule_level_audit.csv rule_name order != spectrum_labels.npz order")
    if [r["category"] for r in audit_sorted] != rule_category:
        raise RuntimeError("rule_level_audit.csv category order != spectrum_labels.npz order")

    n_rules = len(rule_name)
    labels = labels_npz["labels"]  # (n_spectra, n_rules) uint8
    prevalence = labels.mean(axis=0).astype(np.float64)  # per-rule prevalence over 140k pool spectra

    # Continuous features (raw), one per rule.
    def fcol(key: str) -> np.ndarray:
        return np.asarray([float(r[key]) for r in audit_sorted], dtype=np.float64)

    raw = np.stack([
        fcol("net_screening_score"),
        fcol("error_stability"),
        fcol("control_stability"),
        fcol("all_clean_coverage"),
        prevalence,
    ], axis=1)  # (n_rules, 5)

    # Categorical: category one-hot (6) + is_main (source != MassBank) binary (1).
    cat_onehot = np.zeros((n_rules, len(CATEGORIES)), dtype=np.float64)
    for i, c in enumerate(rule_category):
        cat_onehot[i, CATEGORIES.index(c)] = 1.0
    is_main = np.asarray(
        [float(r["source"] != "MassBank record-derived") for r in audit_sorted],
        dtype=np.float64,
    )[:, None]

    if cat_only:
        X = np.concatenate([cat_onehot, is_main], axis=1)  # (n_rules, 7)
        feature_names = [f"cat_{c}" for c in CATEGORIES] + ["is_main"]
    else:
        X = np.concatenate([cat_onehot, is_main, raw], axis=1)  # (n_rules, 12)
        feature_names = (
            [f"cat_{c}" for c in CATEGORIES] + ["is_main"]
            + ["net_screening_score", "error_stability", "control_stability", "all_clean_coverage", "prevalence_140k"]
        )

    return {
        "X": X,
        "feature_names": feature_names,
        "n_rules": n_rules,
        "rule_name": rule_name,
        "rule_category": rule_category,
        "source": [r["source"] for r in audit_sorted],
        "panel_class": [r["panel_class"] for r in audit_sorted],
    }


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score the continuous columns only; one-hot / binary columns are left untouched."""
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    # Only standardize the trailing continuous block (everything after the 6 one-hot + is_main).
    cont_start = len(CATEGORIES) + 1
    Xs = X.copy()
    Xs[:, cont_start:] = (X[:, cont_start:] - mean[:, cont_start:]) / std[:, cont_start:]
    return Xs, mean, std


# --------------------------------------------------------------------------- #
# 权重头
# --------------------------------------------------------------------------- #
class RuleWeightHead(torch.nn.Module):
    """MLP(rule meta-features) -> scalar -> bounded weight, one global weight per rule.

    w = 0.5 + delta * tanh(z). The tanh bound hard-caps |w - 0.5| <= delta, so no
    single rule can go extreme regardless of its gradient. This implements the
    plan's "压住、不许极端" by construction, not by a soft penalty alone.
    """

    def __init__(self, n_features: int, hidden: int, delta: float = 0.25):
        super().__init__()
        self.delta = delta
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_features, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden // 2, 1),
        )
        for module in self.net.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, 0.0, 0.02)
                torch.nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x).squeeze(-1)  # (n_rules,)
        return 0.5 + self.delta * torch.tanh(z)


def weighted_scores(w_fp: torch.Tensor, w_fn: torch.Tensor, wrong: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    wrong_w = wrong @ w_fp
    true_w = true @ w_fn
    return (wrong_w - true_w) / torch.clamp(wrong_w + true_w, min=1.0)


def ranking_logistic_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    err = scores[labels == 1]
    corr = scores[labels == 0]
    if err.numel() == 0 or corr.numel() == 0:
        return torch.zeros((), device=scores.device, requires_grad=True)
    diff = err[:, None] - corr[None, :]  # (E, C)
    return torch.log1p(torch.exp(-diff)).mean()


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
def train_heads(
    X: torch.Tensor,
    wrong_train: torch.Tensor,
    true_train: torch.Tensor,
    labels_train: torch.Tensor,
    n_features: int,
    hidden: int,
    lr: float,
    epochs: int,
    reg: float,
    delta: float,
    seed: int,
) -> tuple[RuleWeightHead, RuleWeightHead, list[dict[str, float]]]:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    head_fp = RuleWeightHead(n_features, hidden, delta)
    head_fn = RuleWeightHead(n_features, hidden, delta)
    params = list(head_fp.parameters()) + list(head_fn.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        w_fp = head_fp(X)
        w_fn = head_fn(X)
        scores = weighted_scores(w_fp, w_fn, wrong_train, true_train)
        rank_loss = ranking_logistic_loss(scores, labels_train)
        reg_loss = reg * ((w_fp - 0.5).pow(2).mean() + (w_fn - 0.5).pow(2).mean())
        loss = rank_loss + reg_loss
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 200 == 0 or epoch == epochs:
            w = torch.cat([w_fp.detach(), w_fn.detach()])
            history.append({
                "epoch": epoch,
                "rank_loss": float(rank_loss.detach()),
                "reg_loss": float(reg_loss.detach()),
                "total_loss": float(loss.detach()),
                "mean_w": float(w.mean()),
                "mean_abs_dw": float((w - 0.5).abs().mean()),
                "max_abs_dw": float((w - 0.5).abs().max()),
                "fraction_saturated": float(((w - 0.5).abs() > 0.9 * delta).float().mean()),
            })
    return head_fp, head_fn, history


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Stage 0: 元特征 + 一致性门 ----
    rule_audit = read_csv(args.rule_audit)
    labels_npz = np.load(args.labels_npz)
    meta = load_meta_features(rule_audit, labels_npz, cat_only=args.cat_only)
    n_rules = meta["n_rules"]
    print(f"[meta] {n_rules} rules, {meta['X'].shape[1]} features, aligned with Step-1 labels", flush=True)
    Xs, feat_mean, feat_std = standardize(meta["X"])

    # ---- Stage 1: 查询选择（复刻 P1 的分子不相交协议）----
    audit = read_csv(args.audit_csv)
    pilot_selected = read_csv(args.screen_dir / "selected_queries.csv")
    manifest = json.loads((args.manifest).read_text(encoding="utf-8"))
    manifest_by_id = {row["spectrum_id"]: row for row in manifest}
    screen_iks = {manifest_by_id[row["spectrum_id"]]["inchikey_14"] for row in pilot_selected}

    all_error_iks = {row["query_ik14"] for row in audit if as_bool(row["is_top1_error"])}
    error_pool = [row for row in audit if as_bool(row["is_top1_error"]) and row["query_ik14"] not in screen_iks]
    control_pool = [
        row for row in audit
        if not as_bool(row["is_top1_error"])
        and row["query_ik14"] not in all_error_iks
        and row["query_ik14"] not in screen_iks
    ]
    rng = np.random.default_rng(args.seed)
    errors = one_per_molecule(error_pool, rng, 10**9)
    controls = one_per_molecule(control_pool, rng, 10**9)
    if args.max_molecules > 0:
        errors = errors[: max(args.max_molecules // 2, 1)]
        controls = controls[: max(args.max_molecules // 2, 1)]
    print(f"[queries] {len(errors)} error molecules + {len(controls)} control molecules "
          f"(screen {len(screen_iks)} excluded)", flush=True)

    # ---- Stage 2: 分子不相交 train/eval 分层划分 ----
    def split_rows(rows: list[dict[str, str]], frac: float, seed: int) -> tuple[list, list]:
        rng2 = np.random.default_rng(seed)
        order = rng2.permutation(len(rows))
        n_train = max(1, int(round(len(rows) * frac)))
        idx = [int(i) for i in order]
        return [rows[i] for i in idx[:n_train]], [rows[i] for i in idx[n_train:]]

    err_train, err_eval = split_rows(errors, args.frac, pilot.stable_seed(args.seed, "err"))
    ctrl_train, ctrl_eval = split_rows(controls, args.frac, pilot.stable_seed(args.seed, "ctrl"))
    train_rows = err_train + ctrl_train
    eval_rows = err_eval + ctrl_eval
    train_iks = {r["query_ik14"] for r in train_rows}
    eval_iks = {r["query_ik14"] for r in eval_rows}
    assert not (train_iks & eval_iks), "train/eval molecule overlap — leak!"
    print(f"[split] train {len(train_rows)} ({len(err_train)} err / {len(ctrl_train)} ctrl), "
          f"eval {len(eval_rows)} ({len(err_eval)} err / {len(ctrl_eval)} ctrl), disjoint", flush=True)

    # ---- Stage 3: 计算所有谱的规则向量 + 每条查询的 wrong_only / true_only ----
    all_rows = train_rows + eval_rows
    wanted_ids = set()
    for row in all_rows:
        wanted_ids.update([row["query_spectrum_id"], row["correct_best_spectrum_id"], row["best_negative_spectrum_id"]])
    h5_rows = pilot.hdf5_index(args.data, wanted_ids)
    RuleEngine = pilot.load_rule_engine_class()
    engine = RuleEngine(tolerance=args.rule_tolerance, use_massbank=True)
    matcher = pilot.FastRuleMatcher(engine, args.rule_tolerance)
    assert len(engine.rules) == n_rules, f"engine has {len(engine.rules)} rules, expected {n_rules}"

    vectors: dict[str, np.ndarray] = {}
    with h5py.File(args.data, "r") as handle:
        for i, (sid, index) in enumerate(h5_rows.items()):
            vectors[sid] = matcher(pilot.preprocess_spectrum(handle, index, None), float(handle["precursor_mz"][index]))
            if (i + 1) % 1000 == 0:
                print(f"[match] {i + 1}/{len(h5_rows)} spectra", flush=True)
    print(f"[match] {len(vectors)} spectra rule-encoded", flush=True)

    def evidence(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        wrong = np.zeros((len(rows), n_rules), dtype=np.uint8)
        true = np.zeros((len(rows), n_rules), dtype=np.uint8)
        labels = np.zeros(len(rows), dtype=np.int64)
        for i, row in enumerate(rows):
            q = vectors[row["query_spectrum_id"]]
            t = vectors[row["correct_best_spectrum_id"]]
            w = vectors[row["best_negative_spectrum_id"]]
            wrong[i] = (q & w & ~t).astype(np.uint8)
            true[i] = (q & t & ~w).astype(np.uint8)
            labels[i] = 1 if as_bool(row["is_top1_error"]) else 0
        return wrong, true, labels

    wrong_train, true_train, labels_train = evidence(train_rows)
    wrong_eval, true_eval, labels_eval = evidence(eval_rows)

    # Uniform baseline (P1 exact) on both splits, for the record and the gate.
    uni_eval_point, uni_eval_lo, uni_eval_hi = bootstrap_auc(
        labels_eval, uniform_scores(wrong_eval, true_eval),
        pilot.stable_seed(args.seed, "uni", "eval"), args.n_bootstrap,
    )
    print(f"[baseline] uniform-weight eval ROC-AUC {uni_eval_point:.4f} "
          f"[{uni_eval_lo:.4f}, {uni_eval_hi:.4f}]", flush=True)

    # ---- Stage 4 + 5: 训练双头 ----
    X_t = torch.from_numpy(Xs.astype(np.float32))
    wrong_train_t = torch.from_numpy(wrong_train.astype(np.float32))
    true_train_t = torch.from_numpy(true_train.astype(np.float32))
    labels_train_t = torch.from_numpy(labels_train.astype(np.int64))

    head_fp, head_fn, history = train_heads(
        X_t, wrong_train_t, true_train_t, labels_train_t,
        meta["X"].shape[1], args.hidden, args.lr, args.epochs, args.reg, args.delta, args.seed,
    )

    with torch.no_grad():
        w_fp = head_fp(X_t).numpy()
        w_fn = head_fn(X_t).numpy()

    # ---- Stage 6: G2 门（eval 冻结）----
    def weighted_np_scores(w_fp: np.ndarray, w_fn: np.ndarray, wrong: np.ndarray, true: np.ndarray) -> np.ndarray:
        wrong_w = wrong.astype(np.float64) @ w_fp
        true_w = true.astype(np.float64) @ w_fn
        return (wrong_w - true_w) / np.maximum(1.0, wrong_w + true_w)

    w_eval_point, w_eval_lo, w_eval_hi = bootstrap_auc(
        labels_eval, weighted_np_scores(w_fp, w_fn, wrong_eval, true_eval),
        pilot.stable_seed(args.seed, "w", "eval"), args.n_bootstrap,
    )

    # G2-1 非极端检查（参数化已硬限 |w-0.5| <= delta；此门检查权重是否整体塌到界上）
    all_w = np.concatenate([w_fp, w_fn])
    in_band_05 = float((np.abs(all_w - 0.5) <= 0.10).mean())
    in_band_10 = float((np.abs(all_w - 0.5) <= 0.20).mean())
    max_abs_dw = float(np.abs(all_w - 0.5).max())
    fraction_saturated = float((np.abs(all_w - 0.5) > 0.9 * args.delta).mean())
    non_extreme = max_abs_dw <= args.delta + 1e-6 and fraction_saturated < 0.05

    # G2-2 加权 >= 全同权
    gate_weighted_beats_uniform = w_eval_point > uni_eval_point

    # G2-3 高 |dw| 规则 vs 审计签名
    importance = np.abs(w_fp - 0.5) + np.abs(w_fn - 0.5)
    top_k = min(100, n_rules)
    top_idx = np.argsort(-importance)[:top_k]
    bot_idx = np.argsort(importance)[:top_k]
    top_cat = Counter(meta["rule_category"][i] for i in top_idx)
    bot_cat = Counter(meta["rule_category"][i] for i in bot_idx)
    top_main = float(np.mean([meta["source"][i] != "MassBank record-derived" for i in top_idx]))
    bot_main = float(np.mean([meta["source"][i] != "MassBank record-derived" for i in bot_idx]))

    summary = {
        "status": "step2_rule_weight_head",
        "gate_consistency": True,
        "n_rules": n_rules,
        "n_features": meta["X"].shape[1],
        "feature_names": meta["feature_names"],
        "queries": {
            "n_error_molecules": len(errors),
            "n_control_molecules": len(controls),
            "n_screen_excluded": len(screen_iks),
            "train": {"n": len(train_rows), "n_error": len(err_train), "n_control": len(ctrl_train)},
            "eval": {"n": len(eval_rows), "n_error": len(err_eval), "n_control": len(ctrl_eval)},
            "train_eval_molecule_disjoint": True,
        },
        "training": {
            "lr": args.lr, "epochs": args.epochs, "reg": args.reg, "hidden": args.hidden,
            "final": history[-1],
        },
        "g2_1_non_extreme": {
            "pass": non_extreme,
            "delta": args.delta,
            "max_abs_dw": max_abs_dw,
            "fraction_saturated": fraction_saturated,
            "fraction_within_0_10": in_band_05,
            "fraction_within_0_20": in_band_10,
            "mean_abs_dw": float(np.abs(all_w - 0.5).mean()),
        },
        "g2_2_weighted_beats_uniform": {
            "pass": gate_weighted_beats_uniform,
            "uniform_roc_auc": uni_eval_point,
            "uniform_roc_auc_95ci": [uni_eval_lo, uni_eval_hi],
            "weighted_roc_auc": w_eval_point,
            "weighted_roc_auc_95ci": [w_eval_lo, w_eval_hi],
        },
        "g2_3_audit_signature": {
            "top100_importance": {
                "category": dict(top_cat),
                "fraction_main_library": top_main,
            },
            "bottom100_importance": {
                "category": dict(bot_cat),
                "fraction_main_library": bot_main,
            },
        },
        "gate_overall_pass": non_extreme and gate_weighted_beats_uniform,
    }

    (args.output_dir / "STEP2_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez(
        args.output_dir / "rule_weights.npz",
        w_fp=w_fp.astype(np.float32),
        w_fn=w_fn.astype(np.float32),
        importance=importance.astype(np.float32),
        rule_name=np.asarray(meta["rule_name"]),
        rule_category=np.asarray(meta["rule_category"]),
        source=np.asarray(meta["source"]),
        panel_class=np.asarray(meta["panel_class"]),
    )
    (args.output_dir / "training_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 顶部权重规则表（解释性）
    top_rows = []
    for rank, i in enumerate(top_idx, 1):
        top_rows.append({
            "rank": rank,
            "rule_index": int(i),
            "rule_name": meta["rule_name"][i],
            "category": meta["rule_category"][i],
            "source": meta["source"][i],
            "panel_class": meta["panel_class"][i],
            "w_fp": float(w_fp[i]),
            "w_fn": float(w_fn[i]),
            "importance": float(importance[i]),
        })
    with (args.output_dir / "top_weighted_rules.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(top_rows[0]))
        writer.writeheader()
        writer.writerows(top_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
