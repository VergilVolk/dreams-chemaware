"""P1: risk-controlled selective ranking switch (learned intervention gate).

AUDIT STATUS (2026-08-22): HOLD -- DO NOT USE FOR A FORMAL RUN YET.

Frozen RAW-v1 evidence that this stage must preserve in its reports:
  * g8r_val DEV: Recall@1 0.8081 -> 0.8516, corrected/introduced 44/17.
  * Original-protocol frozen tests: Test-A +0.45 pp (59/50) and Test-B
    +0.73 pp (77/64); both confidence intervals cross zero.  Test-B is a
    challenge view with substantial IK14 overlap with Test-A, not an
    independent replication.
  * RAW-v1 is a second-stage candidate reranker.  It does not update DreaMS
    weights or establish an improved embedding.

Locked roadmap:
  P1 -- learn a risk-controlled intervention policy from fully out-of-fold
        RAW-ranker predictions; do not tune on the consumed g8r_val/Test-A/B.
  P2 -- replace pair classification with query-group/listwise candidate
        ranking, while retaining DreaMS as the protected base score.
  P3 -- pre-register a new IK14/formula/scaffold-audited holdout before any
        P2/P3 model or threshold selection, then evaluate it exactly once.

This rewrite attempts to clear the four blockers from
docs/g8r_reranker_roadmap_20260822.md:
  1. Full-pipeline OOF: the RAW ranker is re-fitted inside each formula fold and
     only held-out queries are scored (no in-sample RAW predictions).
  2. Utility label u(q) in {-1,0,+1}; two classifiers P(correct)=P(u=+1) and
     P(introduce)=P(u=-1); fixed-risk utility U(q)=P(correct)-2*P(introduce).
  3. Fixed risk preference lambda=2 (not swept as both train weight and eval).
  4. Loads the FROZEN RAW-v1 artifact and reproduces 0.8516/44/17/46.45% as the
     archived reference (the P1 model is a separate full-OOF system).
This is a HARD SWITCH between two full rankings (renamed "selective switch"),
not an additive residual model.  g8r_val is DEV only (already viewed).

POST-RUN AUDIT (2026-08-22): the reported DEV result (29 corrected, 7
introduced, 8.06% coverage) is diagnostic only.  Formal sign-off remains on
hold until (i) nested upstream-ranker/gate OOF and candidate-role purging are
audited, (ii) the agreement-only and confidence-only gates are reported at
matched coverage, (iii) score ties/unique utility values are exposed, and
(iv) P1-vs-frozen-RAW-v1 paired uncertainty is reported.  The current
``intervention_precision`` is conditional on a correctness-changing switch;
it is not precision over all gated queries.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from train_g8r_raw_reranker import (  # noqa: E402
    fit_ranker, score_frame, retrieval_query, RAW_FEATURES, fold_for_formula,
)

DEFAULT_CACHE = ROOT / "data/validation/g8r_raw_reranker_cache.npz"
DEFAULT_VAL_CACHE = ROOT / "data/validation/g8r_raw_reranker_cache_val.npz"
DEFAULT_ARTIFACT = ROOT / "data/validation/g8r_raw_reranker_v1_artifact.json"
DEFAULT_OUT = ROOT / "data/validation/g8r_p1_risk_gate.json"

LAMBDA = 2.0  # fixed risk preference: introduced costs 2x a missed correction


def query_weighted_formula_bootstrap(delta, formula, n_boot, seed):
    """Query-weighted mean delta with formula-cluster resampling (CI matches the
    point estimate)."""
    df = pd.DataFrame({"delta": delta, "formula": formula})
    point = float(df["delta"].mean())
    by_f = {f: g["delta"].to_numpy() for f, g in df.groupby("formula")}
    formulas = list(by_f)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(formulas), len(formulas))
        draws[b] = np.concatenate([by_f[formulas[i]] for i in idx]).mean()
    return {"mean": point, "ci_low": float(np.percentile(draws, 2.5)),
            "ci_high": float(np.percentile(draws, 97.5))}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--val-cache", type=Path, default=DEFAULT_VAL_CACHE)
    p.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--hard-k", type=int, default=5)
    p.add_argument("--C", type=float, default=0.01)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--gate-c", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    p.add_argument("--seed", type=int, default=20260823)
    return p.parse_args()


def oof_retrieval(frame, features, hard_k, C, folds, seed):
    """Full-pipeline OOF: per formula fold, fit RAW ranker on non-held and score
    held queries.  Returns (base, oof_reranked) query-level frames."""
    assignments = {f: fold_for_formula(f, folds) for f in frame["formula"].unique()}
    scored_parts = []
    for fold in range(folds):
        held = {f for f, v in assignments.items() if v == fold}
        trn = frame[~frame["formula"].isin(held)]
        hld = frame[frame["formula"].isin(held)]
        if len(trn) == 0 or len(hld) == 0:
            continue
        scaler, model = fit_ranker(trn, features, hard_k, C)
        if scaler is None:
            continue
        scored_parts.append(score_frame(hld, features, scaler, model))
    oof = pd.concat(scored_parts)
    base = retrieval_query(oof, "dreams_similarity")
    reranked = retrieval_query(oof, "score")
    return base, reranked


def gate_features(base, reranked, frame):
    """Deployable features + utility label u in {-1,0,+1}."""
    merged = base[["query_index", "ik14", "formula", "chosen_ik14", "confidence_margin", "top1"]].merge(
        reranked[["query_index", "chosen_ik14", "confidence_margin", "top1"]],
        on="query_index", suffixes=("_b", "_r"))
    counts = frame.groupby("query").size().to_frame("candidate_count").reset_index()
    merged = merged.merge(counts, left_on="query_index", right_on="query", how="left").drop(columns="query")
    merged["agree"] = (merged["chosen_ik14_b"] == merged["chosen_ik14_r"]).astype(float)
    X = merged[["confidence_margin_b", "confidence_margin_r", "agree", "candidate_count"]].to_numpy(float)
    base_wrong = ~merged["top1_b"].astype(bool)
    rk_right = merged["top1_r"].astype(bool)
    u = np.where(base_wrong & rk_right, 1, np.where((~base_wrong) & (~rk_right), -1, 0))
    return X, u, merged


def train_binary(X, y, C, seed):
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(C=C, max_iter=5000, random_state=seed).fit(scaler.transform(X), y)
    return scaler, model


def oof_predict(X, y, fold_assign, gate_c, seed):
    pred = np.zeros(len(y), dtype=float)
    for fold in set(fold_assign):
        trn = fold_assign != fold
        hld = fold_assign == fold
        if not trn.any() or not hld.any():
            continue
        sc, m = train_binary(X[trn], y[trn], gate_c, seed)
        pred[hld] = m.predict_proba(sc.transform(X[hld]))[:, 1]
    return pred


def main() -> None:
    a = parse_args()
    cache = np.load(a.cache, allow_pickle=True)
    tr = pd.DataFrame({k: cache[k] for k in cache.files})
    va_cache = np.load(a.val_cache, allow_pickle=True)
    va = pd.DataFrame({k: va_cache[k] for k in va_cache.files})
    features = ["dreams_similarity"] + RAW_FEATURES

    # 1. full-pipeline OOF RAW predictions on train
    tr_base, tr_reranked = oof_retrieval(tr, features, a.hard_k, a.C, a.folds, a.seed)
    X_tr, u_tr, tr_merged = gate_features(tr_base, tr_reranked, tr)
    print(f"[oof] train queries={len(u_tr)} u=+1:{(u_tr == 1).sum()} u=-1:{(u_tr == -1).sum()} "
          f"u=0:{(u_tr == 0).sum()}", flush=True)

    # 2. train P(correct) and P(introduce), formula-group OOF for C selection
    fold_assign = tr_merged["formula"].map(
        {f: fold_for_formula(f, a.folds) for f in tr_merged["formula"].unique()}).to_numpy()
    y_correct = (u_tr == 1).astype(int)
    y_intro = (u_tr == -1).astype(int)

    best = None
    for gc in a.gate_c:
        p_c = oof_predict(X_tr, y_correct, fold_assign, gc, a.seed)
        p_i = oof_predict(X_tr, y_intro, fold_assign, gc, a.seed)
        U = p_c - LAMBDA * p_i
        for t in np.quantile(U, [0.5, 0.6, 0.7, 0.8, 0.9]):
            intervene = U > t
            final = np.where(intervene, tr_reranked["top1"].to_numpy(), tr_base["top1"].to_numpy())
            corrected = int(((~tr_base["top1"]) & final).sum())
            introduced = int((tr_base["top1"] & (~final)).sum())
            obj = corrected - LAMBDA * introduced
            if best is None or obj > best[0]:
                best = (obj, gc, t, corrected, introduced, float(intervene.mean()))
    _, best_gc, best_t, b_corr, b_intro, b_cov = best
    print(f"[gate OOF] gc={best_gc} t={best_t:.4f} corrected={b_corr} introduced={b_intro} "
          f"coverage={b_cov:.4f}", flush=True)

    # 3. final classifiers on full train; apply to val (DEV descriptive only)
    sc_c, m_c = train_binary(X_tr, y_correct, best_gc, a.seed)
    sc_i, m_i = train_binary(X_tr, y_intro, best_gc, a.seed)

    scaler_rk, rk = fit_ranker(tr, features, a.hard_k, a.C)
    va_base = retrieval_query(va, "dreams_similarity")
    va_reranked = retrieval_query(score_frame(va, features, scaler_rk, rk), "score")
    X_va, u_va, va_merged = gate_features(va_base, va_reranked, va)
    p_c_va = m_c.predict_proba(sc_c.transform(X_va))[:, 1]
    p_i_va = m_i.predict_proba(sc_i.transform(X_va))[:, 1]
    U_va = p_c_va - LAMBDA * p_i_va
    intervene = U_va > best_t
    final_top1 = np.where(intervene, va_reranked["top1"].to_numpy(), va_base["top1"].to_numpy())
    final_mrr = np.where(intervene, va_reranked["mrr"].to_numpy(), va_base["mrr"].to_numpy())
    corrected = int(((~va_base["top1"]) & final_top1).sum())
    introduced = int((va_base["top1"] & (~final_top1)).sum())

    # audit: CI, intervention precision, correction recall, risk-coverage curve
    val_delta = final_top1.astype(float) - va_base["top1"].to_numpy().astype(float)
    val_formula = va_merged["formula"].to_numpy()
    recall1_ci = query_weighted_formula_bootstrap(val_delta, val_formula, 2000, a.seed)
    total_possible_corrections = int((u_va == 1).sum())
    total_possible_introduced = int((u_va == -1).sum())
    intervention_precision = (corrected / (corrected + introduced)
                              if (corrected + introduced) > 0 else None)
    correction_recall = (corrected / total_possible_corrections
                         if total_possible_corrections > 0 else None)
    introduced_avoidance = (1.0 - introduced / total_possible_introduced
                            if total_possible_introduced > 0 else None)
    # risk-coverage curve over UNIQUE U values (handles ties correctly)
    uniq_U = np.unique(U_va)
    sample_idx = np.unique(np.linspace(0, len(uniq_U) - 1, min(15, len(uniq_U))).astype(int))
    coverage_curve = []
    for j in sample_idx:
        t = uniq_U[j]
        m = U_va > t
        ft = np.where(m, va_reranked["top1"].to_numpy(), va_base["top1"].to_numpy())
        c = int(((~va_base["top1"]) & ft).sum())
        i = int((va_base["top1"] & (~ft)).sum())
        coverage_curve.append({"threshold": float(t), "coverage": float(m.mean()),
                               "corrected": c, "introduced": i, "lambda_net": c - LAMBDA * i})

    # ---- four cheap controls at the SAME ~8% coverage ----
    n_open = int(round(0.0806 * len(va_base)))
    agree_arr = va_merged["agree"].to_numpy() == 1.0
    disagree_idx = np.where(~agree_arr)[0]

    def control_result(mask):
        ft = np.where(mask, va_reranked["top1"].to_numpy(), va_base["top1"].to_numpy())
        c = int(((~va_base["top1"]) & ft).sum())
        i = int((va_base["top1"] & (~ft)).sum())
        opened = int(mask.sum())
        neutral = opened - c - i
        return {"coverage": float(mask.mean()), "corrected": c, "introduced": i, "neutral": neutral,
                "all_open_correction_rate": c / opened if opened else None,
                "all_open_breakage_rate": i / opened if opened else None,
                "nonneutral_win_rate": c / (c + i) if (c + i) else None,
                "lambda_net": c - LAMBDA * i}

    order = np.argsort(-U_va)
    full_mask = np.zeros(len(va_base), dtype=bool); full_mask[order[:n_open]] = True
    dis_order = disagree_idx[np.argsort(-U_va[disagree_idx])]
    dis_mask = np.zeros(len(va_base), dtype=bool)
    dis_mask[dis_order[:min(n_open, len(disagree_idx))]] = True
    lc_order = np.argsort(va_merged["confidence_margin_b"].to_numpy())[:n_open]
    lc_mask = np.zeros(len(va_base), dtype=bool); lc_mask[lc_order] = True
    rng = np.random.default_rng(a.seed)
    rand_mask = np.zeros(len(va_base), dtype=bool)
    rand_mask[rng.choice(len(va_base), n_open, replace=False)] = True

    controls = {
        "full_learned_gate": control_result(full_mask),
        "disagreement_only": control_result(dis_mask),
        "low_confidence_only": control_result(lc_mask),
        "random": control_result(rand_mask),
    }
    uniq_counts = np.unique(np.round(U_va, 10), return_counts=True)
    tie_diag = {"n_unique_U": int(len(uniq_counts[0])), "max_tie_block": int(uniq_counts[1].max()),
                "n_disagree_queries": int(len(disagree_idx))}

    # ---- P1 final audits (no new model) ----
    # audit 1: ALL 66 disagreement queries (no 50-subset)
    all_dis_mask = np.zeros(len(va_base), dtype=bool)
    all_dis_mask[disagree_idx] = True
    all_disagree = control_result(all_dis_mask)

    # audit 2: random 50-subset of disagreements, 1000 draws -> lambda-net distribution
    rng2 = np.random.default_rng(a.seed + 100)
    nets = []
    for _ in range(1000):
        sub = rng2.choice(disagree_idx, min(50, len(disagree_idx)), replace=False)
        mm = np.zeros(len(va_base), dtype=bool); mm[sub] = True
        nets.append(control_result(mm)["lambda_net"])
    random50 = {"n_draws": len(nets), "mean_lambda_net": float(np.mean(nets)),
                "ci_low": float(np.percentile(nets, 2.5)),
                "ci_high": float(np.percentile(nets, 97.5))}

    # audit 3: can learned-U distinguish correction vs introduced WITHIN disagreements?
    nn_mask = (u_va != 0)
    labels = (u_va[nn_mask] == 1).astype(int)
    scores = U_va[nn_mask]
    from sklearn.metrics import roc_auc_score, average_precision_score
    auc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else None
    auprc = average_precision_score(labels, scores) if len(np.unique(labels)) > 1 else None
    learned_u_discrim = {"n_nonneutral_disagreement": int(nn_mask.sum()),
                         "n_correction": int(labels.sum()),
                         "n_introduced": int((labels == 0).sum()),
                         "auc": auc, "auprc": auprc}

    # 4. load FROZEN RAW-v1 artifact and reproduce the reference
    art = json.loads(a.artifact.read_text(encoding="utf-8"))
    mean = np.asarray(art["scaler_mean"]); scale = np.asarray(art["scaler_scale"])
    coef = np.asarray(art["model_coef"]); thr = art["gate_threshold"]

    def raw_score(vec):
        return float(coef @ ((vec - mean) / scale))

    va["rawv1_score"] = [raw_score(va[features].iloc[i].to_numpy(dtype=np.float64)) for i in range(len(va))]
    rv_base = retrieval_query(va, "dreams_similarity")
    rv_rk = retrieval_query(va, "rawv1_score")
    rv_merged = rv_base[["query_index", "ik14", "formula", "top1", "confidence_margin", "mrr"]].merge(
        rv_rk[["query_index", "top1", "rank", "mrr"]], on="query_index", suffixes=("_b", "_r"))
    rv_use = rv_merged["confidence_margin"] <= thr
    rv_top1 = np.where(rv_use, rv_merged["top1_r"], rv_merged["top1_b"])
    rv_corr = int(((~rv_merged["top1_b"]) & rv_top1).sum())
    rv_intro = int((rv_merged["top1_b"] & (~rv_top1)).sum())
    rv_r1 = float(rv_top1.mean()); rv_cov = float(rv_use.mean())
    print(f"[rawv1 repro] r1={rv_r1:.4f} corrected={rv_corr} introduced={rv_intro} "
          f"coverage={rv_cov:.4f}", flush=True)
    assert abs(rv_r1 - 0.8516129032258064) < 1e-3 and rv_corr == 44 and rv_intro == 17 and abs(rv_cov - 0.4645161290322581) < 1e-3
    print("[rawv1 repro] PASS: frozen RAW-v1 reproduced", flush=True)

    report = {
        "status": "g8r_p1_selective_switch",
        "lambda": LAMBDA, "best_gate_C": best_gc, "best_threshold": best_t,
        "baseline_recall1": float(va_base["top1"].mean()),
        "raw_v1_recall1": rv_r1, "raw_v1_corrected_introduced": [rv_corr, rv_intro],
        "raw_v1_coverage": rv_cov,
        "p1_recall1": float(final_top1.mean()),
        "p1_mrr": float(final_mrr.mean()),
        "p1_gate_coverage": float(intervene.mean()),
        "p1_corrected": corrected, "p1_introduced": introduced,
        "p1_unweighted_net": corrected - introduced,
        "p1_lambda_net": corrected - LAMBDA * introduced,
        "recall1_delta_queryweighted_formula_bootstrap": recall1_ci,
        "intervention_precision_nonneutral": intervention_precision,
        "correction_recall": correction_recall,
        "introduced_avoidance": introduced_avoidance,
        "total_possible_corrections": total_possible_corrections,
        "total_possible_introduced": total_possible_introduced,
        "risk_coverage_curve": coverage_curve,
        "controls_at_8pct_coverage": controls,
        "tie_diagnostics": tie_diag,
        "p1_final_audits": {
            "all_disagreement_queries": all_disagree,
            "random_50_subset_lambda_net": random50,
            "learned_u_discrimination": learned_u_discrim,
        },
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved: {a.output}")


if __name__ == "__main__":
    main()
