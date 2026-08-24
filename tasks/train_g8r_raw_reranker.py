"""Groupwise RAW reranker for g8r: DreaMS cosine + RAW spectrum features ONLY.

Per the 2026-08-22 audit (first version): NO token, NO old 8-peak panel.
Training is within-query ranking f(a,p) > f(a,n) via the same difference logic
as train_pairwise_delta_reranker.py (NOT a global pair classifier).  Deployment
uses a low-confidence gate on the DreaMS Top1-minus-Top2 score gap.

Candidate-graph protocol (identical train/val/final-test):
  strict-10ppm same-adduct, exclude self, per-IK14 molecule dedup (max cosine),
  query must have a positive (same IK14) and a negative (different IK14).

Model selection is formula-group OOF on g8r_train only; g8r_val is DEV eval
(stratified by seen/unseen formula & scaffold, near/mid MCES).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from train_e1_identity import load_base_model, preprocess_spectrum  # noqa: E402
from step5_gate_eval import embed  # noqa: E402
from audit_e0_observability_residual import pair_features  # noqa: E402
from audit_large_observability_residual import symmetric_features  # noqa: E402

DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_TRAIN = ROOT / "tasks/massspecgym_isomers/g8r_locked/train.json"
DEFAULT_VAL = ROOT / "tasks/massspecgym_isomers/g8r_locked/val.json"
DEFAULT_BASE = ROOT / "data/e1/official_embedding_slim.pt"
DEFAULT_ARCH = ROOT / "dreams/models/pretrained/ssl_model_server.pt"
DEFAULT_OUT = ROOT / "data/validation/g8r_raw_reranker"
DEFAULT_CACHE = ROOT / "data/validation/g8r_raw_reranker_cache.npz"

RAW_FEATURES = [
    "sqrt_cosine", "linear_cosine", "entropy_similarity",
    "intensity_coverage_min", "intensity_coverage_mean",
    "matched_peak_fraction_min", "top10_match_fraction",
    "neutral_loss_sqrt_cosine", "neutral_loss_coverage_min",
    "neutral_loss_coverage_mean", "peak_count_ratio",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--val", type=Path, default=DEFAULT_VAL)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--base-ckpt", type=Path, default=DEFAULT_BASE)
    p.add_argument("--architecture-ckpt", type=Path, default=DEFAULT_ARCH)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-highest-peaks", type=int, default=100)
    p.add_argument("--ppm-tol", type=float, default=10.0)
    p.add_argument("--peak-tolerance", type=float, default=0.02)
    p.add_argument("--hard-k", type=int, default=5)
    p.add_argument("--c-values", type=float, nargs="+", default=[0.0001, 0.001, 0.01])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260822)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-anchors", type=int, default=0)
    return p.parse_args()


def fold_for_formula(formula, folds):
    return int.from_bytes(hashlib.blake2b(str(formula).encode(), digest_size=8).digest(), "little") % folds


def read_str(h, key):
    raw = h[key][:]
    return np.asarray([x.decode("utf-8", "ignore") if isinstance(x, bytes) else str(x)
                       for x in raw], dtype=object)


def build_pairs(a, entries, cache_path, device):
    """Build per-anchor candidate pairs: cosine + RAW features + label."""
    if cache_path.exists() and not a.smoke:
        d = np.load(cache_path, allow_pickle=True)
        return pd.DataFrame({k: d[k] for k in d.files})

    iks = np.asarray([e["ik14"] for e in entries], dtype=object)
    adducts = np.asarray([e["adduct"] for e in entries], dtype=object)
    pmzs = np.asarray([e["precursor_mz"] for e in entries], dtype=float)
    anchor_rows = [int(e["anchor_row"]) for e in entries]

    model, _ = load_base_model(a.base_ckpt, a.architecture_ckpt, device, a.n_highest_peaks)
    model.eval()
    with h5py.File(a.data, "r") as h:
        pmz_all = np.asarray(h["precursor_mz"][:], dtype=float)
        formula_all = read_str(h, "FORMULA")
        spectra = {r: np.asarray(h["spectrum"][r]) for r in anchor_rows}
        specs = [preprocess_spectrum(np.asarray(h["spectrum"][r]), float(pmz_all[r]), a.n_highest_peaks)
                 for r in anchor_rows]
    z = embed(model, specs, device, a.batch_size).numpy()
    z = z / np.clip(np.linalg.norm(z, axis=1, keepdims=True), 1e-12, None)

    rows = []
    n_anchor = len(entries)
    for qi in range(n_anchor):
        ppm_da = a.ppm_tol * 1e-6 * pmzs[qi]
        cand = (np.abs(pmzs - pmzs[qi]) <= ppm_da) & (np.arange(n_anchor) != qi) & (adducts == adducts[qi])
        idx = np.where(cand)[0]
        if len(idx) == 0:
            continue
        # per-IK14 dedup by max cosine
        best = {}
        for j in idx:
            c = float(z[qi] @ z[j])
            if iks[j] not in best or c > best[iks[j]][0]:
                best[iks[j]] = (c, j)
        has_pos = any(ik == iks[qi] for ik in best)
        negs = {ik: v for ik, v in best.items() if ik != iks[qi]}
        if not has_pos or not negs:
            continue
        for ik, (c, j) in best.items():
            label = 1 if ik == iks[qi] else 0
            f = symmetric_features(spectra[anchor_rows[qi]], float(pmzs[qi]),
                                   spectra[anchor_rows[j]], float(pmzs[j]), a.peak_tolerance)
            rows.append({
                "query": qi, "candidate": int(j), "query_ik14": iks[qi],
                "candidate_ik14": ik, "label": label,
                "formula": formula_all[anchor_rows[qi]],
                "dreams_similarity": float(c),
                **f,
            })
    df = pd.DataFrame(rows)
    out = {k: df[k].to_numpy() for k in df.columns}
    if not a.smoke:
        np.savez_compressed(cache_path, **{k: df[k].to_numpy() for k in df.columns})
        print(f"[cache] saved {cache_path}", flush=True)
    return df


def ranking_examples(frame, features, hard_k):
    diffs, formulas = [], []
    for _, group in frame.groupby("query", sort=False):
        positives = group[group["label"] == 1]
        negatives = group[group["label"] == 0]
        if positives.empty or negatives.empty:
            continue
        positive = positives.loc[positives["dreams_similarity"].idxmax()]
        mol_best = negatives.sort_values("dreams_similarity", ascending=False).drop_duplicates("candidate_ik14")
        hard = mol_best.head(hard_k)
        delta = positive[features].to_numpy(float)[None, :] - hard[features].to_numpy(float)
        diffs.append(delta)
        formulas.extend([positive.formula] * len(delta))
    x_pos = np.concatenate(diffs)
    formula = np.asarray(formulas, object)
    x = np.vstack([x_pos, -x_pos])
    y = np.r_[np.ones(len(x_pos), dtype=int), np.zeros(len(x_pos), dtype=int)]
    return x, y, np.r_[formula, formula]


def fit_ranker(frame, features, hard_k, c_value):
    x, y, formula = ranking_examples(frame, features, hard_k)
    if len(x) == 0:
        return None, None
    scaler = StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    counts = pd.Series(formula).map(pd.Series(formula).value_counts()).to_numpy(float)
    w = 1.0 / counts
    w *= len(w) / w.sum()
    model = LogisticRegression(C=c_value, fit_intercept=False, max_iter=5000, random_state=20260822).fit(x_scaled, y, sample_weight=w)
    return scaler, model


def score_frame(frame, features, scaler, model):
    out = frame.copy()
    out["score"] = model.decision_function(scaler.transform(out[features].to_numpy()))
    return out


def retrieval_query(frame, score_col):
    rows = []
    for qi, group in frame.groupby("query", sort=False):
        pos = group[group["label"] == 1]
        neg = group[group["label"] == 0]
        if pos.empty or neg.empty:
            continue
        pscore = pos[score_col].max()
        mol = group.sort_values(score_col, ascending=False).drop_duplicates("candidate_ik14")
        top = mol[score_col].to_numpy(float)
        conf_margin = float(top[0] - top[1]) if len(top) > 1 else float("inf")
        neg_scores = neg.groupby("candidate_ik14")[score_col].max().to_numpy(float)
        rank = 1 + int(np.sum(neg_scores >= pscore))
        rows.append({
            "query_index": int(qi), "ik14": group.iloc[0].query_ik14,
            "formula": group.iloc[0].formula,
            "chosen_ik14": mol.iloc[0].candidate_ik14,
            "margin": float(pscore - neg_scores.max()),
            "confidence_margin": conf_margin,
            "top1": bool(rank == 1), "rank": rank, "mrr": 1.0 / rank,
        })
    return pd.DataFrame(rows)


def gated(baseline, reranked, threshold, require_disagreement):
    merged = baseline.merge(reranked, on=["query_index", "ik14", "formula"], suffixes=("_base", "_rer"))
    use = merged["confidence_margin_base"] <= threshold
    if require_disagreement:
        use &= merged["chosen_ik14_base"] != merged["chosen_ik14_rer"]
    out = pd.DataFrame({"query_index": merged["query_index"], "ik14": merged["ik14"],
                        "formula": merged["formula"], "gate_used": use})
    for col in ("chosen_ik14", "margin", "confidence_margin", "top1", "rank", "mrr"):
        out[col] = np.where(use, merged[f"{col}_rer"], merged[f"{col}_base"])
    return out


def main() -> None:
    a = parse_args()
    device = a.device
    train = json.loads(a.train.read_text(encoding="utf-8"))["entries"]
    val = json.loads(a.val.read_text(encoding="utf-8"))["entries"]
    if a.max_anchors > 0:
        train = train[: a.max_anchors]
        val = val[: a.max_anchors]

    tr = build_pairs(a, train, a.cache, device)
    va = build_pairs(a, val, Path(str(a.cache).replace(".npz", "_val.npz")), device)

    features = ["dreams_similarity"] + RAW_FEATURES
    base_tr = retrieval_query(tr, "dreams_similarity")
    base_va = retrieval_query(va, "dreams_similarity")

    # C selection via formula-group OOF on train
    assignments = {f: fold_for_formula(f, a.folds) for f in tr["formula"].unique()}
    best_c, best_score = None, -1
    for c in a.c_values:
        scored_parts = []
        for fold in range(a.folds):
            held = {f for f, v in assignments.items() if v == fold}
            trn = tr[~tr["formula"].isin(held)]
            hld = tr[tr["formula"].isin(held)]
            if len(trn) == 0 or len(hld) == 0:
                continue
            scaler, model = fit_ranker(trn, features, a.hard_k, c)
            if scaler is None:
                continue
            scored_parts.append(score_frame(hld, features, scaler, model))
        if not scored_parts:
            continue
        q = retrieval_query(pd.concat(scored_parts), "score")
        s = float(q["top1"].mean())
        if s > best_score:
            best_score, best_c = s, c
    print(f"[train OOF] best C={best_c} top1={best_score:.4f}", flush=True)

    # OOF predictions for the best C (formula-group), so the gate threshold is
    # selected on out-of-fold predictions, NOT in-sample (fixes gate leakage).
    oof_parts = []
    for fold in range(a.folds):
        held = {f for f, v in assignments.items() if v == fold}
        trn = tr[~tr["formula"].isin(held)]
        hld = tr[tr["formula"].isin(held)]
        if len(trn) == 0 or len(hld) == 0:
            continue
        scaler_f, model_f = fit_ranker(trn, features, a.hard_k, best_c)
        if scaler_f is None:
            continue
        oof_parts.append(score_frame(hld, features, scaler_f, model_f))
    oof_scored = pd.concat(oof_parts)
    reranked_oof = retrieval_query(oof_scored, "score")

    best_gate, best_gate_score = None, -1
    for thr in np.quantile(base_tr["confidence_margin"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        for req in (False, True):
            g = gated(base_tr, reranked_oof, float(thr), req)
            s = float(g["top1"].mean())
            if s > best_gate_score:
                best_gate_score, best_gate = s, (float(thr), req)
    thr, req = best_gate
    print(f"[gate OOF] threshold={thr:.4f} require_disagreement={req} oof_top1={best_gate_score:.4f}", flush=True)

    # fit the final ranker on the FULL train, then apply the OOF-frozen gate to val
    scaler, model = fit_ranker(tr, features, a.hard_k, best_c)
    va_scored = score_frame(va, features, scaler, model)
    reranked_va = retrieval_query(va_scored, "score")
    g_va = gated(base_va, reranked_va, thr, req)

    def summarize(name, base, final):
        gate_frac = float(final["gate_used"].mean()) if "gate_used" in final.columns else 0.0
        return {
            "name": name,
            "n_queries": int(len(final)),
            "top1": float(final["top1"].mean()),
            "mrr": float(final["mrr"].mean()),
            "recall1": float(final["top1"].mean()),
            "gate_fraction": gate_frac,
            "corrected": int(((~base["top1"]) & final["top1"]).sum()),
            "introduced": int((base["top1"] & (~final["top1"])).sum()),
        }

    # seen/unseen-formula stratification (g8r_train/val are IK14-disjoint but share
    # 75.7% of formulas; report both so the "seen-formula" softening is explicit).
    train_formulas = set(tr["formula"].unique())

    def stratify_by_formula(base, final):
        seen = final["formula"].isin(train_formulas).to_numpy()
        out = {}
        for name, mask in [("overall", np.ones(len(final), dtype=bool)),
                           ("seen_formula", seen),
                           ("unseen_formula", ~seen)]:
            b = base[mask].reset_index(drop=True)
            f = final[mask].reset_index(drop=True)
            out[name] = {
                "n_queries": int(len(f)),
                "n_unique_formula": int(f["formula"].nunique()),
                "top1": float(f["top1"].mean()),
                "mrr": float(f["mrr"].mean()),
                "corrected": int(((~b["top1"]) & f["top1"]).sum()),
                "introduced": int((b["top1"] & (~f["top1"])).sum()),
            }
        return out

    def formula_bootstrap_top1(base, final, n_boot=2000):
        merged = base[["query_index", "formula", "top1"]].merge(
            final[["query_index", "top1"]], on="query_index", suffixes=("_b", "_f"))
        merged["delta"] = merged["top1_f"].astype(float) - merged["top1_b"].astype(float)
        by_formula = merged.groupby("formula")["delta"].mean().to_numpy(float)
        rng = np.random.default_rng(a.seed)
        draws = np.array([rng.choice(by_formula, len(by_formula), replace=True).mean()
                          for _ in range(n_boot)])
        return {"mean_delta": float(merged["delta"].mean()),
                "ci_low": float(np.percentile(draws, 2.5)),
                "ci_high": float(np.percentile(draws, 97.5))}

    report = {
        "status": "g8r_raw_reranker",
        "features": features, "hard_k": a.hard_k, "best_C": best_c,
        "gate_threshold": thr, "gate_require_disagreement": req,
        "baseline_val": summarize("baseline", base_va, base_va),
        "reranker_val": summarize("reranker", base_va, g_va),
        "baseline_stratified": stratify_by_formula(base_va, base_va),
        "reranker_stratified": stratify_by_formula(base_va, g_va),
        "top1_delta_formula_bootstrap": formula_bootstrap_top1(base_va, g_va),
    }
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved: {a.output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
