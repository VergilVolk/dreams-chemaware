"""P1: pairwise embedding reranker over the frozen DreaMS representation.

Per the 2026-08-22 audit: stop distorting the global embedding geometry; add a
lightweight pairwise reranker over strict-10ppm same-adduct candidates reading
  cosine   = z_q . z_c
  hadamard = z_q (.) z_c
  absdiff  = |z_q - z_c|
The frozen DreaMS embedding is preserved (no backbone/head training).

Training pairs come from the D0.4 PSD cache (Atr/Btr/ytr = locked train pos/neg
pair embeddings, IK14-disjoint from val).  Test = rerank the locked val set.
Reports macro-AUC / Recall@1 / near-vs-mid same-anchor pairwise accuracy vs the
cosine baseline.  Read-only backbone.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from train_e1_identity import load_base_model, preprocess_spectrum  # noqa: E402
from step5_gate_eval import embed, query_auc  # noqa: E402

DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_TRAIN = ROOT / "tasks/massspecgym_isomers/g8r_locked/train.json"
DEFAULT_VAL = ROOT / "tasks/massspecgym_isomers/g8r_locked/val.json"
DEFAULT_BASE = ROOT / "data/e1/official_embedding_slim.pt"
DEFAULT_ARCH = ROOT / "dreams/models/pretrained/ssl_model_server.pt"
DEFAULT_CACHE = ROOT / "data/validation/g8r_d04_psd_probe_cache.npz"
DEFAULT_VAL_EMB = ROOT / "data/validation/m1b_reranker_val_emb.npz"
DEFAULT_OUT = ROOT / "data/validation/m1b_embedding_reranker.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--val", type=Path, default=DEFAULT_VAL)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--base-ckpt", type=Path, default=DEFAULT_BASE)
    p.add_argument("--architecture-ckpt", type=Path, default=DEFAULT_ARCH)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--val-emb-cache", type=Path, default=DEFAULT_VAL_EMB)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-highest-peaks", type=int, default=100)
    p.add_argument("--ppm-tol", type=float, default=10.0)
    p.add_argument("--C", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=20260821)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-anchors", type=int, default=0)
    return p.parse_args()


def build_sibling(entries):
    groups = defaultdict(list)
    for i, e in enumerate(entries):
        groups[(e["ik14"], e["adduct"])].append(i)
    sib = [-1] * len(entries)
    for rows in groups.values():
        if len(rows) == 2:
            sib[rows[0]] = rows[1]; sib[rows[1]] = rows[0]
        else:
            for a, b in zip(rows, rows[1:]):
                sib[a] = b; sib[b] = a
    return sib


def pair_feats(A, B):
    cos = (A * B).sum(-1, keepdims=True)
    return np.concatenate([cos, A * B, np.abs(A - B)], axis=1)


def embed_val(a, val):
    """Embed val anchors + unique neg rows (official model, eval). Cached."""
    if a.val_emb_cache.exists() and not a.smoke:
        d = np.load(a.val_emb_cache, allow_pickle=True)
        return d["z_val"], d["va_rows"], d["n_anchor"]
    device = torch.device(a.device)
    model, _ = load_base_model(a.base_ckpt, a.architecture_ckpt, device, a.n_highest_peaks)
    model.eval()
    anchor_rows = [int(e["anchor_row"]) for e in val]
    anchor_set = set(anchor_rows)
    neg_rows = sorted({int(n["row"]) for e in val for n in e["neg"]} - anchor_set)
    all_rows = anchor_rows + neg_rows
    with h5py.File(a.data, "r") as h:
        pmz_all = np.asarray(h["precursor_mz"][:], dtype=float)
        specs = [preprocess_spectrum(np.asarray(h["spectrum"][r]), float(pmz_all[r]), a.n_highest_peaks)
                 for r in all_rows]
    z = embed(model, specs, device, a.batch_size).numpy()
    if not a.smoke:
        np.savez_compressed(a.val_emb_cache, z_val=z.astype(np.float32),
                            va_rows=np.asarray(all_rows, dtype=np.int64),
                            n_anchor=len(anchor_rows))
    return z, np.asarray(all_rows, dtype=np.int64), len(anchor_rows)


def rerank(z_anchor, iks, pmzs, adducts, ppm_tol, scorer):
    rows = []
    for qi in range(len(iks)):
        ppm_da = ppm_tol * 1e-6 * pmzs[qi]
        cand = (np.abs(pmzs - pmzs[qi]) <= ppm_da) & (np.arange(len(iks)) != qi) & (adducts == adducts[qi])
        idx = np.where(cand)[0]
        if len(idx) == 0:
            continue
        labels = (iks[idx] == iks[qi]).astype(int)
        if labels.sum() == 0 or (labels == 0).sum() == 0:
            continue
        scores = scorer(z_anchor[qi], z_anchor[idx])
        best = {}
        for j, s in zip(idx, scores):
            ik = iks[j]
            if ik not in best or s > best[ik]:
                best[ik] = float(s)
        order = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        ranks = [ik for ik, _ in order]
        top1 = bool(ranks and ranks[0] == iks[qi])
        rows.append({"auc": query_auc(labels, scores), "recall1": 1.0 if top1 else 0.0,
                     "mrr": 1.0 / (ranks.index(iks[qi]) + 1) if iks[qi] in ranks else 0.0,
                     "top1": top1})
    return rows


def aggregate(rows):
    if not rows:
        return {"n_queries": 0, "macro_auc": 0.5, "recall1": 0.0, "mrr": 0.0}
    return {"n_queries": len(rows), "macro_auc": float(np.mean([r["auc"] for r in rows])),
            "recall1": float(np.mean([r["recall1"] for r in rows])),
            "mrr": float(np.mean([r["mrr"] for r in rows]))}


def near_mid_pairwise(entries, z, sibling, row_to_index, scorer):
    out = {"near": [], "mid": []}
    for i, e in enumerate(entries):
        sib = sibling[i]
        if sib < 0 or not e["neg"]:
            continue
        p = float(scorer(z[i], z[sib]))
        for nn in e["neg"]:
            j = row_to_index[int(nn["row"])]
            n = float(scorer(z[i], z[j]))
            out[nn.get("grade", "mid")].append(int(p > n))
    return {g: (float(np.mean(v)) if v else float("nan")) for g, v in out.items()}


def main() -> None:
    a = parse_args()
    cache = np.load(a.cache, allow_pickle=True)
    Atr, Btr, ytr = cache["Atr"], cache["Btr"], cache["ytr"]
    if a.smoke:
        Atr, Btr, ytr = Atr[:2000], Btr[:2000], ytr[:2000]
    Xtr = pair_feats(Atr, Btr)
    scaler = StandardScaler().fit(Xtr)
    model = LogisticRegression(C=a.C, max_iter=5000, random_state=a.seed).fit(scaler.transform(Xtr), ytr)

    def scorer(zq, zc):
        X = pair_feats(zq[None, :], zc)
        return model.decision_function(scaler.transform(X))

    val = json.loads(a.val.read_text(encoding="utf-8"))["entries"]
    if a.max_anchors > 0:
        val = val[: a.max_anchors]
    z_val, va_rows, n_anchor = embed_val(a, val)
    z_anchor = z_val[:n_anchor]
    va_row_to_idx = {int(r): i for i, r in enumerate(va_rows)}
    iks = [e["ik14"] for e in val]
    pmzs = np.array([e["precursor_mz"] for e in val])
    adducts = np.array([e["adduct"] for e in val])
    sib_va = build_sibling(val)

    def cosine_scorer(zq, zc):
        return (zq * zc).sum(-1)

    base_rows = rerank(z_anchor, iks, pmzs, adducts, a.ppm_tol, cosine_scorer)
    rk_rows = rerank(z_anchor, iks, pmzs, adducts, a.ppm_tol, scorer)
    base_nm = near_mid_pairwise(val, z_anchor, sib_va, va_row_to_idx, cosine_scorer)
    rk_nm = near_mid_pairwise(val, z_anchor, sib_va, va_row_to_idx, scorer)

    report = {
        "status": "m1b_embedding_reranker", "C": a.C, "ppm_tol": a.ppm_tol,
        "n_train_pairs": int(len(ytr)), "n_train_pos": int((ytr == 1).sum()),
        "n_train_neg": int((ytr == 0).sum()),
        "baseline": {"retrieval": aggregate(base_rows), "near_pairwise": base_nm["near"],
                     "mid_pairwise": base_nm["mid"]},
        "reranker": {"retrieval": aggregate(rk_rows), "near_pairwise": rk_nm["near"],
                     "mid_pairwise": rk_nm["mid"]},
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved: {a.output}")


if __name__ == "__main__":
    main()
