"""Decisive per-query audit of the RAW reranker's near/mid effect.

Fixes the definitional inconsistency between the D0/PSD probe and the earlier
near/mid eval, and produces the transition table the reviewer requires.

UNIFIED definition:
  - positive = cross-condition sibling spectrum (same IK14+adduct);
  - near anchor = has >=1 hard negative with grade "near" (MCES 0-2);
  - mid anchor  = has >=1 hard negative with grade "mid" (MCES 3-5);
  - near accuracy = pos_score > max over the anchor's NEAR negatives;
  - mid accuracy  = pos_score > max over the anchor's MID negatives.
Also reports the D0/PSD metric (pos vs max over ALL negatives) for the 186
near-classified anchors, to reconcile the 0.6344 vs 0.6720 difference.

Outputs: perquery.csv + summary.json (near/mid corrected-vs-introduced, formula
-cluster paired bootstrap, and the "allneg" reconciliation number).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from train_e1_identity import load_base_model, preprocess_spectrum  # noqa: E402
from step5_gate_eval import embed  # noqa: E402
from audit_large_observability_residual import symmetric_features  # noqa: E402
from train_g8r_raw_reranker import fit_ranker, RAW_FEATURES  # noqa: E402

DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_TRAIN = ROOT / "tasks/massspecgym_isomers/g8r_locked/train.json"
DEFAULT_VAL = ROOT / "tasks/massspecgym_isomers/g8r_locked/val.json"
DEFAULT_BASE = ROOT / "data/e1/official_embedding_slim.pt"
DEFAULT_ARCH = ROOT / "dreams/models/pretrained/ssl_model_server.pt"
DEFAULT_CACHE = ROOT / "data/validation/g8r_raw_reranker_cache.npz"
DEFAULT_OUT = ROOT / "data/validation/g8r_raw_reranker_perquery_audit"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--val", type=Path, default=DEFAULT_VAL)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--base-ckpt", type=Path, default=DEFAULT_BASE)
    p.add_argument("--architecture-ckpt", type=Path, default=DEFAULT_ARCH)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-highest-peaks", type=int, default=100)
    p.add_argument("--peak-tolerance", type=float, default=0.02)
    p.add_argument("--hard-k", type=int, default=5)
    p.add_argument("--C", type=float, default=0.01)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260822)
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


def paired_bootstrap_by_formula(deltas, formulas, n_boot, seed):
    d = pd.DataFrame({"delta": deltas, "formula": formulas})
    by_f = d.groupby("formula")["delta"].mean().to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.array([rng.choice(by_f, len(by_f), replace=True).mean() for _ in range(n_boot)])
    return {"mean": float(d["delta"].mean()),
            "ci_low": float(np.percentile(draws, 2.5)),
            "ci_high": float(np.percentile(draws, 97.5))}


def main() -> None:
    a = parse_args()
    device = a.device
    val = json.loads(a.val.read_text(encoding="utf-8"))["entries"]
    sib = build_sibling(val)

    anchor_rows = [int(e["anchor_row"]) for e in val]
    anchor_set = set(anchor_rows)
    neg_rows = sorted({int(n["row"]) for e in val for n in e["neg"]} - anchor_set)
    all_rows = anchor_rows + neg_rows
    row_to_idx = {r: i for i, r in enumerate(all_rows)}

    model, _ = load_base_model(a.base_ckpt, a.architecture_ckpt, device, a.n_highest_peaks)
    model.eval()
    with h5py.File(a.data, "r") as h:
        pmz_all = np.asarray(h["precursor_mz"][:], dtype=float)
        formula_all = np.asarray([x.decode("utf-8", "ignore") if isinstance(x, bytes) else str(x)
                                  for x in h["FORMULA"][:]], dtype=object)
        inst_all = np.asarray([x.decode("utf-8", "ignore") if isinstance(x, bytes) else str(x)
                               for x in h["INSTRUMENT_TYPE"][:]], dtype=object)
        ce_all = np.asarray(h["COLLISION_ENERGY"][:], dtype=float)
        adduct_all = np.asarray([x.decode("utf-8", "ignore") if isinstance(x, bytes) else str(x)
                                 for x in h["adduct"][:]], dtype=object)
        specs = [preprocess_spectrum(np.asarray(h["spectrum"][r]), float(pmz_all[r]), a.n_highest_peaks)
                 for r in all_rows]
        spectra = {r: np.asarray(h["spectrum"][r]) for r in all_rows}
    z = embed(model, specs, device, a.batch_size).numpy()
    z = z / np.clip(np.linalg.norm(z, axis=1, keepdims=True), 1e-12, None)

    cache = np.load(a.cache, allow_pickle=True)
    tr = pd.DataFrame({k: cache[k] for k in cache.files})
    train_formulas = set(tr["formula"].unique())
    scaler, rk = fit_ranker(tr, ["dreams_similarity"] + RAW_FEATURES, a.hard_k, a.C)

    def pair_score(anchor_row, other_row, pmz_other):
        ia = row_to_idx[anchor_row]; ib = row_to_idx[other_row]
        cos = float(z[ia] @ z[ib])
        f = symmetric_features(spectra[anchor_row], float(pmz_all[anchor_row]),
                               spectra[other_row], float(pmz_other), a.peak_tolerance)
        vec = np.concatenate([[cos], [f[k] for k in RAW_FEATURES]]).astype(np.float32)
        rks = float(rk.decision_function(scaler.transform(vec[None, :]))[0])
        return cos, rks

    records = []
    for i, e in enumerate(val):
        if sib[i] < 0:
            continue
        pos_row = int(val[sib[i]]["anchor_row"])
        cos_pos, rks_pos = pair_score(int(e["anchor_row"]), pos_row,
                                      float(val[sib[i]]["precursor_mz"]))
        near_negs = [n for n in e["neg"] if n.get("grade") == "near"]
        mid_negs = [n for n in e["neg"] if n.get("grade") == "mid"]

        rec = {
            "anchor_row": int(e["anchor_row"]), "ik14": e["ik14"],
            "formula": str(formula_all[int(e["anchor_row"])]),
            "instrument": str(inst_all[int(e["anchor_row"])]),
            "ce_finite": bool(np.isfinite(ce_all[int(e["anchor_row"])])),
            "adduct": str(adduct_all[int(e["anchor_row"])]),
            "formula_seen": str(formula_all[int(e["anchor_row"])]) in train_formulas,
            "cos_pos": cos_pos, "rks_pos": rks_pos,
        }

        if near_negs:
            h = max(near_negs, key=lambda n: float(z[row_to_idx[int(e["anchor_row"])]] @ z[row_to_idx[int(n["row"])]]))
            c, r = pair_score(int(e["anchor_row"]), int(h["row"]), float(pmz_all[int(h["row"])]))
            rec.update({"near_neg_row": int(h["row"]), "near_mces": h.get("mces_raw"),
                        "cos_near_hard": c, "rks_near_hard": r,
                        "near_cos_ok": int(cos_pos > c), "near_rks_ok": int(rks_pos > r)})
        if mid_negs:
            h = max(mid_negs, key=lambda n: float(z[row_to_idx[int(e["anchor_row"])]] @ z[row_to_idx[int(n["row"])]]))
            c, r = pair_score(int(e["anchor_row"]), int(h["row"]), float(pmz_all[int(h["row"])]))
            rec.update({"mid_neg_row": int(h["row"]), "mid_mces": h.get("mces_raw"),
                        "cos_mid_hard": c, "rks_mid_hard": r,
                        "mid_cos_ok": int(cos_pos > c), "mid_rks_ok": int(rks_pos > r)})
        # "allneg" (D0/PSD definition): pos vs max over ALL negatives, near-classified
        if near_negs:
            all_neg = list(e["neg"])
            h = max(all_neg, key=lambda n: float(z[row_to_idx[int(e["anchor_row"])]] @ z[row_to_idx[int(n["row"])]]))
            c, r = pair_score(int(e["anchor_row"]), int(h["row"]), float(pmz_all[int(h["row"])]))
            rec.update({"allneg_cos_ok": int(cos_pos > c), "allneg_rks_ok": int(rks_pos > r)})
        records.append(rec)

    df = pd.DataFrame(records)

    def transition(mask, cos_col, rks_col):
        m = df[mask]
        b = m[cos_col].astype(int).to_numpy()
        c = m[rks_col].astype(int).to_numpy()
        return {"n": int(len(m)), "base_acc": float(b.mean()), "cand_acc": float(c.mean()),
                "corrected": int(((b == 0) & (c == 1)).sum()),
                "introduced": int(((b == 1) & (c == 0)).sum()),
                "unchanged": int((b == c).sum())}

    near_mask = df["near_neg_row"].notna().to_numpy()
    mid_mask = df["mid_neg_row"].notna().to_numpy()
    allneg_mask = df["allneg_cos_ok"].notna().to_numpy()

    def delta(mask, cos_col, rks_col):
        m = df[mask]
        return (m[rks_col].astype(float) - m[cos_col].astype(float)).to_numpy(), m["formula"].to_numpy()

    nd, nf = delta(near_mask, "near_cos_ok", "near_rks_ok")
    md, mf = delta(mid_mask, "mid_cos_ok", "mid_rks_ok")

    summary = {
        "status": "g8r_raw_reranker_perquery_audit",
        "unified_definition": "pos vs max over the grade's own negatives",
        "near": transition(near_mask, "near_cos_ok", "near_rks_ok"),
        "mid": transition(mid_mask, "mid_cos_ok", "mid_rks_ok"),
        "allneg_reconciliation": transition(allneg_mask, "allneg_cos_ok", "allneg_rks_ok"),
        "near_delta_formula_bootstrap": paired_bootstrap_by_formula(nd, nf, a.bootstrap, a.seed),
        "mid_delta_formula_bootstrap": paired_bootstrap_by_formula(md, mf, a.bootstrap, a.seed),
    }

    a.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.output_dir / "perquery.csv", index=False)
    (a.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved perquery.csv + summary.json to {a.output_dir}")


if __name__ == "__main__":
    main()
