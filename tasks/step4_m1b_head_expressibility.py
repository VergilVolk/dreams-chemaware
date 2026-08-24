"""Step 4: exact head expressibility experiment on cached PRE-HEAD representations.

Rationale (2026-08-22 audit): M1b-v2 failed identically to v1, so the missing
hard-stream preservation was NOT the root cause.  The remaining hypotheses are
(a) the linear head z=normalize(Wh+b) cannot express the hard-negative margin at
all, vs (b) M1b trains the head on dropout-NOISY inputs (backbone in train mode,
~0.11 cosine noise vs ~0.005 margin signal).

This isolates (a) by training the head DIRECTLY on the cached 1024-dim pre-head
representations, computed with the frozen backbone in EVAL mode (no dropout).
Everything else mirrors M1b-v2 exactly: same dual-stream ranking loss, same
margin/tau/lambdas, same locked data.  Sweep lam_preserve x seeds; report the
same gate metrics (same-anchor margin, pairwise accuracy, macro-AUC, Recall@1,
preservation) computed inline from the cached h.

Read-only backbone.  CPU is fine once the cache is built (first run ~40 min to
embed; later runs are seconds).
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from train_e1_identity import load_base_model, preprocess_spectrum  # noqa: E402
from eval_g8r_inner_gate import retrieval_per_query, aggregate_retrieval  # noqa: E402

DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_TRAIN = ROOT / "tasks/massspecgym_isomers/g8r_locked/train.json"
DEFAULT_VAL = ROOT / "tasks/massspecgym_isomers/g8r_locked/val.json"
DEFAULT_BASE = ROOT / "data/e1/official_embedding_slim.pt"
DEFAULT_ARCH = ROOT / "dreams/models/pretrained/ssl_model_server.pt"
DEFAULT_OUT = ROOT / "data/validation/m1b_head_expressibility"
DEFAULT_CACHE = ROOT / "data/validation/m1b_head_expressibility_cache.npz"


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
    p.add_argument("--seeds", type=int, nargs="+", default=[20260821, 20260822, 20260823])
    p.add_argument("--lam-preserve-list", type=float, nargs="+", default=[0.0, 1.0, 5.0, 20.0])
    p.add_argument("--margin", type=float, default=0.05)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--lam-posfloor", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--ppm-tol", type=float, default=10.0)
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


def prehead_embed(model, spectra_list, device, batch_size):
    out = []
    for i in range(0, len(spectra_list), batch_size):
        batch = torch.stack(spectra_list[i:i + batch_size]).to(device)
        with torch.inference_mode():
            h = model.backbone(batch, None)[:, 0, :]
            out.append(h.cpu())
    return torch.cat(out, dim=0).numpy()


def load_or_build_cache(a):
    if a.cache.exists() and not a.smoke:
        d = np.load(a.cache, allow_pickle=True)
        out = {k: d[k] for k in d.files}
        if int(out.get("n_train_rows", 0)) >= 1000:
            print(f"[cache] loaded {a.cache}", flush=True)
            return out
    train = json.loads(a.train.read_text(encoding="utf-8"))["entries"]
    val = json.loads(a.val.read_text(encoding="utf-8"))["entries"]
    if a.max_anchors > 0:
        train = train[: a.max_anchors]
        val = val[: a.max_anchors]
    device = torch.device(a.device)
    model, _ = load_base_model(a.base_ckpt, a.architecture_ckpt, device, a.n_highest_peaks)
    model.eval()
    with h5py.File(a.data, "r") as f:
        pmz_all = np.asarray(f["precursor_mz"][:], dtype=float)

    def rows_and_h(entries):
        anchor_rows = [int(e["anchor_row"]) for e in entries]
        anchor_set = set(anchor_rows)
        neg_rows = sorted({int(n["row"]) for e in entries for n in e["neg"]} - anchor_set)
        all_rows = anchor_rows + neg_rows
        specs = []
        with h5py.File(a.data, "r") as f:
            for r in all_rows:
                specs.append(preprocess_spectrum(np.asarray(f["spectrum"][r]),
                                                 float(pmz_all[r]), a.n_highest_peaks))
        hh = prehead_embed(model, specs, device, a.batch_size)
        return all_rows, hh

    tr_rows, h_tr = rows_and_h(train)
    va_rows, h_va = rows_and_h(val)
    out = {
        "h_train": h_tr.astype(np.float32), "h_val": h_va.astype(np.float32),
        "tr_rows": np.asarray(tr_rows, dtype=np.int64),
        "va_rows": np.asarray(va_rows, dtype=np.int64),
        "W_official": model.head.weight.detach().cpu().numpy().astype(np.float32),
        "b_official": model.head.bias.detach().cpu().numpy().astype(np.float32),
        "n_train_rows": len(tr_rows), "n_val_rows": len(va_rows),
    }
    if not a.smoke:
        np.savez_compressed(a.cache, **out)
        print(f"[cache] saved {a.cache}", flush=True)
    return out


def normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def apply_head(h, W, b):
    return normalize(h @ W.T + b)


def main() -> None:
    a = parse_args()
    d = load_or_build_cache(a)
    h_tr, h_va = d["h_train"], d["h_val"]
    tr_rows, va_rows = d["tr_rows"], d["va_rows"]
    W_off, b_off = d["W_official"], d["b_official"]

    train = json.loads(a.train.read_text(encoding="utf-8"))["entries"]
    val = json.loads(a.val.read_text(encoding="utf-8"))["entries"]
    if a.max_anchors > 0:
        train = train[: a.max_anchors]; val = val[: a.max_anchors]

    tr_row_to_idx = {int(r): i for i, r in enumerate(tr_rows)}
    va_row_to_idx = {int(r): i for i, r in enumerate(va_rows)}

    sib_tr = build_sibling(train)
    sib_va = build_sibling(val)

    # official embeddings (frozen constants)
    z_off_tr = apply_head(h_tr, W_off, b_off)
    z_off_va = apply_head(h_va, W_off, b_off)

    # ---- hard + safety indices (mirror step4_m1b_train) ----
    hard_idx = [i for i, e in enumerate(train) if e["neg"]]
    if a.smoke:
        hard_sel = hard_idx[:120]
        safety_sel = list(range(min(120, len(train))))
    else:
        hard_sel = hard_idx
        safety_sel = list(range(len(train)))
    hard_anchor_spec = [tr_row_to_idx[int(train[i]["anchor_row"])] for i in hard_sel]
    hard_pos_spec = [tr_row_to_idx[int(train[sib_tr[i]]["anchor_row"])] for i in hard_sel]
    hard_neg_spec = [[tr_row_to_idx[int(n["row"])] for n in train[i]["neg"]] for i in hard_sel]
    off_pos_cos = np.array([
        float((z_off_tr[hard_anchor_spec[k]] * z_off_tr[hard_pos_spec[k]]).sum())
        for k in range(len(hard_sel))], dtype=np.float32)
    safety_spec = [tr_row_to_idx[int(train[i]["anchor_row"])] for i in safety_sel]

    # ---- val eval machinery ----
    val_iks = [e["ik14"] for e in val]
    val_pmzs = [e["precursor_mz"] for e in val]
    val_adducts = [e["adduct"] for e in val]
    n_val_anchor = len(val)

    def val_margin(z):
        margins, pos_cos, neg_cos = [], [], []
        for i, e in enumerate(val):
            sib = sib_va[i]
            if sib < 0 or not e["neg"]:
                continue
            p = float(np.dot(z[i], z[sib]))
            n = max(float(np.dot(z[i], z[va_row_to_idx[int(nn["row"])]])) for nn in e["neg"])
            margins.append(p - n); pos_cos.append(p); neg_cos.append(n)
        margins = np.asarray(margins)
        return {"n": len(margins), "margin_mean": float(margins.mean()),
                "pairwise_accuracy": float((margins > 0).mean()),
                "pos_cosine_mean": float(np.mean(pos_cos)),
                "max_neg_cosine_mean": float(np.mean(neg_cos))}

    def val_retrieval(z):
        z_anchor = z[:n_val_anchor]
        rows = retrieval_per_query(z_anchor, val_iks, val_pmzs, val_adducts, a.ppm_tol)
        return aggregate_retrieval(rows)

    def val_preserve(z):
        return float(np.sum(z[:n_val_anchor] * z_off_va[:n_val_anchor], axis=1).mean())

    # ---- baseline (official head) ----
    baseline_margin = val_margin(z_off_va)
    baseline_ret = val_retrieval(z_off_va)

    device = torch.device(a.device)
    report = {"status": "m1b_head_expressibility", "baseline": {
        "margin": baseline_margin, "retrieval": baseline_ret}, "runs": []}

    h_tr_t = torch.as_tensor(h_tr, dtype=torch.float32)
    hard_anchor_t = torch.as_tensor(hard_anchor_spec, dtype=torch.long)
    hard_pos_t = torch.as_tensor(hard_pos_spec, dtype=torch.long)
    hard_neg_flat = torch.as_tensor([j for nsl in hard_neg_spec for j in nsl], dtype=torch.long)
    hard_neg_ptr = torch.cat([torch.zeros(1, dtype=torch.long),
                              torch.as_tensor([len(ns) for ns in hard_neg_spec]).cumsum(0)])
    safety_t = torch.as_tensor(safety_spec, dtype=torch.long)
    off_pos_cos_t = torch.as_tensor(off_pos_cos)

    for lam_p in a.lam_preserve_list:
        for seed in a.seeds:
            torch.manual_seed(seed)
            W = torch.as_tensor(W_off.copy(), dtype=torch.float32, device=device).clone()
            b = torch.as_tensor(b_off.copy(), dtype=torch.float32, device=device).clone()
            W = torch.nn.Parameter(W); b = torch.nn.Parameter(b)
            opt = torch.optim.Adam([W, b], lr=a.lr, weight_decay=1e-4)

            def head_fwd(idx, hh):
                return F.normalize(hh[idx.to(device)] @ W.T + b, p=2, dim=-1)

            hard_anchor_t_d = hard_anchor_t.to(device)
            hard_pos_t_d = hard_pos_t.to(device)
            hard_neg_flat_d = hard_neg_flat.to(device)
            hard_neg_ptr_d = hard_neg_ptr.to(device)
            safety_t_d = safety_t.to(device)
            off_pos_cos_d = off_pos_cos_t.to(device)
            z_off_tr_d = torch.as_tensor(z_off_tr, dtype=torch.float32, device=device)
            h_tr_d = h_tr_t.to(device)

            for _epoch in range(a.epochs):
                perm_hard = torch.randperm(len(hard_sel))
                perm_safety = torch.randperm(len(safety_sel))
                hard_batches = [perm_hard[i:i + a.batch_size] for i in range(0, len(perm_hard), a.batch_size)]
                for step, hb in enumerate(hard_batches):
                    start = (step * a.batch_size) % max(1, len(safety_sel))
                    sb = perm_safety[start:start + a.batch_size]
                    if len(sb) < a.batch_size:
                        sb = torch.cat([sb, perm_safety[: a.batch_size - len(sb)]])
                    a_idx = hard_anchor_t_d[hb]
                    p_idx = hard_pos_t_d[hb]
                    z_a = head_fwd(a_idx, h_tr_d)
                    z_p = head_fwd(p_idx, h_tr_d)
                    lo = hard_neg_ptr_d[hb]
                    hi = hard_neg_ptr_d[hb + 1]
                    z_n_flat = head_fwd(hard_neg_flat_d, h_tr_d)
                    B = len(hb)
                    s_p = (z_a * z_p).sum(1)
                    s_n = torch.empty(B, device=device)
                    preserve_n = torch.empty(B, device=device)
                    for k in range(B):
                        seg = z_n_flat[lo[k]:hi[k]]
                        cs = (z_a[k] * seg).sum(1)
                        s_n[k] = cs.max()
                        preserve_n[k] = (1.0 - (seg * z_off_tr_d[hard_neg_flat_d[lo[k]:hi[k]]]).sum(1)).mean()
                    L_rank = F.softplus((a.margin + s_n - s_p) / a.tau).mean()
                    L_posfloor = F.relu(off_pos_cos_d[hb] - s_p).mean()
                    preserve_a = (1.0 - (z_a * z_off_tr_d[a_idx]).sum(1))
                    preserve_p = (1.0 - (z_p * z_off_tr_d[p_idx]).sum(1))
                    L_preserve_hard = ((preserve_a + preserve_p + preserve_n) / 3.0).mean()
                    z_s = head_fwd(safety_t_d[sb], h_tr_d)
                    L_preserve_safety = (1.0 - (z_s * z_off_tr_d[safety_t_d[sb]]).sum(1)).mean()
                    L_preserve = (L_preserve_hard + L_preserve_safety) / 2.0
                    loss = L_rank + a.lam_posfloor * L_posfloor + lam_p * L_preserve
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            with torch.no_grad():
                W_np = W.detach().cpu().numpy().astype(np.float32)
                b_np = b.detach().cpu().numpy().astype(np.float32)
            z_val = apply_head(h_va, W_np, b_np)
            run = {"lam_preserve": lam_p, "seed": seed,
                   "margin": val_margin(z_val),
                   "retrieval": val_retrieval(z_val),
                   "preservation": val_preserve(z_val)}
            report["runs"].append(run)
            print(f"[run] lam_p={lam_p} seed={seed} margin={run['margin']['margin_mean']:.4f} "
                  f"pairwise={run['margin']['pairwise_accuracy']:.4f} "
                  f"macro_auc={run['retrieval']['macro_auc']:.4f} "
                  f"preserve={run['preservation']:.4f}", flush=True)

    out_path = a.output_dir / "expressibility_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
