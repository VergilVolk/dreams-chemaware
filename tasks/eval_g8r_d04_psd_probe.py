"""D0.4 rigor check #3: PSD-constrained metric probes on the FROZEN embedding.

Decisive question (docs/G8R_M1_POSTMORTEM_AND_M1B_DECISION_20260822.md §9 D0.4
items 1 and 3): the linear projection head z = normalize(W h + b) induces a PSD
metric (W^T W).  Does the frozen embedding's hard-negative separability survive
under PSD constraints?  This is DIAGNOSTIC ONLY — it has no veto power over the
unfreeze decision (see the 2026-08-22 audit conclusion).

Probes (fit on --train, evaluated on --val; IK14-isolated by construction):
  cosine         : global score z_a . z_b                          (baseline)
  hadamard_free  : unconstrained w (+bias) on z_a (.) z_b          (non-PSD upper bound)
  hadamard_nn    : w >= 0 (no bias) on z_a (.) z_b                 (diagonal PSD)
  lowrank_psd    : s = z_a^T (U U^T) z_b, U in R^{d x r}           (rank-r PSD)

Per the audit Step-1 list, each probe reports:
  - global AUC on val (query_auc);
  - AUC stratified by hard-negative grade (near vs mid);
  - same-anchor pairwise ranking accuracy (pos score > max neg score), matching
    the M1b gate's pairwise-accuracy metric;
  - IK14-cluster bootstrap 95% CI on the global AUC (cosine only; others report
    multi-seed mean/std);
  - multi-seed mean over --seeds.

Embeddings + pair arrays are cached to --cache so rank/seed sweeps do not re-run
the 116M backbone.  Read-only: no checkpoint written, no weight modified.
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
from step5_gate_eval import embed, query_auc  # noqa: E402

DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_TRAIN = ROOT / "tasks/massspecgym_isomers/g8r_locked/train.json"
DEFAULT_VAL = ROOT / "tasks/massspecgym_isomers/g8r_locked/val.json"
DEFAULT_BASE = ROOT / "data/e1/official_embedding_slim.pt"
DEFAULT_ARCH = ROOT / "dreams/models/pretrained/ssl_model_server.pt"
DEFAULT_OUT = ROOT / "data/validation/g8r_d04_psd_probe.json"
DEFAULT_CACHE = ROOT / "data/validation/g8r_d04_psd_probe_cache.npz"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--val", type=Path, default=DEFAULT_VAL)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--base-ckpt", type=Path, default=DEFAULT_BASE)
    p.add_argument("--architecture-ckpt", type=Path, default=DEFAULT_ARCH)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-highest-peaks", type=int, default=100)
    p.add_argument("--probe-steps", type=int, default=800)
    p.add_argument("--rank-list", type=int, nargs="+", default=[32, 128, 256],
                   help="low-rank PSD ranks to sweep")
    p.add_argument("--seeds", type=int, nargs="+", default=[20260821, 20260822, 20260823],
                   help="probe-fitting seeds")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                   help="npz cache of embeddings + pair arrays")
    p.add_argument("--smoke", action="store_true", help="cap pairs for a fast mechanism check")
    p.add_argument("--max-anchors", type=int, default=0, help="0=all; >0 smoke cap per subset")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def build_sibling(entries: list[dict]) -> list[int]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, e in enumerate(entries):
        groups[(e["ik14"], e["adduct"])].append(i)
    sib = [-1] * len(entries)
    for rows in groups.values():
        if len(rows) == 2:
            sib[rows[0]] = rows[1]
            sib[rows[1]] = rows[0]
        else:
            for a, b in zip(rows, rows[1:]):
                sib[a] = b
                sib[b] = a
    return sib


def embed_subset(entries, h, pmz_all, base_model, device, n_highest, batch_size):
    anchor_rows = [int(e["anchor_row"]) for e in entries]
    anchor_set = set(anchor_rows)
    neg_rows = sorted({int(n["row"]) for e in entries for n in e["neg"]} - anchor_set)
    all_rows = anchor_rows + neg_rows
    specs = [preprocess_spectrum(np.asarray(h["spectrum"][r]), float(pmz_all[r]), n_highest)
             for r in all_rows]
    z = embed(base_model, specs, device, batch_size).numpy()
    row_to_index = {r: i for i, r in enumerate(all_rows)}
    return z, row_to_index


def pair_samples(entries, z, sibling, row_to_index):
    A, B, Y, COS, GRADE, ANCHOR, IK = [], [], [], [], [], [], []
    for i, e in enumerate(entries):
        sib = sibling[i]
        if sib is None:
            continue
        if sib > i:
            A.append(z[i]); B.append(z[sib]); Y.append(1.0); COS.append(float(z[i] @ z[sib]))
            GRADE.append("pos"); ANCHOR.append(i); IK.append(e["ik14"])
        for nn in e["neg"]:
            j = row_to_index[int(nn["row"])]
            A.append(z[i]); B.append(z[j]); Y.append(0.0); COS.append(float(z[i] @ z[j]))
            GRADE.append(nn.get("grade", "mid")); ANCHOR.append(i); IK.append(e["ik14"])
    return (np.asarray(A, dtype=np.float32), np.asarray(B, dtype=np.float32),
            np.asarray(Y), np.asarray(COS), np.asarray(GRADE),
            np.asarray(ANCHOR), np.asarray(IK))


def load_or_build_cache(a, device):
    if a.cache.exists() and not a.smoke:
        d = np.load(a.cache, allow_pickle=True)
        out = {k: d[k] for k in d.files}
        if int(out.get("n_train_pairs", 0)) >= 1000:
            print(f"[cache] loaded {a.cache}", flush=True)
            return out
    train = json.loads(a.train.read_text(encoding="utf-8"))["entries"]
    val = json.loads(a.val.read_text(encoding="utf-8"))["entries"]
    if a.max_anchors > 0:
        train = train[: a.max_anchors]
        val = val[: a.max_anchors]
    with h5py.File(a.data, "r") as h:
        pmz_all = np.asarray(h["precursor_mz"][:], dtype=float)
        base_model, _ = load_base_model(a.base_ckpt, a.architecture_ckpt, device, a.n_highest_peaks)
        base_model.eval()
        z_tr, rti_tr = embed_subset(train, h, pmz_all, base_model, device,
                                    a.n_highest_peaks, a.batch_size)
        z_va, rti_va = embed_subset(val, h, pmz_all, base_model, device,
                                    a.n_highest_peaks, a.batch_size)
    sib_tr = build_sibling(train)
    sib_va = build_sibling(val)
    Atr, Btr, ytr, cos_tr, g_tr, an_tr, ik_tr = pair_samples(train, z_tr, sib_tr, rti_tr)
    Ava, Bva, yva, cos_va, g_va, an_va, ik_va = pair_samples(val, z_va, sib_va, rti_va)
    out = {
        "Atr": Atr, "Btr": Btr, "ytr": ytr, "cos_tr": cos_tr,
        "g_tr": g_tr, "an_tr": an_tr, "ik_tr": ik_tr,
        "Ava": Ava, "Bva": Bva, "yva": yva, "cos_va": cos_va,
        "g_va": g_va, "an_va": an_va, "ik_va": ik_va,
        "n_train_pairs": int(len(ytr)), "n_val_pairs": int(len(yva)),
    }
    if not a.smoke:
        np.savez_compressed(a.cache, **out)
        print(f"[cache] saved {a.cache}", flush=True)
    return out


def fit_hadamard_free(Htr, ytr, Hte, steps, seed, device):
    torch.manual_seed(seed)
    lin = torch.nn.Linear(Htr.shape[1], 1).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-3)
    Htr_t = torch.as_tensor(Htr, dtype=torch.float32, device=device)
    ytr_t = torch.as_tensor(ytr, dtype=torch.float32, device=device)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(lin(Htr_t).squeeze(1), ytr_t)
        loss.backward(); opt.step()
    with torch.no_grad():
        s = lin(torch.as_tensor(Hte, dtype=torch.float32, device=device)).squeeze(1).cpu().numpy()
    return s


def fit_hadamard_nn(Htr, ytr, Hte, steps, seed, device):
    torch.manual_seed(seed)
    w = torch.zeros(Htr.shape[1], device=device)
    w = torch.nn.Parameter(w)
    opt = torch.optim.Adam([w], lr=1e-2)
    Htr_t = torch.as_tensor(Htr, dtype=torch.float32, device=device)
    ytr_t = torch.as_tensor(ytr, dtype=torch.float32, device=device)
    for _ in range(steps):
        opt.zero_grad()
        s = Htr_t @ w
        loss = F.binary_cross_entropy_with_logits(s, ytr_t)
        loss.backward(); opt.step()
        with torch.no_grad():
            w.clamp_(min=0.0)
    with torch.no_grad():
        s = (torch.as_tensor(Hte, dtype=torch.float32, device=device) @ w).cpu().numpy()
    return s


def fit_lowrank_psd(Atr, Btr, ytr, Ate, Bte, r, steps, seed, device):
    torch.manual_seed(seed)
    d = Atr.shape[1]
    U = torch.randn(d, r, device=device) * (1.0 / (d ** 0.5))
    U = torch.nn.Parameter(U)
    opt = torch.optim.Adam([U], lr=1e-2)
    Atr_t = torch.as_tensor(Atr, dtype=torch.float32, device=device)
    Btr_t = torch.as_tensor(Btr, dtype=torch.float32, device=device)
    ytr_t = torch.as_tensor(ytr, dtype=torch.float32, device=device)
    for _ in range(steps):
        opt.zero_grad()
        s = ((Atr_t @ U) * (Btr_t @ U)).sum(1)
        loss = F.binary_cross_entropy_with_logits(s, ytr_t)
        loss.backward(); opt.step()
    with torch.no_grad():
        Ate_t = torch.as_tensor(Ate, dtype=torch.float32, device=device)
        Bte_t = torch.as_tensor(Bte, dtype=torch.float32, device=device)
        s = ((Ate_t @ U) * (Bte_t @ U)).sum(1).cpu().numpy()
    return s


def same_anchor_ranking(scores, Y, ANCHOR):
    pos_score: dict[int, float] = {}
    neg_scores: dict[int, list[float]] = defaultdict(list)
    for s, y, a in zip(scores, Y, ANCHOR):
        if y == 1:
            pos_score[int(a)] = float(s)
        else:
            neg_scores[int(a)].append(float(s))
    acc, cnt = 0, 0
    for a, ps in pos_score.items():
        if a in neg_scores:
            acc += int(ps > max(neg_scores[a]))
            cnt += 1
    return (acc / cnt) if cnt else float("nan"), cnt


def cluster_bootstrap_auc(Y, scores, IK, n_boot, seed):
    by_ik = defaultdict(list)
    for y, s, ik in zip(Y, scores, IK):
        by_ik[str(ik)].append((float(y), float(s)))
    keys = list(by_ik)
    rng = np.random.default_rng(seed)
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(keys), len(keys))
        ys, ss = [], []
        for kk in idx:
            for y, s in by_ik[keys[kk]]:
                ys.append(y); ss.append(s)
        aucs[b] = query_auc(np.asarray(ys), np.asarray(ss))
    return {"mean": float(np.mean(aucs)), "ci_low": float(np.percentile(aucs, 2.5)),
            "ci_high": float(np.percentile(aucs, 97.5))}


def evaluate_probe(scores, Y, GRADE, ANCHOR, IK, n_boot, seed):
    out = {"auc": float(query_auc(Y, scores))}
    out["auc_cluster_boot"] = cluster_bootstrap_auc(Y, scores, IK, n_boot, seed)
    for g in ("near", "mid"):
        m = (GRADE == "pos") | (GRADE == g)
        if m.sum() and (Y[m] == 0).sum() and (Y[m] == 1).sum():
            out[f"auc_{g}"] = float(query_auc(Y[m], scores[m]))
    acc, cnt = same_anchor_ranking(scores, Y, ANCHOR)
    out["same_anchor_pairwise_accuracy"] = float(acc)
    out["same_anchor_n"] = int(cnt)
    return out


def main() -> None:
    a = parse_args()
    device = torch.device(a.device)
    d = load_or_build_cache(a, device)

    Atr, Btr, ytr = d["Atr"], d["Btr"], d["ytr"]
    Ava, Bva, yva = d["Ava"], d["Bva"], d["yva"]
    cos_va, g_va, an_va, ik_va = d["cos_va"], d["g_va"], d["an_va"], d["ik_va"]
    Htr, Hva = Atr * Btr, Ava * Bva

    if a.smoke:
        cap = 2000
        Atr, Btr, ytr, Htr = Atr[:cap], Btr[:cap], ytr[:cap], Htr[:cap]
        Ava, Bva, yva, cos_va = Ava[:cap], Bva[:cap], yva[:cap], cos_va[:cap]
        Hva = Hva[:cap]; g_va = g_va[:cap]; an_va = an_va[:cap]; ik_va = ik_va[:cap]

    report = {
        "status": "g8r_d04_psd_probe",
        "n_train_pairs": int(len(ytr)), "n_val_pairs": int(len(yva)),
        "n_train_pos": int((ytr == 1).sum()), "n_train_neg": int((ytr == 0).sum()),
        "n_val_pos": int((yva == 1).sum()), "n_val_neg": int((yva == 0).sum()),
        "rank_list": a.rank_list, "seeds": a.seeds, "bootstrap": a.bootstrap,
    }

    report["cosine"] = evaluate_probe(cos_va, yva, g_va, an_va, ik_va, a.bootstrap, a.seeds[0])

    def run_multi_seed(fn):
        aucs, near_aucs, mid_aucs, ranks = [], [], [], []
        for s in a.seeds:
            sc = fn(s)
            aucs.append(float(query_auc(yva, sc)))
            for g in ("near", "mid"):
                m = (g_va == "pos") | (g_va == g)
                if m.sum() and (yva[m] == 0).sum() and (yva[m] == 1).sum():
                    (near_aucs if g == "near" else mid_aucs).append(float(query_auc(yva[m], sc[m])))
            acc, _ = same_anchor_ranking(sc, yva, an_va)
            ranks.append(acc)
        return {"auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
                "auc_near_mean": float(np.mean(near_aucs)) if near_aucs else float("nan"),
                "auc_mid_mean": float(np.mean(mid_aucs)) if mid_aucs else float("nan"),
                "same_anchor_accuracy_mean": float(np.mean(ranks))}

    report["hadamard_free"] = run_multi_seed(
        lambda s: fit_hadamard_free(Htr, ytr, Hva, a.probe_steps, s, device))
    report["hadamard_nn"] = run_multi_seed(
        lambda s: fit_hadamard_nn(Htr, ytr, Hva, a.probe_steps, s, device))
    for r in a.rank_list:
        report[f"lowrank_psd_r{r}"] = run_multi_seed(
            lambda s, r=r: fit_lowrank_psd(Atr, Btr, ytr, Ava, Bva, r, a.probe_steps, s, device))

    c = report["cosine"]["auc"]
    report["gain_over_cosine"] = {
        "hadamard_free": report["hadamard_free"]["auc_mean"] - c,
        "hadamard_nn": report["hadamard_nn"]["auc_mean"] - c,
    }
    for r in a.rank_list:
        report["gain_over_cosine"][f"lowrank_psd_r{r}"] = report[f"lowrank_psd_r{r}"]["auc_mean"] - c

    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved: {a.output}")


if __name__ == "__main__":
    main()
