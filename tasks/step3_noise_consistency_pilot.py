"""Step 3 pilot: MS2DeepScore-style noise consistency (head-only) + G3 gate.

Freeze the DreaMS backbone at the official fine-tuned weights, learn only the
1024->1024 projection head, and ask one question: can the head be nudged so that
a spectrum and its intensity-jittered variant land closer than two different
molecules -- without degrading strict-10ppm retrieval?

Noise (faithful MS2DeepScore menu, transferred to DreaMS's raw peak list before
tokenization; see docs/MS2DEEPSCORE_NOISE_METHOD_RESEARCH_20260817.md):
  1. remove weak peaks  -- randomly drop 0-20% of peaks with intensity < 0.4
                          (measured on the max-normalized scale, before jitter)
  2. intensity jitter   -- multiply each peak intensity by U[0.6, 1.4]  (+/-40%)
  3. add peaks          -- 0-10 new peaks at random m/z with intensity U[0, 0.01]

The backbone is frozen, so its precursor embeddings are computed ONCE and cached;
training then reduces to a cheap linear-head optimisation on 1024-dim vectors.

G3 gate (see plan doc §3 Step 3):
  G3-1  noise consistency UP  : mean cos(orig, jitter) - mean cos(orig, diff-mol)
                                on the eval set is higher for the trained head
                                than for the official head, and stays > 0.
  G3-2  retrieval not worse   : strict-10ppm macro ROC-AUC and Recall@1 on the
                                eval set do not drop below the official head.
  G3-3  3 seeds               : rerun with --seed 1/2/3; direction must agree.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from pilot_paired_layer_cka import preprocess_spectrum
from train_e1_identity import cpu_state_dict, load_base_model


ROOT = Path(__file__).resolve().parent.parent


def stable_seed(*parts: object) -> int:
    digest = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--base-ckpt", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt",
                        help="slim official fine-tuned checkpoint (prepare_official_embedding_checkpoint.py)")
    parser.add_argument("--architecture-ckpt", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/noise_consistency_pilot")
    parser.add_argument("--fold", type=str, default="val")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--split-seed", type=int, default=0,
                        help="fixed train/eval split seed; identical across --seed runs so orig embeddings are cacheable")
    parser.add_argument("--n-spectra", type=int, default=600,
                        help="total spectra sampled (CPU: backbone frozen but slow; ~2x this many forwards)")
    parser.add_argument("--frac", type=float, default=0.7, help="train fraction of molecules")
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--triplet-margin", type=float, default=0.05)
    parser.add_argument("--triplet-weight", type=float, default=1.0)
    parser.add_argument("--preserve-weight", type=float, default=5.0)
    # noise menu toggles (for future ablation)
    parser.add_argument("--remove-weak", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jitter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--add-peaks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jitter-range", type=float, nargs=2, default=[0.6, 1.4])
    parser.add_argument("--remove-frac", type=float, nargs=2, default=[0.0, 0.2])
    parser.add_argument("--weak-threshold", type=float, default=0.4)
    parser.add_argument("--ppm-tol", type=float, default=10.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny subset: 120 spectra, 1 epoch (fast end-to-end check)")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# 噪声（MS2DeepScore 三件套，作用于原始峰列表）
# --------------------------------------------------------------------------- #
def apply_ms2deepscore_noise(
    raw_2_n: np.ndarray,
    rng: np.random.Generator,
    remove_weak: bool,
    jitter: bool,
    add_peaks: bool,
    remove_frac: tuple[float, float],
    weak_threshold: float,
    jitter_range: tuple[float, float],
) -> np.ndarray:
    """Return a noise-augmented (2, m) spectrum from a raw (2, n) spectrum.

    Intensities are first normalised to max=1 so the weak-peak threshold (0.4)
    is on the same scale preprocess_spectrum re-normalises into. The returned
    array is m/z-sorted, matching what preprocess_spectrum / DreaMS expect.
    """
    raw = np.asarray(raw_2_n, dtype=np.float64)
    mz = raw[0].copy()
    intens = raw[1].copy()
    valid = mz > 0
    if valid.any() and intens[valid].max() > 0:
        intens[valid] = intens[valid] / intens[valid].max()

    # 1) remove weak peaks: drop a random 0-20% of peaks below the threshold
    if remove_weak:
        widx = np.where(valid & (intens < weak_threshold))[0]
        if len(widx):
            frac = rng.uniform(*remove_frac)
            n_rm = int(round(frac * len(widx)))
            n_rm = max(0, min(n_rm, len(widx)))
            rm = rng.choice(widx, size=n_rm, replace=False)
            keep = np.ones(len(mz), dtype=bool)
            keep[rm] = False
            mz, intens, valid = mz[keep], intens[keep], valid[keep]

    # 2) intensity jitter +/-40%
    if jitter:
        intens = intens * rng.uniform(*jitter_range, size=len(mz))

    # 3) add 0-10 weak peaks within the spectrum's own m/z range
    if add_peaks and valid.any():
        n_add = int(rng.integers(0, 11))
        if n_add:
            lo, hi = float(mz[valid].min()), float(mz[valid].max())
            if hi > lo:
                new_mz = rng.uniform(lo, hi, size=n_add)
                new_intens = rng.uniform(0.0, 0.01, size=n_add)
                mz = np.concatenate([mz, new_mz])
                intens = np.concatenate([intens, new_intens])

    order = np.argsort(mz, kind="stable")
    return np.stack([mz[order], intens[order]], axis=0)


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def decode_bytes(x: Any) -> str:
    return x.decode("utf-8", errors="ignore") if isinstance(x, bytes) else str(x)


def load_fold_rows(data: Path, fold: str) -> list[dict[str, Any]]:
    with h5py.File(data, "r") as handle:
        all_folds = np.array([decode_bytes(x) for x in handle["fold"][:]])
        indices = np.where(all_folds == fold)[0]
        rows = []
        for idx in indices:
            rows.append({
                "row": int(idx),
                "ik14": decode_bytes(handle["INCHIKEY"][idx])[:14],
                "precursor_mz": float(handle["precursor_mz"][idx]),
                "adduct": decode_bytes(handle["adduct"][idx]),
            })
    return rows


def sample_split(rows: list[dict[str, Any]], seed: int, n_spectra: int, frac: float):
    """Sample spectra and split molecules into train/eval (disjoint IK14)."""
    rng = np.random.default_rng(seed)
    if len(rows) > n_spectra:
        idx = rng.choice(len(rows), size=n_spectra, replace=False)
        rows = [rows[int(i)] for i in idx]
    by_ik: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_ik.setdefault(r["ik14"], []).append(r)
    iks = list(by_ik)
    rng.shuffle(iks)
    n_train_mol = max(1, int(round(len(iks) * frac)))
    train_iks, eval_iks = set(iks[:n_train_mol]), set(iks[n_train_mol:])
    train = [r for r in rows if r["ik14"] in train_iks]
    eval_ = [r for r in rows if r["ik14"] in eval_iks]
    return train, eval_


def negative_indices(ik14s: list[str], seed: int) -> list[int]:
    """One different-molecule index per anchor, sampled from the same pool.

    Returns indices into the same list, so negatives reuse the already-computed
    orig backbone embeddings (no extra forward passes).
    """
    r = np.random.default_rng(seed)
    n = len(ik14s)
    out = []
    for i in range(n):
        cand = [j for j in range(n) if ik14s[j] != ik14s[i]]
        if not cand:
            cand = list(range(n))
        out.append(int(cand[int(r.integers(len(cand)))]))
    return out


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
def backbone_embeddings(model, tokens: list[torch.Tensor], device: torch.device, batch_size: int,
                        label: str = "") -> torch.Tensor:
    model.backbone.eval()
    out = []
    n = len(tokens)
    t0 = time.time()
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            batch = torch.stack(tokens[start:start + batch_size]).to(device)
            out.append(model.backbone(batch, None)[:, 0, :].cpu())
            done = min(start + batch_size, n)
            if done % (batch_size * 5) == 0 or done == n:
                elapsed = time.time() - t0
                eta = elapsed / done * (n - done) if done else 0.0
                print(f"  [{label}] {done}/{n} spectra ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)", flush=True)
    return torch.cat(out, dim=0)


def cache_signature(args: argparse.Namespace, data_size: int) -> str:
    return hashlib.md5(
        f"v2:{args.data}:{data_size}:{args.fold}:{args.n_spectra}:{args.frac}:"
        f"{args.split_seed}:{args.n_highest_peaks}".encode()
    ).hexdigest()[:16]


def load_or_compute_orig(model, train_tokens, eval_tokens, cache_path, device, batch_size):
    """Orig backbone embeddings are seed-independent (frozen backbone + fixed split),
    so they are cached to disk and reused across --seed runs."""
    if cache_path.exists():
        try:
            data = np.load(cache_path)
            print(f"[embed] loading cached orig embeddings from {cache_path.name}", flush=True)
            return (torch.from_numpy(data["Z_train_orig"]).clone(),
                    torch.from_numpy(data["Z_eval_orig"]).clone())
        except Exception as exc:  # noqa: BLE001
            print(f"[embed] cache unreadable ({exc}), recomputing", flush=True)
    Z_tr = backbone_embeddings(model, train_tokens, device, batch_size, "train-orig")
    Z_ev = backbone_embeddings(model, eval_tokens, device, batch_size, "eval-orig")
    np.savez(cache_path, Z_train_orig=Z_tr.numpy(), Z_eval_orig=Z_ev.numpy())
    print(f"[embed] orig embeddings cached -> {cache_path.name}", flush=True)
    return Z_tr, Z_ev


# --------------------------------------------------------------------------- #
# 紧凑检索评估（10ppm，子集）
# --------------------------------------------------------------------------- #
def query_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    diff = pos[:, None] - neg[None, :]
    return float((np.count_nonzero(diff > 0) + 0.5 * np.count_nonzero(diff == 0)) / diff.size)


def retrieval_metrics(emb: np.ndarray, meta: list[dict[str, Any]], ppm_tol: float):
    """Strict-10ppm, same-adduct retrieval on the eval subset (molecule-aggregated)."""
    pmzs = np.array([m["precursor_mz"] for m in meta])
    iks = np.array([m["ik14"] for m in meta])
    adducts = np.array([m["adduct"] for m in meta])
    aucs, recalls1, mrrs = [], [], []
    n_eligible = 0
    for qi in range(len(meta)):
        ppm_da = ppm_tol * 1e-6 * pmzs[qi]
        cand = (np.abs(pmzs - pmzs[qi]) <= ppm_da) & (np.arange(len(meta)) != qi) & (adducts == adducts[qi])
        idx = np.where(cand)[0]
        if len(idx) == 0:
            continue
        labels = (iks[idx] == iks[qi]).astype(int)
        if labels.sum() == 0 or (labels == 0).sum() == 0:
            continue
        scores = (emb[qi:qi + 1] * emb[idx]).sum(axis=1)
        aucs.append(query_auc(labels, scores))
        # molecule-aggregated rank
        best = {}
        for j, s in zip(idx, scores):
            ik = iks[j]
            if ik not in best or s > best[ik]:
                best[ik] = float(s)
        order = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        ranks = [ik for ik, _ in order]
        rank = ranks.index(iks[qi]) + 1 if iks[qi] in ranks else len(ranks) + 1
        recalls1.append(1.0 if rank <= 1 else 0.0)
        mrrs.append(1.0 / rank)
        n_eligible += 1
    return {
        "n_queries": n_eligible,
        "macro_auc": float(np.mean(aucs)) if aucs else 0.5,
        "recall1": float(np.mean(recalls1)) if recalls1 else 0.0,
        "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    if args.smoke:
        args.n_spectra = 120
        args.epochs = 1
    seed_everything(args.seed)
    device = torch.device(args.device)
    run_dir = args.output_dir / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pilot] device={device} seed={args.seed} n_spectra={args.n_spectra}", flush=True)

    # ---- load model (backbone frozen + official head) ----
    model, kind = load_base_model(args.base_ckpt, args.architecture_ckpt, device, args.n_highest_peaks)
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True

    # ---- sample + split ----
    rows = load_fold_rows(args.data, args.fold)
    train_rows, eval_rows = sample_split(rows, args.split_seed, args.n_spectra, args.frac)
    print(f"[data] {len(rows)} fold spectra -> train {len(train_rows)} / eval {len(eval_rows)} "
          f"(molecule-disjoint)", flush=True)

    # ---- tokenise (original + jittered) ----
    rng = np.random.default_rng(stable_seed(args.seed, "noise"))
    with h5py.File(args.data, "r") as handle:
        def tokens(row_dict: dict[str, Any], noisy: bool) -> torch.Tensor:
            raw = np.asarray(handle["spectrum"][row_dict["row"]])
            if noisy:
                raw = apply_ms2deepscore_noise(
                    raw, rng, args.remove_weak, args.jitter, args.add_peaks,
                    tuple(args.remove_frac), args.weak_threshold, tuple(args.jitter_range),
                )
            return preprocess_spectrum(raw, row_dict["precursor_mz"], args.n_highest_peaks)

        train_orig = [tokens(r, False) for r in train_rows]
        train_jitt = [tokens(r, True) for r in train_rows]
        eval_orig = [tokens(r, False) for r in eval_rows]
        eval_jitt = [tokens(r, True) for r in eval_rows]

    # ---- negatives reuse the orig pool (indices -> no extra forward passes) ----
    train_neg_idx = negative_indices([r["ik14"] for r in train_rows], stable_seed(args.seed, "train_neg"))
    eval_neg_idx = negative_indices([r["ik14"] for r in eval_rows], stable_seed(args.seed, "eval_neg"))

    # ---- frozen backbone embeddings ----
    # orig is seed-independent (frozen backbone + fixed split) -> cached on disk;
    # jitter depends on the noise seed -> computed fresh each run.
    sig = cache_signature(args, args.data.stat().st_size)
    cache_path = args.output_dir / f"orig_embeddings_{sig}.npz"
    print("[embed] precomputing frozen backbone embeddings (orig cached, jitter fresh)...", flush=True)
    t0 = time.time()
    Z_train_orig, Z_eval_orig = load_or_compute_orig(model, train_orig, eval_orig, cache_path, device, args.batch_size)
    Z_train_jitt = backbone_embeddings(model, train_jitt, device, args.batch_size, "train-jitter")
    Z_eval_jitt = backbone_embeddings(model, eval_jitt, device, args.batch_size, "eval-jitter")
    Z_train_neg = Z_train_orig[train_neg_idx]
    Z_eval_neg = Z_eval_orig[eval_neg_idx]
    print(f"[embed] done in {time.time() - t0:.1f}s", flush=True)

    # official-head embeddings = baseline + teachers
    with torch.no_grad():
        official = lambda z: F.normalize(model.head(z.to(device)), dim=-1).cpu()
        B_eval_orig = official(Z_eval_orig)
        B_eval_jitt = official(Z_eval_jitt)
        B_eval_neg = official(Z_eval_neg)
        T_q = official(Z_train_orig)   # teacher for anchor
        T_n = official(Z_train_neg)    # teacher for negative

    # ---- train head (linear 1024->1024) ----
    print("[train] head-only...", flush=True)
    head = model.head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    Zq = Z_train_orig.to(device); Zp = Z_train_jitt.to(device); Zn = Z_train_neg.to(device)
    Tq = T_q.to(device); Tn = T_n.to(device)
    n_train = len(Zq)
    history = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        perm = torch.randperm(n_train)
        losses, trips, pres = [], [], []
        for start in range(0, n_train, args.batch_size):
            idx = perm[start:start + args.batch_size]
            q = F.normalize(head(Zq[idx]), dim=-1)
            p = F.normalize(head(Zp[idx]), dim=-1)
            n = F.normalize(head(Zn[idx]), dim=-1)
            pos_cos = (q * p).sum(1)
            neg_cos = (q * n).sum(1)
            triplet = F.relu(args.triplet_margin - (pos_cos - neg_cos)).mean()
            preserve = ((1 - (q * Tq[idx]).sum(1)).mean() + (1 - (n * Tn[idx]).sum(1)).mean()) / 2
            loss = args.triplet_weight * triplet + args.preserve_weight * preserve
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach())); trips.append(float(triplet.detach())); pres.append(float(preserve.detach()))
        history.append({
            "epoch": epoch, "loss": float(np.mean(losses)),
            "triplet": float(np.mean(trips)), "preserve": float(np.mean(pres)),
        })
        print(f"  epoch {epoch:02d} loss={history[-1]['loss']:.4f} "
              f"triplet={history[-1]['triplet']:.4f} preserve={history[-1]['preserve']:.4f}", flush=True)

    # ---- G3 eval ----
    print("[eval] G3 gate...", flush=True)
    with torch.no_grad():
        head.eval()
        T_eval_orig = F.normalize(head(Z_eval_orig.to(device)), dim=-1).cpu()
        T_eval_jitt = F.normalize(head(Z_eval_jitt.to(device)), dim=-1).cpu()
        T_eval_neg = F.normalize(head(Z_eval_neg.to(device)), dim=-1).cpu()

    def noise_consistency(orig, jitt, neg):
        pos = (orig * jitt).sum(1).numpy()
        negc = (orig * neg).sum(1).numpy()
        return float(pos.mean()), float(negc.mean()), float(pos.mean() - negc.mean())

    base_pos, base_neg, base_sep = noise_consistency(B_eval_orig, B_eval_jitt, B_eval_neg)
    trn_pos, trn_neg, trn_sep = noise_consistency(T_eval_orig, T_eval_jitt, T_eval_neg)

    ret_base = retrieval_metrics(B_eval_orig.numpy(), eval_rows, args.ppm_tol)
    ret_train = retrieval_metrics(T_eval_orig.numpy(), eval_rows, args.ppm_tol)

    g3_1 = trn_sep > base_sep and trn_sep > 0
    g3_2 = ret_train["macro_auc"] >= ret_base["macro_auc"] - 0.01 and ret_train["recall1"] >= ret_base["recall1"] - 0.01

    summary = {
        "status": "step3_noise_consistency_pilot",
        "seed": args.seed, "kind": kind,
        "n_spectra": args.n_spectra, "n_train": len(train_rows), "n_eval": len(eval_rows),
        "config": {
            "lr": args.lr, "epochs": args.epochs, "triplet_margin": args.triplet_margin,
            "triplet_weight": args.triplet_weight, "preserve_weight": args.preserve_weight,
            "jitter_range": args.jitter_range, "remove_frac": args.remove_frac,
            "weak_threshold": args.weak_threshold,
        },
        "noise_consistency": {
            "baseline": {"pos": base_pos, "neg": base_neg, "separation": base_sep},
            "trained": {"pos": trn_pos, "neg": trn_neg, "separation": trn_sep},
        },
        "retrieval": {
            "baseline": ret_base, "trained": ret_train,
        },
        "g3_1_noise_consistency_up": bool(g3_1),
        "g3_2_retrieval_not_worse": bool(g3_2),
        "gate_overall_pass": bool(g3_1 and g3_2),
        "training_history": history,
    }

    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    checkpoint = {
        "format": "e1_identity_v1", "stage": "noise_consistency", "seed": args.seed,
        "architecture_checkpoint": str(args.architecture_ckpt.resolve()),
        "base_checkpoint": str(args.base_ckpt.resolve()),
        "backbone_state_dict": cpu_state_dict(model.backbone),
        "head_state_dict": cpu_state_dict(model.head),
        "config": {"n_highest_peaks": args.n_highest_peaks},
    }
    torch.save(checkpoint, run_dir / "best_noise_consistency.pt")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    del model
    gc.collect()


if __name__ == "__main__":
    main()
