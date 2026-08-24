"""Fix the 157/314 same-anchor ranking bug in the PSD probe.

Bug: pair_samples anchored each pos pair only at i where sib[i] > i, so the
same-anchor metric saw ~half the anchors.  All pair features (cosine, hadamard,
absdiff, lowrank PSD) are SYMMETRIC, so one pos pair's score can be assigned to
BOTH anchors.  Recompute the same-anchor pairwise accuracy over the full 314
hard anchors, retraining the probes from the cached pair arrays (no backbone
re-embed, no backbone retrain).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from eval_g8r_d04_psd_probe import (  # noqa: E402
    build_sibling, fit_hadamard_free, fit_hadamard_nn, fit_lowrank_psd,
)
from step5_gate_eval import query_auc  # noqa: E402

DEFAULT_VAL = ROOT / "tasks/massspecgym_isomers/g8r_locked/val.json"
DEFAULT_CACHE = ROOT / "data/validation/g8r_d04_psd_probe_cache.npz"
DEFAULT_OUT = ROOT / "data/validation/g8r_d04_psd_probe_314.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--val", type=Path, default=DEFAULT_VAL)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", default="cpu")
    p.add_argument("--probe-steps", type=int, default=800)
    p.add_argument("--seed", type=int, default=20260821)
    return p.parse_args()


def same_anchor_from_pairs(scores, yva, an_va, sibling):
    """Assign symmetric pair scores to both anchors, then per-anchor pos vs max neg."""
    pos_score: dict[int, float] = {}
    neg_scores: dict[int, list[float]] = defaultdict(list)
    for s, y, a in zip(scores, yva, an_va):
        a = int(a)
        if y == 1:
            sib = sibling[a]
            # symmetric feature => same score for both anchors
            pos_score[a] = float(s)
            if sib >= 0:
                pos_score[sib] = float(s)
        else:
            neg_scores[a].append(float(s))
    acc, cnt = 0, 0
    near_acc = near_cnt = 0
    mid_acc = mid_cnt = 0
    for a, ps in pos_score.items():
        if a not in neg_scores:
            continue
        ok = int(ps > max(neg_scores[a]))
        acc += ok; cnt += 1
        # grade assignment: near if any neg has grade near, else mid
        # (grades are not in neg_scores; use a parallel grade map from yva/g_va)
    return acc / cnt if cnt else float("nan"), cnt


def same_anchor_full(scores, yva, an_va, g_va, sibling):
    pos_score: dict[int, float] = {}
    neg_scores: dict[int, list[float]] = defaultdict(list)
    neg_near: dict[int, bool] = {}
    for s, y, a, g in zip(scores, yva, an_va, g_va):
        a = int(a)
        if y == 1:
            pos_score[a] = float(s)
            if sibling[a] >= 0:
                pos_score[sibling[a]] = float(s)
        else:
            neg_scores[a].append(float(s))
            if g == "near":
                neg_near[a] = True
    acc = cnt = near_acc = near_cnt = mid_acc = mid_cnt = 0
    for a, ps in pos_score.items():
        if a not in neg_scores:
            continue
        ok = int(ps > max(neg_scores[a]))
        acc += ok; cnt += 1
        if neg_near.get(a, False):
            near_acc += ok; near_cnt += 1
        else:
            mid_acc += ok; mid_cnt += 1
    return {
        "same_anchor_pairwise_accuracy": float(acc / cnt) if cnt else float("nan"),
        "same_anchor_n": cnt,
        "near_pairwise_accuracy": float(near_acc / near_cnt) if near_cnt else float("nan"),
        "near_n": near_cnt,
        "mid_pairwise_accuracy": float(mid_acc / mid_cnt) if mid_cnt else float("nan"),
        "mid_n": mid_cnt,
    }


def main() -> None:
    a = parse_args()
    cache = np.load(a.cache, allow_pickle=True)
    Atr, Btr, ytr = cache["Atr"], cache["Btr"], cache["ytr"]
    Ava, Bva, yva = cache["Ava"], cache["Bva"], cache["yva"]
    cos_va = cache["cos_va"]
    an_va = cache["an_va"]
    g_va = cache["g_va"]
    Htr, Hva = Atr * Btr, Ava * Bva

    val = json.loads(a.val.read_text(encoding="utf-8"))["entries"]
    sibling = build_sibling(val)

    report = {"status": "g8r_d04_psd_probe_314", "n_val_pairs": int(len(yva)),
              "n_val_pos": int((yva == 1).sum()), "n_val_neg": int((yva == 0).sum())}

    # cosine baseline (scores already in cache)
    report["cosine"] = same_anchor_full(cos_va, yva, an_va, g_va, sibling)

    # retrain probes from cached pair arrays
    def run(fn, *args):
        sc = fn(*args, a.seed)
        return same_anchor_full(sc, yva, an_va, g_va, sibling)

    report["hadamard_free"] = run(
        lambda s: fit_hadamard_free(Htr, ytr, Hva, a.probe_steps, s, a.device))
    report["hadamard_nn"] = run(
        lambda s: fit_hadamard_nn(Htr, ytr, Hva, a.probe_steps, s, a.device))
    for r in (32, 128, 256):
        report[f"lowrank_psd_r{r}"] = run(
            lambda s, r=r: fit_lowrank_psd(Atr, Btr, ytr, Ava, Bva, r, a.probe_steps, s, a.device))

    # also global AUC + near/mid AUC for completeness (same as before, from scores)
    def aucs(sc):
        out = {"auc": float(query_auc(yva, sc))}
        for g in ("near", "mid"):
            m = (g_va == "pos") | (g_va == g)
            if m.sum() and (yva[m] == 0).sum() and (yva[m] == 1).sum():
                out[f"auc_{g}"] = float(query_auc(yva[m], sc[m]))
        return out

    report["global_auc"] = {"cosine": aucs(cos_va)}
    report["global_auc"]["hadamard_free"] = aucs(
        fit_hadamard_free(Htr, ytr, Hva, a.probe_steps, a.seed, a.device))
    report["global_auc"]["hadamard_nn"] = aucs(
        fit_hadamard_nn(Htr, ytr, Hva, a.probe_steps, a.seed, a.device))
    for r in (32, 128, 256):
        report["global_auc"][f"lowrank_psd_r{r}"] = aucs(
            fit_lowrank_psd(Atr, Btr, ytr, Ava, Bva, r, a.probe_steps, a.seed, a.device))

    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved: {a.output}")


if __name__ == "__main__":
    main()
