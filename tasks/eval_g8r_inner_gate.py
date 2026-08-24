"""Paired inner validation for the corrected G8R noise fine-tuning protocol.

The evaluator uses a locked, IK14-disjoint validation subset and compares a
candidate checkpoint with the official DreaMS initialization on the identical
spectra.  It implements the full M1 gate of
docs/DREAMS_ERROR_GUIDED_FINETUNING_MASTER_PLAN_20260821.md §6 (six criteria):

1. true cross-condition same-molecule cosine (must not fall);
2. explicit local hard-negative cosine (must fall — measured over ALL locked
   hard-negative spectra, not just the anchor-anchor subset);
3. strict 10 ppm, same-adduct molecular retrieval macro-AUC (non-inferior);
4. strict 10 ppm Recall@1 (non-inferior);
5. error transition: corrected >= introduced errors;
6. embedding preservation cosine vs official (>= 0.995).

It also reports query-level bootstrap (>= 2000) CIs for macro-AUC / Recall@1 and
pair-level paired-bootstrap CIs for the two cosine gates, per §8 rule 2.

It is deliberately separate from training.  Training loss is not a selection
metric and the historical G5--G7 runs showed why.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tasks"))

from train_e1_identity import load_base_model, preprocess_spectrum  # noqa: E402
from step5_gate_eval import embed, load_trained, query_auc  # noqa: E402

DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_VAL = ROOT / "tasks/massspecgym_isomers/g8r_locked/val.json"
DEFAULT_BASE = ROOT / "data/e1/official_embedding_slim.pt"
DEFAULT_ARCH = ROOT / "dreams/models/pretrained/ssl_model_server.pt"

# Pre-registered non-inferiority margins (master plan §8 rule 7 — do not loosen
# after seeing results).
MACRO_AUC_MARGIN = 0.005
RECALL1_MARGIN = 0.003
PRESERVATION_FLOOR = 0.995


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trained", type=Path, required=True)
    p.add_argument("--val-subset", type=Path, default=DEFAULT_VAL)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--base-ckpt", type=Path, default=DEFAULT_BASE)
    p.add_argument("--architecture-ckpt", type=Path, default=DEFAULT_ARCH)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-highest-peaks", type=int, default=100)
    p.add_argument("--ppm-tol", type=float, default=10.0)
    p.add_argument("--bootstrap", type=int, default=2000, help="paired bootstrap resamples (>= 2000 per §8)")
    p.add_argument("--seed", type=int, default=20260821, help="bootstrap RNG seed")
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def cross_pairs(entries: list[dict], instrument: np.ndarray, ce: np.ndarray) -> list[tuple[int, int]]:
    """(anchor_i, anchor_j) indices into `entries`; one pair per identity-adduct."""
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, e in enumerate(entries):
        groups[(e["ik14"], e["adduct"])].append(i)
    pairs = []
    for rows in groups.values():
        found = False
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                ra, rb = int(entries[rows[a]]["anchor_row"]), int(entries[rows[b]]["anchor_row"])
                if instrument[ra] != instrument[rb] or (
                    np.isfinite(ce[ra]) and np.isfinite(ce[rb]) and abs(ce[ra] - ce[rb]) >= 10
                ):
                    pairs.append((rows[a], rows[b])); found = True; break
            if found:
                break
    return pairs


def hard_pairs_full(entries: list[dict]) -> list[tuple[int, int]]:
    """(anchor_i, neg_row) for EVERY locked hard-negative pair (not just the
    anchor-anchor subset).  The neg row is resolved to an embedding index by the
    caller via row_to_index."""
    return [(i, int(n["row"])) for i, e in enumerate(entries) for n in e["neg"]]


def pair_cos_values(emb: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    """Per-pair cosine values (not just the mean) so the caller can bootstrap."""
    return np.array([float(np.dot(emb[i], emb[j])) for i, j in pairs], dtype=float)


def retrieval_per_query(emb: np.ndarray, iks: list[str], pmzs: list[float],
                        adducts: list[str], ppm_tol: float) -> list[dict]:
    """Per-query strict-10ppm same-adduct retrieval (molecule aggregation),
    returning one dict per valid query so bootstrap + error transition reuse the
    same alignable rows."""
    pmzs = np.asarray(pmzs); iks = np.asarray(iks); adducts = np.asarray(adducts)
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
        scores = (emb[qi:qi + 1] * emb[idx]).sum(axis=1)
        best: dict[str, float] = {}
        for j, s in zip(idx, scores):
            ik = iks[j]
            if ik not in best or s > best[ik]:
                best[ik] = float(s)
        order = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        ranks = [ik for ik, _ in order]
        top1_correct = bool(ranks and ranks[0] == iks[qi])
        rows.append({
            "query_index": int(qi),
            "auc": query_auc(labels, scores),
            "recall1": 1.0 if top1_correct else 0.0,
            "mrr": 1.0 / (ranks.index(iks[qi]) + 1) if iks[qi] in ranks else 0.0,
            "top1_correct": top1_correct,
        })
    return rows


def aggregate_retrieval(rows: list[dict]) -> dict:
    if not rows:
        return {"n_queries": 0, "macro_auc": 0.5, "recall1": 0.0, "mrr": 0.0}
    return {
        "n_queries": len(rows),
        "macro_auc": float(np.mean([r["auc"] for r in rows])),
        "recall1": float(np.mean([r["recall1"] for r in rows])),
        "mrr": float(np.mean([r["mrr"] for r in rows])),
    }


def paired_bootstrap_delta(base_vals: np.ndarray, cand_vals: np.ndarray,
                           rng: np.random.Generator, n_boot: int) -> dict:
    """Paired bootstrap of (cand - base) over aligned units (queries or pairs)."""
    base_vals = np.asarray(base_vals); cand_vals = np.asarray(cand_vals)
    n = len(base_vals)
    mean_delta = float(cand_vals.mean() - base_vals.mean())
    if n == 0:
        return {"mean_delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    deltas = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[b] = float(cand_vals[idx].mean() - base_vals[idx].mean())
    return {
        "mean_delta": mean_delta,
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
    }


def error_transition(base_rows: list[dict], cand_rows: list[dict]) -> dict:
    """baseline wrong->candidate right (corrected) vs right->wrong (introduced)."""
    corrected = introduced = persistent_wrong = persistent_right = 0
    for b, c in zip(base_rows, cand_rows):
        if b["top1_correct"] and c["top1_correct"]:
            persistent_right += 1
        elif b["top1_correct"] and not c["top1_correct"]:
            introduced += 1
        elif not b["top1_correct"] and c["top1_correct"]:
            corrected += 1
        else:
            persistent_wrong += 1
    return {
        "corrected": corrected,
        "introduced": introduced,
        "persistent_wrong": persistent_wrong,
        "persistent_right": persistent_right,
    }


def preservation_cosine(base_emb: np.ndarray, cand_emb: np.ndarray) -> float:
    """Mean diagonal cosine between candidate and official embeddings."""
    return float(np.sum(base_emb * cand_emb, axis=1).mean())


def main() -> None:
    a = parse_args()
    payload = json.loads(a.val_subset.read_text(encoding="utf-8"))
    entries = payload["entries"]
    device = torch.device(a.device)

    with h5py.File(a.data, "r") as h:
        pmz_all = np.asarray(h["precursor_mz"][:], dtype=float)
        inst = np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in h["INSTRUMENT_TYPE"][:]], dtype=object)
        ce = np.asarray(h["COLLISION_ENERGY"][:], dtype=float)
        # Anchors first, then unique non-anchor hard-negative spectra, so the
        # full locked hard-negative set is measured, not just anchor-anchor.
        anchor_rows = [int(e["anchor_row"]) for e in entries]
        anchor_set = set(anchor_rows)
        neg_rows = sorted({int(n["row"]) for e in entries for n in e["neg"]} - anchor_set)
        all_rows = anchor_rows + neg_rows
        specs_all = [preprocess_spectrum(np.asarray(h["spectrum"][r]), float(pmz_all[r]), a.n_highest_peaks)
                     for r in all_rows]
    row_to_index = {row: i for i, row in enumerate(all_rows)}

    pairs_pos = cross_pairs(entries, inst, ce)                    # (anchor_i, anchor_j)
    pairs_neg = [(i, row_to_index[r]) for i, r in hard_pairs_full(entries)]
    if not pairs_pos:
        raise RuntimeError("Validation set has no real cross-condition pairs")
    iks = [e["ik14"] for e in entries]
    pmzs = [e["precursor_mz"] for e in entries]
    adducts = [e["adduct"] for e in entries]
    n_anchor = len(entries)

    base, _ = load_base_model(a.base_ckpt, a.architecture_ckpt, device, a.n_highest_peaks)
    base.eval()
    candidate, _ = load_trained(a.base_ckpt, a.architecture_ckpt, device, a.n_highest_peaks, a.trained)

    base_all = embed(base, specs_all, device, a.batch_size).numpy()
    cand_all = embed(candidate, specs_all, device, a.batch_size).numpy()
    base_anchor = base_all[:n_anchor]
    cand_anchor = cand_all[:n_anchor]

    # Per-pair cosines (for mean + paired bootstrap).
    base_pos = pair_cos_values(base_all, pairs_pos)
    cand_pos = pair_cos_values(cand_all, pairs_pos)
    base_neg = pair_cos_values(base_all, pairs_neg)
    cand_neg = pair_cos_values(cand_all, pairs_neg)

    # Per-query retrieval (for mean + query bootstrap + error transition).
    base_rows = retrieval_per_query(base_anchor, iks, pmzs, adducts, a.ppm_tol)
    cand_rows = retrieval_per_query(cand_anchor, iks, pmzs, adducts, a.ppm_tol)

    rng = np.random.default_rng(a.seed)
    boot = {
        "cross_condition_positive_cosine": paired_bootstrap_delta(base_pos, cand_pos, rng, a.bootstrap),
        "hard_negative_cosine": paired_bootstrap_delta(base_neg, cand_neg, rng, a.bootstrap),
        "macro_auc": paired_bootstrap_delta(
            np.array([r["auc"] for r in base_rows]), np.array([r["auc"] for r in cand_rows]), rng, a.bootstrap),
        "recall1": paired_bootstrap_delta(
            np.array([r["recall1"] for r in base_rows]), np.array([r["recall1"] for r in cand_rows]), rng, a.bootstrap),
    }

    base_ret = aggregate_retrieval(base_rows)
    cand_ret = aggregate_retrieval(cand_rows)
    trans = error_transition(base_rows, cand_rows)
    preserv = preservation_cosine(base_anchor, cand_anchor)

    d_pos = boot["cross_condition_positive_cosine"]["mean_delta"]
    d_neg = boot["hard_negative_cosine"]["mean_delta"]
    d_auc = boot["macro_auc"]["mean_delta"]
    d_rec = boot["recall1"]["mean_delta"]
    ci_pos_low = boot["cross_condition_positive_cosine"]["ci_low"]
    ci_neg_high = boot["hard_negative_cosine"]["ci_high"]

    gates = {
        "positive_not_degraded": bool(d_pos >= 0.0 or ci_pos_low >= -MACRO_AUC_MARGIN),
        "hard_negative_reduced": bool(d_neg < 0.0 and ci_neg_high < 0.0),
        "retrieval_noninferior_macro_auc": bool(d_auc >= -MACRO_AUC_MARGIN),
        "retrieval_noninferior_recall1": bool(d_rec >= -RECALL1_MARGIN),
        "corrected_ge_introduced": bool(trans["corrected"] >= trans["introduced"]),
        "preservation_cosine_ok": bool(preserv >= PRESERVATION_FLOOR),
    }
    gates["pass"] = bool(all(gates.values()))

    report = {
        "status": "g8r_inner_paired_gate_m1",
        "trained": str(a.trained), "val_subset": str(a.val_subset),
        "n_spectra": len(entries),
        "n_cross_condition_pairs": len(pairs_pos),
        "n_hard_negative_pairs": len(pairs_neg),
        "n_unique_hard_negative_spectra": len(neg_rows),
        "bootstrap_resamples": a.bootstrap,
        "baseline": {
            "cross_condition_positive_cosine": float(base_pos.mean()),
            "hard_negative_cosine": float(base_neg.mean()),
            "retrieval": base_ret,
        },
        "candidate": {
            "cross_condition_positive_cosine": float(cand_pos.mean()),
            "hard_negative_cosine": float(cand_neg.mean()),
            "retrieval": cand_ret,
        },
        "delta_candidate_minus_baseline": {
            "cross_condition_positive_cosine": d_pos,
            "hard_negative_cosine": d_neg,
            "macro_auc": d_auc,
            "recall1": d_rec,
        },
        "bootstrap_delta_ci": boot,
        "preservation_cosine": preserv,
        "error_transition": trans,
        "gates": gates,
    }
    output = a.output or a.trained.with_name("g8r_inner_gate.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
