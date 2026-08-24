"""M1b gate evaluator: local-ranking margin + retrieval/preservation, no noise.

Implements docs/G8R_M1_POSTMORTEM_AND_M1B_DECISION_20260822.md §7.  Compares the
M1b candidate against the official frozen DreaMS on the SAME locked val subset.

Main gate:
  1. delta(s_p - max_n s_n) cluster-bootstrap 95% CI lower bound > 0;
  2. hard-panel pairwise accuracy improved by >=1 percentage point;
  3. strict-10ppm macro-AUC >= baseline - 0.005;
  4. Recall@1 >= baseline - 0.003;
  5. corrected > introduced;
  6. preservation cosine >= 0.995.

Secondary gate:
  7. cross-condition positive cosine not degraded (candidate >= official);
  8. near subgroup margin + violation rate improve; mid subgroup not degraded.

Panel status is "pilot" unless >=500 unique hard anchors are present (val has 314).

Reuses the retrieval / preservation / pair machinery from eval_g8r_inner_gate.py.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from train_e1_identity import load_base_model, preprocess_spectrum  # noqa: E402
from step5_gate_eval import embed, load_trained  # noqa: E402
from eval_g8r_inner_gate import (  # noqa: E402
    retrieval_per_query, aggregate_retrieval, error_transition,
    preservation_cosine, cross_pairs, hard_pairs_full, pair_cos_values,
    paired_bootstrap_delta,
)

DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_VAL = ROOT / "tasks/massspecgym_isomers/g8r_locked/val.json"
DEFAULT_BASE = ROOT / "data/e1/official_embedding_slim.pt"
DEFAULT_ARCH = ROOT / "dreams/models/pretrained/ssl_model_server.pt"

MACRO_AUC_MARGIN = 0.005
RECALL1_MARGIN = 0.003
PRESERVATION_FLOOR = 0.995
PAIRWISE_ACC_GAIN = 0.01
MID_DEGRADE_MARGIN = 0.01


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trained", type=Path, required=True, help="M1b best.pt")
    p.add_argument("--val-subset", type=Path, default=DEFAULT_VAL)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--base-ckpt", type=Path, default=DEFAULT_BASE)
    p.add_argument("--architecture-ckpt", type=Path, default=DEFAULT_ARCH)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-highest-peaks", type=int, default=100)
    p.add_argument("--ppm-tol", type=float, default=10.0)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260821)
    p.add_argument("--output", type=Path, default=None)
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


def margin_per_anchor(entries, z, sibling, row_to_index) -> dict:
    margins, pos_cos, neg_cos, iks = [], [], [], []
    for i, e in enumerate(entries):
        sib = sibling[i]
        if sib is None or not e["neg"]:
            continue
        p = float(np.dot(z[i], z[sib]))
        n = max(float(np.dot(z[i], z[row_to_index[int(nn["row"])]])) for nn in e["neg"])
        margins.append(p - n)
        pos_cos.append(p)
        neg_cos.append(n)
        iks.append(e["ik14"])
    margins = np.asarray(margins)
    return {
        "n_hard_anchors": int(len(margins)),
        "margin_mean": float(margins.mean()),
        "pos_cosine_mean": float(np.asarray(pos_cos).mean()),
        "max_neg_cosine_mean": float(np.asarray(neg_cos).mean()),
        "pairwise_accuracy": float((margins > 0).mean()),
        "violation_rate": float((margins < 0).mean()),
        "margins": margins,
        "iks": iks,
    }


def paired_cluster_bootstrap_delta(g_base, g_cand, iks, n_boot, seed):
    """Cluster (by anchor IK14) paired bootstrap of mean(cand - base) margin."""
    g_base = np.asarray(g_base)
    g_cand = np.asarray(g_cand)
    mean_delta = float(g_cand.mean() - g_base.mean())
    by_ik: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for ik, b, c in zip(iks, g_base, g_cand):
        by_ik[ik].append((float(b), float(c)))
    keys = list(by_ik)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, len(keys), len(keys))
        sb = sc = cnt = 0.0
        for kk in idx:
            for b, c in by_ik[keys[kk]]:
                sb += b
                sc += c
                cnt += 1
        boot[k] = (sc - sb) / cnt if cnt else float("nan")
    return {"mean_delta": mean_delta,
            "ci_low": float(np.percentile(boot, 2.5)),
            "ci_high": float(np.percentile(boot, 97.5))}


def stratify(entries, z, sibling, row_to_index) -> dict:
    strata = defaultdict(lambda: {"n": 0, "margin_sum": 0.0, "viol": 0})
    for i, e in enumerate(entries):
        sib = sibling[i]
        if sib is None or not e["neg"]:
            continue
        p = float(np.dot(z[i], z[sib]))
        n = max(float(np.dot(z[i], z[row_to_index[int(nn["row"])]])) for nn in e["neg"])
        m = p - n
        grades = {nn.get("grade") for nn in e["neg"]}
        g = "near" if "near" in grades else ("mid" if "mid" in grades else "other")
        st = strata[g]
        st["n"] += 1
        st["margin_sum"] += m
        st["viol"] += int(m < 0)
    return {f"grade={g}": {"n": st["n"], "margin_mean": st["margin_sum"] / st["n"],
                           "violation_rate": st["viol"] / st["n"]}
            for g, st in strata.items()}


def main() -> None:
    a = parse_args()
    payload = json.loads(a.val_subset.read_text(encoding="utf-8"))
    entries = payload["entries"]
    device = torch.device(a.device)

    with h5py.File(a.data, "r") as h:
        pmz_all = np.asarray(h["precursor_mz"][:], dtype=float)
        inst = np.asarray([x.decode() if isinstance(x, bytes) else str(x)
                           for x in h["INSTRUMENT_TYPE"][:]], dtype=object)
        ce = np.asarray(h["COLLISION_ENERGY"][:], dtype=float)
        anchor_rows = [int(e["anchor_row"]) for e in entries]
        anchor_set = set(anchor_rows)
        neg_rows = sorted({int(n["row"]) for e in entries for n in e["neg"]} - anchor_set)
        all_rows = anchor_rows + neg_rows
        specs_all = [preprocess_spectrum(np.asarray(h["spectrum"][r]), float(pmz_all[r]), a.n_highest_peaks)
                     for r in all_rows]
    row_to_index = {row: i for i, row in enumerate(all_rows)}
    n_anchor = len(entries)

    base, _ = load_base_model(a.base_ckpt, a.architecture_ckpt, device, a.n_highest_peaks)
    base.eval()
    cand, _ = load_trained(a.base_ckpt, a.architecture_ckpt, device, a.n_highest_peaks, a.trained)
    z_base = embed(base, specs_all, device, a.batch_size).numpy()
    z_cand = embed(cand, specs_all, device, a.batch_size).numpy()
    z_base_anchor = z_base[:n_anchor]
    z_cand_anchor = z_cand[:n_anchor]

    sibling = build_sibling(entries)

    base_m = margin_per_anchor(entries, z_base, sibling, row_to_index)
    cand_m = margin_per_anchor(entries, z_cand, sibling, row_to_index)
    delta = paired_cluster_bootstrap_delta(base_m["margins"], cand_m["margins"],
                                           base_m["iks"], a.bootstrap, a.seed)
    # Paired cluster CI on the pairwise-accuracy delta (binary correct/incorrect),
    # per §9 "同时报告 McNemar/cluster CI".
    acc_delta = paired_cluster_bootstrap_delta(
        (base_m["margins"] > 0).astype(float),
        (cand_m["margins"] > 0).astype(float),
        base_m["iks"], a.bootstrap, a.seed)
    base_s = stratify(entries, z_base, sibling, row_to_index)
    cand_s = stratify(entries, z_cand, sibling, row_to_index)

    # Step-3 diagnostics: per-anchor margin-delta distribution + hard-panel flips.
    margin_deltas = cand_m["margins"] - base_m["margins"]
    base_viol = base_m["margins"] < 0
    cand_viol = cand_m["margins"] < 0
    flip_good = int((base_viol & ~cand_viol).sum())   # baseline violated -> fixed
    flip_bad = int((~base_viol & cand_viol).sum())    # baseline correct -> broken
    margin_dist = {
        "delta_p10": float(np.percentile(margin_deltas, 10)),
        "delta_median": float(np.percentile(margin_deltas, 50)),
        "delta_p90": float(np.percentile(margin_deltas, 90)),
        "delta_mean": float(margin_deltas.mean()),
    }
    hard_flips = {
        "n_baseline_violated": int(base_viol.sum()),
        "n_baseline_correct": int((~base_viol).sum()),
        "violated_fixed_flip_good": flip_good,
        "correct_broken_flip_bad": flip_bad,
    }

    iks = [e["ik14"] for e in entries]
    pmzs = [e["precursor_mz"] for e in entries]
    adducts = [e["adduct"] for e in entries]
    base_rows = retrieval_per_query(z_base_anchor, iks, pmzs, adducts, a.ppm_tol)
    cand_rows = retrieval_per_query(z_cand_anchor, iks, pmzs, adducts, a.ppm_tol)
    base_ret = aggregate_retrieval(base_rows)
    cand_ret = aggregate_retrieval(cand_rows)
    trans = error_transition(base_rows, cand_rows)
    preserv = preservation_cosine(z_base_anchor, z_cand_anchor)
    # Which molecules flipped at Top-1 (corrected vs introduced), keyed by IK14.
    corrected_iks, introduced_iks = [], []
    for b, c in zip(base_rows, cand_rows):
        qi = b.get("query_index", None)
        ik = iks[qi] if qi is not None else "?"
        if b["top1_correct"] and not c["top1_correct"]:
            introduced_iks.append(ik)
        elif not b["top1_correct"] and c["top1_correct"]:
            corrected_iks.append(ik)
    retrieval_flips = {"corrected_ik14": corrected_iks, "introduced_ik14": introduced_iks}
    # §8 rule 2: macro-AUC / Recall@1 / margin must carry query-level bootstrap
    # (>=2000). The margin gate already uses a cluster bootstrap; add the missing
    # query-level paired bootstrap for the two retrieval non-inferiority gates.
    rng = np.random.default_rng(a.seed)
    retrieval_boot = {
        "macro_auc": paired_bootstrap_delta(
            np.array([r["auc"] for r in base_rows]),
            np.array([r["auc"] for r in cand_rows]), rng, a.bootstrap),
        "recall1": paired_bootstrap_delta(
            np.array([r["recall1"] for r in base_rows]),
            np.array([r["recall1"] for r in cand_rows]), rng, a.bootstrap),
    }

    pairs_pos = cross_pairs(entries, inst, ce)
    pairs_neg = [(i, row_to_index[r]) for i, r in hard_pairs_full(entries)]
    base_pos = pair_cos_values(z_base, pairs_pos)
    cand_pos = pair_cos_values(z_cand, pairs_pos)
    base_neg = pair_cos_values(z_base, pairs_neg)
    cand_neg = pair_cos_values(z_cand, pairs_neg)

    near_b = base_s.get("grade=near", {})
    near_c = cand_s.get("grade=near", {})
    mid_b = base_s.get("grade=mid", {})
    mid_c = cand_s.get("grade=mid", {})

    gates = {
        "margin_delta_ci_low_gt0": bool(delta["ci_low"] > 0),
        "pairwise_accuracy_improved_1pp": bool(
            cand_m["pairwise_accuracy"] - base_m["pairwise_accuracy"] >= PAIRWISE_ACC_GAIN),
        "macro_auc_noninferior": bool(cand_ret["macro_auc"] >= base_ret["macro_auc"] - MACRO_AUC_MARGIN),
        "recall1_noninferior": bool(cand_ret["recall1"] >= base_ret["recall1"] - RECALL1_MARGIN),
        "corrected_gt_introduced": bool(trans["corrected"] > trans["introduced"]),
        "preservation_ok": bool(preserv >= PRESERVATION_FLOOR),
        "cross_cond_pos_not_degraded": bool(cand_pos.mean() >= base_pos.mean()),
        "near_margin_improved": bool(near_c.get("margin_mean", -1e9) > near_b.get("margin_mean", -1e9)),
        "near_violation_reduced": bool(near_c.get("violation_rate", 1e9) < near_b.get("violation_rate", 1e9)),
        "mid_not_degraded": bool(mid_c.get("margin_mean", -1e9) >= mid_b.get("margin_mean", -1e9) - MID_DEGRADE_MARGIN),
    }
    main_keys = ["margin_delta_ci_low_gt0", "pairwise_accuracy_improved_1pp",
                 "macro_auc_noninferior", "recall1_noninferior",
                 "corrected_gt_introduced", "preservation_ok"]
    sec_keys = ["cross_cond_pos_not_degraded", "near_margin_improved",
                "near_violation_reduced", "mid_not_degraded"]
    gates["pass_main"] = bool(all(gates[k] for k in main_keys))
    gates["pass_secondary"] = bool(all(gates[k] for k in sec_keys))
    gates["pass"] = bool(gates["pass_main"] and gates["pass_secondary"])
    # Informational only (do NOT change pass): CI-based non-inferiority reading.
    gates["macro_auc_noninferior_ci"] = bool(
        retrieval_boot["macro_auc"]["ci_low"] >= -MACRO_AUC_MARGIN)
    gates["recall1_noninferior_ci"] = bool(
        retrieval_boot["recall1"]["ci_low"] >= -RECALL1_MARGIN)

    report = {
        "status": "m1b_rank_gate",
        "trained": str(a.trained),
        "val_subset": str(a.val_subset),
        "panel_status": "formal" if base_m["n_hard_anchors"] >= 500 else "pilot",
        "margin": {
            "baseline": {k: v for k, v in base_m.items() if k not in ("margins", "iks")},
            "candidate": {k: v for k, v in cand_m.items() if k not in ("margins", "iks")},
            "delta_candidate_minus_baseline": delta,
            "pairwise_accuracy_delta_candidate_minus_baseline": acc_delta,
        },
        "stratification": {"baseline": base_s, "candidate": cand_s},
        "margin_delta_distribution": margin_dist,
        "hard_panel_flips": hard_flips,
        "retrieval": {"baseline": base_ret, "candidate": cand_ret},
        "retrieval_delta_bootstrap": retrieval_boot,
        "retrieval_flips": retrieval_flips,
        "error_transition": trans,
        "preservation_cosine": preserv,
        "cross_condition_positive_cosine": {"baseline": float(base_pos.mean()),
                                            "candidate": float(cand_pos.mean())},
        "hard_negative_cosine": {"baseline": float(base_neg.mean()),
                                 "candidate": float(cand_neg.mean())},
        "gates": gates,
    }
    out = a.output or a.trained.parent / "m1b_rank_gate.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
