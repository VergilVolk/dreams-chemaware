"""D0 diagnostic: decompose WHY the G8R M1 gate failed, without training.

Read-only: no checkpoint is written, no weight is modified.  Implements
docs/G8R_M1_POSTMORTEM_AND_M1B_DECISION_20260822.md §5 (D0.1, D0.2, D0.4).
D0.3 (per-term gradient decomposition) requires the G8R candidate checkpoint and
is intentionally deferred to a follow-up (see notes at the bottom of main()).

  D0.1  same-anchor paired margin   g_i = s(a_i,p_i) - max_j s(a_i,n_ij)
        over every anchor that has BOTH a real cross-condition positive and
        >=1 hard negative.  Reports mean/median/frac>0 (pairwise accuracy),
        margin-violation rate, and an IK14-cluster bootstrap CI.  With
        --candidate-ckpt also reports the paired delta candidate-minus-baseline.

  D0.2  coverage stratification: has-hard-neg vs not, near vs mid grade, and a
        coarse bio class (nucleoside/purine, amino_acid, other) via bio_class.

  D0.4  frozen-representation separability probe.  On the OFFICIAL frozen
        embedding only, classify pos (same-molecule cross-condition) vs
        hard-negative (isomer) pairs with a linear probe on |z_a-z_b| and/or
        z_a (.) z_b, IK14-isolated, versus the raw cosine.  If even a linear
        readout cannot beat raw cosine, head-only has no room; if it can, the
        signal lives in the frozen embedding and M1b (loss/coverage) is the
        right lever.

The probe is fit on --probe-train-subset and evaluated on --subset when the
former is given (highest power, zero IK14 overlap by construction); otherwise it
is k-fold cross-validated on --subset alone.
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
from step5_gate_eval import embed, query_auc, load_trained  # noqa: E402
from bio_class import bio_tags  # noqa: E402

DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_VAL = ROOT / "tasks/massspecgym_isomers/g8r_locked/val.json"
DEFAULT_BASE = ROOT / "data/e1/official_embedding_slim.pt"
DEFAULT_ARCH = ROOT / "dreams/models/pretrained/ssl_model_server.pt"
DEFAULT_OUT = ROOT / "data/validation/g8r_d0_diagnostic.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subset", type=Path, default=DEFAULT_VAL, help="locked JSON to analyze")
    p.add_argument("--probe-train-subset", type=Path, default=None,
                   help="if set, fit the D0.4 probe here and evaluate on --subset")
    p.add_argument("--candidate-ckpt", type=Path, default=None,
                   help="optional G8R checkpoint; adds candidate deltas to D0.1/D0.2")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--base-ckpt", type=Path, default=DEFAULT_BASE)
    p.add_argument("--architecture-ckpt", type=Path, default=DEFAULT_ARCH)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-highest-peaks", type=int, default=100)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--probe-folds", type=int, default=5)
    p.add_argument("--probe-steps", type=int, default=800)
    p.add_argument("--seed", type=int, default=20260821)
    p.add_argument("--max-anchors", type=int, default=0, help="0=all; >0 smoke limit")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  pair construction (pure metadata)
# --------------------------------------------------------------------------- #
def build_sibling(entries: list[dict]) -> list[int | None]:
    """For each entry index i, the index of its cross-condition positive sibling.

    The G8R builder samples one cross-condition pair per (ik14, adduct) group, so
    every retained group holds exactly two entries (audit max_entries_per_identity_adduct=2).
    """
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, e in enumerate(entries):
        groups[(e["ik14"], e["adduct"])].append(i)
    sibling: list[int | None] = [None] * len(entries)
    for rows in groups.values():
        if len(rows) == 2:
            sibling[rows[0]] = rows[1]
            sibling[rows[1]] = rows[0]
        else:  # defensive: pair the first two (audit says this never happens)
            for a, b in zip(rows, rows[1:]):
                sibling[a] = b
                sibling[b] = a
    return sibling


def embed_all(entries: list[dict], h: h5py.File, pmz_all: np.ndarray,
              model, device, n_highest: int, batch_size: int):
    """Embed anchors + unique non-anchor hard-negative rows; return (z, row_to_index)."""
    anchor_rows = [int(e["anchor_row"]) for e in entries]
    anchor_set = set(anchor_rows)
    neg_rows = sorted({int(n["row"]) for e in entries for n in e["neg"]} - anchor_set)
    all_rows = anchor_rows + neg_rows
    specs = [preprocess_spectrum(np.asarray(h["spectrum"][r]), float(pmz_all[r]), n_highest)
             for r in all_rows]
    z = embed(model, specs, device, batch_size).numpy()
    row_to_index = {row: i for i, row in enumerate(all_rows)}
    return z, row_to_index, neg_rows


# --------------------------------------------------------------------------- #
#  D0.1 same-anchor paired margin
# --------------------------------------------------------------------------- #
def d01_margin(entries, z, sibling, row_to_index, bootstrap, seed):
    margins, pos_cos, neg_cos = [], [], []
    ik14_of = []
    for i, e in enumerate(entries):
        sib = sibling[i]
        if sib is None or not e["neg"]:
            continue
        p = float(np.dot(z[i], z[sib]))
        n = max(float(np.dot(z[i], z[row_to_index[int(nn["row"])]])) for nn in e["neg"])
        margins.append(p - n)
        pos_cos.append(p)
        neg_cos.append(n)
        ik14_of.append(e["ik14"])
    margins = np.asarray(margins)
    pos_cos = np.asarray(pos_cos)
    neg_cos = np.asarray(neg_cos)

    def ik_boot_ci(vals: np.ndarray) -> dict:
        by_ik: dict[str, list[float]] = defaultdict(list)
        for ik, v in zip(ik14_of, vals):
            by_ik[ik].append(v)
        keys = list(by_ik)
        rng = np.random.default_rng(seed)
        boot_means = np.empty(bootstrap)
        for b in range(bootstrap):
            k = rng.integers(0, len(keys), len(keys))
            s = 0.0
            cnt = 0
            for kk in k:
                s += sum(by_ik[keys[kk]])
                cnt += len(by_ik[keys[kk]])
            boot_means[b] = s / cnt if cnt else float("nan")
        return {"mean": float(vals.mean()),
                "ci_low": float(np.percentile(boot_means, 2.5)),
                "ci_high": float(np.percentile(boot_means, 97.5))}

    out = {
        "n_anchors_with_pos_and_neg": int(len(margins)),
        "margin": ik_boot_ci(margins),
        "pos_cosine_mean": float(pos_cos.mean()) if len(pos_cos) else float("nan"),
        "max_neg_cosine_mean": float(neg_cos.mean()) if len(neg_cos) else float("nan"),
        "pairwise_accuracy_frac_margin_gt0": float((margins > 0).mean()),
        "margin_violation_rate_frac_lt0": float((margins < 0).mean()),
    }
    return out


def d01_delta(base_out, cand_out):
    """Paired delta summary of the two margin dicts (means already computed)."""
    return {
        "delta_margin_mean": cand_out["margin"]["mean"] - base_out["margin"]["mean"],
        "delta_pos_cosine_mean": cand_out["pos_cosine_mean"] - base_out["pos_cosine_mean"],
        "delta_max_neg_cosine_mean": cand_out["max_neg_cosine_mean"] - base_out["max_neg_cosine_mean"],
    }


# --------------------------------------------------------------------------- #
#  D0.2 coverage stratification
# --------------------------------------------------------------------------- #
def d02_stratify(entries, z, sibling, row_to_index, smiles_by_row):
    strata = defaultdict(lambda: {"n": 0, "n_with_neg": 0, "pos_sum": 0.0, "neg_sum": 0.0,
                                  "margin_sum": 0.0, "margin_n": 0})
    for i, e in enumerate(entries):
        sib = sibling[i]
        has_neg = bool(e["neg"])
        grade = "none" if not has_neg else (
            "near" if any(nn.get("grade") == "near" for nn in e["neg"]) else
            "mid" if any(nn.get("grade") == "mid" for nn in e["neg"]) else "mixed")
        smi = smiles_by_row.get(int(e["anchor_row"]), "")
        tags = bio_tags(smi) if smi else "other"
        bio = "nucleoside_purine" if ("purine" in tags or "pyrimidine" in tags or "nucleoside" in tags) else (
            "amino_acid" if "amino_acid" in tags else "other")

        keys = [f"has_neg={has_neg}", f"grade={grade}", f"bio={bio}"]
        for k in keys:
            st = strata[k]
            st["n"] += 1
            if sib is not None:
                st["pos_sum"] += float(np.dot(z[i], z[sib]))
            if has_neg:
                st["n_with_neg"] += 1
                m = max(float(np.dot(z[i], z[row_to_index[int(nn["row"])]])) for nn in e["neg"])
                st["neg_sum"] += m
                if sib is not None:
                    st["margin_sum"] += float(np.dot(z[i], z[sib])) - m
                    st["margin_n"] += 1

    out = {}
    for k, st in sorted(strata.items()):
        out[k] = {
            "n": st["n"],
            "n_with_neg": st["n_with_neg"],
            "pos_cosine_mean": st["pos_sum"] / st["n"] if st["n"] else float("nan"),
            "max_neg_cosine_mean": st["neg_sum"] / st["n_with_neg"] if st["n_with_neg"] else float("nan"),
            "margin_mean": st["margin_sum"] / st["margin_n"] if st["margin_n"] else float("nan"),
        }
    return out


# --------------------------------------------------------------------------- #
#  D0.4 frozen separability probe
# --------------------------------------------------------------------------- #
def _probe_samples(entries, z, sibling, row_to_index):
    """Return (X_absdiff, X_hadamard, X_cat, y, keys, cosine_scores).

    pos sample = (anchor, sibling); neg sample = (anchor, hard-neg row), one per neg.
    """
    absd, had, cat, y, keys, cos = [], [], [], [], [], []
    for i, e in enumerate(entries):
        sib = sibling[i]
        if sib is None:
            continue
        if sib > i:  # count each pos pair once
            zi, zp = z[i], z[sib]
            d = np.abs(zi - zp)
            h = zi * zp
            absd.append(d); had.append(h); cat.append(np.concatenate([d, h]))
            y.append(1.0); keys.append(e["ik14"]); cos.append(float(np.dot(zi, zp)))
        for nn in e["neg"]:
            zi, zn = z[i], z[row_to_index[int(nn["row"])]]
            d = np.abs(zi - zn)
            h = zi * zn
            absd.append(d); had.append(h); cat.append(np.concatenate([d, h]))
            y.append(0.0); keys.append(e["ik14"]); cos.append(float(np.dot(zi, zn)))
    return (np.asarray(absd), np.asarray(had), np.asarray(cat),
            np.asarray(y), keys, np.asarray(cos))


def _logreg_auc(Xtr, ytr, Xte, yte, device, steps, seed):
    torch.manual_seed(seed)
    Xtr = torch.as_tensor(Xtr, dtype=torch.float32)
    ytr = torch.as_tensor(ytr, dtype=torch.float32)
    Xte = torch.as_tensor(Xte, dtype=torch.float32)
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True) + 1e-8
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    lin = torch.nn.Linear(Xtr.shape[1], 1).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-3)
    Xtr, ytr = Xtr.to(device), ytr.to(device)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(lin(Xtr).squeeze(1), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        scores = torch.sigmoid(lin(Xte.to(device)).squeeze(1)).cpu().numpy()
    return query_auc(yte, scores)


def _fold_aucs(X, y, keys, folds, device, steps, seed):
    aucs = []
    for f, test_keys in enumerate(folds):
        tset = set(test_keys)
        tr = np.array([k not in tset for k in keys])
        te = ~tr
        if tr.sum() == 0 or te.sum() == 0 or len(set(y[te])) < 2:
            aucs.append(float("nan"))
            continue
        aucs.append(_logreg_auc(X[tr], y[tr], X[te], y[te], device, steps, seed + f))
    return aucs


def d04_probe(entries, z, sibling, row_to_index, fit_entries, fit_z, fit_sibling,
              fit_row_to_index, folds, steps, seed, device):
    """Fit (optionally on a separate train set), evaluate on --subset by IK14."""
    Xa, Xh, Xc, y, keys, cos = _probe_samples(entries, z, sibling, row_to_index)
    cos_auc = query_auc(y, cos)  # raw cosine as a single global score

    if fit_entries is not None:
        Fa, Fh, Fc, fy, fkeys, fcos = _probe_samples(fit_entries, fit_z, fit_sibling, fit_row_to_index)
        # train on the fit set, test on the whole --subset
        out = {
            "fit_on": "probe-train-subset",
            "n_train_samples": int(len(fy)),
            "n_test_samples": int(len(y)),
            "cosine_auc_test": float(cos_auc),
            "probe_absdiff_auc_test": float(_logreg_auc(Fa, fy, Xa, y, device, steps, seed)),
            "probe_hadamard_auc_test": float(_logreg_auc(Fh, fy, Xh, y, device, steps, seed)),
            "probe_concat_auc_test": float(_logreg_auc(Fc, fy, Xc, y, device, steps, seed)),
        }
    else:
        rng = np.random.default_rng(seed)
        uniq = sorted(set(keys))
        rng.shuffle(uniq)
        fold_list = [list(a) for a in np.array_split(uniq, folds)]
        def summary(arr):
            a = np.asarray(arr)
            return {"mean": float(np.nanmean(a)), "std": float(np.nanstd(a)),
                    "folds": [float(x) for x in a]}
        out = {
            "fit_on": f"kfold({folds})",
            "n_samples": int(len(y)),
            "n_pos": int((y == 1).sum()),
            "n_neg": int((y == 0).sum()),
            "cosine_auc": summary([cos_auc]),
            "probe_absdiff_auc": summary(_fold_aucs(Xa, y, keys, fold_list, device, steps, seed)),
            "probe_hadamard_auc": summary(_fold_aucs(Xh, y, keys, fold_list, device, steps, seed)),
            "probe_concat_auc": summary(_fold_aucs(Xc, y, keys, fold_list, device, steps, seed)),
        }
    return out


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #
def main() -> None:
    a = parse_args()
    device = torch.device(a.device)
    payload = json.loads(a.subset.read_text(encoding="utf-8"))
    entries = payload["entries"]
    if a.max_anchors > 0:
        entries = entries[: a.max_anchors]

    with h5py.File(a.data, "r") as h:
        pmz_all = np.asarray(h["precursor_mz"][:], dtype=float)
        smiles_all = np.asarray([x.decode("utf-8", "ignore") if isinstance(x, bytes) else str(x)
                                 for x in h["smiles"][:]])
        base_model, _ = load_base_model(a.base_ckpt, a.architecture_ckpt, device, a.n_highest_peaks)
        base_model.eval()
        z, row_to_index, neg_rows = embed_all(entries, h, pmz_all, base_model, device,
                                              a.n_highest_peaks, a.batch_size)
        smiles_by_row = {int(e["anchor_row"]): smiles_all[int(e["anchor_row"])] for e in entries}

    sibling = build_sibling(entries)

    report: dict = {
        "status": "g8r_d0_diagnostic",
        "subset": str(a.subset),
        "n_anchors": len(entries),
        "n_unique_hard_negative_spectra": len(neg_rows),
        "d01_margin_baseline": d01_margin(entries, z, sibling, row_to_index, a.bootstrap, a.seed),
        "d02_stratification_baseline": d02_stratify(entries, z, sibling, row_to_index, smiles_by_row),
    }

    if a.candidate_ckpt is not None:
        with h5py.File(a.data, "r") as h:
            cand_model, _ = load_trained(a.base_ckpt, a.architecture_ckpt, device,
                                         a.n_highest_peaks, a.candidate_ckpt)
            zc, _, _ = embed_all(entries, h, pmz_all, cand_model, device, a.n_highest_peaks, a.batch_size)
        cand_margin = d01_margin(entries, zc, sibling, row_to_index, a.bootstrap, a.seed)
        report["d01_margin_candidate"] = cand_margin
        report["d01_delta_candidate_minus_baseline"] = d01_delta(report["d01_margin_baseline"], cand_margin)
        report["d02_stratification_candidate"] = d02_stratify(entries, zc, sibling, row_to_index, smiles_by_row)

    # D0.4 probe (optionally fit on a separate train set for power + isolation)
    fit_entries = fit_z = fit_sibling = fit_row_to_index = None
    if a.probe_train_subset is not None:
        fit_payload = json.loads(a.probe_train_subset.read_text(encoding="utf-8"))
        fit_entries = fit_payload["entries"]
        with h5py.File(a.data, "r") as h:
            fit_z, fit_row_to_index, _ = embed_all(fit_entries, h, pmz_all, base_model, device,
                                                   a.n_highest_peaks, a.batch_size)
        fit_sibling = build_sibling(fit_entries)
    report["d04_probe"] = d04_probe(entries, z, sibling, row_to_index,
                                    fit_entries, fit_z, fit_sibling, fit_row_to_index,
                                    a.probe_folds, a.probe_steps, a.seed, device)

    report["note_d03_deferred"] = (
        "D0.3 per-term gradient decomposition requires the G8R candidate checkpoint and a "
        "faithful replay of the step4 loss; it is intentionally deferred and not computed here.")

    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved: {a.output}")


if __name__ == "__main__":
    main()
