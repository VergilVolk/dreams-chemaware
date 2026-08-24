"""Near/mid hardest-negative accuracy of the RAW reranker (val.json neg grades).

Uses the val.json `neg` field (grade near = MCES 0-2, mid = MCES 3-5) to compute,
per anchor, whether the score of the cross-condition positive exceeds the score
of the hardest near / mid hard negative.  Compares the cosine baseline vs the
RAW reranker score.  The reranker is re-trained from the train feature cache
(no backbone re-embed; only val anchors + neg rows are embedded for scoring).
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
DEFAULT_OUT = ROOT / "data/validation/g8r_raw_reranker_nearmid.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--val", type=Path, default=DEFAULT_VAL)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--base-ckpt", type=Path, default=DEFAULT_BASE)
    p.add_argument("--architecture-ckpt", type=Path, default=DEFAULT_ARCH)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-highest-peaks", type=int, default=100)
    p.add_argument("--peak-tolerance", type=float, default=0.02)
    p.add_argument("--hard-k", type=int, default=5)
    p.add_argument("--C", type=float, default=0.01)
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


def main() -> None:
    a = parse_args()
    device = a.device
    val = json.loads(a.val.read_text(encoding="utf-8"))["entries"]
    sib = build_sibling(val)

    # Build pos (sibling) + neg (neg-row) pairs with grade.
    pairs = []  # (anchor_entry_idx, row, grade)
    for i, e in enumerate(val):
        if sib[i] >= 0:
            pairs.append((i, int(val[sib[i]]["anchor_row"]), "pos"))
        for nn in e["neg"]:
            pairs.append((i, int(nn["row"]), nn.get("grade", "mid")))

    # Unique rows to embed (anchors + neg rows).
    anchor_rows = [int(e["anchor_row"]) for e in val]
    anchor_set = set(anchor_rows)
    neg_rows = sorted({int(n["row"]) for e in val for n in e["neg"]} - anchor_set)
    all_rows = anchor_rows + neg_rows
    row_to_idx = {r: i for i, r in enumerate(all_rows)}

    model, _ = load_base_model(a.base_ckpt, a.architecture_ckpt, device, a.n_highest_peaks)
    model.eval()
    with h5py.File(a.data, "r") as h:
        pmz_all = np.asarray(h["precursor_mz"][:], dtype=float)
        specs = [preprocess_spectrum(np.asarray(h["spectrum"][r]), float(pmz_all[r]), a.n_highest_peaks)
                 for r in all_rows]
        spectra = {r: np.asarray(h["spectrum"][r]) for r in all_rows}
    z = embed(model, specs, device, a.batch_size).numpy()
    z = z / np.clip(np.linalg.norm(z, axis=1, keepdims=True), 1e-12, None)

    # Per-pair: cosine + RAW features.
    feat_rows = []
    for anchor_i, row, grade in pairs:
        ia = row_to_idx[int(val[anchor_i]["anchor_row"])]
        ib = row_to_idx[row]
        cos = float(z[ia] @ z[ib])
        f = symmetric_features(spectra[int(val[anchor_i]["anchor_row"])],
                               float(val[anchor_i]["precursor_mz"]),
                               spectra[row], float(pmz_all[row]), a.peak_tolerance)
        feat_rows.append({"anchor": anchor_i, "grade": grade, "cosine": cos, **f})

    df = pd.DataFrame(feat_rows)
    features = ["cosine"] + RAW_FEATURES

    # Re-train the reranker from the train cache (same C/hard_k as the report).
    cache = np.load(a.cache, allow_pickle=True)
    tr = pd.DataFrame({k: cache[k] for k in cache.files})
    scaler, model_rk = fit_ranker(tr, ["dreams_similarity"] + RAW_FEATURES, a.hard_k, a.C)

    # Score: reranker (note: training used "dreams_similarity"; here the cosine column
    # is named "cosine", so build a frame with the exact feature names).
    score_frame = df.copy()
    score_frame = score_frame.rename(columns={"cosine": "dreams_similarity"})
    df["reranker_score"] = model_rk.decision_function(
        scaler.transform(score_frame[["dreams_similarity"] + RAW_FEATURES].to_numpy()))

    def hard_accuracy(score_col):
        out = {}
        for grade in ("near", "mid"):
            pos = df[df["grade"] == "pos"]
            negs = df[df["grade"] == grade]
            pos_by_a = pos.set_index("anchor")[score_col].to_dict()
            neg_by_a = defaultdict(list)
            for anchor, sc in zip(negs["anchor"], negs[score_col]):
                neg_by_a[int(anchor)].append(float(sc))
            acc, cnt = 0, 0
            for anchor, psc in pos_by_a.items():
                if anchor in neg_by_a:
                    acc += int(psc > max(neg_by_a[anchor]))
                    cnt += 1
            out[f"{grade}_hardest_neg_accuracy"] = float(acc / cnt) if cnt else float("nan")
            out[f"{grade}_n_anchors"] = cnt
        return out

    report = {
        "status": "g8r_raw_reranker_nearmid",
        "C": a.C, "hard_k": a.hard_k,
        "n_pos_pairs": int((df["grade"] == "pos").sum()),
        "n_near_neg_pairs": int((df["grade"] == "near").sum()),
        "n_mid_neg_pairs": int((df["grade"] == "mid").sum()),
        "cosine": hard_accuracy("cosine"),
        "reranker": hard_accuracy("reranker_score"),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved: {a.output}")


if __name__ == "__main__":
    main()
