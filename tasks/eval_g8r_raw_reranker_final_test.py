"""One-shot final-test eval of the frozen RAW reranker v1 on Test-A/B/C.

Freezes the model (re-trained from the g8r train cache, C=0.01, hard_k=5, RAW
features, gate 0.24098341166973114).  Candidate library = full HDF5 minus g8r
IK14.  Per query: strict-10ppm same-adduct candidates, per-molecule dedup by
max cosine, RAW rerank, gate (Top1-Top2 gap), then Recall@1 / MRR.
This is the ONE-SHOT final test; no parameter may change after it runs.
"""
from __future__ import annotations

import argparse
import json
import sys
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
DEFAULT_TEST_DIR = ROOT / "data/validation/g8r_final_test_large"
DEFAULT_OUT = ROOT / "data/validation/g8r_raw_reranker_final_test.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--val", type=Path, default=DEFAULT_VAL)
    p.add_argument("--base-ckpt", type=Path, default=DEFAULT_BASE)
    p.add_argument("--architecture-ckpt", type=Path, default=DEFAULT_ARCH)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-highest-peaks", type=int, default=100)
    p.add_argument("--peak-tolerance", type=float, default=0.02)
    p.add_argument("--ppm-tol", type=float, default=10.0)
    p.add_argument("--hard-k", type=int, default=5)
    p.add_argument("--C", type=float, default=0.01)
    p.add_argument("--gate-threshold", type=float, default=0.24098341166973114)
    return p.parse_args()


def read_str(h, key):
    raw = h[key][:]
    return np.asarray([x.decode("utf-8", "ignore") if isinstance(x, bytes) else str(x)
                       for x in raw], dtype=object)


def main() -> None:
    a = parse_args()
    device = a.device
    g8r_ik = set()
    for p in (a.train, a.val):
        for e in json.loads(p.read_text(encoding="utf-8"))["entries"]:
            g8r_ik.add(e["ik14"])

    with h5py.File(a.data, "r") as h:
        ikf = read_str(h, "INCHIKEY")
        adduct = read_str(h, "adduct")
        pmz = np.asarray(h["precursor_mz"][:], dtype=float)
        lib_spectra = None

    n = len(pmz)
    ik14 = np.asarray([k[:14] for k in ikf], dtype=object)
    lib_mask = np.array([ik not in g8r_ik for ik in ik14], dtype=bool)
    lib_idx = np.where(lib_mask)[0]

    # re-train frozen model
    cache = np.load(a.cache, allow_pickle=True)
    tr = pd.DataFrame({k: cache[k] for k in cache.files})
    scaler, rk = fit_ranker(tr, ["dreams_similarity"] + RAW_FEATURES, a.hard_k, a.C)

    # embed library once + load spectra for RAW features
    model, _ = load_base_model(a.base_ckpt, a.architecture_ckpt, device, a.n_highest_peaks)
    model.eval()
    pmz_lib = pmz[lib_idx]
    with h5py.File(a.data, "r") as h:
        lib_spectra = np.asarray(h["spectrum"][lib_idx])  # bulk read (fast)
    specs = []
    for i in range(len(lib_idx)):
        specs.append(preprocess_spectrum(lib_spectra[i], float(pmz_lib[i]), a.n_highest_peaks))
        if (i + 1) % 20000 == 0:
            print(f"[preprocess] {i+1}/{len(lib_idx)}", flush=True)
    print(f"[preprocess] done {len(lib_idx)} spectra", flush=True)
    z_lib = embed(model, specs, device, a.batch_size).numpy()
    z_lib = z_lib / np.clip(np.linalg.norm(z_lib, axis=1, keepdims=True), 1e-12, None)

    ad_lib = adduct[lib_idx]
    ik_lib = ik14[lib_idx]
    lib_pos = {int(r): i for i, r in enumerate(lib_idx)}

    report = {"status": "g8r_raw_reranker_final_test",
              "gate_threshold": a.gate_threshold, "C": a.C, "hard_k": a.hard_k,
              "panels": {}}
    for panel in ("a", "b", "c"):
        qdf = pd.read_csv(a.test_dir / f"test_{panel}_queries.csv")
        rows = qdf["row"].to_numpy(dtype=np.int64)
        q_pos = np.array([lib_pos[int(r)] for r in rows])
        q_pmz = pmz[rows]
        q_ad = adduct[rows]
        q_ik = ik14[rows]
        q_spec = lib_spectra[q_pos]
        q_emb = z_lib[q_pos]

        base_corr = rk_corr = 0
        base_mrr = rk_mrr = 0.0
        n_valid = 0
        for qi in range(len(rows)):
            if (qi + 1) % 2000 == 0:
                print(f"[panel {panel}] query {qi+1}/{len(rows)}", flush=True)
            ppm_da = a.ppm_tol * 1e-6 * q_pmz[qi]
            cand = (np.abs(pmz_lib - q_pmz[qi]) <= ppm_da) & (ad_lib == q_ad[qi]) & \
                   (np.arange(len(lib_idx)) != q_pos[qi])
            idx = np.where(cand)[0]
            if len(idx) == 0:
                continue
            cik = ik_lib[idx]
            if not (cik == q_ik[qi]).any() or not (cik != q_ik[qi]).any():
                continue
            n_valid += 1
            cos = z_lib[idx] @ q_emb[qi]
            # per-molecule dedup: keep the max-cosine spectrum per IK14
            best = {}
            for j, c in zip(idx, cos):
                ik = ik_lib[j]
                if ik not in best or c > best[ik][0]:
                    best[ik] = (c, j)
            # reranker score for each candidate molecule (max-cosine representative)
            score = {}
            for ik, (c, j) in best.items():
                f = symmetric_features(lib_spectra[j], float(pmz_lib[j]),
                                       q_spec[qi], float(q_pmz[qi]), a.peak_tolerance)
                vec = np.concatenate([[c], [f[k] for k in RAW_FEATURES]]).astype(np.float32)
                score[ik] = float(rk.decision_function(scaler.transform(vec[None, :]))[0])

            cos_scores = {ik: v[0] for ik, v in best.items()}
            b_order = sorted(cos_scores, key=cos_scores.get, reverse=True)
            r_order = sorted(score, key=score.get, reverse=True)
            b_ik, r_ik = b_order[0], r_order[0]
            conf_margin = (cos_scores[b_order[0]] - cos_scores[b_order[1]]) if len(b_order) > 1 else float("inf")
            use_rerank = conf_margin <= a.gate_threshold
            final_ik = r_ik if use_rerank else b_ik

            base_corr += int(b_ik == q_ik[qi])
            rk_corr += int(final_ik == q_ik[qi])
            if q_ik[qi] in b_order:
                base_mrr += 1.0 / (b_order.index(q_ik[qi]) + 1)
            if q_ik[qi] in r_order:
                rk_mrr += 1.0 / (r_order.index(q_ik[qi]) + 1)

        report["panels"][panel] = {
            "n_queries": int(len(rows)), "n_valid": n_valid,
            "base_recall1": base_corr / n_valid if n_valid else None,
            "reranker_recall1": rk_corr / n_valid if n_valid else None,
            "base_mrr": base_mrr / n_valid if n_valid else None,
            "reranker_mrr": rk_mrr / n_valid if n_valid else None,
        }

    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved: {a.output}")


if __name__ == "__main__":
    main()
