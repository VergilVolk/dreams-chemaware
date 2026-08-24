"""Evaluate the ORIGINAL locked Test-A/B with the FROZEN RAW-v1 artifact.

Loads the frozen artifact (no fit_ranker), VERIFIES its SHA256 + provenance
(train cache / training script / schema), reproduces the dev +4.35pp before
touching Test-A/B, then evaluates with the dev's strict rank rule
(rank = 1 + #{ s_neg >= s_pos }).  Fail-closed: locked queries cannot be skipped.
"""
from __future__ import annotations

import argparse
import hashlib
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

DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_TRAIN = ROOT / "tasks/massspecgym_isomers/g8r_locked/train.json"
DEFAULT_VAL = ROOT / "tasks/massspecgym_isomers/g8r_locked/val.json"
DEFAULT_BASE = ROOT / "data/e1/official_embedding_slim.pt"
DEFAULT_ARCH = ROOT / "dreams/models/pretrained/ssl_model_server.pt"
DEFAULT_ARTIFACT = ROOT / "data/validation/g8r_raw_reranker_v1_artifact.json"
DEFAULT_CACHE = ROOT / "data/validation/g8r_raw_reranker_cache.npz"
DEFAULT_VAL_CACHE = ROOT / "data/validation/g8r_raw_reranker_cache_val.npz"
DEFAULT_TRAIN_SCRIPT = ROOT / "tasks/train_g8r_raw_reranker.py"
DEFAULT_TEST_DIR = ROOT / "data/validation/g8r_final_test"
DEFAULT_OUT = ROOT / "data/validation/g8r_raw_reranker_original_ab.json"
DEFAULT_CSV = ROOT / "data/validation/g8r_raw_reranker_original_ab_perquery.csv"

EXPECTED_FEATURES = ["dreams_similarity", "sqrt_cosine", "linear_cosine", "entropy_similarity",
                     "intensity_coverage_min", "intensity_coverage_mean", "matched_peak_fraction_min",
                     "top10_match_fraction", "neutral_loss_sqrt_cosine", "neutral_loss_coverage_min",
                     "neutral_loss_coverage_mean", "peak_count_ratio"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--val", type=Path, default=DEFAULT_VAL)
    p.add_argument("--base-ckpt", type=Path, default=DEFAULT_BASE)
    p.add_argument("--architecture-ckpt", type=Path, default=DEFAULT_ARCH)
    p.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--val-cache", type=Path, default=DEFAULT_VAL_CACHE)
    p.add_argument("--train-script", type=Path, default=DEFAULT_TRAIN_SCRIPT)
    p.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-highest-peaks", type=int, default=100)
    p.add_argument("--peak-tolerance", type=float, default=0.02)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260822)
    p.add_argument("--force", action="store_true", help="overwrite existing output")
    return p.parse_args()


def read_str(h, key):
    raw = h[key][:]
    return np.asarray([x.decode("utf-8", "ignore") if isinstance(x, bytes) else str(x)
                       for x in raw], dtype=object)


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_artifact(a, art) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], float]:
    """Verify artifact SHA256 + provenance + schema.  Raises on any mismatch."""
    raw = a.artifact.read_text(encoding="utf-8")
    expected_sha = a.artifact.with_suffix(".sha256").read_text(encoding="utf-8").strip()
    actual_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    assert actual_sha == expected_sha, f"artifact SHA256 mismatch: {actual_sha} != {expected_sha}"
    print("[artifact] PASS: frozen artifact SHA256 verified", flush=True)

    assert art["feature_names"] == EXPECTED_FEATURES, "feature names/order mismatch"
    assert art["C"] == 0.01, f"C={art['C']}"
    assert art["hard_k"] == 5, f"hard_k={art['hard_k']}"
    assert abs(art["gate_threshold"] - 0.24098341166973114) < 1e-12, "gate threshold mismatch"
    mean = np.asarray(art["scaler_mean"], dtype=np.float64)
    scale = np.asarray(art["scaler_scale"], dtype=np.float64)
    coef = np.asarray(art["model_coef"], dtype=np.float64)
    intercept = float(art.get("model_intercept", 0.0))
    assert mean.shape == scale.shape == coef.shape == (12,), "model dims mismatch"
    assert np.all(scale > 0), "scaler scale has non-positive entries"
    assert intercept == 0.0, f"intercept must be 0, got {intercept}"
    assert sha256_of_file(a.cache) == art["train_cache_sha256"], "train cache SHA256 mismatch"
    assert sha256_of_file(a.train_script) == art["train_script_sha256"], "train script SHA256 mismatch"
    print("[provenance] PASS: cache/script/schema verified", flush=True)
    return mean, scale, coef, art["feature_names"], art["gate_threshold"]


def strict_rank(scores: dict, qik: str) -> int:
    pos = scores[qik]
    return 1 + sum(1 for ik, s in scores.items() if ik != qik and s >= pos)


def query_weighted_formula_bootstrap(delta, formula, n_boot, seed):
    df = pd.DataFrame({"delta": delta, "formula": formula})
    point = float(df["delta"].mean())
    by_f = {f: g["delta"].to_numpy() for f, g in df.groupby("formula")}
    formulas = list(by_f)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(formulas), len(formulas))
        sampled = np.concatenate([by_f[formulas[i]] for i in idx])
        draws[b] = sampled.mean()
    return {"mean": point, "ci_low": float(np.percentile(draws, 2.5)),
            "ci_high": float(np.percentile(draws, 97.5))}


def mcnemar(c, i):
    t = c + i
    if t == 0:
        return None
    from scipy.stats import binomtest
    return float(binomtest(c, t, 0.5).pvalue)


def main() -> None:
    a = parse_args()
    if a.output.exists() and not a.force:
        raise SystemExit(f"{a.output} exists; use --force to overwrite")
    device = a.device
    art = json.loads(a.artifact.read_text(encoding="utf-8"))
    mean, scale, coef, features, threshold = verify_artifact(a, art)
    RAW_FEATURES = features[1:]

    def raw_score(vec):
        return float(coef @ ((vec - mean) / scale))  # intercept asserted 0

    # ---- reproduction gate on dev (g8r_val) ----
    va_cache = np.load(a.val_cache, allow_pickle=True)
    va = pd.DataFrame({k: va_cache[k] for k in va_cache.files})
    va["score"] = [raw_score(va[features].iloc[i].to_numpy(dtype=np.float64)) for i in range(len(va))]

    def retrieve(frame, score_col):
        rows = []
        for qi, group in frame.groupby("query", sort=False):
            pos = group[group["label"] == 1]; neg = group[group["label"] == 0]
            if pos.empty or neg.empty:
                continue
            pscore = float(pos[score_col].max())
            mol = group.sort_values(score_col, ascending=False).drop_duplicates("candidate_ik14")
            top = mol[score_col].to_numpy(float)
            conf = float(top[0] - top[1]) if len(top) > 1 else float("inf")
            neg_scores = neg.groupby("candidate_ik14")[score_col].max().to_numpy(float)
            rank = 1 + int(np.sum(neg_scores >= pscore))
            rows.append({"query_index": int(qi), "ik14": group.iloc[0].query_ik14,
                         "formula": group.iloc[0].formula, "chosen_ik14": mol.iloc[0].candidate_ik14,
                         "top1": bool(rank == 1), "rank": rank, "mrr": 1.0 / rank, "confidence_margin": conf})
        return pd.DataFrame(rows)

    def gate(base, reranked, thr):
        merged = base[["query_index", "ik14", "formula", "top1", "confidence_margin", "mrr"]].merge(
            reranked[["query_index", "top1", "rank", "mrr"]], on="query_index", suffixes=("_b", "_r"))
        use = merged["confidence_margin"] <= thr
        out = pd.DataFrame({"query_index": merged["query_index"], "ik14": merged["ik14"],
                            "formula": merged["formula"], "gate_used": use,
                            "top1": np.where(use, merged["top1_r"], merged["top1_b"]),
                            "mrr": np.where(use, merged["mrr_r"], merged["mrr_b"])})
        return out

    base_dev = retrieve(va, "dreams_similarity")
    rk_dev = retrieve(va, "score")
    g_dev = gate(base_dev, rk_dev, threshold)
    base_r1 = float(base_dev["top1"].mean()); g_r1 = float(g_dev["top1"].mean())
    base_mrr = float(base_dev["mrr"].mean()); g_mrr = float(g_dev["mrr"].mean())
    corr = int(((~base_dev["top1"]) & g_dev["top1"]).sum())
    intro = int((base_dev["top1"] & (~g_dev["top1"])).sum())
    cov = float(g_dev["gate_used"].mean())
    print(f"[repro] base_r1={base_r1:.4f} gated_r1={g_r1:.4f} corrected={corr} introduced={intro} "
          f"coverage={cov:.4f} gate_on={int(g_dev['gate_used'].sum())}", flush=True)
    assert len(base_dev) == 620, f"repro n_queries {len(base_dev)}"
    assert int(g_dev["gate_used"].sum()) == 288, f"repro gate_on {g_dev['gate_used'].sum()}"
    assert abs(base_r1 - 0.8080645161290323) < 1e-3 and abs(g_r1 - 0.8516129032258064) < 1e-3
    assert abs(base_mrr - 0.8955645161290322) < 1e-3 and abs(g_mrr - 0.9194892473118278) < 1e-3
    assert corr == 44 and intro == 17, f"repro {corr}/{intro}"
    assert abs(cov - 0.4645161290322581) < 1e-3, f"repro coverage {cov}"
    print("[repro] PASS: dev +4.35pp reproduced with the frozen artifact", flush=True)

    # ---- embed HDF5 val fold once ----
    with h5py.File(a.data, "r") as h:
        fold = read_str(h, "fold"); pmz = np.asarray(h["precursor_mz"][:], dtype=float)
        formula_all = read_str(h, "FORMULA")
        val_idx = np.where(fold == "val")[0]
        val_spectra = np.asarray(h["spectrum"][val_idx])
    model, _ = load_base_model(a.base_ckpt, a.architecture_ckpt, device, a.n_highest_peaks)
    model.eval()
    pmz_val = pmz[val_idx]
    specs = [preprocess_spectrum(val_spectra[i], float(pmz_val[i]), a.n_highest_peaks) for i in range(len(val_idx))]
    z_val = embed(model, specs, device, a.batch_size).numpy()
    z_val = z_val / np.clip(np.linalg.norm(z_val, axis=1, keepdims=True), 1e-12, None)
    row_to_pos = {int(r): i for i, r in enumerate(val_idx)}

    report = {"status": "g8r_raw_reranker_original_ab",
              "artifact_sha256": a.artifact.with_suffix(".sha256").read_text().strip(),
              "reproduction_gate": "PASS", "panels": {}}
    perquery_rows = []
    expected = {"a": 2000, "b": 1792}
    panel_ik = {}
    for panel, fname in (("a", "test_a_manifest.json"), ("b", "test_b_manifest.json")):
        m = json.loads((a.test_dir / fname).read_text(encoding="utf-8"))
        queries = m["queries"]
        assert len(queries) == expected[panel], f"Test-{panel} n={len(queries)} != {expected[panel]}"
        print(f"[test] Test-{panel} queries = {len(queries)}", flush=True)
        qiks = [q["ik14"] for q in queries]
        assert len(set(qiks)) == len(qiks), f"Test-{panel} has duplicate IK14"
        panel_ik[panel] = set(qiks)

        base_corr = rk_corr = 0; base_mrr = rk_mrr = 0.0
        gate_used = corrected = introduced = 0
        deltas = []; formulas = []
        for qi, q in enumerate(queries):
            if (qi + 1) % 500 == 0:
                print(f"[panel {panel}] query {qi+1}/{len(queries)}", flush=True)
            qrow = int(q["row"]); qik = q["ik14"]; qformula = q["formula"]
            if qrow not in row_to_pos:
                raise RuntimeError(f"query row {qrow} not in val fold")
            qpos = row_to_pos[qrow]
            cands = q["candidates"]
            best = {}
            for cnd in cands:
                cpos = row_to_pos[int(cnd["row"])]  # KeyError if candidate not in val fold -> fail-closed
                c = float(z_val[qpos] @ z_val[cpos]); ik = cnd["ik14"]
                if ik not in best or c > best[ik][0]:
                    best[ik] = (c, cpos)
            if qik not in best:
                raise RuntimeError(f"query {qrow} has no positive candidate")
            score = {}
            for ik, (c, cpos) in best.items():
                f = symmetric_features(val_spectra[cpos], float(pmz_val[cpos]),
                                       val_spectra[qpos], float(pmz_val[qpos]), a.peak_tolerance)
                vec = np.concatenate([[c], [f[k] for k in RAW_FEATURES]]).astype(np.float64)
                score[ik] = raw_score(vec)
            cos_scores = {ik: v[0] for ik, v in best.items()}

            b_rank = strict_rank(cos_scores, qik); r_rank = strict_rank(score, qik)
            b_order = sorted(cos_scores, key=cos_scores.get, reverse=True)
            r_order = sorted(score, key=score.get, reverse=True)
            conf = (cos_scores[b_order[0]] - cos_scores[b_order[1]]) if len(b_order) > 1 else float("inf")
            use = conf <= threshold
            final_scores = score if use else cos_scores
            f_rank = strict_rank(final_scores, qik)
            f_order = sorted(final_scores, key=final_scores.get, reverse=True)

            b_correct = b_rank == 1; f_correct = f_rank == 1
            base_corr += int(b_correct); rk_corr += int(f_correct)
            base_mrr += 1.0 / b_rank; rk_mrr += 1.0 / f_rank
            gate_used += int(use)
            if not b_correct and f_correct:
                corrected += 1
            elif b_correct and not f_correct:
                introduced += 1
            deltas.append(float(f_correct - b_correct)); formulas.append(qformula)
            perquery_rows.append({
                "panel": panel, "row": qrow, "ik14": qik, "formula": qformula,
                "n_candidates": len(cands), "n_candidate_ik14": len(best),
                "baseline_rank": b_rank, "reranker_rank": r_rank, "final_rank": f_rank,
                "baseline_top1_ik14": b_order[0], "reranker_top1_ik14": r_order[0],
                "final_top1_ik14": f_order[0],
                "baseline_top1": int(b_correct), "final_top1": int(f_correct),
                "gate_used": int(use), "confidence_margin": conf,
            })

        n = len(queries)
        report["panels"][panel] = {
            "n_queries": n, "base_recall1": base_corr / n, "reranker_recall1": rk_corr / n,
            "base_mrr": base_mrr / n, "reranker_mrr": rk_mrr / n, "gate_coverage": gate_used / n,
            "corrected": corrected, "introduced": introduced,
            "recall1_delta_queryweighted_formula_bootstrap": query_weighted_formula_bootstrap(
                np.array(deltas), np.array(formulas), a.bootstrap, a.seed),
            "mcnemar_p": mcnemar(corrected, introduced),
        }

    overlap = panel_ik["a"] & panel_ik["b"]
    report["test_a_b_ik14_overlap"] = len(overlap)

    a.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(perquery_rows).to_csv(a.csv, index=False)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved {a.output} + {a.csv}")


if __name__ == "__main__":
    main()
