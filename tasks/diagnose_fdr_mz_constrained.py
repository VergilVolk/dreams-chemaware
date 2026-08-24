"""m/z 约束下的 TDA —— 拿我们真实运行点（cos>=0.7 + m/z<=20ppm）的诚实 FDR。

背景：现有 fdr.py 的 target/decoy top1 是"纯余弦"（对全库取 max），没算 m/z。
而 m/z 才是真过滤器。本脚本对 target 和 decoy 都施加同样的 20ppm 前体 m/z 硬约束，
再用**修复后的** target_decoy_qvalues 算 fdr(s) 与 q(s)，得到与注释管线一致的 FDR。

关键事实（shuffle 诱饵生成时保留 precursor_mz）：decoy[i] 的前体 m/z = library[i] 的
前体 m/z，所以 decoy 的 m/z 掩码与 library 完全一致。

数据:
    query   = data/mtbls13729/smoke/neg_emb/embeddings.npy + manifest.csv
    library = data/models/mona_neg_dreams_emb/embeddings.npy + manifest.csv
    decoy   = data/models/mona_neg_decoy_emb.npy  (m/z = library 同序)

用法 (conda dreams_env, CPU):
    python tasks/diagnose_fdr_mz_constrained.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from annotation.fdr import target_decoy_qvalues  # noqa: E402
from annotation.params import DEFAULT  # noqa: E402

QUERY_DIR = ROOT / "data/mtbls13729/smoke/neg_emb"
LIB_DIR = ROOT / "data/models/mona_neg_dreams_emb"
DECOY = ROOT / "data/models/mona_neg_decoy_emb.npy"

PPM = DEFAULT.ppm_tolerance
COS = DEFAULT.cosine_confident


def mz_masked_top1(query, q_mz, library, l_mz):
    """每 query 在 m/z 硬约束内的最高余弦（无匹配则 -inf）。"""
    q = query / np.linalg.norm(query, axis=1, keepdims=True)
    lib = library / np.linalg.norm(library, axis=1, keepdims=True)
    best = np.full(len(q), -np.inf, dtype=np.float32)
    for start in range(0, len(q), 512):
        stop = min(start + 512, len(q))
        sim = q[start:stop] @ lib.T
        dppm = np.abs(q_mz[start:stop, None] - l_mz[None, :]) / np.maximum(
            np.abs(l_mz[None, :]), 1e-9) * 1e6
        sim = np.where(dppm <= PPM, sim, -np.inf)
        best[start:stop] = sim.max(axis=1)
    return best


def main() -> None:
    query = np.load(QUERY_DIR / "embeddings.npy")
    q_manifest = pd.read_csv(QUERY_DIR / "manifest.csv")
    library = np.load(LIB_DIR / "embeddings.npy")
    l_manifest = pd.read_csv(LIB_DIR / "manifest.csv")
    decoy = np.load(DECOY)

    q_mz = q_manifest["precursor_mz"].to_numpy(dtype=np.float64)
    l_mz = l_manifest["precursor_mz"].to_numpy(dtype=np.float64)
    assert len(l_mz) == library.shape[0] == decoy.shape[0]

    # ---- 纯余弦（现状，对照）----
    qn = query / np.linalg.norm(query, axis=1, keepdims=True)
    libn = library / np.linalg.norm(library, axis=1, keepdims=True)
    decn = decoy / np.linalg.norm(decoy, axis=1, keepdims=True)
    tgt_cos = (qn @ libn.T).max(axis=1)
    dec_cos = (qn @ decn.T).max(axis=1)

    # ---- m/z 约束（正确口径）----
    tgt_mz = mz_masked_top1(query, q_mz, library, l_mz)
    dec_mz = mz_masked_top1(query, q_mz, decoy, l_mz)

    print("=== 纯余弦 vs m/z约束 top1 分布 ===", flush=True)
    for name, arr in (("target纯cos", tgt_cos), ("decoy纯cos", dec_cos),
                      ("target+mz", tgt_mz), ("decoy+mz", dec_mz)):
        pct = np.percentile(arr[~np.isneginf(arr)], [0, 50, 75, 90, 95, 99, 100]).round(4)
        n_fin = int(np.isfinite(arr).sum())
        print(f"  {name:12s} 有限{n_fin}  percentiles[0,50,75,90,95,99,max]={pct}",
              flush=True)

    # ---- 修复后的 TDA，纯余弦 vs m/z约束 ----
    print("\n=== 修复后 TDA：纯余弦 vs m/z约束 ===", flush=True)
    for name, tgt, dec in (("纯cos", tgt_cos, dec_cos), ("m/z约束", tgt_mz, dec_mz)):
        fin = np.isfinite(tgt) & np.isfinite(dec)
        t, d = tgt[fin], dec[fin]
        q = target_decoy_qvalues(t, d)
        print(f"\n  [{name}] 有效 query {len(t)}/{len(query)}", flush=True)
        print(f"    q percentiles[0,25,50,75,90,99,max] = "
              f"{np.percentile(q, [0,25,50,75,90,99,100]).round(4)}", flush=True)
        # 逐阈值 fdr（正确公式）
        print(f"    {'thr':>5s} {'N_t':>6s} {'N_d':>6s} {'fdr':>7s}", flush=True)
        for th in (0.9, 0.8, 0.7, 0.6):
            nt = int((t >= th).sum())
            nd = int((d >= th).sum())
            fdr = (nd + 1.0) / (nt + 1.0)
            print(f"    {th:5.2f} {nt:6d} {nd:6d} {fdr:7.4f}", flush=True)
        # 运行点 cos>=0.7 的 fdr
        nt07 = int((t >= COS).sum())
        nd07 = int((d >= COS).sum())
        print(f"    ** 运行点 cos>=0.7: N_t={nt07} N_d={nd07} "
              f"fdr={(nd07+1.0)/(nt07+1.0):.4f} **", flush=True)


if __name__ == "__main__":
    main()
