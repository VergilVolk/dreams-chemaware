"""验证 TDA FDR 估计是否校准 —— 库内自检索真值 vs TDA 估计。

决定性实验：query = 库里抽出的已知谱（真值 = 自身 InChIKey），留一法搜其余库，
在 cos>=0.7 + m/z<=20ppm 运行点比较：
    (a) 实际 FDR = top-1 命中 InChIKey != query InChIKey 的比例（真值）
    (b) TDA 估计 = (N_decoy+1)/(N_target+1)（用 shuffle 诱饵）
若两者一致 → TDA 校准，67%（暗物质）可信；若 TDA 系统性偏 → 量化偏差。

注意：
  - 留一法：query i 搜库时屏蔽自身（对角线），避免 cos=1 自匹配。
  - 诱饵 = shuffle 全库（保留前体 m/z），诱饵[i] 是库[i] 的打乱版，是合法诱饵（非自匹配）。
  - m/z 约束对 target 和 decoy 一致施加（20ppm）。

数据:
    library = data/models/mona_neg_dreams_emb/{embeddings.npy,manifest.csv}
    decoy   = data/models/mona_neg_decoy_emb.npy
用法 (conda dreams_env, CPU):
    python tasks/validate_fdr_calibration.py --n-query 2000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from annotation.params import DEFAULT  # noqa: E402

LIB_DIR = ROOT / "data/models/mona_neg_dreams_emb"
DECOY = ROOT / "data/models/mona_neg_decoy_emb.npy"
PPM = DEFAULT.ppm_tolerance
COS = DEFAULT.cosine_confident


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-query", type=int, default=2000)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    lib = np.load(LIB_DIR / "embeddings.npy")
    man = pd.read_csv(LIB_DIR / "manifest.csv")
    decoy = np.load(DECOY)
    l_mz = man["precursor_mz"].to_numpy(dtype=np.float64)
    inchikey = man["inchikey"].tolist()
    n = lib.shape[0]

    # L2 归一化
    libn = lib / np.linalg.norm(lib, axis=1, keepdims=True)
    decn = decoy / np.linalg.norm(decoy, axis=1, keepdims=True)

    idxs = list(range(args.start, min(args.start + args.n_query, n)))
    qn = libn[idxs]
    q_mz = l_mz[idxs]
    q_ik = [inchikey[i] for i in idxs]

    tgt_best = np.full(len(idxs), -np.inf)
    dec_best = np.full(len(idxs), -np.inf)
    tgt_ik = [""] * len(idxs)

    for k in range(len(idxs)):
        # target：全库余弦 + m/z 约束，屏蔽自身
        sim = qn[k] @ libn.T
        dppm = np.abs(q_mz[k] - l_mz) / np.maximum(np.abs(l_mz), 1e-9) * 1e6
        ok = (dppm <= PPM)
        ok[idxs[k]] = False  # 留一法：屏蔽自身
        sim_masked = np.where(ok, sim, -np.inf)
        j = int(np.argmax(sim_masked))
        tgt_best[k] = sim_masked[j]
        tgt_ik[k] = inchikey[j]
        # decoy：同 m/z 掩码（诱饵与库同序），不打屏蔽（诱饵是合法伪谱）
        simd = qn[k] @ decn.T
        simd_masked = np.where(ok, simd, -np.inf)
        dec_best[k] = simd_masked.max()

    tgt_best = np.array(tgt_best)
    dec_best = np.array(dec_best)

    # 运行点 cos>=0.7（m/z 已约束）
    confident = tgt_best >= COS
    n_conf = int(confident.sum())
    correct = np.array([tgt_ik[k] == q_ik[k] for k in range(len(idxs))])
    actual_fpr = float((~correct[confident]).sum()) if n_conf else 0.0
    print(f"\n=== 自检索真值 (n_query={len(idxs)}) ===", flush=True)
    print(f"  confident(cos>={COS}+m/z) = {n_conf}", flush=True)
    print(f"  真值: 正确 {int(correct[confident].sum())} / 错 "
          f"{int((~correct[confident]).sum())}  →  实际 FDR = {actual_fpr:.4f}",
          flush=True)

    # TDA 估计（逐阈值，正确公式）
    print(f"\n=== TDA 估计 (shuffle 诱饵, m/z 约束) ===", flush=True)
    print(f"  {'thr':>5s} {'N_t':>6s} {'N_d':>6s} {'fdr_est':>8s}", flush=True)
    for th in (0.9, 0.8, 0.7, 0.6):
        nt = int((tgt_best >= th).sum())
        nd = int((dec_best >= th).sum())
        fdr = (nd + 1.0) / (nt + 1.0)
        print(f"  {th:5.2f} {nt:6d} {nd:6d} {fdr:8.4f}", flush=True)

    nt07 = int((tgt_best >= COS).sum())
    nd07 = int((dec_best >= COS).sum())
    fdr_est = (nd07 + 1.0) / (nt07 + 1.0)
    print(f"\n  ** TDA 估计 FDR(cos>=0.7) = {fdr_est:.4f}  vs  实际 FDR = "
          f"{actual_fpr:.4f} **", flush=True)
    print(f"  比值 TDA/实际 = {fdr_est/max(actual_fpr,1e-9):.2f}x "
          f"(>1 高估=保守, <1 低估=冒进)", flush=True)


if __name__ == "__main__":
    main()
