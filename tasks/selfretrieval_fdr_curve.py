"""库内自检索真值 FDR 校准曲线 —— 最终可交付的"置信度刻度"。

不依赖任何诱饵（诱饵已证明 miscalibrated：shuffle 高估4x、precursor-swap 低估5x）。
直接测：query = 库里抽出的已知谱，留一法搜其余库，在 m/z<=20ppm 硬约束下，
逐 cosine 阈值报告 top-1 命中 InChIKey 是否等于 query 自身 InChIKey（真值 FDR）。

输出每一档：n_confident / n_correct / n_wrong / FDR / precision。
这就是"对已知化合物，cosine=t + m/z 硬约束 的鉴定 FDR"。

数据:
    library = data/models/mona_neg_dreams_emb/{embeddings.npy,manifest.csv}
用法 (conda dreams_env, CPU):
    python tasks/selfretrieval_fdr_curve.py --n-query 5000
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
PPM = DEFAULT.ppm_tolerance


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-query", type=int, default=5000)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    lib = np.load(LIB_DIR / "embeddings.npy")
    man = pd.read_csv(LIB_DIR / "manifest.csv")
    l_mz = man["precursor_mz"].to_numpy(dtype=np.float64)
    inchikey = man["inchikey"].tolist()
    n = lib.shape[0]

    libn = lib / np.linalg.norm(lib, axis=1, keepdims=True)
    idxs = list(range(args.start, min(args.start + args.n_query, n)))
    qn = libn[idxs]
    q_mz = l_mz[idxs]
    q_ik = [inchikey[i] for i in idxs]

    tgt_best = np.full(len(idxs), -np.inf)
    tgt_ik = [""] * len(idxs)
    for k in range(len(idxs)):
        i = idxs[k]
        sim = qn[k] @ libn.T
        dppm = np.abs(q_mz[k] - l_mz) / np.maximum(np.abs(l_mz), 1e-9) * 1e6
        ok = (dppm <= PPM)
        ok[i] = False
        sm = np.where(ok, sim, -np.inf)
        j = int(np.argmax(sm))
        tgt_best[k] = sm[j]
        tgt_ik[k] = inchikey[j]

    tgt_best = np.array(tgt_best)
    correct = np.array([tgt_ik[k] == q_ik[k] for k in range(len(idxs))])

    print(f"=== 库内自检索真值 FDR 校准曲线 (n_query={len(idxs)}, m/z<={PPM}ppm) ===",
          flush=True)
    print(f"  {'cos>=thr':>9s} {'n_conf':>7s} {'n_corr':>7s} {'n_wrong':>7s} "
          f"{'FDR':>7s} {'precision':>9s}", flush=True)
    for th in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
        conf = tgt_best >= th
        nc = int(conf.sum())
        nw = int((~correct[conf]).sum())
        ncr = nc - nw
        fdr = nw / max(nc, 1)
        prec = ncr / max(nc, 1)
        print(f"  {th:9.2f} {nc:7d} {ncr:7d} {nw:7d} {fdr:7.4f} {prec:9.4f}",
              flush=True)


if __name__ == "__main__":
    main()
