"""对比三种诱饵策略在库内自检索上的 TDA FDR 校准性。

真值 = 自检索 top-1 InChIKey 正确率（query 在库里，留一法）。
三种诱饵：
  (a) shuffle      —— 打乱强度、保留 m/z 轴（现状 mona_neg_decoy_emb.npy）
  (b) precursor-swap —— 真实库谱 + 打乱前体 m/z 标签（derangement，在流形内）
分别算 TDA fdr(cos>=0.7 + m/z<=20ppm)，与真值 FDR 比较，看谁更竞争（更接近真值）。

数据:
    library = data/models/mona_neg_dreams_emb/{embeddings.npy,manifest.csv}
    decoy(shuffle) = data/models/mona_neg_decoy_emb.npy
用法 (conda dreams_env, CPU):
    python tasks/compare_decoy_strategies.py --n-query 2000
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
DECOY_SHUFFLE = ROOT / "data/models/mona_neg_decoy_emb.npy"
PPM = DEFAULT.ppm_tolerance
COS = DEFAULT.cosine_confident


def derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """长度 n 的错排（无不动点）。"""
    perm = rng.permutation(n)
    fixed = np.where(perm == np.arange(n))[0]
    while len(fixed) > 1:
        a, b = fixed[0], fixed[1]
        perm[[a, b]] = perm[[b, a]]
        fixed = np.where(perm == np.arange(n))[0]
    if len(fixed) == 1:  # 只剩一个不动点，与邻位交换
        i = fixed[0]
        j = (i + 1) % n
        perm[[i, j]] = perm[[j, i]]
    return perm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-query", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lib = np.load(LIB_DIR / "embeddings.npy")
    man = pd.read_csv(LIB_DIR / "manifest.csv")
    shuffle_decoy = np.load(DECOY_SHUFFLE)
    l_mz = man["precursor_mz"].to_numpy(dtype=np.float64)
    inchikey = man["inchikey"].tolist()
    n = lib.shape[0]

    libn = lib / np.linalg.norm(lib, axis=1, keepdims=True)
    shuffle_decoy_n = shuffle_decoy / np.linalg.norm(shuffle_decoy, axis=1, keepdims=True)

    # precursor-swap：真实谱 + 错排 m/z 标签
    rng = np.random.default_rng(args.seed)
    perm = derangement(n, rng)
    swap_mz = l_mz[perm]  # decoy[j] 的"伪前体 m/z" = 真实谱 j 配了别的化合物 m/z

    idxs = list(range(args.n_query))
    qn = libn[idxs]
    q_mz = l_mz[idxs]
    q_ik = [inchikey[i] for i in idxs]

    tgt_best = np.full(len(idxs), -np.inf)
    shuf_best = np.full(len(idxs), -np.inf)
    swap_best = np.full(len(idxs), -np.inf)
    tgt_ik = [""] * len(idxs)

    for k in range(len(idxs)):
        i = idxs[k]
        sim = qn[k] @ libn.T
        dppm = np.abs(q_mz[k] - l_mz) / np.maximum(np.abs(l_mz), 1e-9) * 1e6
        ok = (dppm <= PPM)
        ok[i] = False  # 留一法
        sm = np.where(ok, sim, -np.inf)
        j = int(np.argmax(sm))
        tgt_best[k] = sm[j]
        tgt_ik[k] = inchikey[j]

        # shuffle 诱饵：m/z 掩码与库同序
        simd = qn[k] @ shuffle_decoy_n.T
        shuf_best[k] = np.where(ok, simd, -np.inf).max()

        # precursor-swap 诱饵：真实谱 embedding + 错排 m/z 掩码
        dppm_swap = np.abs(q_mz[k] - swap_mz) / np.maximum(np.abs(swap_mz), 1e-9) * 1e6
        ok_swap = (dppm_swap <= PPM)
        sims = qn[k] @ libn.T  # decoy embedding = 真实库谱
        swap_best[k] = np.where(ok_swap, sims, -np.inf).max()

    tgt_best = np.array(tgt_best)
    shuf_best = np.array(shuf_best)
    swap_best = np.array(swap_best)

    confident = tgt_best >= COS
    correct = np.array([tgt_ik[k] == q_ik[k] for k in range(len(idxs))])
    actual_fpr = float((~correct[confident]).sum()) / max(int(confident.sum()), 1)

    def tda_fdr(tgt, dec):
        nt = int((tgt >= COS).sum())
        nd = int((dec >= COS).sum())
        return (nd + 1.0) / (nt + 1.0), nt, nd

    f_shuf, nt, nd_s = tda_fdr(tgt_best, shuf_best)
    f_swap, nt2, nd_w = tda_fdr(tgt_best, swap_best)

    print(f"=== 诱饵策略校准对比 (n_query={len(idxs)}, cos>={COS}+m/z<={PPM}ppm) ===",
          flush=True)
    print(f"  真值: confident={int(confident.sum())} 实际 FDR = {actual_fpr:.4f}",
          flush=True)
    print(f"  (a) shuffle 诱饵:       N_d={nd_s:5d}  TDA fdr = {f_shuf:.4f}  "
          f"(×{f_shuf/max(actual_fpr,1e-9):.2f})", flush=True)
    print(f"  (b) precursor-swap 诱饵: N_d={nd_w:5d}  TDA fdr = {f_swap:.4f}  "
          f"(×{f_swap/max(actual_fpr,1e-9):.2f})", flush=True)
    print(f"\n  结论: 更接近 1.0x 的诱饵更竞争/更校准", flush=True)


if __name__ == "__main__":
    main()
