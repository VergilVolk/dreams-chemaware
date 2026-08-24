"""MolForge decoder 接入注释流程的最小可跑演示（de novo 结构提案的雏形）。

诚实的管线形状（三步，只有第 2 步是占位）：

    query MS2 ──► DreaMS 1024-d embedding   [已存在  annotation.embed]
         ──► embedding ──► ECFP4 指纹       [STUB  emb->fp head 未训练]
         ──► MolForge fp ──► SMILES         [已存在  annotation.molforge_decoder]

这个脚本**只证明 decoder 这一条腿**在真实 query 谱上端到端可跑，并让你看清楚
「emb->fp 头」到底该插在哪。当前 `embedding_to_fp()` 是**占位实现**：它返回
top-1 检索命中的自身 ECFP4 指纹，所以产出的 candidate 只是「MolForge 把检索
命中重建了一遍」——**不是** de novo 提案。de novo 只有等 emb->fp 头训练好、
替换进 `embedding_to_fp()` 之后才成立。

用法 (conda dreams_env, 服务器/本机 CPU 均可；本机 ~1.8s/条):
    python tasks/molforge_decoder_demo.py \
        --query-dir data/mtbls13729/smoke/emb \
        --annotations data/mtbls13729/smoke/ann/annotations.csv \
        --checkpoint third_party/MolForge/saved_models/ECFP4_selfies_checkpoint.pth \
        --n 12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Windows 控制台默认 GBK，无法输出 emoji/部分中文——统一到 UTF-8。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from annotation.molforge_decoder import MolForgeDecoder, ecfp4_bits  # noqa: E402
from rdkit import Chem, DataStructs  # noqa: E402
from rdkit.Chem import rdMolDescriptors  # noqa: E402

DEFAULT_QUERY = REPO / "data" / "mtbls13729" / "smoke" / "emb"
DEFAULT_ANN = REPO / "data" / "mtbls13729" / "smoke" / "ann" / "annotations.csv"
DEFAULT_CKPT = REPO / "third_party" / "MolForge" / "saved_models" / "ECFP4_selfies_checkpoint.pth"


def tanimoto(smiles_a: str, smiles_b: str) -> float:
    ma, mb = Chem.MolFromSmiles(smiles_a), Chem.MolFromSmiles(smiles_b)
    if ma is None or mb is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(
        rdMolDescriptors.GetMorganFingerprintAsBitVect(ma, 2, 2048),
        rdMolDescriptors.GetMorganFingerprintAsBitVect(mb, 2, 2048),
    )


def embedding_to_fp(query_emb: np.ndarray, retrieval_smiles: str) -> list[int]:
    """STUB：embedding -> ECFP4 指纹。

    真实实现 = 一个小 head（多标签 BCE，2048 个二分类头）把 1024-d embedding
    映射成 2048-bit ECFP4 开位。当前未训练，因此用 top-1 检索命中的指纹顶替，
    只为了让下游 decoder 能跑起来。换掉这个函数 = 完成 Part 3 的科研核心。

    ``query_emb`` 参数在此占位实现中未被使用——这正是缺的那块。
    """
    del query_emb  # 占位：emb 还没用上
    return ecfp4_bits(retrieval_smiles)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query-dir", type=Path, default=DEFAULT_QUERY)
    ap.add_argument("--annotations", type=Path, default=DEFAULT_ANN)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--n", type=int, default=12, help="演示解码的 query 谱数")
    ap.add_argument("--cosine-min", type=float, default=0.7,
                    help="只取 cosine >= 该阈值且 mz_pass 的 top-1 命中")
    args = ap.parse_args()

    if not args.checkpoint.exists():
        print(f"[demo] checkpoint 不存在: {args.checkpoint}", file=sys.stderr)
        return 1

    # 1) 加载 query embedding（1024-d）与检索结果
    emb = np.load(args.query_dir / "embeddings.npy")
    q_manifest = pd.read_csv(args.query_dir / "manifest.csv")
    ann = pd.read_csv(args.annotations)
    top1 = ann[(ann["rank"] == 1) & (ann["mz_pass"]) & (ann["cosine"] >= args.cosine_min)].copy()
    top1 = top1.drop_duplicates("query_idx").head(args.n)

    print(f"query embedding: {emb.shape}  (DreaMS 1024-d, L2-normalized)")
    print(f"top-1 命中(cos>={args.cosine_min} & mz_pass): {len(top1)} 条\n")

    # 2) 加载 decoder（自包含，无 sentencepiece/selfies 导入依赖）
    dec = MolForgeDecoder(model_type="selfies")
    dec.load_checkpoint(args.checkpoint)

    rows = []
    for _, r in top1.iterrows():
        qi = int(r["query_idx"])
        q_emb = emb[qi]
        retrieval_smiles = str(r["lib_smiles"])
        if not retrieval_smiles or Chem.MolFromSmiles(retrieval_smiles) is None:
            continue
        fp = embedding_to_fp(q_emb, retrieval_smiles)          # ← 占位头
        candidate = dec.decode_bits_to_smiles(fp)              # ← 真 decoder
        t = tanimoto(candidate, retrieval_smiles)
        rows.append({
            "query_idx": qi,
            "cosine": float(r["cosine"]),
            "dppm": float(r["dppm"]),
            "retrieval_smiles": retrieval_smiles,
            "molforge_candidate": candidate,
            "tanimoto_vs_retrieval": round(t, 3),
            "exact": int(candidate == retrieval_smiles),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        print("[demo] 没有可解码的命中（SMILES 均无效）", file=sys.stderr)
        return 1

    out_path = args.query_dir / "molforge_demo_candidates.csv"
    out.to_csv(out_path, index=False)

    print("=" * 96)
    for _, r in out.iterrows():
        flag = "exact" if r["exact"] else f"T={r['tanimoto_vs_retrieval']:.2f}"
        print(f"[q{r['query_idx']:>5}] cos={r['cosine']:.2f}  {flag}")
        print(f"          检索命中  {r['retrieval_smiles']}")
        print(f"          MolForge  {r['molforge_candidate']}")
    print("=" * 96)
    print(f"\n{len(out)} 条完成，Tanimoto(vs 检索命中) 均值 = "
          f"{out['tanimoto_vs_retrieval'].mean():.3f}")
    print(f"写入 -> {out_path}")

    print("\n[!] 诚实说明：")
    print("  candidate ≈ 检索命中，是因为 embedding_to_fp() 是占位（直接用了检索命中的指纹）。")
    print("  MolForge 把那个指纹重建回 SMILES，Tanimoto≈0.89 即 decoder 腿的能力上限。")
    print("  真正的 de novo（暗物质特征 → 新结构）需要训练 emb->fp 头替换该占位函数。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
