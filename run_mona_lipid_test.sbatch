#!/bin/bash
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=mona_lipid_test_%j.out
#SBATCH --error=mona_lipid_test_%j.err

# MONA 脂质组覆盖测试 —— 只用 Lipidomics/neg 两个测试文件（HF_1, PF_1）
# 目的：用真实脂质组数据确认 MONA 能捞到多少脂质 / 鞘脂，给「是否补脂质库」一个基线。
# 库的静态上限已经查过：MONA-neg 37303 条里鞘脂 ~0.2%（70 鞘磷脂 + ~17 神经酰胺，其余鞘脂类别 0）。

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "===== MONA 脂质组覆盖测试（Lipidomics/neg: HF_1, PF_1）====="
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# 前置：data/msv100574/Lipidomics/neg/HF_1.mzML 和 PF_1.mzML 已在服务器上

# (1) mzML -> hdf5（2 个文件，秒级）
python -u tasks/convert_met_neg_hdf5.py --mzml-dir data/msv100574/Lipidomics/neg

# (2) 嵌入（2 个测试文件 -> embeddings/lipid_neg）
python -u tasks/encode_msv100574_spectra.py \
    --hdf5 data/msv100574/Lipidomics/neg/HF_1.hdf5 \
            data/msv100574/Lipidomics/neg/PF_1.hdf5 \
    --out data/msv100574/embeddings/lipid_neg \
    --device cuda

# (3) 注释（复用已嵌入的 MONA-neg 库，不做规则注入）
python -u -m annotation.cli annotate \
    --query-dir data/msv100574/embeddings/lipid_neg \
    --library-dir data/models/mona_neg_dreams_emb \
    --out data/msv100574/annotation/lipid_neg

# (4) 鞘脂 / 脂质覆盖统计（top-1 命中里按名字扫脂质关键词）
python -u - <<'PY'
import re
import pandas as pd
ann = pd.read_csv("data/msv100574/annotation/lipid_neg/annotations.csv")
top1 = ann[ann["rank"] == 1]
conf = top1[(top1["cosine"] >= 0.7) & top1["mz_pass"]]

sph = ["Sphingomyelin","Ceramide","Sphingosine","Sphinganine","Phytosphingosine",
       "Ganglioside","Cerebroside","GlcCer","HexCer","LacCer","S1P","Sphingoid"]
lipid = sph + ["PC(","PE(","PI(","PG(","PS(","PA(","TG(","DG(","MG(","LPC(","LPE(","FA(","Acyl"]

pat_sph = "|".join(re.escape(k) for k in sph)
pat_lip = "|".join(re.escape(k) for k in lipid)

def hit(df, pat):
    m = df["lib_name"].astype(str).str.contains(pat, case=False, regex=True, na=False)
    return int(m.sum()), float(m.mean())

n_sph, r_sph = hit(top1, pat_sph)
n_lip, r_lip = hit(top1, pat_lip)
n_csph, r_csph = hit(conf, pat_sph)
print(f"[coverage] top1 总命中={len(top1)}  鞘脂名={n_sph} ({r_sph:.4f})  脂质名={n_lip} ({r_lip:.4f})")
print(f"[coverage] confident(c>=0.7&mz_pass)={len(conf)}  其中鞘脂名={n_csph} ({r_csph:.4f})")
print("[coverage] 鞘脂名命中条目（前 30）:")
m = top1["lib_name"].astype(str).str.contains(pat_sph, case=False, regex=True, na=False)
print(top1.loc[m, ["lib_name","cosine","mz_pass"]].head(30).to_string(index=False))
PY

echo "Finished: $(date)"
