#!/bin/bash
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=annotate_met_neg_restart_%j.out
#SBATCH --error=annotate_met_neg_restart_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "===== Met/neg 注释续跑（hdf5 + query 嵌入已完成，跳过）====="
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# (1) 库嵌入：在服务器上重嵌 MONA-neg 库
#     与 query 嵌入用同一条代码路径 + 同权重（encode_msv100574_spectra.py 的同款 reconstruct_backbone）
#     需要 data/models/mona_neg_full.mgf 已在服务器上（下一步 FDR 生成诱饵也要用）
python -u tasks/encode_mona_neg_library.py \
    --mgf data/models/mona_neg_full.mgf \
    --out data/models/mona_neg_dreams_emb \
    --device cuda

# (2) 规则命中缓存（CPU）：给全部 query 谱算 335 规则命中矩阵 -> rule_hits.npy
#     必须在 annotate 之前跑，annotate --rules 才能读到（否则只是警告、没有规则证据列）
python -u tasks/explore_rule_evidence.py

# (3) 注释 + 规则（cosine top-k + 前体 m/z 约束 + Schymanski 分级）
python -u -m annotation.cli annotate \
    --query-dir data/msv100574/embeddings/met_neg \
    --library-dir data/models/mona_neg_dreams_emb \
    --out data/msv100574/annotation/met_neg \
    --rules

# (4) FDR（诱饵嵌入 GPU + target/decoy q-value）
python -u tasks/run_fdr_met_neg.py --device cuda

# (5) 差异分析 PF vs HF（Fisher + BH）
python -u tasks/run_diff.py \
    --annotations data/msv100574/annotation/met_neg/annotations_fdr.csv \
    --out data/msv100574/annotation/met_neg/diff.csv \
    --group-a PF --group-b HF

echo "Finished: $(date)"
