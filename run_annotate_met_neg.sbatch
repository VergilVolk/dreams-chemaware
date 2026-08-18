#!/bin/bash
#SBATCH --gpus=2
#SBATCH --time=24:00:00
#SBATCH --output=annotate_met_neg_%j.out
#SBATCH --error=annotate_met_neg_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "===== Met/neg 全量注释管线 (126 文件) ====="
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# (1) mzML -> hdf5（断点续传，7s/文件，126 个 ~15 min）
python -u tasks/convert_met_neg_hdf5.py --mzml-dir data/msv100574/Metabolomics/neg

# (2) 嵌入 87 万谱（GPU，--device cuda；CPU 版去掉 --device 则 ~27h）
python -u tasks/encode_msv100574_spectra.py \
    --hdf5 $(ls data/msv100574/Metabolomics/neg/*.hdf5) \
    --out data/msv100574/embeddings/met_neg \
    --device cuda

# (3) 注释 + 规则注入（numpy，CPU 秒级）
python -u -m annotation.cli annotate \
    --query-dir data/msv100574/embeddings/met_neg \
    --library-dir data/models/mona_neg_dreams_emb \
    --out data/msv100574/annotation/met_neg \
    --rules

# (4) FDR（诱饵嵌入，--device cuda；分块断点续传）
python -u tasks/run_fdr_met_neg.py --device cuda

# (5) 差异分析 PF vs HF（Fisher + BH，CPU 秒级）
python -u tasks/run_diff.py \
    --annotations data/msv100574/annotation/met_neg/annotations_fdr.csv \
    --out data/msv100574/annotation/met_neg/diff.csv \
    --group-a PF --group-b HF

echo "Finished: $(date)"
