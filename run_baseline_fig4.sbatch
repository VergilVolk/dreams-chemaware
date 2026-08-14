#!/bin/bash
#SBATCH --gpus=1
#SBATCH -o baseline_fig4_%j.out
#SBATCH -e baseline_fig4_%j.err
#SBATCH -t 04:00:00

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH=$PWD:$PYTHONPATH

echo "===== DreaMS Zero-Shot Baseline (Figure 4a reproduction) ====="
echo "Started at $(date)"
echo "Hostname: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# 论文协议:
#   Morgan r=2 nBits=4096
#   分层采样: 10 bins × 8000 pairs = 80,000 pairs
#   同分子vs不同分子 AUC
#   Cos vs Tanimoto 相关性

python -u dreams_zeroshot_baseline.py

echo "===== Job finished at $(date) ====="
echo ""
echo "Output: dreams_zeroshot_baseline.png + dreams_zeroshot_baseline.json"
