#!/bin/bash
#SBATCH --gpus=1
#SBATCH -o triplet_t1_v2_%j.out
#SBATCH -e triplet_t1_v2_%j.err
#SBATCH -t 12:00:00

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

cd /data02/run01/scv7tsl/DreaMS
git pull
export PYTHONPATH=$PWD:$PYTHONPATH

echo "===== T1 V2 MCES [3,5] Hard Negative Triplet Training ====="
echo "Started at $(date)"
echo "Hostname: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# T1 V2 Triplet 训练 — MCES [3,5] Hard Negatives
#   Anchor+Positive: 同分子式 MCES[0,2]
#   Negative: MCES[3,5] (真正的hard negatives, 之前被排除!)
#   Loss: L_mask + α·L_triplet + β·L_preservation
#   ChemAwareDreaMS: 335 rules, last-layer injection
#
# 参数:
#   margin=0.4  (MCES[3,5]分离度~0.2-0.3, 需要更宽margin才有梯度)
#   α=0.05 β=0.02 (v5 sweep验证)

python -u -m dreams.models.chem_aware.train_triplet_t1 \
    --ckpt_path ./dreams/models/pretrained/ssl_model_server.pt \
    --epochs 10 \
    --batch_size 32 \
    --lr 5e-6 \
    --alpha 0.05 \
    --beta 0.02 \
    --margin 0.4 \
    --triplet_dir tasks/T1_v2_hard \
    --save_dir ./triplet_t1_v2_checkpoints \
    --val_every 1 \
    --save_best_only \
    --auc_pairs 2000

echo "===== Job finished at $(date) ====="
echo ""
echo "Best checkpoint: ./triplet_t1_v2_checkpoints/best.pt"
