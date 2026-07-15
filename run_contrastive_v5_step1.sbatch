#!/bin/bash
#SBATCH --gpus=1
#SBATCH -o v5_step1_%j.out
#SBATCH -e v5_step1_%j.err
#SBATCH -t 08:00:00

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH=$PWD:$PYTHONPATH

echo "===== v5 Step 1: MSE + Random pairs (等量) + α=0.5 + β=0 ====="
echo "验证目标：确认 MSE 回归本身不导致坍缩（与 v4 硬 triplet 对比）"
echo "Started at $(date)"
echo "Hostname: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# v5 Step 1（消融实验）：
#   - 随机配对，数量与 hard-pair 模式对齐（~batch_size 对，而非全排列 N²）
#   - MSE 回归 cos_sim → rule_overlap
#   - α=0.5（高对比权重，模拟 v4 的 triplet_weight）
#   - β=0.0（无正则化）
#   - 随机 shuffle dataloader（不按质量分桶）
#   - 每 500 步 held-out AUC 监控（fold='test'）

python -u -m dreams.models.chem_aware.train_contrastive_v5 \
    --dataset_path ./data/MassSpecGym_MurckoHist_split.hdf5 \
    --ckpt_path ./dreams/models/pretrained/ssl_model_server.pt \
    --save_dir ./contrastive_checkpoints_v5_step1 \
    --train_fold train \
    --val_fold test \
    --epochs 5 \
    --batch_size 64 \
    --lr 5e-6 \
    --alpha 0.5 \
    --beta 0.0 \
    --no_hard_mining \
    --auc_monitor_steps 500 \
    --auc_n_spectra 500 \
    --auc_n_pairs 2000 \
    --overlap_categories "NL,CF,ISO"

echo "===== Job finished at $(date) ====="
