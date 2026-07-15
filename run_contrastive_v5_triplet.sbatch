#!/bin/bash
#SBATCH --gpus=1
#SBATCH -o triplet_%j.out
#SBATCH -e triplet_%j.err
#SBATCH -t 08:00:00

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH=$PWD:$PYTHONPATH

echo "===== P0: Rule-guided Triplet Sampling ====="
echo "Started at $(date)"
echo "Hostname: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# P0 实验：规则重叠度引导的 Triplet 采样
#   核心变化：规则重叠度不做回归目标，只做采样依据
#   - 正样本：batch 内规则重叠度最高且 > 0.3 的谱图
#   - 负样本：batch 内规则重叠度最低且 < 0.1 的谱图
#   - Loss: Triplet Margin Loss (margin=0.2, 比 v4 的 1.0 保守得多)
#   - α=0.05, β=0.02
#   - 随机 batch（无需 precursor mass 分桶）
#   - 每 500 步报告 Sep、AUC、Triplet Loss

python -u -m dreams.models.chem_aware.train_contrastive_v5 \
    --dataset_path ./data/MassSpecGym_MurckoHist_split.hdf5 \
    --ckpt_path ./dreams/models/pretrained/ssl_model_server.pt \
    --save_dir ./contrastive_checkpoints_v5_triplet \
    --train_fold train \
    --val_fold test \
    --epochs 5 \
    --batch_size 64 \
    --lr 5e-6 \
    --loss_mode triplet \
    --triplet_margin 0.2 \
    --overlap_high 0.3 \
    --overlap_low 0.1 \
    --k_pairs 2 \
    --alpha 0.05 \
    --beta 0.02 \
    --no_hard_mining \
    --auc_monitor_steps 500 \
    --auc_n_spectra 500 \
    --auc_n_pairs 2000 \
    --overlap_categories "NL,CF,ISO"

echo "===== Job finished at $(date) ====="
