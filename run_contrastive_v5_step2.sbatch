#!/bin/bash
#SBATCH --gpus=1
#SBATCH -o v5_step2_%j.out
#SBATCH -e v5_step2_%j.err
#SBATCH -t 08:00:00

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH=$PWD:$PYTHONPATH

echo "===== v5 Step 2: MSE + Hard-pair (mass-sorted) + α=0.5 + β=0 ====="
echo "验证目标：确认 hard-pair 采样相比 Step 1 随机配对有增量价值"
echo "Started at $(date)"
echo "Hostname: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# v5 Step 2（消融实验）：
#   - Hard-pair 采样（按 precursor mass 排序分桶 → 桶内最近邻配对）
#   - 50% 质量分桶 batch + 50% 随机 shuffle batch（保持全局检索多样性）
#   - MSE 回归 cos_sim → rule_overlap
#   - α=0.5（保持与 Step 1 一致，唯一变量是采样策略）
#   - β=0.0（保持与 Step 1 一致）
#   - 每 500 步 held-out AUC 监控（fold='test'）

python -u -m dreams.models.chem_aware.train_contrastive_v5 \
    --dataset_path ./data/MassSpecGym_MurckoHist_split.hdf5 \
    --ckpt_path ./dreams/models/pretrained/ssl_model_server.pt \
    --save_dir ./contrastive_checkpoints_v5_step2 \
    --train_fold train \
    --val_fold test \
    --epochs 3 \
    --batch_size 64 \
    --lr 5e-6 \
    --alpha 0.5 \
    --beta 0.0 \
    --sorted_batch_fraction 0.5 \
    --easy_pair_weight 0.2 \
    --auc_monitor_steps 500 \
    --auc_n_spectra 500 \
    --auc_n_pairs 2000 \
    --overlap_categories "NL,CF,ISO"

echo "===== Job finished at $(date) ====="
