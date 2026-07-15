#!/bin/bash
#SBATCH --gpus=1
#SBATCH -o v5_full_%j.out
#SBATCH -e v5_full_%j.err
#SBATCH -t 08:00:00

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH=$PWD:$PYTHONPATH

echo "===== v5 Step 3 (Full): Hard-pair + MSE + α=0.05 + β=0.01 ====="
echo "Started at $(date)"
echo "Hostname: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# v5 完整版（Step 3）：
#   - Hard-pair 采样（按 precursor mass 排序分桶 → 最近邻配对）
#   - MSE 回归 cos_sim → rule_overlap
#   - α=0.05（低对比权重，mask loss 主导）
#   - β=0.01（preservation = MSE(emb_current, emb_frozen_original)）
#   - 每 500 步 held-out AUC 监控（严格按 fold='test' 隔离）
#   - 全模型微调 lr=5e-6

python -u -m dreams.models.chem_aware.train_contrastive_v5 \
    --dataset_path ./data/MassSpecGym_MurckoHist_split.hdf5 \
    --ckpt_path ./dreams/models/pretrained/ssl_model_server.pt \
    --save_dir ./contrastive_checkpoints_v5 \
    --train_fold train \
    --val_fold test \
    --epochs 5 \
    --batch_size 64 \
    --lr 5e-6 \
    --alpha 0.05 \
    --beta 0.01 \
    --easy_pair_weight 0.2 \
    --auc_monitor_steps 500 \
    --auc_n_spectra 500 \
    --auc_n_pairs 2000 \
    --overlap_categories "NL,CF,ISO"

echo "===== Job finished at $(date) ====="
