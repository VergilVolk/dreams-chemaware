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

# P0 实验 v2：335条规则 + Triplet 采样
#   更新：
#   - 规则库 127→335 (CompMS2Miner+MS-FINDER HR+ESI)，含 HR 类别
#   - 阈值基于新overlap分布重定: high=0.23 (batch P70), low=0.09 (batch P30)
#   - 预期每batch ~61个有效三元组
#   - Loss: Triplet Margin Loss (margin=0.2)
#   - α=0.05, β=0.02

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
    --overlap_high 0.23 \
    --overlap_low 0.09 \
    --k_pairs 2 \
    --alpha 0.05 \
    --beta 0.02 \
    --no_hard_mining \
    --auc_monitor_steps 500 \
    --auc_n_spectra 500 \
    --auc_n_pairs 2000 \
    --overlap_categories "NL,CF,ISO,HR"

echo "===== Job finished at $(date) ====="
