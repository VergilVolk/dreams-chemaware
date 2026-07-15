#!/bin/bash
#SBATCH --gpus=1
#SBATCH -o contrastive_%j.out
#SBATCH -e contrastive_%j.err
#SBATCH -t 06:00:00

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH=$PWD:$PYTHONPATH

echo "===== Contrastive v4.1 Job started at $(date) ====="
echo "Hostname: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# v4.1: 软对比 + preservation loss + 更保守的参数
# 关键改动:
#   - triplet_weight 0.5→0.1 (mask loss 主导，对比只做引导)
#   - overlap_threshold 0.3→0.5 (更严格的正样本)
#   - lr 1e-5→5e-6 (更慢的学习)
#   - 全模型微调 (不是只有最后2层)
#   - MSE (cos_sim→rule_overlap) 替代硬 triplet margin

python -u -m dreams.models.chem_aware.train_contrastive \
    --dataset_path ./data/MassSpecGym_MurckoHist_split.hdf5 \
    --ckpt_path ./dreams/models/pretrained/ssl_model_server.pt \
    --save_dir ./contrastive_checkpoints_v4.1 \
    --epochs 5 \
    --batch_size 64 \
    --lr 5e-6 \
    --triplet_weight 0.1 \
    --triplet_margin 0.3 \
    --overlap_threshold 0.5 \
    --overlap_categories "NL,CF,ISO"

echo "===== Job finished at $(date) ====="
