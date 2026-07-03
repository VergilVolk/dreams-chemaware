#!/bin/bash
#SBATCH --gpus=1
#SBATCH -o chem_v3_%j.out
#SBATCH -e chem_v3_%j.err
#SBATCH -t 02:00:00

# 强制禁用所有缓冲
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH=$PWD:$PYTHONPATH

echo "===== ChemAware v3 Job started at $(date) ====="
echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# v3 训练：奖励式化学偏置 + 逐规则独立权重 + 仅最后一层注入
python -u -m dreams.models.chem_aware.train_chem_aware \
    --dataset_path ./data/MassSpecGym_MurckoHist_split.hdf5 \
    --ckpt_path ./dreams/models/pretrained/ssl_model_server.pt \
    --save_dir ./chem_aware_checkpoints_v3 \
    --epochs 5 \
    --batch_size 16

echo "===== Job finished at $(date) ====="
