#!/bin/bash
#SBATCH --gpus=1
#SBATCH -o triplet_t1_%j.out
#SBATCH -e triplet_t1_%j.err
#SBATCH -t 12:00:00

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH=$PWD:$PYTHONPATH

echo "===== T1 MCES Triplet Fine-Tuning ====="
echo "Started at $(date)"
echo "Hostname: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# 数据准备（首次运行需要，后续可注释）
# python build_annotated01.py
# python -c "from tasks.build_utils import load_indices; load_indices(force_rebuild=True)"
# python tasks/T1_near_isomers/code/build_test_cases.py --max_groups 2000 --max_pairs_per_group 20

# Triplet 训练
#   - 7030 train + 780 val triplets
#   - Anchor+Positive: 同分子式 MCES[0,2] (近同分异构体)
#   - Negative: MCES[6,10] (明确不同异构体)
#   - Loss: L_mask + α·L_triplet + β·L_preservation
#   - 全模型微调 lr=5e-6

python -u -m dreams.models.chem_aware.train_triplet_t1 \
    --ckpt_path ./dreams/models/pretrained/ssl_model_server.pt \
    --epochs 10 \
    --batch_size 32 \
    --lr 5e-6 \
    --alpha 0.5 \
    --beta 0.01 \
    --margin 0.3 \
    --save_dir ./triplet_t1_checkpoints

echo "===== Job finished at $(date) ====="
echo ""
echo "Resume:"
echo "  python -m dreams.models.chem_aware.train_triplet_t1 --ckpt_path ... --resume_from ./triplet_t1_checkpoints/epoch_N.pt"
