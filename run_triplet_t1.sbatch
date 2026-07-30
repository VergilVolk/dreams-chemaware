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

# T1 MCES Triplet 训练
#   7030 train + 780 val triplets
#   Anchor+Positive: 同分子式 MCES[0,2] (近同分异构体)
#   Negative: MCES[6,10] (明确不同异构体)
#   Loss: L_mask + α·L_triplet + β·L_preservation
#
# 参数选择（v5 方向二验证经验）:
#   α=0.05  (v4的α=0.5导致坍缩到AUC=0.59)
#   β=0.02  (preservation需足够强防嵌入漂移)
#   margin=0.2 (v5验证最优值)
#   lr=5e-6 (低学习率,防止灾难性遗忘)
#   val_every=1 (每个epoch评估验证集Separation+Accuracy)
#   save_best_only (只保存val_sep最高的checkpoint)

python -u -m dreams.models.chem_aware.train_triplet_t1 \
    --ckpt_path ./dreams/models/pretrained/ssl_model_server.pt \
    --epochs 10 \
    --batch_size 32 \
    --lr 5e-6 \
    --alpha 0.05 \
    --beta 0.02 \
    --margin 0.2 \
    --save_dir ./triplet_t1_checkpoints \
    --val_every 1 \
    --save_best_only

echo "===== Job finished at $(date) ====="
echo ""
echo "Best checkpoint: ./triplet_t1_checkpoints/best.pt"
echo ""
echo "Resume:"
echo "  python -m dreams.models.chem_aware.train_triplet_t1 --ckpt_path ... --resume_from ./triplet_t1_checkpoints/epoch_N.pt"
