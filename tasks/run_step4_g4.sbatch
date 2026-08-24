#!/bin/bash
#SBATCH --gpus=1
#SBATCH --array=0-2%1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=step4_g4_%A_%a.out
#SBATCH --error=step4_g4_%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# ============================================================================
# G4 门：全量规模（89,555 锚 × 3 种子）InfoNCE 训练 + Step5 评估。
#   Slurm array 0-2%1，每个 task 一个 seed（独立目录 + 独立续训）。
#   %1 = 同一时刻只跑 1 个 task（串行）：若某 seed 出 bug 只烧 1 个 seed 的机时，不 3 倍一起烧。
#
# 前置（G3 教训，已修）：
#   1) step4 脚本已加 per-epoch last.pt（~678MB，含 optimizer state）+ 断点续训；
#   2) 本 sbatch 启动前做配额预检（写 1MB canary），超配额当场 exit，不烧机时。
#
# 时间/磁盘估算：
#   - 单 seed ≈ 18h（89,555/32 ≈ 2799 步/epoch × 8 epoch；G3 30k 锚 8 epoch ≈ 6h 实测）
#   - 单 seed 磁盘峰值 ≈ last.pt 678MB + best_infonce.pt 470MB ≈ 1.2GB；3 seed ≈ 3.5GB
#   - 启动前须确保配额 ≥ 5GB 余量；不够先清空间（旧 .out/下载物/旧 checkpoint）
#
# 失败恢复：某 seed 挂了，清完空间后原样 resubmit（array 同一 task）即可，
#   脚本会自动从该 seed 的 last.pt 续训，不重头来。
# ============================================================================

SEED="$SLURM_ARRAY_TASK_ID"
EPOCHS=8
BATCH=32
UNFREEZE=2
LR=3e-5
TRAINED="data/validation/noise_isomer_infonce/seed_${SEED}/best_infonce.pt"

# ---- 配额预检（写 1MB canary，超配额当场失败，避免重演 G3） ----
QUOTA_TEST="data/validation/.quota_test_${SLURM_JOB_ID}_${SEED}"
if ! dd if=/dev/zero of="$QUOTA_TEST" bs=1M count=1 2>/dev/null; then
    echo "配额预检失败：磁盘/配额不可写。请先清空间再 resubmit（本 seed 不会重头训）。" >&2
    exit 4
fi
rm -f "$QUOTA_TEST"
echo "配额预检 OK（可写）"

for REQUIRED in \
    data/models/MassSpecGym_MurckoHist_split.hdf5 \
    tasks/massspecgym_isomers/dataset_manifest.json \
    data/e1/official_embedding_slim.pt \
    dreams/models/pretrained/ssl_model_server.pt \
    tasks/step4_infonce_train.py \
    tasks/step5_gate_eval.py \
    tasks/noise_augment.py \
    tasks/train_e1_identity.py ; do
    if [[ ! -f "$REQUIRED" ]]; then
        echo "MISSING DEPENDENCY: $REQUIRED" >&2
        exit 2
    fi
done
echo "输入依赖齐。"

echo "===== G4 全量 seed=${SEED}：89,555 锚 × 8 epoch，lr=${LR}，no-amp，unfreeze=${UNFREEZE} ====="
echo "Started: $(date)"

echo ""
echo "===== [1/2] 训练 InfoNCE（seed=${SEED}）====="
# --no-amp：G3 首跑发现 AMP fp16 在 epoch0 末溢出 NaN，纯 fp32 无溢出天花板。
# --lr 3e-5：PartB/C 定案——1e-4/3e-4 会 rank 坍缩 + 检索退化，3e-5 唯一四门全过。
# 不传 --max-anchors（默认 0 = 全量 89,555 锚）。
python -u tasks/step4_infonce_train.py \
    --device cuda \
    --no-amp \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH}" \
    --unfreeze-layers "${UNFREEZE}" \
    --lr "${LR}"

if [[ ! -f "${TRAINED}" ]]; then
    echo "训练未产出 checkpoint: ${TRAINED}" >&2
    exit 3
fi
echo "训练完成 -> ${TRAINED}"

echo ""
echo "===== [2/2] Step5 评估（seed=${SEED}）====="
python -u tasks/step5_gate_eval.py \
    --device cuda \
    --trained "${TRAINED}" \
    --max-eval 2000 \
    --noise-draws 1

echo ""
echo "===== G4 seed=${SEED} DONE $(date) ====="
echo "评估结果: data/validation/noise_isomer_infonce/seed_${SEED}/gate_eval.json"
