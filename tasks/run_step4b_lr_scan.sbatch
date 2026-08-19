#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=step4b_lr_scan_%j.out
#SBATCH --error=step4b_lr_scan_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# ============================================================================
# Part B：学习率扫描 + 训练中坍缩监控（定 lr）。--no-amp 已内建（脚本不用 AMP）。
#
# 对每个 lr 短训 3 epoch，每 epoch 在 held-out eval 分子上采三套指标，
# 读 Part A baseline.json 做参照，选「不坍缩前提下最高 lr」。
#
# DEPENDENCIES（须已同步/安装到服务器，否则 exit 2）：
#   1. data/models/MassSpecGym_MurckoHist_split.hdf5    # 谱数据（旧，应在）
#   2. tasks/massspecgym_isomers/dataset_manifest.json  # Step 3 产物（新，须同步）
#   3. data/e1/official_embedding_slim.pt               # 起点（旧，应在）
#   4. dreams/models/pretrained/ssl_model_server.pt     # 架构（旧，应在）
#   5. tasks/step4b_lr_scan.py                          # 本脚本（新，须同步）
#   6. tasks/step4_infonce_train.py                     # 复用训练组件（新，须同步）
#   7. tasks/noise_augment.py                           # 四轴噪声（新，须同步）
#   8. tasks/train_e1_identity.py                       # load_base_model（旧，应在）
#   9. tasks/metrics_space.py                           # 三套指标（新，须同步）
#  10. data/validation/space_audit/baseline.json        # Part A 产物（新，须同步！决策阈值来源）
#
# GENERATED（本 job 产出）：
#   data/validation/lr_scan/lr_scan.json   # 每个 lr 的逐 epoch 指标 + 决策表
# ============================================================================

LRS="3e-6,1e-5,3e-5,1e-4,3e-4"
EPOCHS=3
MAX_TRAIN=4000
EVAL_MOLECULES=300
BASELINE="data/validation/space_audit/baseline.json"

echo "===== Part B：LR 扫描（lrs=${LRS}, epochs=${EPOCHS}, train=${MAX_TRAIN} 锚, eval=${EVAL_MOLECULES} 分子）====="
echo "Started: $(date)"

for REQUIRED in \
    data/models/MassSpecGym_MurckoHist_split.hdf5 \
    tasks/massspecgym_isomers/dataset_manifest.json \
    data/e1/official_embedding_slim.pt \
    dreams/models/pretrained/ssl_model_server.pt \
    tasks/step4b_lr_scan.py \
    tasks/step4_infonce_train.py \
    tasks/noise_augment.py \
    tasks/train_e1_identity.py \
    tasks/metrics_space.py \
    "${BASELINE}" ; do
    if [[ ! -f "$REQUIRED" ]]; then
        echo "MISSING DEPENDENCY: $REQUIRED  (see DEPENDENCIES block above)" >&2
        exit 2
    fi
done
echo "输入依赖齐。"
echo ""

python -u tasks/step4b_lr_scan.py \
    --device cuda \
    --lrs "${LRS}" \
    --epochs "${EPOCHS}" \
    --max-train-anchors "${MAX_TRAIN}" \
    --eval-molecules "${EVAL_MOLECULES}" \
    --baseline-json "${BASELINE}"

echo ""
echo "===== Part B DONE $(date) ====="
echo "结果: data/validation/lr_scan/lr_scan.json"
echo "  看末尾决策表：'✓ 选' 的那行即选定 lr；无 '✓ 选' 则全坍缩，回查噪声/负例配比。"
