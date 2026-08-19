#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=step4_g3_%j.out
#SBATCH --error=step4_g3_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# ============================================================================
# G3 门（单尺度方向检查，plan doc §7）：训练 InfoNCE（噪声不变性 + 异构体区分）
#   -> 跑 Step 5 评估三件事：噪声一致性↑ / 异构体推远(FP守卫) / 10ppm检索不降。
#
# 通过后 G4 才是「全量 3 种子」。本 job 只跑 1 个 seed、30k 锚、8 epoch，
# 先验证方向，避免方向错就烧 3 seed 全量。
#
# DEPENDENCIES（须已同步/安装到服务器，否则 exit 2）：
#   1. data/models/MassSpecGym_MurckoHist_split.hdf5    # 谱数据（旧文件，应在）
#   2. tasks/massspecgym_isomers/dataset_manifest.json  # Step 3 产物（新，须同步）
#   3. data/e1/official_embedding_slim.pt               # 起点（旧文件，应在）
#   4. dreams/models/pretrained/ssl_model_server.pt     # 架构（旧文件，应在）
#   5. tasks/step4_infonce_train.py                     # 训练脚本（新，须同步）
#   6. tasks/step5_gate_eval.py                         # 评估脚本（新，须同步）
#   7. tasks/noise_augment.py                           # 四轴噪声（新，须同步）
#   8. tasks/train_e1_identity.py                       # load_base_model/preprocess（旧，应在）
#   注：pairs.json / ik_to_rows.json 是 Step1/3 输入，Step4/5 运行时已不读（负例行号已
#       内嵌进 dataset_manifest.json），故不在此校验。
#
# GENERATED（本 job 产出）：
#   data/validation/noise_isomer_infonce/seed_0/best_infonce.pt   # 训练后模型
#   data/validation/noise_isomer_infonce/seed_0/summary.json      # G2 门+训练摘要
#   data/validation/noise_isomer_infonce/seed_0/gate_eval.json    # G3 门评估结果
# ============================================================================

SEED=0
MAX_ANCHORS=30000
EPOCHS=8
BATCH=32
UNFREEZE=2
LR=3e-5
TRAINED="data/validation/noise_isomer_infonce/seed_${SEED}/best_infonce.pt"

echo "===== G3 门：InfoNCE 训练 + Step5 评估（seed=${SEED}, anchors=${MAX_ANCHORS}, epochs=${EPOCHS}）====="
echo "Started: $(date)"

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
        echo "MISSING DEPENDENCY: $REQUIRED  (see DEPENDENCIES block above)" >&2
        exit 2
    fi
done
echo "输入依赖齐。"
echo ""

echo "===== [1/2] 训练 InfoNCE（seed=${SEED}）====="
# --no-amp：G3 首跑发现 AMP fp16 在 epoch0 末(step~880)溢出 NaN——模型变自信后
#   last-2 层 scale 漂移使 fp16 激活溢出。纯 fp32 无溢出天花板（G2 CPU fp32 已验证）。
# --lr 3e-5：PartB/C 定案——1e-4/3e-4 会 rank 坍缩 + 检索退化（auc 0.739→0.718/0.675），
#   3e-5 是唯一四门全过（pos↑/neg↓/rank 42.5 更健康/检索 0.744 不退化）。
python -u tasks/step4_infonce_train.py \
    --device cuda \
    --no-amp \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH}" \
    --unfreeze-layers "${UNFREEZE}" \
    --lr "${LR}" \
    --max-anchors "${MAX_ANCHORS}"

if [[ ! -f "${TRAINED}" ]]; then
    echo "训练未产出 checkpoint: ${TRAINED}" >&2
    exit 3
fi
echo "训练完成 -> ${TRAINED}"
echo ""

echo "===== [2/2] Step5 评估（G3 门：噪声一致性 / 异构体 / 检索）====="
python -u tasks/step5_gate_eval.py \
    --device cuda \
    --trained "${TRAINED}" \
    --max-eval 2000 \
    --noise-draws 1

echo ""
echo "===== G3 DONE $(date) ====="
echo "评估结果: data/validation/noise_isomer_infonce/seed_${SEED}/gate_eval.json"
echo "  gate_overall=true  -> 方向对，进 G4 全量 3 种子。"
echo "  gate_overall=false -> 见 gate_eval.json 逐项 base->trained 定位哪一门挂。"
