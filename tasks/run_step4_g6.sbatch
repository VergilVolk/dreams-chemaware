#!/bin/bash
#SBATCH --gpus=1
#SBATCH --array=0-1%1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=step4_g6_%A_%a.out
#SBATCH --error=step4_g6_%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# ============================================================================
# G6 门：修复 in-batch 同分子负例污染 —— 验证「推远异构体」本身不伤检索，
#        真正伤检索的是「同分子不同谱」被误当负例推远。
#
# 背景（G5 λ_p=5 实证）：
#   separation 0.43→0.66（↑ 异构体确实被推远）、噪声一致性守住 0.891，
#   但检索 macro-AUC 0.8676→0.8216（−4.6pp）、Recall@1 −1pp。
#   根因（代码+数据已证）：infonce_forward 的 in-batch 负例
#     negs=[sim_aa[i,mask], sim_ap[i,mask]] 未按 ik14 过滤，把「同分子不同谱」
#     （train 97.8% 锚属多谱分子、2.48M 同 ik14 谱对）误当负例推远；
#     而 neg（异构体）字段 100% 干净。→ 本版已在 step4 里 mask 掉同 ik14 对。
#
# λ_p 扫描（array 0-1%1，串行）：
#   task 0: λ_p=0  —— mask 开、无保持项。判据：若检索回基线，说明污染=唯一伤因，
#                        L_preserve 可退役；同时 separation 应仍↑。
#   task 1: λ_p=5  —— mask 开、中等保持。与 G5 λ_p=5(mask 关) 直接 A/B。
#   判据（两者都要）：separation↑ 且 检索 macro-AUC/Recall@1 相对基线不降（≥基线−0.01）。
#
# 前置：
#   1) 已同步改动版 step4_infonce_train.py（含 same_mol mask，默认开）；
#   2) manifest 仍为连通分量版（脚本有断言兜底）。
# ============================================================================

case "$SLURM_ARRAY_TASK_ID" in
  0) LAM=0.0 ;;
  1) LAM=5.0 ;;
esac

SEED=0
EPOCHS=8
BATCH=32
UNFREEZE=2
LR=3e-5
MAXANCHORS=10000
OUTDIR="data/validation/noise_isomer_infonce_g6"   # mask 修复版，独立目录

# ---- 输入依赖检查 ----
for REQUIRED in \
    data/models/MassSpecGym_MurckoHist_split.hdf5 \
    tasks/massspecgym_isomers/dataset_manifest.json \
    data/e1/official_embedding_slim.pt \
    dreams/models/pretrained/ssl_model_server.pt \
    tasks/step4_infonce_train.py \
    tasks/step5_gate_eval.py \
    tasks/noise_augment.py \
    tasks/train_e1_identity.py ; do
    [[ -f "$REQUIRED" ]] || { echo "MISSING DEPENDENCY: $REQUIRED" >&2; exit 2; }
done

# ---- 关键前置：manifest 必须是连通分量版（泄漏已修），否则结果作废 ----
python - <<'PY'
import json
m = json.load(open("tasks/massspecgym_isomers/dataset_manifest.json"))
meta = m["meta"]
assert meta["split"] == "isomer_connected_component", \
    "manifest 仍是旧泄漏版！先把本地重生成的 dataset_manifest.json 同步到服务器。"
assert meta["leak_molecule_overlap"] == 0 and meta["leak_spectrum_row_overlap"] == 0, \
    "manifest 仍有跨 split 重叠，检查 step3 输出。"
print(f"manifest OK: 连通分量划分，重叠全 0（train={meta['n_train_anchors']}, eval={meta['n_eval_anchors']}）")
PY

echo ""
echo "===== G6 λ_p=${LAM} seed=${SEED}（same_mol mask 修复版）：10k 锚 × 8 epoch，lr=${LR}，no-amp ====="
echo "Started: $(date)"

echo ""
echo "===== [1/2] 训练（λ_p=${LAM}，same_mol mask 开）====="
python -u tasks/step4_infonce_train.py \
    --device cuda \
    --no-amp \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH}" \
    --unfreeze-layers "${UNFREEZE}" \
    --lr "${LR}" \
    --max-anchors "${MAXANCHORS}" \
    --preserve-lambda "${LAM}" \
    --output-dir "${OUTDIR}"

CKPT="${OUTDIR}/seed_${SEED}/best_infonce.pt"
if [[ ! -f "${CKPT}" ]]; then
    echo "训练未产出 checkpoint: ${CKPT}" >&2
    exit 3
fi
echo "训练完成 -> ${CKPT}"

echo ""
echo "===== [2/2] Step5 门评估（配对 + 全量候选池）====="
python -u tasks/step5_gate_eval.py \
    --device cuda \
    --trained "${CKPT}" \
    --max-eval 2000 \
    --noise-draws 1

echo ""
echo "===== G6 λ_p=${LAM} seed=${SEED} DONE $(date) ====="
echo "评估结果: ${OUTDIR}/seed_${SEED}/gate_eval.json"
echo "判据：separation↑ 且 macro-AUC/Recall@1 相对基线不降（≥基线−0.01）。"
echo "对照：G5 λ_p=5(mask 关) macro-AUC 0.8676→0.8216。若 G6 恢复≥0.8576，污染=唯一伤因成立。"
