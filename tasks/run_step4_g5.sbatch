#!/bin/bash
#SBATCH --gpus=1
#SBATCH --array=0-2%1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=step4_g5_%A_%a.out
#SBATCH --error=step4_g5_%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# ============================================================================
# G5 门：L_preserve 表示保持项 —— 能否在保住「异构体推远」的同时，把 10ppm 检索拉回来？
#
# 背景：G4 三个种子的 InfoNCE（无保持项）都赢了 separation（0.278→0.54~0.62）、
#   输了检索（Recall@1 全降 1~2pp）。这正是「空间被推散」的征兆。
#   L_preserve = λ_p · mean(1 − cos(z_train, z_official))，把每个 clean 锚的表示
#   拉回官方冻结嵌入，防整体漂移。
#
# λ_p 扫描（array 0-2%1，串行，一次只烧 1 个 task）：
#   task 0: λ_p=0    —— 干净基线（新连通分量 manifest 上，无保持项的检索退化到底多少）
#   task 1: λ_p=5    —— 中等锚定
#   task 2: λ_p=20   —— 强锚定
#   判据：λ_p>0 时 separation 仍 ↑ 且 检索 Recall@1/AUC 相对 λ_p=0 基线不降。
#   若 λ_p=20 仍救不回检索 → L_preserve 单独不够，下一步改 near-only 困难负例（两条腿）。
#
# 规模：10k 锚 × 1 seed × 8 epoch（师兄审计定的「先小规模过门」）。
# 前置（务必先做，否则白烧）：
#   1) 已同步新 manifest（连通分量划分）到服务器 —— 本脚本有断言兜底；
#   2) 已同步改动代码：step3/step4/step5/noise_augment（含 L_preserve + 配对评估 + do_shift 关）。
# ============================================================================

case "$SLURM_ARRAY_TASK_ID" in
  0) LAM=0.0 ;;
  1) LAM=5.0 ;;
  2) LAM=20.0 ;;
esac

SEED=0
EPOCHS=8
BATCH=32
UNFREEZE=2
LR=3e-5
MAXANCHORS=10000
OUTDIR="data/validation/noise_isomer_infonce_g5"   # 独立目录，不污染旧 seed_0/1/2

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
print(f"manifest OK: 连通分量划分，anchor/分子/谱行重叠全 0（train={meta['n_train_anchors']}, eval={meta['n_eval_anchors']}）")
PY

echo ""
echo "===== G5 λ_p=${LAM} seed=${SEED}：10k 锚 × 8 epoch，lr=${LR}，no-amp，unfreeze=${UNFREEZE} ====="
echo "Started: $(date)"

echo ""
echo "===== [1/2] 训练（λ_p=${LAM}）====="
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
echo "===== G5 λ_p=${LAM} seed=${SEED} DONE $(date) ====="
echo "评估结果: ${OUTDIR}/seed_${SEED}/gate_eval.json"
echo "看三件事：噪声一致性↑(pos_cos) / 异构体↓(neg_cos) / 检索不降(auc, recall1)。"
