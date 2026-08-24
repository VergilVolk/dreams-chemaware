#!/bin/bash
#SBATCH --gpus=1
#SBATCH --array=0-1%1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=step4_g8_%A_%a.out
#SBATCH --error=step4_g8_%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# ============================================================================
# G8 门：显式同分子正例（--same-mol-as-positive 的新实现）—— 修复 G7 的稀疏性缺陷。
#
# G7 前审计（2026-08-21，硬证据）：
#   1) batch 内同分子对稀疏：train 池 median 仅 3 谱/分子，batch=32 下同分子对期望
#      0.318/批。旧实现把同分子正例绑定 batch 内 shuffle 偶遇 → 分支 ~99.7% 不触发，
#      promote 在真实训练里几乎不生效（demo 8×16 密集构造高 750×，不可外推）。
#   2) ik14 立体异构体混并：InChIKey 字段只有 14 位（连通性），同 ik14 组内 8.86% 混有
#      不同 SMILES（立体异构体）。旧 promote 会把对映体/非对映体错误拉近。
#
# 本版 step4 修复：
#   - promote 改为「显式同分子正例」：__getitem__ 直接携带同完整 SMILES 的另一条谱作为
#     正例，独立于 batch 采样 → 81.1% 多谱锚每步都有确定的正例信号。
#   - 判据用完整 SMILES（含 @/@@ 立体标记），不再把立体异构体当同分子拉近。
#   - mask（负例排除）仍用 ik14：保守地不推远任何同骨架（含立体异构体）。
#
# λ_p 扫描（array 0-1%1，串行）：
#   task 0: λ_p=0  —— 显式正例、无保持项。与 G7 λ_p=0（旧偶遇路径）直接 A/B：
#                      若 G8 检索回基线而 G7 没回，证明稀疏性=根因、显式正例=解。
#   task 1: λ_p=5  —— 显式正例 + 中等保持。看 preserve 是否还有叠加价值。
#   判据（两者都要）：separation↑ 且 macro-AUC/Recall@1 相对基线不降（≥基线−0.01）。
#
# 前置（关键）：
#   1) 已同步新版 step4_infonce_train.py（显式同分子正例 + SMILES 判据）；
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
OUTDIR="data/validation/noise_isomer_infonce_g8"   # 显式正例版，独立目录

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

# ---- 关键前置：step4 必须是显式正例版（不是 G7 的 batch 内偶遇版）----
python - <<'PY'
import inspect
from step4_infonce_train import infonce_forward
sig = str(inspect.signature(infonce_forward))
assert "sm_pos" in sig and "sm_mask" in sig, \
    f"服务器 step4 仍是旧版（batch 内偶遇），签名 {sig}。请先同步本地改动版 step4。"
print(f"step4 OK: 显式同分子正例版（{sig}）")
PY

echo ""
echo "===== G8 λ_p=${LAM} seed=${SEED}（显式同分子正例）：10k 锚 × 8 epoch，lr=${LR}，no-amp ====="
echo "Started: $(date)"

echo ""
echo "===== [1/2] 训练（λ_p=${LAM}，same_mol mask 开 + 显式同分子正例）====="
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
    --same-mol-as-positive \
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
echo "===== G8 λ_p=${LAM} seed=${SEED} DONE $(date) ====="
echo "评估结果: ${OUTDIR}/seed_${SEED}/gate_eval.json"
echo "判据：separation↑ 且 macro-AUC/Recall@1 相对基线不降（≥基线−0.01）。"
echo "对照：G7 λ_p=0（旧偶遇路径）macro-AUC。若 G8 恢复≥0.8576 而 G7 没恢复，稀疏性=根因成立。"
