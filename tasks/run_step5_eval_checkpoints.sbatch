#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=step5_eval_checkpoints_%j.out
#SBATCH --error=step5_eval_checkpoints_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# ============================================================================
# 训练中评估：把当前已有的 checkpoint（seed_0/1/2 的 last.pt）分别与官方基线
# official_embedding_slim.pt 对比，看三件套（噪声一致性↑ / 异构体推远↓ / 检索不降）。
#
# step5_gate_eval.py 内部已加载官方基线并打印 base->trained 对比，所以「官方微调权重」
# 就是每个输出里的 baseline 列；本 sbatch 循环跑已有的 checkpoint 即可。
#
# 安全性：
#   - last.pt 是原子写（.tmp -> rename），训练还在跑也能读到完整快照，不脏读。
#   - 只评估已存在的 last.pt（--f 守卫），seed 还没跑出来的自动跳过。
# ============================================================================

for REQUIRED in \
    data/models/MassSpecGym_MurckoHist_split.hdf5 \
    tasks/massspecgym_isomers/dataset_manifest.json \
    data/e1/official_embedding_slim.pt \
    dreams/models/pretrained/ssl_model_server.pt \
    tasks/step5_gate_eval.py \
    tasks/noise_augment.py \
    tasks/train_e1_identity.py ; do
    if [[ ! -f "$REQUIRED" ]]; then
        echo "MISSING DEPENDENCY: $REQUIRED" >&2
        exit 2
    fi
done
echo "输入依赖齐。"

for SEED in 0 1 2; do
    CKPT="data/validation/noise_isomer_infonce/seed_${SEED}/last.pt"
    if [[ ! -f "$CKPT" ]]; then
        echo "跳过 seed_${SEED}：无 ${CKPT}"
        continue
    fi
    echo ""
    echo "===== 评估 seed_${SEED}/last.pt（vs 官方基线）$(date) ====="
    python -u tasks/step5_gate_eval.py \
        --device cuda \
        --trained "$CKPT" \
        --max-eval 2000 \
        --noise-draws 1
done

echo ""
echo "===== 训练中评估 DONE $(date) ====="
echo "结果: data/validation/noise_isomer_infonce/seed_{0,1,2}/gate_eval.json"
echo "看每段末尾三个门：噪声一致性↑(pos_cos) / 异构体↓(neg_cos) / 检索不降(auc, recall1)。"
