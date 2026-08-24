#!/bin/bash
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=annotation_ladder_%j.out
#SBATCH --error=annotation_ladder_%j.err

# Step 1 —— 降阈值报告阶梯（纯 pandas 读 CSV）。
# N26 区内存跟卡走，只写 --gpus=1、不写 --mem/--cpus-per-task（和 all_panels 那个一致）。
# 输出在 annotation_ladder_<jobid>.out，四板各一张表。

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "===== 注释阶梯（cos ∈ 0.5..0.9 × 纯余弦 / +m/z 约束）====="
echo "Started: $(date)"

python -u tasks/annotation_ladder.py \
    --annotations data/mtbls13729/annotation/pos_rp/annotations.csv \
                   data/mtbls13729/annotation/neg_rp/annotations.csv \
                   data/mtbls13729/annotation/pos_hilic/annotations.csv \
                   data/mtbls13729/annotation/neg_hilic/annotations.csv

echo "Finished: $(date)"
