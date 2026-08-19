#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=step4c_retrieval_report_%j.out
#SBATCH --error=step4c_retrieval_report_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# ============================================================================
# Part C：从 lr_scan.json 提取检索指标（零训练、零 GPU、单核 <1 分钟）。
# 检索其实已在 Part B 里算好，本 job 只读 JSON 打印，不重新前向。
#
# DEPENDENCIES（须已存在，否则 exit 2）：
#   1. data/validation/lr_scan/lr_scan.json        # Part B 产出（检索在这里面）
#   2. data/validation/space_audit/baseline.json   # Part A 产出（baseline 检索对照）
#   3. tasks/step4c_retrieval_report.py            # 本脚本（新，须同步）
# ============================================================================

for REQUIRED in \
    data/validation/lr_scan/lr_scan.json \
    data/validation/space_audit/baseline.json \
    tasks/step4c_retrieval_report.py ; do
    if [[ ! -f "$REQUIRED" ]]; then
        echo "MISSING DEPENDENCY: $REQUIRED  (see DEPENDENCIES block above)" >&2
        exit 2
    fi
done
echo "输入依赖齐。"
echo ""

python -u tasks/step4c_retrieval_report.py \
    --scan data/validation/lr_scan/lr_scan.json \
    --baseline data/validation/space_audit/baseline.json

echo ""
echo "===== Part C DONE $(date) ====="
echo "看末尾：[建议] 加检索门槛后选 lr；[对照] Part B 原决策（较松门槛）。"
