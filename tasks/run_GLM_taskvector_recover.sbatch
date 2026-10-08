#!/bin/bash
#SBATCH --job-name=glm_tv_recover
#SBATCH --gpus=1
#SBATCH --output=data/validation/GLM_taskvector_recover_%j.out
#SBATCH --error=data/validation/GLM_taskvector_recover_%j.err

# GLM: task-vector 7-arm recovery + internal protection gate.
# Pulls run 2349962 output and runs the protection gate evaluation.
set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=== checking run 2349962 outputs ==="
for f in data/validation/*2349962*.out data/validation/*2349962*.json; do
  if [[ -e "$f" ]]; then
    echo "--- $f ---"
    tail -80 "$f"
  fi
done

# if the seven-arm gate evaluation script exists, run it
if [[ -f tasks/GLM_taskvector_seven_arm_gate.py ]]; then
  python -X utf8 tasks/GLM_taskvector_seven_arm_gate.py
else
  echo "gate evaluation script not found; raw output above"
fi
echo "TASK-VECTOR RECOVERY COMPLETE"
