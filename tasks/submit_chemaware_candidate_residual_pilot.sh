#!/bin/bash
set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

: "${CLEAN_OBSERVABILITY_REPORT:?set the formal clean-observability PASS report}"
AUTH="data/validation/chemaware_gpu_authorizations/candidate_residual_$(date -u +%Y%m%dT%H%M%SZ).json"
python tasks/authorize_chemaware_gpu_submission.py \
  --route rule_candidate_residual \
  --route-report "$CLEAN_OBSERVABILITY_REPORT" \
  --data-semantics-report data/validation/chemaware_data_semantics_gate_v1/report.json \
  --rule-admission-report data/validation/chemaware_rule_library_admission_v1/report.json \
  --output "$AUTH" --array-tasks 5 --gpus-per-task 1 --hours-per-task 2 \
  --epochs 2 --max-train-identities 512 --max-total-gpu-hours 10
sbatch --export=ALL,CHEMAWARE_GPU_AUTHORIZATION="$AUTH",CLEAN_OBSERVABILITY_REPORT="$CLEAN_OBSERVABILITY_REPORT" \
  tasks/run_chemaware_candidate_residual_arm.sbatch
