#!/bin/bash
#SBATCH --job-name=e4deb_resume
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=data/validation/noise_e4_deb_resume_2331540_%j.out
#SBATCH --error=data/validation/noise_e4_deb_resume_2331540_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED=20260905
RUN_ROOT="data/validation/g8r_noise_final_e4_deb/fold_0_run_2331540"
ARMS_ROOT="$RUN_ROOT/arms"
SUMMARY="$RUN_ROOT/summary_resume_${SLURM_JOB_ID}"

[[ -d "$ARMS_ROOT" ]] || { echo "Missing completed job-2331540 arms: $ARMS_ROOT" >&2; exit 3; }
for LABEL in clean_duplicate matched_random candidate_boundary; do
  DECISION_COUNT=$(find "$ARMS_ROOT/$LABEL" -name decision.json -type f 2>/dev/null | wc -l)
  CHECKPOINT_COUNT=$(find "$ARMS_ROOT/$LABEL" -name final_shared_encoder.pt -type f 2>/dev/null | wc -l)
  HELD_COUNT=$(find "$ARMS_ROOT/$LABEL" -name held_per_query.csv.gz -type f 2>/dev/null | wc -l)
  [[ "$DECISION_COUNT" -eq 1 && "$CHECKPOINT_COUNT" -eq 1 && "$HELD_COUNT" -eq 1 ]] || {
    echo "Incomplete $LABEL arm: decision=$DECISION_COUNT checkpoint=$CHECKPOINT_COUNT held=$HELD_COUNT" >&2
    exit 4
  }
done

echo "=== E4-DEB job 2331540 evaluation-only resume started: $(date) ==="
python -m py_compile \
  tasks/summarize_noise_final_e4_deb.py \
  tasks/audit_noise_final_e4_deb_implementation.py
python -u tasks/audit_noise_final_e4_deb_implementation.py

python -u tasks/summarize_noise_final_e4_deb.py \
  --arms-root "$ARMS_ROOT" \
  --output-dir "$SUMMARY" --bootstrap-resamples 10000 --seed "$SEED"

echo "E4-DEB summary: $SUMMARY/report.json"
echo "=== E4-DEB job 2331540 evaluation-only resume completed: $(date) ==="
