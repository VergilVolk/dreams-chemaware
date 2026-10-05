#!/bin/bash
#SBATCH --job-name=chem_candres
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --array=0-4%1
#SBATCH --output=data/validation/chem_candres_%A_%a.out
#SBATCH --error=data/validation/chem_candres_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# This array is intentionally fail-closed. It cannot start until the clean-only
# observability audit has a formal, formula-disjoint confirmation PASS.
: "${CLEAN_OBSERVABILITY_REPORT:?set CLEAN_OBSERVABILITY_REPORT to a formal PASS report}"
: "${CHEMAWARE_GPU_AUTHORIZATION:?submit through submit_chemaware_candidate_residual_pilot.sh}"
test -f "$CLEAN_OBSERVABILITY_REPORT"
test -f data/validation/chemaware_data_semantics_gate_v1/report.json
python tasks/verify_chemaware_gpu_authorization.py \
  --authorization "$CHEMAWARE_GPU_AUTHORIZATION" \
  --expected-route-report "$CLEAN_OBSERVABILITY_REPORT"

LABELS=(none mass rule rule_shifted rule_permuted)
TEACHERS=(none mass rule_response rule_response_shifted rule_response_row_permuted)
BETAS=(0.00 0.10 0.20 0.20 0.20)
GRADIENT_RATIOS=(0.00 0.25 0.25 0.25 0.25)

LABEL="${LABELS[$SLURM_ARRAY_TASK_ID]}"
TEACHER="${TEACHERS[$SLURM_ARRAY_TASK_ID]}"
BETA="${BETAS[$SLURM_ARRAY_TASK_ID]}"
GRADIENT_RATIO="${GRADIENT_RATIOS[$SLURM_ARRAY_TASK_ID]}"
ROOT="data/validation/chemaware_candidate_residual_phase_c/run_${SLURM_ARRAY_JOB_ID}"
OUT="$ROOT/$LABEL"
[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }

python -u tasks/train_chemaware_full_candidate_direct.py \
  --output "$OUT" --gpu-authorization "$CHEMAWARE_GPU_AUTHORIZATION" \
  --inner-fold 3 --outer-fold 4 --seed 20260905 \
  --epochs 2 --batch-queries 2 --references-per-molecule 2 \
  --max-train-identities 512 --error-identity-fraction 0.50 \
  --clean-safety-selection random --training-mass formula_identity \
  --max-eval-identities 0 --unfreeze-blocks 1 \
  --backbone-lr 2e-6 --head-lr 1e-5 --weight-decay 1e-4 \
  --temperature 0.10 --lambda-spectrum 1.0 --lambda-inbatch-spectrum 0.25 \
  --lambda-teacher 1.0 --teacher-arm "$TEACHER" --teacher-beta "$BETA" \
  --teacher-temperature 0.07 --teacher-objective candidate_residual_huber \
  --teacher-scope all --candidate-residual-alpha 0.50 \
  --candidate-residual-cap 0.0 --candidate-residual-huber 0.02 \
  --teacher-gradient-ratio "$GRADIENT_RATIO" --teacher-gradient-scale-cap 4.0 \
  --clean-observability-report "$CLEAN_OBSERVABILITY_REPORT" \
  --data-semantics-report data/validation/chemaware_data_semantics_gate_v1/report.json \
  --lambda-margin-floor 2.0 --margin-floor-slack 0.005 \
  --lambda-preserve 5.0 --grad-clip 1.0 \
  --top-peaks 32 --kernel-dim 2048 --bin-width 0.02 --grid-offsets 4 \
  --intensity-power 0.5 --mass-shift-da 0.137 \
  --eval-batch-size 96 --bootstrap-draws 10000 \
  --device cuda --no-amp --torch-threads 8

test -f "$OUT/report.json"
test -f "$OUT/inner_per_query.npz"
test -f "$OUT/final_shared_encoder.pt"
