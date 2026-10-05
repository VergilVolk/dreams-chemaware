#!/bin/bash
#SBATCH --job-name=chem_rule_pmt
##SBATCH --partition=gpu
##SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:01:00
##SBATCH --array=0-7
#SBATCH --output=data/validation/chem_rule_pmt_%A_%a.out
#SBATCH --error=data/validation/chem_rule_pmt_%A_%a.err

echo "BLOCKED: rejected Phase-B KL/scalar-PMT route; historical reproduction only" >&2
exit 64

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Phase B removes the ranking-irrelevant teacher-score compression and tests
# positive paired-margin transfer.  The shifted-rule and mass-only arms are
# schedule/dose matched controls; no candidate-side input survives inference.
LABELS=(none rule_rankkl rule_rankkl_balanced shifted_rankkl_balanced rule_pmt_a25 rule_pmt_a50 shifted_pmt_a50 mass_pmt_a50)
TEACHERS=(none rule_mass rule_mass rule_mass_shifted rule_mass rule_mass rule_mass_shifted mass)
OBJECTIVES=(rank_equivalent_kl rank_equivalent_kl rank_equivalent_kl rank_equivalent_kl positive_margin_transfer positive_margin_transfer positive_margin_transfer positive_margin_transfer)
BETAS=(0.00 0.20 0.20 0.20 0.20 0.20 0.20 0.10)
ALPHAS=(0.50 0.50 0.50 0.50 0.25 0.50 0.50 0.50)
GRADIENT_RATIOS=(0.00 0.00 0.50 0.50 0.50 0.50 0.50 0.50)

LABEL="${LABELS[$SLURM_ARRAY_TASK_ID]}"
ARM="${TEACHERS[$SLURM_ARRAY_TASK_ID]}"
OBJECTIVE="${OBJECTIVES[$SLURM_ARRAY_TASK_ID]}"
BETA="${BETAS[$SLURM_ARRAY_TASK_ID]}"
ALPHA="${ALPHAS[$SLURM_ARRAY_TASK_ID]}"
GRADIENT_RATIO="${GRADIENT_RATIOS[$SLURM_ARRAY_TASK_ID]}"
ROOT="data/validation/chemaware_rule_margin_transfer_phase_b/run_${SLURM_ARRAY_JOB_ID}"
OUT="$ROOT/$LABEL"
[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }

python -u tasks/train_chemaware_full_candidate_direct.py \
  --output "$OUT" --inner-fold 3 --outer-fold 4 --seed 20260905 \
  --epochs 4 --batch-queries 2 --references-per-molecule 2 \
  --max-train-identities 2048 --error-identity-fraction 0.50 \
  --clean-safety-selection random --training-mass formula_identity \
  --max-eval-identities 0 --unfreeze-blocks 1 \
  --backbone-lr 2e-6 --head-lr 1e-5 --weight-decay 1e-4 \
  --temperature 0.10 --lambda-spectrum 1.0 \
  --lambda-inbatch-spectrum 0.25 --lambda-teacher 1.0 \
  --teacher-arm "$ARM" --teacher-beta "$BETA" --teacher-temperature 0.07 \
  --teacher-objective "$OBJECTIVE" --teacher-scope all \
  --margin-transfer-alpha "$ALPHA" --margin-transfer-cap 0.05 \
  --margin-transfer-huber 0.01 \
  --teacher-gradient-ratio "$GRADIENT_RATIO" --teacher-gradient-scale-cap 4.0 \
  --lambda-margin-floor 2.0 --margin-floor-slack 0.005 \
  --lambda-preserve 5.0 --grad-clip 1.0 \
  --top-peaks 32 --kernel-dim 2048 --bin-width 0.02 --grid-offsets 4 \
  --intensity-power 0.5 --mass-shift-da 0.137 \
  --eval-batch-size 96 --bootstrap-draws 10000 \
  --device cuda --no-amp --torch-threads 8

test -f "$OUT/report.json"
test -f "$OUT/inner_per_query.npz"
test -f "$OUT/final_shared_encoder.pt"
echo "completed label=$LABEL teacher=$ARM objective=$OBJECTIVE alpha=$ALPHA gradient_ratio=$GRADIENT_RATIO output=$OUT"
