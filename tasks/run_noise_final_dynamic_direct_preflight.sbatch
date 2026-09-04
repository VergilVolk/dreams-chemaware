#!/bin/bash
#SBATCH --job-name=noise_dd_preflight
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=04:00:00
#SBATCH --output=data/validation/noise_dd_preflight_%j.out
#SBATCH --error=data/validation/noise_dd_preflight_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

FOLD=0
SEED=20260828
INITIAL="data/validation/g8r_noise_final_e4a_causal_attribution/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_e4a_causal_v1_20260901_causal_clean_duplicate/seed_${SEED}/fold_${FOLD}/final_shared_encoder.pt"
OUTPUT="data/validation/g8r_noise_final_dynamic_direct_preflight/fold_${FOLD}_run_${SLURM_JOB_ID}"
LEDGER="data/validation/g8r_noise_final_dynamic_direct_ledger/fold_${FOLD}_run_${SLURM_JOB_ID}"

echo "=== dynamic direct model-free preflight fold=${FOLD} started: $(date) ==="
python -u tasks/test_noise_final_dynamic_direct_core.py
python -u tasks/test_noise_final_dynamic_direct_preflight.py
python -u tasks/test_noise_final_dynamic_direct_ledger.py
python -u tasks/audit_noise_final_dynamic_direct_implementation.py
echo "=== exact artifact hashes and provenance ==="
python -u tasks/audit_noise_final_dynamic_direct_preflight.py \
  --outer-fold "$FOLD" --formula-fold-seed 20260825 \
  --initial-checkpoint "$INITIAL" --output-dir "$OUTPUT" --hash-large-files
python -u tasks/validate_noise_final_dynamic_direct_preflight.py --output-dir "$OUTPUT"
echo "=== outer-train 30-cell N/P ledger and formula-crossfit ==="
python -u tasks/build_noise_final_dynamic_direct_ledger.py \
  --preflight-dir "$OUTPUT" --outer-fold "$FOLD" --seed 20260904 \
  --output-dir "$LEDGER"
python -u tasks/validate_noise_final_dynamic_direct_ledger.py --output-dir "$LEDGER"
echo "=== dynamic direct model-free preflight completed: $(date) ==="
