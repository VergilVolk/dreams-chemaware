#!/bin/bash
#SBATCH --job-name=noise_dd_replay
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/noise_dd_replay_%j.out
#SBATCH --error=data/validation/noise_dd_replay_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FOLD=0
SOURCE_JOB="${SOURCE_JOB:?submit with sbatch --export=ALL,SOURCE_JOB=<successful-preflight-jobid> tasks/run_noise_final_dynamic_direct_replay.sbatch}"
SEED=20260828
INITIAL="data/validation/g8r_noise_final_e4a_causal_attribution/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_e4a_causal_v1_20260901_causal_clean_duplicate/seed_${SEED}/fold_${FOLD}/final_shared_encoder.pt"
PREFLIGHT="data/validation/g8r_noise_final_dynamic_direct_preflight/fold_${FOLD}_run_${SOURCE_JOB}"
LEDGER="data/validation/g8r_noise_final_dynamic_direct_ledger/fold_${FOLD}_run_${SOURCE_JOB}"
OUTPUT="data/validation/g8r_noise_final_dynamic_direct_replay/fold_${FOLD}_run_${SLURM_JOB_ID}"

echo "=== dynamic direct current-geometry replay fold=${FOLD} started: $(date) ==="
python -u tasks/test_noise_final_dynamic_direct_core.py
python -u tasks/test_noise_final_dynamic_direct_replay.py
python -u tasks/validate_noise_final_dynamic_direct_preflight.py --output-dir "$PREFLIGHT"
python -u tasks/validate_noise_final_dynamic_direct_ledger.py --output-dir "$LEDGER"
python -u tasks/audit_noise_final_dynamic_direct_replay.py \
  --preflight-dir "$PREFLIGHT" --ledger-dir "$LEDGER" \
  --clean-checkpoint "$INITIAL" --output-dir "$OUTPUT" \
  --batch-size 64 --fp32-retry-batch-size 16 --actions-per-chunk 256 \
  --device cuda --amp
python -u tasks/validate_noise_final_dynamic_direct_replay.py --output-dir "$OUTPUT"
echo "=== dynamic direct current-geometry replay completed: $(date) ==="
