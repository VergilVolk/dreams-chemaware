#!/bin/bash
#SBATCH --job-name=noise_cpg0
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --chdir=/data02/run01/scv7tsl/DreaMS
#SBATCH --output=/data02/run01/scv7tsl/DreaMS/data/validation/noise_cpg0_%j.out
#SBATCH --error=/data02/run01/scv7tsl/DreaMS/data/validation/noise_cpg0_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTER_FOLD="${OUTER_FOLD:-0}"
OUTPUT="data/validation/g8r_noise_final_cpg0_residual_teacher/fold_${OUTER_FOLD}_run_${SLURM_JOB_ID}"

REQUIRED=(
  tasks/audit_noise_final_cpg_implementation.py
  tasks/noise_final_cpg_core.py
  tasks/test_noise_final_cpg_core.py
  tasks/build_noise_final_cpg0_residual_teacher.py
  tasks/test_noise_final_cpg0_residual_teacher.py
  tasks/validate_noise_final_cpg0_residual_teacher.py
  data/validation/g8r_error_atlas_listwise_cache.npz
  data/validation/g8r_noise_final_r0_faithful_s3a/report.json
  data/validation/g8r_noise_final_r0_faithful_s3a/training_actions.csv.gz
  data/validation/g8r_noise_final_positive_guided_matrix/report.json
  data/validation/g8r_noise_final_positive_guided_matrix/action_manifest.csv.gz
  data/validation/g8r_noise_final_positive_peak_transfer/report.json
  data/validation/g8r_noise_final_positive_peak_transfer/action_manifest.csv.gz
  data/models/MassSpecGym_MurckoHist_split.hdf5
  data/e1/official_embedding_slim.pt
  dreams/models/pretrained/ssl_model_server.pt
)
for path in "${REQUIRED[@]}"; do
  [[ -f "$path" ]] || { echo "Missing required CPG0 file: $path" >&2; exit 2; }
done

CLEAN="data/validation/g8r_noise_final_e4a_causal_attribution/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_e4a_causal_v1_20260901_causal_clean_duplicate/seed_20260828/fold_${OUTER_FOLD}/final_shared_encoder.pt"
[[ -f "$CLEAN" ]] || { echo "Missing fold-matched mature clean checkpoint: $CLEAN" >&2; exit 2; }

echo "=== CPG0 mature N/P full-candidate residual teacher started: $(date) ==="
python -u tasks/audit_noise_final_cpg_implementation.py
python -u tasks/test_noise_final_cpg_core.py
python -u tasks/test_noise_final_cpg0_residual_teacher.py
python -u tasks/build_noise_final_cpg0_residual_teacher.py \
  --output-dir "$OUTPUT" --outer-fold "$OUTER_FOLD" --formula-fold-seed 20260825 \
  --clean-checkpoint "$CLEAN" --batch-size 64 --fp32-retry-batch-size 8 \
  --actions-per-chunk 128 --n-highest-peaks 100 --device cuda --amp
python -u tasks/validate_noise_final_cpg0_residual_teacher.py \
  --output-dir "$OUTPUT" --expected-outer-fold "$OUTER_FOLD"
echo "=== CPG0 mature N/P full-candidate residual teacher completed: $(date) ==="
