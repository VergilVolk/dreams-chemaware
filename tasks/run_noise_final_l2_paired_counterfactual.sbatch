#!/bin/bash
#SBATCH --job-name=noise_l2_pair
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --chdir=/data02/run01/scv7tsl/DreaMS
#SBATCH --output=/data02/run01/scv7tsl/DreaMS/data/validation/noise_l2_pair_%j.out
#SBATCH --error=/data02/run01/scv7tsl/DreaMS/data/validation/noise_l2_pair_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTPUT="data/validation/g8r_noise_final_l2_paired_counterfactual/run_${SLURM_JOB_ID}"
REQUIRED=(
  tasks/train_noise_final_l2_paired_counterfactual.py
  tasks/test_noise_final_l2_paired_counterfactual.py
  tasks/validate_noise_final_l2_paired_counterfactual.py
  data/validation/g8r_noise_final_l1_clean_action_learnability/report.json
  data/validation/g8r_noise_final_l1_clean_action_learnability/action_oof_predictions.csv.gz
  data/validation/g8r_error_atlas_listwise_cache.npz
  data/models/MassSpecGym_MurckoHist_split.hdf5
  data/e1/official_embedding_slim.pt
  dreams/models/pretrained/ssl_model_server.pt
  data/validation/g8r_noise_final_e4a_causal_attribution/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_e4a_causal_v1_20260901_causal_clean_duplicate/seed_20260828/fold_0/final_shared_encoder.pt
  data/validation/g8r_noise_final_e4a_causal_attribution/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_e4a_causal_v1_20260901_causal_clean_duplicate/seed_20260828/fold_0/decision.json
)
for path in "${REQUIRED[@]}"; do
  [[ -f "$path" ]] || { echo "Missing required L2 file: $path" >&2; exit 2; }
done

echo "=== L2 paired counterfactual shared-encoder pilot started: $(date) ==="
python -u tasks/test_noise_final_l2_paired_counterfactual.py
python -u tasks/train_noise_final_l2_paired_counterfactual.py \
  --output-dir "$OUTPUT" --outer-fold 0 --formula-fold-seed 20260825 \
  --seed 20260903 --epochs 2 \
  --maximum-train-queries 1024 --actions-per-query-per-epoch 2 \
  --positive-spectra 4 --unfreeze-blocks 1 \
  --backbone-lr 2e-6 --head-lr 1e-5 --weight-decay 1e-4 \
  --rank-margin 0.05 --temperature 0.10 --advantage-margin 0.01 \
  --lambda-clean-rank 1 --lambda-action-rank 1 --lambda-advantage 1 \
  --lambda-consistency 0.05 --lambda-margin-floor 1 --lambda-preserve 5 \
  --margin-floor-slack 0.005 --grad-clip 1 --n-highest-peaks 100 \
  --eval-batch-size 128 --bootstrap-resamples 5000 --device cuda --no-amp
python -u tasks/validate_noise_final_l2_paired_counterfactual.py --output-dir "$OUTPUT"
echo "=== L2 paired counterfactual shared-encoder pilot completed: $(date) ==="
