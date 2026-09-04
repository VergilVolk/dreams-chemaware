#!/bin/bash
#SBATCH --job-name=noise_l1_learn
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/noise_l1_learn_%j.out
#SBATCH --error=data/validation/noise_l1_learn_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTPUT=data/validation/g8r_noise_final_l1_clean_action_learnability
REQUIRED=(
  tasks/audit_noise_final_l1_clean_action_learnability.py
  tasks/test_noise_final_l1_clean_action_learnability.py
  tasks/validate_noise_final_l1_clean_action_learnability.py
  data/validation/g8r_noise_final_l0_action_learnability_ledger/report.json
  data/validation/g8r_noise_final_l0_action_learnability_ledger/action_labels.csv.gz
  data/validation/g8r_noise_final_f1_full_tokens/report.json
  data/validation/g8r_noise_final_f1_full_tokens/rows.npy
  data/validation/g8r_noise_final_f1_full_tokens/tokens_f16.npy
  data/validation/g8r_noise_final_f1_full_tokens/mz_f32.npy
  data/validation/g8r_noise_final_f1_full_tokens/intensity_f32.npy
  data/validation/g8r_noise_final_f1_full_tokens/valid.npy
  data/validation/g8r_error_atlas_listwise_cache.npz
  data/models/MassSpecGym_MurckoHist_split.hdf5
  data/e1/official_embedding_slim.pt
  dreams/models/pretrained/ssl_model_server.pt
  data/validation/g8r_noise_final_e4a_causal_attribution/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_e4a_causal_v1_20260901_causal_clean_duplicate/seed_20260828/fold_0/final_shared_encoder.pt
  data/validation/g8r_noise_final_e4a_causal_attribution/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_e4a_causal_v1_20260901_causal_clean_duplicate/seed_20260828/fold_0/decision.json
)
for path in "${REQUIRED[@]}"; do
  [[ -f "$path" ]] || { echo "Missing required L1 file: $path" >&2; exit 2; }
done

echo "=== L1 clean-input action learnability started: $(date) ==="
python -u tasks/test_noise_final_l1_clean_action_learnability.py
if [[ -d "$OUTPUT" ]]; then
  python -u tasks/validate_noise_final_l1_clean_action_learnability.py --output-dir "$OUTPUT"
  echo "Reused already-complete L1 output: $OUTPUT"
else
  python -u tasks/audit_noise_final_l1_clean_action_learnability.py \
    --output-dir "$OUTPUT" --embedding-projection-dim 64 --token-projection-dim 16 \
    --metadata-hash-bins 8 --n-highest-peaks 100 --batch-size 64 --bootstrap-resamples 5000 \
    --seed 20260903 --device cuda --amp
  python -u tasks/validate_noise_final_l1_clean_action_learnability.py --output-dir "$OUTPUT"
fi
echo "=== L1 clean-input action learnability completed: $(date) ==="
