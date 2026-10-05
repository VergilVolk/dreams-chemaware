#!/bin/bash
#SBATCH --job-name=chem_official_pilot
##SBATCH --partition=gpu
##SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:01:00
#SBATCH --output=data/validation/chem_official_pilot_%j.out
#SBATCH --error=data/validation/chem_official_pilot_%j.err

echo "BLOCKED: historical spectrum-only adapter pilot is not a ChemAware causal experiment" >&2
exit 64

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TOKEN_DIR=data/validation/chemaware_corrected_manifest_tokens_v1
OUT="data/validation/chemaware_full_candidate_official_pilot/run_${SLURM_JOB_ID}"
required=(
  data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz
  "$TOKEN_DIR/report.json"
  "$TOKEN_DIR/rows.npy"
  "$TOKEN_DIR/official_embeddings_f32.npy"
  data/e1/official_embedding_slim.pt
)
for file in "${required[@]}"; do
  [[ -f "$file" ]] || { echo "MISSING: $file" >&2; exit 2; }
done
[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }

python -u tasks/test_chemaware_full_candidate_alignment.py
python -u tasks/train_chemaware_full_candidate_alignment.py \
  --arm spectrum_only --token-dir "$TOKEN_DIR" --output "$OUT" \
  --inner-fold 3 --outer-fold 4 --seed 20260905 \
  --hidden-dim 256 --dropout 0.2 --batch-queries 64 \
  --references-per-molecule 2 --max-steps 500 --warmup-steps 50 \
  --eval-every-steps 50 --learning-rate 3e-5 --weight-decay 1e-4 \
  --temperature 0.07 --lambda-spectrum 1.0 \
  --lambda-inbatch-spectrum 0.25 --lambda-margin-floor 2.0 \
  --margin-floor-slack 0.005 --lambda-preserve 20.0 --grad-clip 5.0 \
  --max-eval-identities 0 --train-diagnostic-identities 512 \
  --max-train-identities 2048 --error-identity-fraction 0.50 \
  --clean-safety-selection random --training-mass identity \
  --bootstrap-draws 10000 --device cuda --torch-threads 8
python -u tasks/validate_chemaware_full_candidate_alignment.py --output-dir "$OUT"
