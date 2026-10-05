#!/bin/bash
#SBATCH --job-name=chem_direct_official
##SBATCH --partition=gpu
##SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:01:00
#SBATCH --output=data/validation/chem_direct_official_%j.out
#SBATCH --error=data/validation/chem_direct_official_%j.err

echo "BLOCKED: superseded ChemAware training entrypoint; use submit_chemaware_candidate_residual_pilot.sh after CPU causal admission" >&2
exit 64

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: "${ADAPTER_RUN:?submit with the passed adapter-screen output directory}"
OUT="data/validation/chemaware_full_candidate_direct_official/run_${SLURM_JOB_ID}"
[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }
required=(
  "$ADAPTER_RUN/report.json"
  data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz
  data/validation/chemaware_corrected_manifest_tokens_v1/report.json
  data/models/MassSpecGym_MurckoHist_split.hdf5
  data/e1/official_embedding_slim.pt
  dreams/models/pretrained/ssl_model_server.pt
)
for file in "${required[@]}"; do
  [[ -f "$file" ]] || { echo "MISSING: $file" >&2; exit 2; }
done

# The expensive direct stage is admitted only by the exact official-space
# screen that passed a strict-positive formula-cluster CI.
python - "$ADAPTER_RUN/report.json" <<'PY'
import json, sys
r=json.load(open(sys.argv[1]))
assert r["status"] == "PASS"
assert r["gates"]["formula_ci_positive"] is True
assert r["final_inner"]["delta_recall1"] > 0
assert r["preflight"]["contracts"]["candidate_input_at_deployment"] is False
print("Passed adapter screen admitted direct official-DreaMS fine-tuning")
PY

python -m py_compile tasks/train_chemaware_full_candidate_direct.py
python -u tasks/train_chemaware_full_candidate_direct.py \
  --output "$OUT" --inner-fold 3 --outer-fold 4 --seed 20260905 \
  --epochs 4 --batch-queries 2 --references-per-molecule 1 \
  --max-train-identities 2048 --error-identity-fraction 0.50 \
  --clean-safety-selection random --training-mass identity \
  --max-eval-identities 0 --unfreeze-blocks 1 \
  --backbone-lr 2e-6 --head-lr 1e-5 --weight-decay 1e-4 \
  --temperature 0.10 --lambda-spectrum 1.0 \
  --lambda-inbatch-spectrum 0.25 --lambda-margin-floor 2.0 \
  --margin-floor-slack 0.005 --lambda-preserve 5.0 --grad-clip 1.0 \
  --eval-batch-size 96 --bootstrap-draws 10000 \
  --device cuda --no-amp --torch-threads 8
python -u tasks/validate_chemaware_full_candidate_direct.py --output-dir "$OUT"
