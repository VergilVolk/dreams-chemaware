#!/bin/bash
#SBATCH --job-name=chem_full_cache
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=data/validation/chem_full_cache_%j.out
#SBATCH --error=data/validation/chem_full_cache_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

OUT=data/validation/chemaware_corrected_manifest_tokens_v1
required=(
  data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz
  data/models/MassSpecGym_MurckoHist_split.hdf5
  data/e1/official_embedding_slim.pt
  dreams/models/pretrained/ssl_model_server.pt
)
for file in "${required[@]}"; do
  [[ -f "$file" ]] || { echo "MISSING: $file" >&2; exit 2; }
done

if [[ -f "$OUT/report.json" ]]; then
  python - "$OUT/report.json" <<'PY'
import json, sys
r=json.load(open(sys.argv[1]))
assert r["status"] == "chemaware_corrected_manifest_token_cache_complete"
assert r["spectra"] == 87848
print("Existing complete cache validated")
PY
else
  python -u tasks/encode_chemaware_corrected_manifest_tokens.py \
    --batch-size 128 --device cuda --output-dir "$OUT"
fi
