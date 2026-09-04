#!/bin/bash
#SBATCH --job-name=noise_c2_tok
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=data/validation/noise_c2_tok_%j.out
#SBATCH --error=data/validation/noise_c2_tok_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
OUTPUT="data/validation/g8r_noise_v3_c2_peak_tokens"
[[ ! -e "$OUTPUT" ]] || { echo "Refusing to overwrite $OUTPUT" >&2; exit 2; }
for path in \
  data/validation/g8r_error_atlas_listwise_cache.npz \
  data/models/MassSpecGym_MurckoHist_split.hdf5 \
  dreams/models/pretrained/ssl_model_server.pt \
  data/e1/official_embedding_slim.pt \
  data/validation/g8r_noise_v3_c1_crossfit_teacher/decision.json
do
  [[ -f "$path" ]] || { echo "Missing C2 dependency: $path" >&2; exit 3; }
done
python - <<'PY'
import json
from pathlib import Path
p = Path('data/validation/g8r_noise_v3_c1_crossfit_teacher/decision.json')
d = json.loads(p.read_text())
if not d.get('pass_to_candidate_aware_student'):
    raise SystemExit('C1 did not pass; refusing C2 token encoding')
print('[C2 preflight] C1 PASS')
PY
echo "=== Noise-v3 C2 contextual peak-token encoding started: $(date) ==="
python -u tasks/encode_noise_v3_c2_peak_tokens.py --output-dir "$OUTPUT" --device cuda --batch-size 64
python -u tasks/validate_noise_v3_c2_peak_tokens.py --output-dir "$OUTPUT"
echo "=== Noise-v3 C2 contextual peak-token encoding completed: $(date) ==="
