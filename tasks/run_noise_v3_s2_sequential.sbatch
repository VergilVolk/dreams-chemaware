#!/bin/bash
#SBATCH --job-name=noise_v3_s2
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/noise_v3_s2_%j.out
#SBATCH --error=data/validation/noise_v3_s2_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"

OUTPUT="data/validation/g8r_noise_v3_s2_sequential"
DECISION="data/validation/g8r_noise_v3_s2_headroom.json"
SMOKE_ROOT="${SLURM_TMPDIR:-/tmp}/g8r_noise_v3_s2_smoke_${SLURM_JOB_ID}"

if [[ -e "$OUTPUT" || -e "$DECISION" ]]; then
  echo "Refusing to overwrite S2 formal output" >&2
  exit 2
fi

echo "=== Noise-v3 S2 sequential audit started: $(date) ==="
echo "=== stage 1/6: core unit tests ==="
python -u tasks/test_noise_v3_core.py

echo "=== stage 2/6: full candidate-graph preflight ==="
python -u tasks/check_noise_v3_candidate_gradient.py

echo "=== stage 3/6: 256-query implementation smoke ==="
python -u tasks/audit_noise_v3_s2_sequential.py \
  --output-dir "$SMOKE_ROOT" \
  --device cuda \
  --selection-batch-size 32 \
  --encode-batch-size 128 \
  --attenuations 0.50 1.00 \
  --selectors candidate_gradient role_confounder \
  --maximum-steps 3 \
  --control-repeats 2 \
  --bootstrap 100 \
  --max-queries 256 \
  --seed 20260824
python - "$SMOKE_ROOT/report.json" <<'PY'
import json, sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text())
assert report["status"] == "noise_v3_s2_sequential_matrix_complete"
assert report["sequences"] > 0 and report["paired_cells"] > 0
assert report["official_forward_cache_preservation"]["p01"] > 0.999
print("[smoke] PASS", flush=True)
PY

echo "=== stage 4/6: formal 23,876-query dynamic sequential matrix ==="
python -u tasks/audit_noise_v3_s2_sequential.py \
  --output-dir "$OUTPUT" \
  --device cuda \
  --selection-batch-size 32 \
  --encode-batch-size 128 \
  --top-k-negatives 5 \
  --softmax-temperature 0.10 \
  --attenuations 0.50 1.00 \
  --selectors candidate_gradient role_confounder \
  --maximum-steps 3 \
  --control-repeats 2 \
  --bootstrap 2000 \
  --seed 20260824

echo "=== stage 5/6: fail-closed sequence validation ==="
python -u tasks/validate_noise_v3_s2_sequential.py --output-dir "$OUTPUT"

echo "=== stage 6/6: locked S1c + S2 headroom decision ==="
python -u tasks/analyze_noise_v3_s2_headroom.py \
  --s1c-dir data/validation/g8r_noise_v3_s1c_topk_matrix \
  --s2-dir "$OUTPUT" \
  --cache data/validation/g8r_error_atlas_listwise_cache.npz \
  --output "$DECISION" \
  --minimum-headroom-pp 2.0 \
  --minimum-recoverable-queries 600 \
  --bootstrap 2000 \
  --seed 20260824
echo "=== Noise-v3 S2 sequential audit completed: $(date) ==="
