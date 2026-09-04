#!/bin/bash
#SBATCH --job-name=noise_v3_a4
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/noise_v3_a4_%j.out
#SBATCH --error=data/validation/noise_v3_a4_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"

OUTPUT="data/validation/g8r_noise_v3_a4_exact_peak_scan"
SMOKE_ROOT="${SLURM_TMPDIR:-/tmp}/g8r_noise_v3_a4_smoke_${SLURM_JOB_ID}"

if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite formal A4 output: $OUTPUT" >&2
  exit 2
fi
for path in \
  data/validation/g8r_error_atlas_listwise_cache.npz \
  data/validation/g8r_p2_official_embeddings.npz \
  data/validation/g8r_p3_test \
  data/validation/g8r_noise_v3_s1c_topk_matrix/paired_interventions.csv.gz \
  data/validation/g8r_noise_v3_s2_sequential/paired_interventions.csv.gz \
  data/validation/g8r_noise_v3_s3a_extended_matrix/paired_interventions.csv.gz
do
  if [[ ! -e "$path" ]]; then
    echo "Missing locked A4 dependency: $path" >&2
    exit 3
  fi
done

echo "=== Noise-v3 A4 exact all-peak scan started: $(date) ==="
echo "=== stage 1/6: unit and provenance gates ==="
python -u tasks/test_noise_v3_core.py
python -u tasks/test_noise_v3_a4.py
python -u tasks/test_noise_v3_a4_artifacts.py
python -u tasks/check_noise_v3_candidate_gradient.py

echo "=== stage 2/6: 32-error exact-scan smoke ==="
python -u tasks/audit_noise_v3_a4_exact_peak_scan.py \
  --output-dir "$SMOKE_ROOT" \
  --device cuda \
  --gradient-batch-size 32 \
  --encode-batch-size 256 \
  --attenuations 0.25 0.50 0.75 1.00 \
  --safety-controls-per-error 1 \
  --maximum-control-reuse 2 \
  --max-errors 32 \
  --seed 20260825
python -u tasks/validate_noise_v3_a4_exact_peak_scan.py \
  --output-dir "$SMOKE_ROOT" --allow-smoke

echo "=== stage 3/6: all 1,805 errors plus matched correct-query safety controls ==="
python -u tasks/audit_noise_v3_a4_exact_peak_scan.py \
  --output-dir "$OUTPUT" \
  --device cuda \
  --gradient-batch-size 64 \
  --encode-batch-size 256 \
  --attenuations 0.25 0.50 0.75 1.00 \
  --safety-controls-per-error 3 \
  --maximum-control-reuse 3 \
  --seed 20260825

echo "=== stage 4/6: fail-closed exact-action validation ==="
python -u tasks/validate_noise_v3_a4_exact_peak_scan.py --output-dir "$OUTPUT"

echo "=== stage 5/6: exact-vs-gradient, error-family, risk and headroom decision ==="
python -u tasks/analyze_noise_v3_a4_exact_peak_scan.py \
  --a4-dir "$OUTPUT" \
  --s1c-dir data/validation/g8r_noise_v3_s1c_topk_matrix \
  --s2-dir data/validation/g8r_noise_v3_s2_sequential \
  --s3a-dir data/validation/g8r_noise_v3_s3a_extended_matrix \
  --four-point-target 0.04 \
  --minimum-policy-headroom 1000 \
  --top-actions-per-query 64

echo "=== stage 6/6: final artifact assertions ==="
python - "$OUTPUT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
validation = json.loads((root / "validation.json").read_text())
decision = json.loads((root / "decision.json").read_text())
assert validation["status"] == "noise_v3_a4_exact_peak_scan_validation_passed"
assert validation["errors"] == 1805
assert decision["status"] == "noise_v3_a4_exact_peak_scan_decision"
for name in (
    "exact_peak_scan.h5", "scan_queries.csv.gz", "safety_control_matches.csv.gz",
    "policy_candidate_actions.csv.gz", "exact_action_matrix.csv",
):
    assert (root / name).is_file(), name
print("[A4 formal artifacts] PASS", flush=True)
PY
echo "=== Noise-v3 A4 exact all-peak scan completed: $(date) ==="
