#!/bin/bash
#SBATCH --job-name=noise_v3_s3a
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/noise_v3_s3a_%j.out
#SBATCH --error=data/validation/noise_v3_s3a_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"

OUTPUT="data/validation/g8r_noise_v3_s3a_extended_matrix"
SMOKE_ROOT="${SLURM_TMPDIR:-/tmp}/g8r_noise_v3_s3a_smoke_${SLURM_JOB_ID}"

if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite S3A formal output: $OUTPUT" >&2
  exit 2
fi

echo "=== Noise-v3 S3A extended mechanism matrix started: $(date) ==="
echo "=== stage 1/7: core unit tests and locked-graph preflight ==="
python -u tasks/test_noise_v3_core.py
python -u tasks/check_noise_v3_candidate_gradient.py

echo "=== stage 2/7: 256-query full-action smoke ==="
python -u tasks/audit_noise_v3_s2_sequential.py \
  --output-dir "$SMOKE_ROOT" \
  --device cuda \
  --selection-batch-size 32 \
  --encode-batch-size 128 \
  --top-k-negatives 5 \
  --softmax-temperature 0.10 \
  --action-specs \
    candidate_gradient:0.50 \
    role_confounder:1.00 \
    role_shared:0.25 role_shared:0.50 role_shared:1.00 \
    role_unmatched:0.25 role_unmatched:0.50 role_unmatched:1.00 \
  --maximum-steps 6 \
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
assert len(report["matrix"]["registered_actions"]) == 8
assert report["matrix"]["maximum_steps"] == 6
assert report["official_forward_cache_preservation"]["p01"] > 0.999
print("[S3A smoke] PASS", flush=True)
PY

echo "=== stage 3/7: formal 23,876-query x 8-action x 6-step matrix ==="
python -u tasks/audit_noise_v3_s2_sequential.py \
  --output-dir "$OUTPUT" \
  --device cuda \
  --selection-batch-size 32 \
  --encode-batch-size 128 \
  --top-k-negatives 5 \
  --softmax-temperature 0.10 \
  --action-specs \
    candidate_gradient:0.50 \
    role_confounder:1.00 \
    role_shared:0.25 role_shared:0.50 role_shared:1.00 \
    role_unmatched:0.25 role_unmatched:0.50 role_unmatched:1.00 \
  --maximum-steps 6 \
  --control-repeats 2 \
  --bootstrap 2000 \
  --seed 20260824

echo "=== stage 4/7: fail-closed matrix validation ==="
python -u tasks/validate_noise_v3_s3a_matrix.py --output-dir "$OUTPUT"

echo "=== stage 5/7: correction + introduced-error + rule-evidence decision ==="
python -u tasks/analyze_noise_v3_s3a_extended_matrix.py \
  --s3a-dir "$OUTPUT" \
  --cache data/validation/g8r_error_atlas_listwise_cache.npz \
  --rule-cache data/validation/g8r_chemaware_g0_rule_cache.npz \
  --s1c-dir data/validation/g8r_noise_v3_s1c_topk_matrix \
  --s2-dir data/validation/g8r_noise_v3_s2_sequential \
  --bootstrap 2000 \
  --seed 20260824

echo "=== stage 6/7: matrix and transition visualisations ==="
python -u tasks/plot_noise_v3_s3a_matrix.py --output-dir "$OUTPUT"

echo "=== stage 7/7: final artefact assertions ==="
python - "$OUTPUT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
decision = json.loads((root / "decision.json").read_text())
assert decision["status"] == "noise_v3_s3a_extended_matrix_decision_complete"
for name in (
    "matrix_validation.json", "cell_summary.csv", "transition_audit.csv.gz",
    "s3a_action_matrix.png", "s3a_transition_destinations.png",
):
    assert (root / name).is_file(), name
assert decision["introduced_error_audit"]["transition_rows"] >= 0
print("[S3A formal artefacts] PASS", flush=True)
PY
echo "=== Noise-v3 S3A extended mechanism matrix completed: $(date) ==="
