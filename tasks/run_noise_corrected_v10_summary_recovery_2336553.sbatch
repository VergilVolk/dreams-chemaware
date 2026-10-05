#!/bin/bash
#SBATCH --job-name=noise_v10_summary
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=data/validation/noise_v10_summary_recovery_2336553_%j.out
#SBATCH --error=data/validation/noise_v10_summary_recovery_2336553_%j.err

set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

FOLD=0
SEED=20260913
ORIGINAL_JOB_ID=2336553
RUN_ROOT="data/validation/.noise_corrected_v10_safe_exact_canary_fold_${FOLD}_seed_${SEED}_run_${ORIGINAL_JOB_ID}.staging"
FINAL_ROOT="data/validation/noise_corrected_v10_safe_exact_canary_fold_${FOLD}_seed_${SEED}_run_${ORIGINAL_JOB_ID}"
ARMS="$RUN_ROOT/arms"
SUMMARY="$RUN_ROOT/summary"
SNAPSHOT_ROOT="$PWD/$RUN_ROOT/source_snapshot"

[[ -d "$RUN_ROOT" ]] || {
  echo "Missing original V10 staging root: $RUN_ROOT" >&2
  exit 4
}
[[ -d "$SNAPSHOT_ROOT" && -f "$SNAPSHOT_ROOT/sha256_manifest.txt" ]] || {
  echo "Missing frozen job-${ORIGINAL_JOB_ID} source snapshot" >&2
  exit 4
}
[[ ! -e "$FINAL_ROOT" ]] || {
  echo "Refusing to overwrite completed V10 result: $FINAL_ROOT" >&2
  exit 5
}

(
  cd "$SNAPSHOT_ROOT"
  sha256sum -c sha256_manifest.txt
  sha256sum -c tasks/noise_best_v10_source_manifest.sha256
)
export PYTHONPATH="$SNAPSHOT_ROOT:$SNAPSHOT_ROOT/tasks"

# This validation executes only inside the allocated Slurm job. It verifies
# that every arm completed atomically and that each checkpoint matches the
# digest recorded by its own frozen decision report.
validate_arm() {
  local ARM_DIR="$1"
  local EXPECTED_ARM="$2"
  local EXPECTED_BRANCH="$3"
  python - "$ARM_DIR" "$EXPECTED_ARM" "$EXPECTED_BRANCH" <<'PY'
import hashlib
import json
import pathlib
import sys

arm_dir = pathlib.Path(sys.argv[1])
expected_arm = sys.argv[2]
expected_branch = sys.argv[3]
checkpoint = arm_dir / "final_shared_encoder.pt"
decision = arm_dir / "decision.json"
held = arm_dir / "held_per_query.csv.gz"
for path in (checkpoint, decision, held):
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing or empty V10 arm artifact: {path}")
report = json.loads(decision.read_text(encoding="utf-8"))
if report.get("status") != "noise_corrected_routed_direct_complete":
    raise SystemExit(f"incomplete V10 decision: {decision}")
if report.get("arm") != expected_arm:
    raise SystemExit(f"wrong arm in {decision}")
if report.get("corrective_branch_mode") != expected_branch:
    raise SystemExit(f"wrong corrective branch in {decision}")
if report.get("direct_contract") != "best_action_v10_safe_exact":
    raise SystemExit(f"wrong V10 contract in {decision}")
digest = hashlib.sha256()
with checkpoint.open("rb") as handle:
    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(block)
expected_digest = report.get("provenance", {}).get("final_shared_encoder_sha256")
if not expected_digest or digest.hexdigest() != expected_digest:
    raise SystemExit(f"checkpoint digest mismatch: {checkpoint}")
print(f"V10 arm verified: {arm_dir.name} sha256={expected_digest}", flush=True)
PY
}

validate_arm "$ARMS/full_action_view" routed_direct full_action_view
validate_arm "$ARMS/scalar_transfer_only" routed_direct scalar_transfer_only
validate_arm "$ARMS/matched_shuffled" shuffled_action_control full_action_view
validate_arm "$ARMS/clean_control" clean_control full_action_view

summary_is_valid() {
  python - "$SUMMARY/report.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.is_file() or path.stat().st_size == 0:
    raise SystemExit(1)
report = json.loads(path.read_text(encoding="utf-8"))
if report.get("status") != "noise_corrected_v10_safe_exact_canary_complete":
    raise SystemExit(1)
if set(report.get("input_artifacts", {})) != {
    "full_action_view", "scalar_transfer_only", "matched_shuffled", "clean_control",
}:
    raise SystemExit(1)
PY
}

if [[ -e "$SUMMARY" ]]; then
  if ! summary_is_valid; then
    echo "Existing V10 summary is incomplete or invalid: $SUMMARY" >&2
    exit 6
  fi
  echo "Preserving already completed V10 summary"
else
  echo "=== V10 summary-only recovery; no encoder will be retrained ==="
  python -u "$SNAPSHOT_ROOT/tasks/summarize_noise_corrected_v10_safe_exact_canary.py" \
    --full-action-view-dir "$ARMS/full_action_view" \
    --scalar-transfer-only-dir "$ARMS/scalar_transfer_only" \
    --matched-shuffled-dir "$ARMS/matched_shuffled" \
    --clean-control-dir "$ARMS/clean_control" \
    --bootstrap-resamples 10000 --seed "$SEED" --output-dir "$SUMMARY"
  summary_is_valid
fi

printf 'original_job_id=%s\nsummary_recovery_job_id=%s\n' \
  "$ORIGINAL_JOB_ID" "$SLURM_JOB_ID" > "$RUN_ROOT/recovery_job.txt"
mv "$RUN_ROOT" "$FINAL_ROOT"
echo "=== V10 summary recovery complete: $FINAL_ROOT ==="
