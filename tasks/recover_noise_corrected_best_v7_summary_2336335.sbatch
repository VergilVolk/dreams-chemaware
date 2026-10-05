#!/bin/bash
#SBATCH --job-name=noise_v7_summary_recover
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=data/validation/noise_v7_summary_recover_%j.out
#SBATCH --error=data/validation/noise_v7_summary_recover_%j.err

set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

FAILED_RUN_ID="${FAILED_RUN_ID:-2336335}"
FOLD=0
SEED=20260911
SOURCE_MANIFEST="tasks/noise_best_v6_source_manifest.sha256"
EXPECTED_SOURCE_MANIFEST_SHA256="888d941645d7e30c7e8dd3e8d45d485d7f058b75675bbd227af6d2c3693f74c1"
RUN_ROOT="data/validation/.noise_corrected_best_v7_corrective_restored_fold_${FOLD}_seed_${SEED}_run_${FAILED_RUN_ID}.staging"
FINAL_ROOT="data/validation/noise_corrected_best_v7_corrective_restored_fold_${FOLD}_seed_${SEED}_run_${FAILED_RUN_ID}"
ARMS="$RUN_ROOT/arms"
SUMMARY="$RUN_ROOT/summary"
RECOVERY_SNAPSHOT="$RUN_ROOT/recovery_source_snapshot"

[[ -d "$RUN_ROOT" ]] || { echo "Missing failed staging run: $RUN_ROOT" >&2; exit 4; }
[[ ! -e "$FINAL_ROOT" ]] || { echo "Refusing to overwrite final run: $FINAL_ROOT" >&2; exit 5; }
[[ ! -e "$SUMMARY" ]] || { echo "Refusing to overwrite existing summary: $SUMMARY" >&2; exit 5; }
[[ ! -e "$RECOVERY_SNAPSHOT" ]] || {
  echo "Refusing to overwrite recovery snapshot: $RECOVERY_SNAPSHOT" >&2
  exit 5
}
for ARM in routed_direct shuffled_action_control clean_control; do
  for REQUIRED in decision.json held_per_query.csv.gz held_metric_evidence.npz final_shared_encoder.pt; do
    [[ -f "$ARMS/$ARM/$REQUIRED" ]] || {
      echo "Incomplete completed arm: $ARMS/$ARM/$REQUIRED" >&2
      exit 4
    }
  done
done

printf '%s  %s\n' "$EXPECTED_SOURCE_MANIFEST_SHA256" "$SOURCE_MANIFEST" | sha256sum -c -
sha256sum -c "$SOURCE_MANIFEST"
mapfile -t SOURCE_FILES < <(awk 'NF == 2 {print $2}' "$SOURCE_MANIFEST")
if (( ${#SOURCE_FILES[@]} != 61 )); then
  echo "Incomplete best-v7 source closure: ${#SOURCE_FILES[@]} files" >&2
  exit 4
fi

mkdir -p "$RECOVERY_SNAPSHOT"
for SOURCE_FILE in "${SOURCE_FILES[@]}"; do
  TARGET="$RECOVERY_SNAPSHOT/$SOURCE_FILE"
  mkdir -p "$(dirname "$TARGET")"
  cp "$SOURCE_FILE" "$TARGET"
done
cp "$SOURCE_MANIFEST" "$RECOVERY_SNAPSHOT/$SOURCE_MANIFEST"
mkdir -p "$RECOVERY_SNAPSHOT/tasks"
cp tasks/recover_noise_corrected_best_v7_summary_2336335.sbatch \
  "$RECOVERY_SNAPSHOT/tasks/"
(
  cd "$RECOVERY_SNAPSHOT"
  sha256sum -c "$SOURCE_MANIFEST"
)

SNAPSHOT_ROOT="$PWD/$RECOVERY_SNAPSHOT"
export PYTHONPATH="$SNAPSHOT_ROOT:$SNAPSHOT_ROOT/tasks"
python -m py_compile \
  "$SNAPSHOT_ROOT/tasks/summarize_noise_corrected_direct_v3_arms.py" \
  "$SNAPSHOT_ROOT/tasks/test_summarize_noise_corrected_direct_v3_arms.py"
python -u "$SNAPSHOT_ROOT/tasks/test_summarize_noise_corrected_direct_v3_arms.py"

echo "=== recovering completed best-v7 three-arm summary without retraining ==="
python -u "$SNAPSHOT_ROOT/tasks/summarize_noise_corrected_direct_v3_arms.py" \
  --routed-dir "$ARMS/routed_direct" \
  --shuffled-dir "$ARMS/shuffled_action_control" \
  --clean-dir "$ARMS/clean_control" \
  --minimum-delta-recall1-pp 4.0 --bootstrap-resamples 10000 \
  --formula-ci-familywise-hypotheses 4 --seed "$SEED" \
  --output-dir "$SUMMARY"

python - "$SUMMARY/report.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["status"] == "noise_corrected_best_action_v7_arm_summary_complete"
assert report["direct_contract"] == "best_action_v7_corrective_restored"
assert report["training_seed"] == 20260911
assert report["minimum_delta_recall1_pp"] == 4.0
assert report["bootstrap_resamples"] == 10000
assert report["formula_ci_familywise_hypotheses"] == 4
assert isinstance(report["single_seed_gate_pass"], bool)
assert report["promote"] is False
assert report["promotion_status"] in {"pending_multiseed", "single_seed_gate_failed"}
print(json.dumps({
    "status": "best_action_v7_summary_recovery_complete",
    "single_seed_gate_pass": report["single_seed_gate_pass"],
    "promotion_status": report["promotion_status"],
    "promotion_checks": report["promotion_checks"],
    "claim_limit": report["claim_limit"],
}, indent=2))
PY

mv "$RUN_ROOT" "$FINAL_ROOT"
echo "=== recovered best-v7 result preserved at: $FINAL_ROOT ==="
