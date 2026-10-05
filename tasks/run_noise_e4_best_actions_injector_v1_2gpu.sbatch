#!/bin/bash
#SBATCH --job-name=noise_e4_best_inj1
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=data/validation/noise_e4_best_inj1_%j.out
#SBATCH --error=data/validation/noise_e4_best_inj1_%j.err

set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ALLOCATED_GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -r -a ALLOCATED_GPUS <<< "$ALLOCATED_GPU_LIST"
if (( ${#ALLOCATED_GPUS[@]} != 2 )); then
  echo "Expected exactly two allocated GPUs, got CUDA_VISIBLE_DEVICES=$ALLOCATED_GPU_LIST" >&2
  exit 3
fi
GPU_ZERO="${ALLOCATED_GPUS[0]}"
GPU_ONE="${ALLOCATED_GPUS[1]}"

FOLD=0
SEED=20260907
SOURCE_RUN="data/validation/noise_corrected_best_v5_canary_fold_0_run_2332784"
GRAPH_DIR="data/validation/noise_corrected_candidate_graph_v1_20260906"
GRAPH="$GRAPH_DIR/candidate_graph.npz"
EMBEDDING_CACHE="$GRAPH_DIR/official_embeddings.npz"
EVAL_METADATA="data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz"
DATA="data/models/MassSpecGym_MurckoHist_split.hdf5"
OFFICIAL="data/e1/official_embedding_slim.pt"
ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
INITIAL="data/validation/g8r_noise_final_e8_direct_transfer/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_e8_baseline_symmetric_shared/seed_20260830/fold_${FOLD}/final_shared_encoder.pt"
LEDGER="$SOURCE_RUN/ledger"
PLAN="$SOURCE_RUN/plan_audit/report.json"
SOURCE_MANIFEST="tasks/noise_e4_best_actions_injector_v1_source_manifest.sha256"
EXPECTED_SOURCE_MANIFEST_SHA256="697e385c2b9530d5be63572c1c93021fe7f931374d37edbcc520548e9be84153"
SBATCH_SOURCE="tasks/run_noise_e4_best_actions_injector_v1_2gpu.sbatch"
SPEC_SOURCE="docs/NOISE_E4_BEST_ACTIONS_INJECTOR_V1_SPEC_20260913.md"

FINAL_ROOT="data/validation/noise_e4_best_actions_injector_v1_fold_${FOLD}_seed_${SEED}_run_${SLURM_JOB_ID}"
RUN_ROOT="data/validation/.noise_e4_best_actions_injector_v1_fold_${FOLD}_seed_${SEED}_run_${SLURM_JOB_ID}.staging"
ARMS="$RUN_ROOT/arms"
SUMMARY="$RUN_ROOT/summary"
SOURCE_SNAPSHOT="$RUN_ROOT/source_snapshot"

for REQUIRED in \
  "$GRAPH" "$GRAPH_DIR/report.json" "$EMBEDDING_CACHE" "$EVAL_METADATA" \
  "$DATA" "$OFFICIAL" "$ARCHITECTURE" "$INITIAL" \
  "$(dirname "$INITIAL")/decision.json" "$LEDGER/report.json" \
  "$LEDGER/training_actions.csv.gz" "$LEDGER/action_spectra.npz" "$PLAN"; do
  [[ -f "$REQUIRED" ]] || { echo "Missing frozen E4 input: $REQUIRED" >&2; exit 4; }
done

printf '%s  %s\n' "$EXPECTED_SOURCE_MANIFEST_SHA256" "$SOURCE_MANIFEST" \
  | sha256sum -c -
sha256sum -c "$SOURCE_MANIFEST"
mapfile -t SOURCE_FILES < <(awk 'NF == 2 {print $2}' "$SOURCE_MANIFEST")
if (( ${#SOURCE_FILES[@]} < 80 )); then
  echo "Incomplete E4+InjectorV1 source closure: ${#SOURCE_FILES[@]} files" >&2
  exit 4
fi

[[ ! -e "$RUN_ROOT" && ! -e "$FINAL_ROOT" ]] || {
  echo "Refusing to overwrite unique E4+InjectorV1 output" >&2
  exit 5
}
mkdir -p "$ARMS" "$SOURCE_SNAPSHOT"

AVAILABLE_KB="$(df -Pk "$PWD" | awk 'NR==2 {print $4}')"
if [[ ! "$AVAILABLE_KB" =~ ^[0-9]+$ ]] || (( AVAILABLE_KB < 6291456 )); then
  echo "E4+InjectorV1 requires at least 6 GiB filesystem headroom; available_kb=$AVAILABLE_KB" >&2
  exit 5
fi

SNAPSHOT_FILES=(
  "${SOURCE_FILES[@]}"
  "$SOURCE_MANIFEST"
  "$SBATCH_SOURCE"
  "$SPEC_SOURCE"
)
for SOURCE_FILE in "${SNAPSHOT_FILES[@]}"; do
  SNAPSHOT_TARGET="$SOURCE_SNAPSHOT/$SOURCE_FILE"
  mkdir -p "$(dirname "$SNAPSHOT_TARGET")"
  cp "$SOURCE_FILE" "$SNAPSHOT_TARGET"
done
(
  cd "$SOURCE_SNAPSHOT"
  sha256sum "${SNAPSHOT_FILES[@]}" > sha256_manifest.txt
  sha256sum -c "$SOURCE_MANIFEST"
)

SNAPSHOT_ROOT="$PWD/$SOURCE_SNAPSHOT"
export PYTHONPATH="$SNAPSHOT_ROOT:$SNAPSHOT_ROOT/tasks"
PYTHON_SOURCES=()
for SOURCE_FILE in "${SOURCE_FILES[@]}"; do
  [[ "$SOURCE_FILE" == *.py ]] && PYTHON_SOURCES+=("$SNAPSHOT_ROOT/$SOURCE_FILE")
done
python -m py_compile "${PYTHON_SOURCES[@]}"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_action_injector_v1.py"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_e4_action_injector_v1_bridge.py"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_final_e4a_direct_augmentation.py"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_e4_multiaction_balanced_sbatch.py"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_e4_best_actions_injector_v1_sbatch.py"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_corrected_action_routing_v3.py"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_corrected_fullgraph_evaluation.py"

python - "$LEDGER/report.json" "$LEDGER/training_actions.csv.gz" "$PLAN" "$INITIAL" "$SNAPSHOT_ROOT/tasks" <<'PY'
import hashlib
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, sys.argv[5])
from noise_final_e4_pmt_core import (
    coverage_first_identity_balanced_schedules,
    identity_family_balanced_exposure_weights,
    select_materialized_best_action_union,
    select_materialized_direct_action_panel,
)

report_path, actions_path, plan_path, initial_path = sys.argv[1:5]
report = json.load(open(report_path, encoding="utf-8"))
plan = json.load(open(plan_path, encoding="utf-8"))
actions = pd.read_csv(actions_path, low_memory=False)

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

expected_sources = {
    "N_mature", "P_guided_original", "E10B", "E11", "E12B",
    "A4_exact", "V4_gradient_path",
}
one_best, corrective = select_materialized_best_action_union(actions)
selected, strict = select_materialized_direct_action_panel(
    actions, mode="multi_action_balanced", margin_floor=5e-6,
)
schedules = coverage_first_identity_balanced_schedules(
    selected.action_id.astype(str).tolist(),
    selected.query_ik14.astype(str).tolist(),
    (
        selected.source.astype(str) + "|" + selected.family.astype(str)
        + "|" + selected.recipe_id.astype(str)
    ).tolist(),
    epochs=4, views_per_identity=4, seed=20260907,
)
flat = [index for epoch in schedules for index in epoch]
weights = identity_family_balanced_exposure_weights(
    selected, flat, total_weight_per_identity=16.0,
)
identity_weight = pd.Series([weights[index] for index in flat]).groupby(
    pd.Series([selected.iloc[index].query_ik14 for index in flat], dtype=str),
).sum()
headroom = float(plan["direct_action_training_headroom"][
    "maximum_training_geometry_recall1_headroom_pp"
])
assert report["status"] == "noise_corrected_routed_action_ledger_complete"
assert report["contracts"]["all_route_clean_ranks_match"] is True
assert report["contracts"]["outer_held_formula_consumed"] is False
assert report["contracts"]["teacher_embedding_target_used"] is False
assert report["contracts"]["P3_consumed"] is False
assert report["model_provenance"]["initial_student_checkpoint_sha256"] == sha256(initial_path)
assert report["provenance"]["training_actions_sha256"] == sha256(actions_path)
assert expected_sources <= set(corrective.source.astype(str))
assert expected_sources <= set(selected.source.astype(str))
assert not corrective.action_id.duplicated().any()
assert selected.query_index.duplicated().any()
assert len(selected) > len(one_best)
assert len(selected) >= int(0.99 * len(corrective))
assert not selected.formula_fold.astype(int).eq(0).any()
assert len(set(flat[:len(selected)])) == len(selected)
assert np.allclose(identity_weight.to_numpy(float), 16.0, rtol=0.0, atol=2e-5)
assert headroom >= 4.0
print(json.dumps({
    "status": "noise_e4_best_actions_injector_v1_preflight_pass",
    "strict_top1_corrective_actions": len(corrective),
    "selected_multi_action_rows": len(selected),
    "selected_multi_action_queries": int(selected.query_index.nunique()),
    "one_best_rows_explicitly_rejected": len(one_best),
    "all_seven_mature_sources_present": sorted(set(selected.source.astype(str))),
    "all_actions_exposed_before_recycling": True,
    "effective_weight_per_identity": 16.0,
    "planned_optimizer_steps": (len(flat) + 3) // 4,
    "training_geometry_action_headroom_pp": headroom,
    "training_kernel": "existing_E4_symmetric_shared_multi_action_balanced",
    "optimizer_boundary": "frozen_ActionInjectorV1",
    "teacher_embedding_or_margin_target": False,
}, indent=2))
PY

COMMON=(
  --graph "$GRAPH" --source-manifest "$EVAL_METADATA"
  --embedding-cache "$EMBEDDING_CACHE" --data "$DATA"
  --official-checkpoint "$OFFICIAL" --architecture-checkpoint "$ARCHITECTURE"
  --initial-student-checkpoint "$INITIAL"
  --action-selection materialized_routed --materialized-action-dir "$LEDGER"
  --materialized-injection-mode multi_action_balanced
  --optimizer-boundary-mode action_injector_v1
  --injector-target-attributable-fraction 0.25
  --injector-minimum-protective-retention 0.90
  --injector-maximum-update-norm-ratio 1.50
  --policy curriculum --action-scope all --outer-fold "$FOLD"
  --formula-fold-seed 20260825 --seed "$SEED"
  --epochs 4 --batch-actions 4 --views-per-identity 4
  --error-views-per-identity 0 --positive-spectra 4 --negative-molecules 8
  --unfreeze-blocks 1 --backbone-lr 2e-6 --head-lr 1e-5
  --weight-decay 1e-4 --rank-margin 0.05 --temperature 0.10
  --lambda-clean-rank 1.0 --lambda-aug-rank 1.0
  --lambda-consistency 0.25 --lambda-margin-floor 2.0
  --lambda-preserve 5.0 --margin-floor-slack 0.005
  --safety-ratio 1.0 --safety-stream-weight 1.0
  --positive-stream-weight 0 --guided-noise-policy none --guided-noise-weight 0
  --direct-transfer-mode symmetric --rank-reference-mode shared
  --grad-clip 1.0 --eval-batch-size 256 --bootstrap-resamples 10000
  --device cuda --no-amp
)

result_dir() {
  local ARM="$1"
  local SUFFIX="e4_best_inj1_${ARM}_run${SLURM_JOB_ID}"
  local TAG="curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_as_materialized_routed_marm_${ARM}_minj_multi_action_balanced_ob_action_injector_v1_warm_${SUFFIX}"
  echo "$ARMS/$ARM/$TAG/seed_${SEED}/fold_${FOLD}"
}

archive_partial_arm() {
  local ARM="$1"
  local PARTIAL
  local FAILED_ROOT="$RUN_ROOT/failed_attempts"
  PARTIAL="$(result_dir "$ARM")"
  if [[ -e "$PARTIAL" ]]; then
    mkdir -p "$FAILED_ROOT"
    local FAILED="$FAILED_ROOT/${ARM}_status137_first_attempt"
    [[ ! -e "$FAILED" ]] || {
      echo "Refusing to overwrite archived $ARM status-137 attempt" >&2
      return 9
    }
    mv "$PARTIAL" "$FAILED"
  fi
}

run_arm() {
  local WORKER_GPU="$1"
  local ARM="$2"
  local SUFFIX="e4_best_inj1_${ARM}_run${SLURM_JOB_ID}"
  export CUDA_VISIBLE_DEVICES="$WORKER_GPU" OMP_NUM_THREADS=8
  python -u "$SNAPSHOT_ROOT/tasks/train_noise_final_e4a_direct_augmentation.py" \
    "${COMMON[@]}" --materialized-action-arm "$ARM" \
    --output-root "$ARMS/$ARM" --run-suffix "$SUFFIX" || return $?
  python -u "$SNAPSHOT_ROOT/tasks/validate_noise_final_e4a_direct_augmentation.py" \
    --output-dir "$(result_dir "$ARM")" || return $?
}

echo "=== Existing E4 + all mature actions + frozen Injector V1 ==="
run_arm "$GPU_ZERO" targeted &
PID_TARGETED=$!
run_arm "$GPU_ONE" shuffled &
PID_SHUFFLED=$!
STATUS_TARGETED=-1
STATUS_SHUFFLED=-1
while (( STATUS_TARGETED == -1 && STATUS_SHUFFLED == -1 )); do
  if ! kill -0 "$PID_TARGETED" 2>/dev/null; then
    STATUS_TARGETED=0
    wait "$PID_TARGETED" || STATUS_TARGETED=$?
  fi
  if ! kill -0 "$PID_SHUFFLED" 2>/dev/null; then
    STATUS_SHUFFLED=0
    wait "$PID_SHUFFLED" || STATUS_SHUFFLED=$?
  fi
  if (( STATUS_TARGETED == -1 && STATUS_SHUFFLED == -1 )); then sleep 2; fi
done
if ((
  ( STATUS_TARGETED != -1 && STATUS_TARGETED != 0 && STATUS_TARGETED != 137 )
  || ( STATUS_SHUFFLED != -1 && STATUS_SHUFFLED != 0 && STATUS_SHUFFLED != 137 )
)); then
  if (( STATUS_TARGETED == -1 )); then
    kill "$PID_TARGETED" 2>/dev/null || true
    STATUS_TARGETED=0
    wait "$PID_TARGETED" || STATUS_TARGETED=$?
  fi
  if (( STATUS_SHUFFLED == -1 )); then
    kill "$PID_SHUFFLED" 2>/dev/null || true
    STATUS_SHUFFLED=0
    wait "$PID_SHUFFLED" || STATUS_SHUFFLED=$?
  fi
  echo "E4+InjectorV1 fail-fast worker failure: $STATUS_TARGETED/$STATUS_SHUFFLED" >&2
  exit 6
fi
if (( STATUS_TARGETED == -1 )); then
  STATUS_TARGETED=0
  wait "$PID_TARGETED" || STATUS_TARGETED=$?
fi
if (( STATUS_SHUFFLED == -1 )); then
  STATUS_SHUFFLED=0
  wait "$PID_SHUFFLED" || STATUS_SHUFFLED=$?
fi
if (( STATUS_TARGETED == 137 )); then
  archive_partial_arm targeted || exit $?
  echo "targeted arm hit status 137; retrying alone on allocated GPU $GPU_ZERO"
  STATUS_TARGETED=0
  run_arm "$GPU_ZERO" targeted || STATUS_TARGETED=$?
fi
if (( STATUS_SHUFFLED == 137 )); then
  archive_partial_arm shuffled || exit $?
  echo "shuffled arm hit status 137; retrying alone on allocated GPU $GPU_ONE"
  STATUS_SHUFFLED=0
  run_arm "$GPU_ONE" shuffled || STATUS_SHUFFLED=$?
fi
if (( STATUS_TARGETED != 0 || STATUS_SHUFFLED != 0 )); then
  echo "E4+InjectorV1 worker failed: $STATUS_TARGETED/$STATUS_SHUFFLED" >&2
  exit 6
fi

python -u "$SNAPSHOT_ROOT/tasks/summarize_noise_e4_native_best_actions.py" \
  --targeted-dir "$(result_dir targeted)" \
  --shuffled-dir "$(result_dir shuffled)" \
  --bootstrap-resamples 10000 --seed "$SEED" --output-dir "$SUMMARY"

mv "$RUN_ROOT" "$FINAL_ROOT"
echo "=== E4+InjectorV1 result preserved at: $FINAL_ROOT ==="
