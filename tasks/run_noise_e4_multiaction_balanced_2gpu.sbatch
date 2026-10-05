#!/bin/bash
#SBATCH --job-name=noise_e4_multi
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=data/validation/noise_e4_multi_%j.out
#SBATCH --error=data/validation/noise_e4_multi_%j.err

set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Job 2332784 already paid the expensive V4 routing/materialization cost. Reuse
# its immutable ledger; do not rerun or mutate that completed result.
SOURCE_RUN="${SOURCE_RUN:-data/validation/noise_corrected_best_v5_canary_fold_0_run_2332784}"
case "$SOURCE_RUN" in
  data/validation/noise_corrected_best_v5_canary_fold_0_run_*) ;;
  *) echo "Unexpected SOURCE_RUN path: $SOURCE_RUN" >&2; exit 2 ;;
esac

ALLOCATED_GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -r -a ALLOCATED_GPUS <<< "$ALLOCATED_GPU_LIST"
if (( ${#ALLOCATED_GPUS[@]} < 2 )); then
  echo "Expected two allocated GPUs, got CUDA_VISIBLE_DEVICES=$ALLOCATED_GPU_LIST" >&2
  exit 3
fi
GPU_ZERO="${ALLOCATED_GPUS[0]}"
GPU_ONE="${ALLOCATED_GPUS[1]}"

FOLD=0
SEED=20260907
GRAPH_DIR="data/validation/noise_corrected_candidate_graph_v1_20260906"
GRAPH="$GRAPH_DIR/candidate_graph.npz"
EMBEDDING_CACHE="$GRAPH_DIR/official_embeddings.npz"
SOURCE_MANIFEST="data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz"
DATA="data/models/MassSpecGym_MurckoHist_split.hdf5"
OFFICIAL="data/e1/official_embedding_slim.pt"
ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
INITIAL="data/validation/g8r_noise_final_e8_direct_transfer/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_e8_baseline_symmetric_shared/seed_20260830/fold_${FOLD}/final_shared_encoder.pt"
LEDGER="$SOURCE_RUN/ledger"
PLAN="$SOURCE_RUN/plan_audit/report.json"

FINAL_ROOT="data/validation/noise_e4_multiaction_balanced_fold_${FOLD}_run_${SLURM_JOB_ID}"
RUN_ROOT="data/validation/.noise_e4_multiaction_balanced_fold_${FOLD}_run_${SLURM_JOB_ID}.staging"
ARMS="$RUN_ROOT/arms"
SUMMARY="$RUN_ROOT/summary"

for REQUIRED in \
  "$GRAPH" "$EMBEDDING_CACHE" "$SOURCE_MANIFEST" "$DATA" "$OFFICIAL" \
  "$ARCHITECTURE" "$INITIAL" "$(dirname "$INITIAL")/decision.json" \
  "$LEDGER/report.json" "$LEDGER/training_actions.csv.gz" \
  "$LEDGER/action_spectra.npz" "$PLAN"; do
  [[ -f "$REQUIRED" ]] || { echo "Missing E4-native input: $REQUIRED" >&2; exit 4; }
done
[[ ! -e "$RUN_ROOT" && ! -e "$FINAL_ROOT" ]] || {
  echo "Refusing to overwrite unique E4-native output" >&2
  exit 5
}
mkdir -p "$ARMS"

python -m py_compile \
  tasks/train_noise_final_e4a_direct_augmentation.py \
  tasks/validate_noise_final_e4a_direct_augmentation.py \
  tasks/summarize_noise_e4_native_best_actions.py \
  tasks/noise_final_e4_pmt_core.py \
  tasks/noise_corrected_shuffled_control_v3.py \
  tasks/noise_corrected_fullgraph_evaluation.py
python -u tasks/test_noise_final_e4a_direct_augmentation.py
python -u tasks/test_noise_e4_multiaction_balanced_sbatch.py
python -u tasks/test_noise_corrected_action_routing_v3.py

python - "$LEDGER/report.json" "$LEDGER/training_actions.csv.gz" "$PLAN" "$INITIAL" <<'PY'
import hashlib
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "tasks")
from noise_final_e4_pmt_core import (
    coverage_first_identity_balanced_schedules,
    identity_family_balanced_exposure_weights,
    select_materialized_best_action_union,
    select_materialized_direct_action_panel,
)

report_path, actions_path, plan_path, initial_path = sys.argv[1:]
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
best_before_floor, corrective = select_materialized_best_action_union(actions)
strict_replay_margin_floor = 5e-6
best, _ = select_materialized_direct_action_panel(
    actions, mode="multi_action_balanced", margin_floor=strict_replay_margin_floor,
)
numerical_boundary = pd.to_numeric(
    corrective.action_margin, errors="raise",
).le(strict_replay_margin_floor).to_numpy(bool)
schedules = coverage_first_identity_balanced_schedules(
    best.action_id.astype(str).tolist(),
    best.query_ik14.astype(str).tolist(),
    (
        best.source.astype(str) + "|" + best.family.astype(str)
        + "|" + best.recipe_id.astype(str)
    ).tolist(),
    epochs=4, views_per_identity=4, seed=20260907,
)
flat_exposures = [index for epoch in schedules for index in epoch]
assert len(set(flat_exposures[:len(best)])) == len(best)
weights = identity_family_balanced_exposure_weights(
    best, flat_exposures, total_weight_per_identity=16.0,
)
robust_headroom_pp = float(
    plan["direct_action_training_headroom"][
        "maximum_training_geometry_recall1_headroom_pp"
    ]
)
assert report["status"] == "noise_corrected_routed_action_ledger_complete"
assert report["contracts"]["all_route_clean_ranks_match"] is True
assert report["contracts"]["outer_held_formula_consumed"] is False
assert report["contracts"]["teacher_embedding_target_used"] is False
assert report["contracts"]["P3_consumed"] is False
assert report["model_provenance"]["initial_student_checkpoint_sha256"] == sha256(initial_path)
assert report["provenance"]["training_actions_sha256"] == sha256(actions_path)
assert expected_sources <= set(corrective.source.astype(str))
assert not corrective.action_id.duplicated().any()
assert best.query_index.duplicated().any()
assert len(best) > len(best_before_floor)
assert len(best) >= int(0.99 * len(corrective))
identity_weight = pd.Series([weights[index] for index in flat_exposures]).groupby(
    pd.Series([best.iloc[index].query_ik14 for index in flat_exposures], dtype=str),
).sum()
assert np.allclose(identity_weight.to_numpy(float), 16.0, rtol=0.0, atol=2e-5)
assert not corrective.formula_fold.astype(int).eq(0).any()
assert len(best_before_floor) == int(plan["direct_action_training_headroom"][
    "errors_with_selected_actual_top1_correcting_action"
])
assert robust_headroom_pp >= 4.0
identity_counts = best.query_ik14.astype(str).value_counts().astype(int)
planned_exposures = len(flat_exposures)
planned_optimizer_steps = (planned_exposures + 4 - 1) // 4
print(json.dumps({
    "status": "noise_e4_multiaction_balanced_preflight_pass",
    "strict_top1_corrective_actions": len(corrective),
    "best_action_union_rows_before_numerical_floor": len(best_before_floor),
    "multi_action_rows": len(best),
    "numerical_boundary_actions_excluded": int(np.sum(numerical_boundary)),
    "strict_replay_action_margin_floor": strict_replay_margin_floor,
    "multi_action_queries": int(best.query_index.nunique()),
    "multi_action_sources": sorted(set(best.source.astype(str))),
    "all_strict_actions_exposed_before_recycling": True,
    "physical_action_exposures": planned_exposures,
    "physical_recycled_exposures": planned_exposures - len(best),
    "effective_action_weight_sum": float(np.sum([
        weights[index] for index in flat_exposures
    ])),
    "effective_weight_per_identity": 16.0,
    "planned_optimizer_steps_across_four_epochs": planned_optimizer_steps,
    "training_geometry_action_headroom_pp": robust_headroom_pp,
    "training_kernel": "E4_symmetric_shared_multi_action_balanced_direct",
    "satisfied_action_rank_gradient_gated": True,
    "preclip_loss_scale": 0.16,
    "teacher_embedding_or_margin_target": False,
}, indent=2))
PY

COMMON=(
  --graph "$GRAPH" --source-manifest "$SOURCE_MANIFEST"
  --embedding-cache "$EMBEDDING_CACHE" --data "$DATA"
  --official-checkpoint "$OFFICIAL" --architecture-checkpoint "$ARCHITECTURE"
  --initial-student-checkpoint "$INITIAL"
  --action-selection materialized_routed --materialized-action-dir "$LEDGER"
  --materialized-injection-mode multi_action_balanced
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
  local SUFFIX="e4_multiaction_${ARM}_run${SLURM_JOB_ID}"
  local TAG="curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_as_materialized_routed_marm_${ARM}_minj_multi_action_balanced_warm_${SUFFIX}"
  echo "$ARMS/$ARM/$TAG/seed_${SEED}/fold_${FOLD}"
}

run_arm() {
  local GPU="$1"
  local ARM="$2"
  local SUFFIX="e4_multiaction_${ARM}_run${SLURM_JOB_ID}"
  export CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=8
  python -u tasks/train_noise_final_e4a_direct_augmentation.py \
    "${COMMON[@]}" --materialized-action-arm "$ARM" \
    --output-root "$ARMS/$ARM" --run-suffix "$SUFFIX" || return $?
  python -u tasks/validate_noise_final_e4a_direct_augmentation.py \
    --output-dir "$(result_dir "$ARM")" || return $?
}

echo "=== E4 multi-action balanced direct: targeted/shuffled on two GPUs ==="
run_arm "$GPU_ZERO" targeted &
PID_TARGETED=$!
run_arm "$GPU_ONE" shuffled &
PID_SHUFFLED=$!
STATUS_TARGETED=-1
STATUS_SHUFFLED=-1
# Detect a non-OOM implementation failure in either worker promptly.  The old
# ordered waits could leave the shuffled control training for hours after the
# targeted arm had already raised, even though no paired summary was possible.
while (( STATUS_TARGETED == -1 && STATUS_SHUFFLED == -1 )); do
  if ! kill -0 "$PID_TARGETED" 2>/dev/null; then
    STATUS_TARGETED=0
    wait "$PID_TARGETED" || STATUS_TARGETED=$?
  fi
  if ! kill -0 "$PID_SHUFFLED" 2>/dev/null; then
    STATUS_SHUFFLED=0
    wait "$PID_SHUFFLED" || STATUS_SHUFFLED=$?
  fi
  if (( STATUS_TARGETED == -1 && STATUS_SHUFFLED == -1 )); then
    sleep 2
  fi
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
  echo "E4 multi-action fail-fast worker failure: $STATUS_TARGETED/$STATUS_SHUFFLED" >&2
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
# Status 137 is specifically a cgroup/process kill. Keep the two-GPU fast
# path, but if node RAM is tighter than expected, retry only the killed arm
# after both concurrent workers have exited. This needs no manual --mem and
# never removes or overwrites a completed output.
if (( STATUS_TARGETED == 137 )); then
  [[ ! -e "$(result_dir targeted)" ]] || {
    echo "Refusing to retry targeted over an existing output" >&2; exit 7;
  }
  echo "targeted arm hit status 137; retrying alone on allocated GPU $GPU_ZERO"
  STATUS_TARGETED=0
  run_arm "$GPU_ZERO" targeted || STATUS_TARGETED=$?
fi
if (( STATUS_SHUFFLED == 137 )); then
  [[ ! -e "$(result_dir shuffled)" ]] || {
    echo "Refusing to retry shuffled over an existing output" >&2; exit 8;
  }
  echo "shuffled arm hit status 137; retrying alone on allocated GPU $GPU_ONE"
  STATUS_SHUFFLED=0
  run_arm "$GPU_ONE" shuffled || STATUS_SHUFFLED=$?
fi
if (( STATUS_TARGETED != 0 || STATUS_SHUFFLED != 0 )); then
  echo "E4 multi-action targeted/shuffled worker failed: $STATUS_TARGETED/$STATUS_SHUFFLED" >&2
  exit 6
fi

python -u tasks/summarize_noise_e4_native_best_actions.py \
  --targeted-dir "$(result_dir targeted)" \
  --shuffled-dir "$(result_dir shuffled)" \
  --bootstrap-resamples 10000 --seed "$SEED" --output-dir "$SUMMARY"

mv "$RUN_ROOT" "$FINAL_ROOT"
echo "=== E4 multi-action balanced result preserved at: $FINAL_ROOT ==="

