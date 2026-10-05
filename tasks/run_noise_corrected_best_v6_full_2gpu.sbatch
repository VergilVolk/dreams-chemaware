#!/bin/bash
#SBATCH --job-name=noise_v7_corr_restore
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=96:00:00
#SBATCH --output=data/validation/noise_v7_corr_restore_%j.out
#SBATCH --error=data/validation/noise_v7_corr_restore_%j.err

set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SOURCE_RUN="data/validation/noise_corrected_best_v5_canary_fold_0_run_2332784"

ALLOCATED_GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -r -a ALLOCATED_GPUS <<< "$ALLOCATED_GPU_LIST"
if (( ${#ALLOCATED_GPUS[@]} != 2 )); then
  echo "Expected exactly two allocated GPUs, got CUDA_VISIBLE_DEVICES=$ALLOCATED_GPU_LIST" >&2
  exit 3
fi
GPU_ZERO="${ALLOCATED_GPUS[0]}"
GPU_ONE="${ALLOCATED_GPUS[1]}"

FOLD=0
SEED=20260911
GRAPH="data/validation/noise_corrected_candidate_graph_v1_20260906"
EVAL_METADATA="data/validation/chemaware_corrected_candidate_manifest_v1"
DATA="data/models/MassSpecGym_MurckoHist_split.hdf5"
OFFICIAL="data/e1/official_embedding_slim.pt"
ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
INITIAL="data/validation/g8r_noise_final_e8_direct_transfer/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_e8_baseline_symmetric_shared/seed_20260830/fold_${FOLD}/final_shared_encoder.pt"
INITIAL_DECISION="$(dirname "$INITIAL")/decision.json"
LEDGER="$SOURCE_RUN/ledger"
SOURCE_MANIFEST="tasks/noise_best_v6_source_manifest.sha256"
EXPECTED_SOURCE_MANIFEST_SHA256="888d941645d7e30c7e8dd3e8d45d485d7f058b75675bbd227af6d2c3693f74c1"
SBATCH_SOURCE="tasks/run_noise_corrected_best_v6_full_2gpu.sbatch"
SPEC_SOURCE="docs/NOISE_BEST_ACTION_V7_CORRECTIVE_RESTORATION_SPEC_20260912.md"

FINAL_ROOT="data/validation/noise_corrected_best_v7_corrective_restored_fold_${FOLD}_seed_${SEED}_run_${SLURM_JOB_ID}"
RUN_ROOT="data/validation/.noise_corrected_best_v7_corrective_restored_fold_${FOLD}_seed_${SEED}_run_${SLURM_JOB_ID}.staging"
ARMS="$RUN_ROOT/arms"
SUMMARY="$RUN_ROOT/summary"
SOURCE_SNAPSHOT="$RUN_ROOT/source_snapshot"

for REQUIRED in \
  "$GRAPH/candidate_graph.npz" "$GRAPH/report.json" \
  "$EVAL_METADATA/manifest.npz" "$DATA" "$OFFICIAL" "$ARCHITECTURE" \
  "$INITIAL" "$INITIAL_DECISION" \
  "$LEDGER/training_actions.csv.gz" "$LEDGER/action_spectra.npz" \
  "$LEDGER/report.json"; do
  [[ -f "$REQUIRED" ]] || { echo "Missing frozen v6 input: $REQUIRED" >&2; exit 4; }
done

# Pin one complete, recursively audited local Python dependency closure before
# model construction.  The manifest itself is pinned by a literal hash in this
# entrypoint, so a stale server file cannot silently satisfy an import.
printf '%s  %s\n' "$EXPECTED_SOURCE_MANIFEST_SHA256" "$SOURCE_MANIFEST" \
  | sha256sum -c -
sha256sum -c "$SOURCE_MANIFEST"
mapfile -t SOURCE_FILES < <(awk 'NF == 2 {print $2}' "$SOURCE_MANIFEST")
if (( ${#SOURCE_FILES[@]} != 61 )); then
  echo "Incomplete best-v7 source closure: ${#SOURCE_FILES[@]} files" >&2
  exit 4
fi

[[ ! -e "$RUN_ROOT" && ! -e "$FINAL_ROOT" ]] || {
  echo "Refusing to overwrite unique v7 output" >&2
  exit 5
}
mkdir -p "$ARMS" "$SOURCE_SNAPSHOT"
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

# Every Python process below executes the byte-exact snapshot, not the mutable
# live checkout.  This prevents a long two-GPU run from loading or reporting a
# source file that changed after the initial manifest verification.
SNAPSHOT_ROOT="$PWD/$SOURCE_SNAPSHOT"
export PYTHONPATH="$SNAPSHOT_ROOT:$SNAPSHOT_ROOT/tasks"

PYTHON_SOURCES=()
for SOURCE_FILE in "${SOURCE_FILES[@]}"; do
  [[ "$SOURCE_FILE" == *.py ]] && PYTHON_SOURCES+=("$SNAPSHOT_ROOT/$SOURCE_FILE")
done
python -m py_compile "${PYTHON_SOURCES[@]}"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_corrected_best_action_v6.py"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_corrected_direct_v3_core.py"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_corrected_direct_v3_trainer.py"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_corrected_action_routing_v3.py"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_corrected_routed_ledger_streaming.py"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_corrected_fullgraph_evaluation.py"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_corrected_transfer_objective_v4.py"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_corrected_update_arbitration_v4.py"
python -u "$SNAPSHOT_ROOT/tasks/test_summarize_noise_corrected_direct_v3_arms.py"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_corrected_best_v6_sbatch.py"

echo "=== best-v7 preflight: immutable seven-source strict Top-1 action bank ==="
python - \
  "$LEDGER/training_actions.csv.gz" "$LEDGER/report.json" \
  "$LEDGER/action_spectra.npz" "$GRAPH/candidate_graph.npz" \
  "$GRAPH/report.json" "$INITIAL" "$INITIAL_DECISION" "$OFFICIAL" \
  "$DATA" "$ARCHITECTURE" <<'PY'
import hashlib
import json
import numpy as np
import pandas as pd
import sys
from noise_final_e4_pmt_core import select_materialized_direct_action_panel
from noise_final_core import CandidateGraph, stable_fold
from noise_corrected_direct_v3_core import v3_schedule_geometry
from train_noise_corrected_routed_direct import (
    _validate_registered_best_action_v6_schedule_geometry,
)

actions = pd.read_csv(sys.argv[1], low_memory=False)
report = json.load(open(sys.argv[2], encoding="utf-8"))
expected_sha256 = {
    sys.argv[9]: "ccda2c4114d9b21413977df03376ca0fc097956a7fa304b861a3154a2b81e64f",
    sys.argv[10]: "9884b62ecadf4bd441d22fec79b6787e5ffef168e15e7d8d5804dbdea08b38b2",
    sys.argv[1]: "93f0785a69b5e323490a0b543059fa213fabb6ff848697831efba5b3fed667aa",
    sys.argv[2]: "246e7e871e6669fec9f2c62330a69bf9eb563a1b97b72b89732cecd682844349",
    sys.argv[3]: "6d57615aebbfb7bd6327bb7edb143c6837d2586221c61aa2ff45e833f5e1512a",
    sys.argv[4]: "8a57bb3a9cccdf69a738f2b093bfad8f6fc4393b23f1342c7a035ac54ce58fa1",
    sys.argv[5]: "1a69db8f14d271adfa454900a206b5bc3cc4d1c51e91703e499ca74837bf557f",
    sys.argv[6]: "8047b3f58c6808c86b320ac94b9e610610384040fa3a03e8d70550eb438a24af",
    sys.argv[7]: "0a7de098701e9ef5f6c3fa2ee62fcc2e1b301dab44878286bd0b61bc31e3e9d9",
    sys.argv[8]: "8928f908606c0bd652c5a4107d3c35102f660622958c225a1f625abe4b1ba245",
}
for path, expected in expected_sha256.items():
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    assert digest.hexdigest() == expected, (path, digest.hexdigest(), expected)
expected_sources = {
    "N_mature", "P_guided_original", "E10B", "E11", "E12B",
    "A4_exact", "V4_gradient_path",
}
assert report["formal_training_authorized"] is True
assert report["contracts"]["three_semantics_are_separate"] is True
assert report["contracts"]["all_route_clean_ranks_match"] is True
assert report["shared_current_E8_clean_geometry"]["rank_disagreement_queries"] == 0
assert len(actions) == 327678
assert actions.supervision_kind.value_counts().to_dict() == {
    "robust": 179289, "harmful": 85959, "corrective": 62430,
}
assert expected_sources == set(actions.source.astype(str))
selected, strict = select_materialized_direct_action_panel(
    actions, mode="multi_action_balanced", margin_floor=5e-6,
)
assert len(strict) == 32127
assert len(selected) == 32114
assert selected.query_index.nunique() == 3482
assert expected_sources == set(selected.source.astype(str))
assert selected.action_id.is_unique
assert selected.clean_rank.ne(1).all() and selected.action_rank.eq(1).all()
assert selected.groupby("query_index").size().max() <= 16
assert len(selected) + 85959 + 179289 == 297362
robust_queries = int(actions.loc[
    actions.supervision_kind.eq("robust"), "query_index"
].nunique())
harmful_queries = int(actions.loc[
    actions.supervision_kind.eq("harmful"), "query_index"
].nunique())
graph = CandidateGraph(sys.argv[4])
outer_folds = np.asarray([
    stable_fold(str(value), 5, 20260825) for value in graph.query_formula
], dtype=np.int8)
protective_queries = int(np.sum(outer_folds != 0))
schedule_geometry = v3_schedule_geometry(
    corrective_queries=int(selected.query_index.nunique()),
    robust_queries=robust_queries,
    harmful_queries=harmful_queries,
    protective_queries=protective_queries,
    corrective_batch_size=4,
    protective_batch_size=8,
    maximum_auxiliary_microbatches_per_step=4,
    maximum_protective_microbatches_per_step=4,
    maximum_corrective_recycle_factor=4.0,
).as_dict()
_validate_registered_best_action_v6_schedule_geometry(
    schedule_geometry, context="allocated-sbatch-preflight",
)
print(json.dumps({
    "status": "best_action_v7_corrective_restoration_preflight_pass",
    "all_ledger_rows": len(actions),
    "strict_top1_rows_before_floor": len(strict),
    "strict_top1_rows_after_floor": len(selected),
    "strict_top1_queries": selected.query_index.nunique(),
    "sources": sorted(set(selected.source.astype(str))),
    "one_best_query_compression_used": False,
    "teacher_embedding_or_distillation_target_used": False,
    "schedule_geometry": schedule_geometry,
    "schedule_geometry_validated_before_model_load": True,
}, indent=2))
PY

COMMON=(
  --graph-dir "$GRAPH" --source-manifest-dir "$EVAL_METADATA"
  --routed-ledger-dir "$LEDGER" --data "$DATA"
  --official-checkpoint "$OFFICIAL" --architecture-checkpoint "$ARCHITECTURE"
  --initial-student-checkpoint "$INITIAL" --outer-fold "$FOLD"
  --formula-fold-seed 20260825 --corrective-objective-mode v3_direct
  --direct-contract best_action_v7_corrective_restored
  --corrective-admission strict_top1 --corrective-margin-floor 0.000005
  --no-arm-invariant-targeted-calibration
  --epochs 4 --batch-queries 4 --protective-batch-queries 8
  --positive-references 4 --action-conditioned-positive-references 32
  --action-conditioned-negative-molecules 32 --negative-molecules 8
  --references-per-negative 2 --reference-refresh-every-epochs 1
  --n-highest-peaks 100 --unfreeze-blocks 1
  --backbone-lr 2e-6 --head-lr 1e-5 --weight-decay 1e-4
  --rank-margin 0.05 --temperature 0.10 --topk-negatives 8
  --action-safety-slack 0.005 --v3-harmful-damage-slack 0.005
  --lambda-margin-transfer 1.0 --lambda-action-rank 0.25
  --lambda-action-safety 0.25 --lambda-v3-corrective-consistency 0.25
  --lambda-v3-robust-rank 0.25 --lambda-v3-robust-floor 0.25
  --lambda-v3-harmful-rank 1.0 --lambda-v3-harmful-floor 1.0
  --margin-transfer-fraction 0.50 --margin-transfer-cap 0.10
  --transfer-target-allocation mass_neutral_monotone
  --maximum-transfer-cap-factor 2.0
  --target-v3-payload-to-transfer-ratio 0.25
  --target-v3-consistency-to-transfer-ratio 0.25
  --target-v3-robust-to-transfer-ratio 0.10
  --target-v3-harmful-to-risk-ratio 0.25
  --v3-branch-scale-cap 16 --minimum-v3-internal-combination-retention 0.50
  --minimum-v3-active-transfer-fraction 0.05
  --v3-maximum-protective-microbatches-per-step 4
  --v3-maximum-auxiliary-microbatches-per-step 4
  --v3-maximum-corrective-recycle-factor 4 --v3-corrective-recycle-full-dose
  --lambda-clean-continuation 1.0 --lambda-risk-floor 2.0
  --lambda-preserve 5.0 --risk-margin-slack 0.005
  --calibration-batches 32 --target-corrective-to-risk-ratio 1.0
  --corrective-scale-cap 256 --minimum-corrective-scale 0.5
  --target-preclip-gradient-norm 0.05 --minimum-action-retention-p10 0.50
  --minimum-optimizer-action-alignment-p10 0.05
  --minimum-optimizer-action-attributable-fraction-p10 0.25
  --minimum-corrective-direction-retention-p10 0.10
  --v4-audit-minimum-attributable-fraction 0.25
  --v4-audit-maximum-action-gain 4.0
  --materialize-optimizer-update-restoration
  --optimizer-restoration-scope corrective_only
  --reconcile-restored-adamw-first-moment
  --optimizer-restoration-minimum-risk-component-retention 0.90
  --minimum-optimizer-restoration-target-reached-fraction 0.90
  --continue-after-signal-gate-failure
  --optimizer-attribution-steps-per-epoch 16 --progress-every-steps 50
  --maximum-clip-event-fraction 0.10 --grad-clip 1.0
  --eval-batch-size 256 --maximum-spectra-per-action-forward 512
  --bootstrap-resamples 10000
  --device cuda --no-amp --seed "$SEED"
)

run_arm() {
  local WORKER_GPU="$1"
  local ARM="$2"
  export CUDA_VISIBLE_DEVICES="$WORKER_GPU" OMP_NUM_THREADS=8
  python -u "$SNAPSHOT_ROOT/tasks/train_noise_corrected_routed_direct.py" \
    "${COMMON[@]}" --arm "$ARM" --output-dir "$ARMS/$ARM"
}

run_arm_async() {
  local WORKER_GPU="$1"
  local ARM="$2"
  export CUDA_VISIBLE_DEVICES="$WORKER_GPU" OMP_NUM_THREADS=8
  # The background PID must be the Python worker itself.  Without exec, killing
  # the shell-function subshell can orphan a live GPU process.
  exec python -u "$SNAPSHOT_ROOT/tasks/train_noise_corrected_routed_direct.py" \
    "${COMMON[@]}" --arm "$ARM" --output-dir "$ARMS/$ARM"
}

echo "=== best-v7 full stage 1/3: targeted and shuffled concurrently ==="
run_arm_async "$GPU_ZERO" routed_direct &
PID_ROUTED=$!
run_arm_async "$GPU_ONE" shuffled_action_control &
PID_SHUFFLED=$!

# Portable fail-fast for Bash 4.2+ (wait -n is unavailable on common CentOS 7
# login/compute images).  jobs -pr distinguishes a completed child without
# consuming its cached wait status.
worker_is_running() {
  jobs -pr | grep -Fxq -- "$1"
}
while worker_is_running "$PID_ROUTED" && worker_is_running "$PID_SHUFFLED"; do
  sleep 2
done
if worker_is_running "$PID_ROUTED"; then
  FIRST_PID="$PID_SHUFFLED"
  SECOND_PID="$PID_ROUTED"
else
  FIRST_PID="$PID_ROUTED"
  SECOND_PID="$PID_SHUFFLED"
fi
FIRST_STATUS=0
wait "$FIRST_PID" || FIRST_STATUS=$?
if (( FIRST_STATUS != 0 )); then
  kill "$SECOND_PID" 2>/dev/null || true
  wait "$SECOND_PID" 2>/dev/null || true
  echo "Best-v7 fail-fast: one causal arm failed with status $FIRST_STATUS" >&2
  exit 6
fi
SECOND_STATUS=0
wait "$SECOND_PID" || SECOND_STATUS=$?
if (( SECOND_STATUS != 0 )); then
  echo "Best-v7 second causal arm failed with status $SECOND_STATUS" >&2
  exit 7
fi

echo "=== best-v7 full stage 2/3: matched clean continuation ==="
run_arm "$GPU_ZERO" clean_control

echo "=== best-v7 full stage 3/3: complete registered metric gate ==="
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
assert report["direct_contract"] == "best_action_v7_corrective_restored"
assert report["training_seed"] == 20260911
assert report["minimum_delta_recall1_pp"] == 4.0
assert report["bootstrap_resamples"] == 10000
assert report["formula_ci_familywise_hypotheses"] == 4
assert isinstance(report["single_seed_gate_pass"], bool)
assert report["promote"] is False
assert report["promotion_status"] in {"pending_multiseed", "single_seed_gate_failed"}
assert report["schedule_and_signal_checks"][
    "post_shuffle_actual_bank_calibration"
] is True
assert report["schedule_and_signal_checks"][
    "strict_top1_multi_action_admission"
] is True
assert report["schedule_and_signal_checks"][
    "registered_formal_direct_configuration_verified"
] is True
assert report["schedule_and_signal_checks"][
    "best_action_v6_cap_safe_871_to_877_schedule_verified"
] is True
assert isinstance(
    report["schedule_and_signal_checks"][
        "optimizer_update_restoration_contract_valid"
    ],
    bool,
)
print(json.dumps({
    "status": "best_action_v7_corrective_restored_complete_metric_gate_evaluated",
    "single_seed_gate_pass": report["single_seed_gate_pass"],
    "promotion_status": report["promotion_status"],
    "promotion_checks": report["promotion_checks"],
    "claim_limit": report["claim_limit"],
}, indent=2))
PY

mv "$RUN_ROOT" "$FINAL_ROOT"
echo "=== best-v7 full result preserved at: $FINAL_ROOT ==="
