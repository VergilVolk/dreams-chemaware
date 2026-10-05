#!/bin/bash
#SBATCH --job-name=noise_e4_live_v3
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=120:00:00
#SBATCH --output=data/validation/noise_e4_live_v3_%j.out
#SBATCH --error=data/validation/noise_e4_live_v3_%j.err

set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ALLOCATED_GPU_LIST="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -r -a ALLOCATED_GPUS <<< "$ALLOCATED_GPU_LIST"
if (( ${#ALLOCATED_GPUS[@]} != 1 )); then
  echo "Expected exactly one allocated GPU, got CUDA_VISIBLE_DEVICES=$ALLOCATED_GPU_LIST" >&2
  exit 3
fi

FOLD=0
SEED=20260830
GPU_ZERO="${ALLOCATED_GPUS[0]}"
E4_BASE="data/validation/noise_corrected_npa4_formal_fold_0_run_2332161_v3routes/routes/n_bank"
BEST_LEDGER="data/validation/noise_corrected_best_v5_canary_fold_0_run_2332784/ledger"
GRAPH_DIR="data/validation/noise_corrected_candidate_graph_v1_20260906"
GRAPH="$GRAPH_DIR/candidate_graph.npz"
EMBEDDING_CACHE="$GRAPH_DIR/official_embeddings.npz"
EVAL_METADATA="data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz"
DATA="data/models/MassSpecGym_MurckoHist_split.hdf5"
OFFICIAL="data/e1/official_embedding_slim.pt"
ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
INITIAL="data/validation/g8r_noise_final_e8_direct_transfer/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_e8_baseline_symmetric_shared/seed_${SEED}/fold_${FOLD}/final_shared_encoder.pt"

RECOVER_RUN_ID="${RECOVER_RUN_ID:-}"
if [[ -n "$RECOVER_RUN_ID" && ! "$RECOVER_RUN_ID" =~ ^[0-9]+$ ]]; then
  echo "RECOVER_RUN_ID must be the numeric Slurm job id" >&2
  exit 5
fi
RUN_ID="${RECOVER_RUN_ID:-${SLURM_JOB_ID}}"
FINAL_ROOT="data/validation/noise_e4_live_shared_hybrid_v3_fold_${FOLD}_seed_${SEED}_run_${RUN_ID}"
STAGING_ROOT="data/validation/.noise_e4_live_shared_hybrid_v3_fold_${FOLD}_seed_${SEED}_run_${RUN_ID}.staging"
RUN_ROOT="$STAGING_ROOT"
ARMS="$RUN_ROOT/arms"
RECOVERY_LAYOUT="fresh"

for REQUIRED in \
  "$GRAPH" "$EMBEDDING_CACHE" "$EVAL_METADATA" "$DATA" "$OFFICIAL" \
  "$ARCHITECTURE" "$INITIAL" "$(dirname "$INITIAL")/decision.json" \
  "$E4_BASE/report.json" "$E4_BASE/training_actions.csv.gz" \
  "$BEST_LEDGER/report.json" "$BEST_LEDGER/training_actions.csv.gz" \
  "$BEST_LEDGER/action_spectra.npz" \
  tasks/noise_e4_live_shared_hybrid_v3.py \
  tasks/noise_corrected_update_arbitration_v10.py \
  tasks/test_noise_e4_live_shared_hybrid_v3.py \
  tasks/test_noise_e4_live_shared_hybrid_v3_sbatch.py \
  tasks/test_noise_corrected_action_routing_v3.py \
  tasks/train_noise_final_e4a_direct_augmentation.py \
  tasks/validate_noise_final_e4a_direct_augmentation.py \
  tasks/summarize_noise_e4_signal_preserving_hybrid_v2.py; do
  [[ -f "$REQUIRED" ]] || { echo "Missing V3 input: $REQUIRED" >&2; exit 4; }
done
if [[ -n "$RECOVER_RUN_ID" ]]; then
  if [[ -e "$STAGING_ROOT" && -e "$FINAL_ROOT" ]]; then
    echo "Ambiguous recovery: both staging and final roots exist" >&2
    exit 5
  elif [[ -d "$STAGING_ROOT" ]]; then
    RUN_ROOT="$STAGING_ROOT"
    RECOVERY_LAYOUT="staging"
  elif [[ -d "$FINAL_ROOT" ]]; then
    # A post-training validator can fail after an earlier entrypoint has
    # already promoted the run directory.  Resume that directory in place;
    # never rename, copy, retrain, or overwrite the completed targeted arm.
    RUN_ROOT="$FINAL_ROOT"
    RECOVERY_LAYOUT="final_in_place"
  else
    echo "Recovery found neither staging nor final root for run ${RUN_ID}" >&2
    exit 5
  fi
  ARMS="$RUN_ROOT/arms"
else
  [[ ! -e "$STAGING_ROOT" && ! -e "$FINAL_ROOT" ]] || {
    echo "Refusing to overwrite V3 output" >&2
    exit 5
  }
fi
if [[ -n "$RECOVER_RUN_ID" ]]; then
  [[ -d "$ARMS" ]] || {
    echo "Recovery root has no arms directory: $ARMS" >&2
    exit 5
  }
else
  mkdir -p "$ARMS"
fi

# All executable Python checks run inside the allocated Slurm job, never on a
# login node. These tests inspect the production loss and optimizer boundary.
python -m py_compile \
  tasks/noise_e4_live_shared_hybrid_v3.py \
  tasks/noise_corrected_update_arbitration_v10.py \
  tasks/test_noise_e4_live_shared_hybrid_v3.py \
  tasks/test_noise_e4_live_shared_hybrid_v3_sbatch.py \
  tasks/test_noise_corrected_action_routing_v3.py \
  tasks/train_noise_final_e4a_direct_augmentation.py \
  tasks/validate_noise_final_e4a_direct_augmentation.py \
  tasks/summarize_noise_e4_signal_preserving_hybrid_v2.py
python -u tasks/test_noise_e4_live_shared_hybrid_v3.py
python -u tasks/test_noise_e4_live_shared_hybrid_v3_sbatch.py
python -u tasks/test_noise_corrected_action_routing_v3.py
python -u tasks/test_noise_final_e4a_direct_augmentation.py
python -u tasks/test_noise_corrected_fullgraph_evaluation.py

python - "$E4_BASE/report.json" "$E4_BASE/training_actions.csv.gz" \
  "$BEST_LEDGER/training_actions.csv.gz" <<'PY'
import json
import sys
import pandas as pd
sys.path.insert(0, "tasks")
from noise_final_e4_pmt_core import select_materialized_direct_action_panel
from noise_e4_live_shared_hybrid_v3 import build_query_equal_action_plan_v3

base_report = json.load(open(sys.argv[1], encoding="utf-8"))
base = pd.read_csv(sys.argv[2], low_memory=False)
best = pd.read_csv(sys.argv[3], low_memory=False)
selected, _ = select_materialized_direct_action_panel(
    best, mode="multi_action_balanced", margin_floor=5e-6,
)
assert base_report["formal_training_authorized"] is True
assert base_report["contracts"]["action_outcomes_computed"] is False
assert base_report["contracts"]["teacher_embedding_or_margin_used"] is False
assert len(base) == 190324
assert base.query_ik14.nunique() == 7624
assert not base.formula_fold.astype(int).eq(0).any()
assert len(selected) == 32114
assert selected.query_index.nunique() == 3482
assert set(selected.source.astype(str)) == {
    "N_mature", "P_guided_original", "E10B", "E11", "E12B",
    "A4_exact", "V4_gradient_path",
}
plan = build_query_equal_action_plan_v3(
    selected.action_id.astype(str).tolist(),
    selected.query_index.astype(int).tolist(),
    (selected.source.astype(str) + "|" + selected.family.astype(str)).tolist(),
    epochs=4,
    views_per_query_per_epoch=4,
    actions_per_step=4,
    optimizer_steps_per_epoch=7624,
    seed=20260830 + 271828,
)
manifest = plan.audit_manifest()
assert manifest["query_dose_equal_within_every_epoch"] is True
assert manifest["same_query_actions_never_share_an_optimizer_step"] is True
assert manifest["active_steps_per_epoch"] == 3484
assert manifest["maximum_actions_per_query"] == 16
assert manifest["unique_actions_exposed"] == 32114
assert manifest["all_unique_actions_exposed"] is True
print(json.dumps({
    "status": "noise_e4_live_shared_hybrid_v3_preflight_pass",
    "base_rows": len(base),
    "action_rows": len(selected),
    "action_queries": int(selected.query_index.nunique()),
    "schedule": manifest,
    "loss": "complete_live_shared_E4",
    "action_rank_gate": False,
    "optimizer_action_fraction": "exact_0.25_on_every_active_group",
    "minimum_e4_projection_retention": 0.90,
    "maximum_update_norm_ratio": 1.50,
    "teacher_embedding_or_margin_target": False,
}, indent=2))
PY

COMMON=(
  --graph "$GRAPH" --source-manifest "$EVAL_METADATA"
  --embedding-cache "$EMBEDDING_CACHE" --data "$DATA"
  --official-checkpoint "$OFFICIAL" --architecture-checkpoint "$ARCHITECTURE"
  --initial-student-checkpoint "$INITIAL"
  --action-selection materialized_routed --materialized-action-dir "$BEST_LEDGER"
  --e4-base-action-dir "$E4_BASE"
  --materialized-injection-mode e4_live_shared_v3
  --optimizer-boundary-mode separated_e4_action_v3
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
  --pmt-arm none --no-candidate-boundary-loss --no-refresh-hard-negatives
  --grad-clip 1.0 --eval-batch-size 256 --bootstrap-resamples 10000
  --device cuda --no-amp
)

result_dir_under() {
  local ARM="$1"
  local OUTPUT_ROOT="$2"
  local SUFFIX="e4_live_v3_${ARM}_run${RUN_ID}"
  local TAG="curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_as_materialized_routed_marm_${ARM}_minj_e4_live_shared_v3_ob_separated_e4_action_v3_warm_${SUFFIX}"
  echo "$OUTPUT_ROOT/$TAG/seed_${SEED}/fold_${FOLD}"
}

result_dir() {
  local ARM="$1"
  result_dir_under "$ARM" "$ARMS/$ARM"
}

arm_dir_is_complete() {
  local DIRECTORY="$1"
  [[ -f "$DIRECTORY/decision.json" \
     && -f "$DIRECTORY/final_shared_encoder.pt" \
     && -f "$DIRECTORY/held_per_query.csv.gz" ]]
}

arm_is_complete() {
  local ARM="$1"
  arm_dir_is_complete "$(result_dir "$ARM")"
}

run_arm_into() {
  local ARM="$1"
  local OUTPUT_ROOT="$2"
  local SUFFIX="e4_live_v3_${ARM}_run${RUN_ID}"
  export CUDA_VISIBLE_DEVICES="$GPU_ZERO" OMP_NUM_THREADS=8
  python -u tasks/train_noise_final_e4a_direct_augmentation.py \
    "${COMMON[@]}" --materialized-action-arm "$ARM" \
    --output-root "$OUTPUT_ROOT" --run-suffix "$SUFFIX"
  python -u tasks/validate_noise_final_e4a_direct_augmentation.py \
    --output-dir "$(result_dir_under "$ARM" "$OUTPUT_ROOT")"
  arm_dir_is_complete "$(result_dir_under "$ARM" "$OUTPUT_ROOT")" || {
    echo "Arm passed training but its three required artifacts are incomplete" >&2
    exit 7
  }
}

run_arm() {
  local ARM="$1"
  run_arm_into "$ARM" "$ARMS/$ARM"
}

echo "=== Full live-shared E4 loss + exact-0.25 separated E4/action optimizer states ==="
if [[ -n "$RECOVER_RUN_ID" ]]; then
  arm_is_complete targeted || {
    echo "Recovery target is incomplete: $(result_dir targeted)" >&2
    exit 6
  }
  echo "=== Reusing completed targeted arm from run ${RUN_ID} (${RECOVERY_LAYOUT}); validating only ==="
  python -u tasks/validate_noise_final_e4a_direct_augmentation.py \
    --output-dir "$(result_dir targeted)"
else
  run_arm targeted
fi

if arm_is_complete shuffled; then
  if [[ -z "$RECOVER_RUN_ID" ]]; then
    echo "Fresh run unexpectedly found a completed shuffled arm" >&2
    exit 7
  fi
  echo "=== Reusing completed shuffled arm from run ${RUN_ID}; validating only ==="
  python -u tasks/validate_noise_final_e4a_direct_augmentation.py \
    --output-dir "$(result_dir shuffled)"
elif [[ -e "$(result_dir shuffled)" ]]; then
  echo "Refusing incomplete shuffled output: $(result_dir shuffled)" >&2
  exit 7
else
  if [[ "$RECOVERY_LAYOUT" == "final_in_place" ]]; then
    RECOVERY_SHUFFLED_ROOT="${FINAL_ROOT}.shuffled_recovery_job_${SLURM_JOB_ID}"
    [[ ! -e "$RECOVERY_SHUFFLED_ROOT" ]] || {
      echo "Refusing to reuse a recovery scratch root: $RECOVERY_SHUFFLED_ROOT" >&2
      exit 7
    }
    run_arm_into shuffled "$RECOVERY_SHUFFLED_ROOT"
    RECOVERY_SHUFFLED_RESULT="$(result_dir_under shuffled "$RECOVERY_SHUFFLED_ROOT")"
    CANONICAL_SHUFFLED_RESULT="$(result_dir shuffled)"
    [[ ! -e "$CANONICAL_SHUFFLED_RESULT" ]] || {
      echo "Canonical shuffled result appeared during recovery" >&2
      exit 7
    }
    mkdir -p "$(dirname "$CANONICAL_SHUFFLED_RESULT")"
    mv -Tn "$RECOVERY_SHUFFLED_RESULT" "$CANONICAL_SHUFFLED_RESULT"
    [[ ! -e "$RECOVERY_SHUFFLED_RESULT" ]] \
      && arm_dir_is_complete "$CANONICAL_SHUFFLED_RESULT" || {
        echo "Atomic shuffled promotion failed" >&2
        exit 7
      }
  else
    run_arm shuffled
  fi
fi

summary_files_complete() {
  local DIRECTORY="$1"
  [[ -f "$DIRECTORY/report.json" \
     && -f "$DIRECTORY/targeted_vs_matched_shuffled_per_query.csv.gz" ]]
}

summary_schema_is_current() {
  local DIRECTORY="$1"
  python - "$DIRECTORY/report.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
gates = report.get("gates", {})
required_gates = {
    "targeted_beats_shuffled_formula_ci",
    "risk_net_lambda2_vs_initial_e8_positive",
    "risk_net_lambda2_vs_shuffled_positive",
    "near_risk_net_lambda2_vs_initial_e8_positive",
    "near_risk_net_lambda2_vs_shuffled_positive",
    "all_registered_metrics_nonnegative_vs_initial_e8",
    "all_registered_metrics_nonnegative_vs_shuffled",
    "all_registered_metrics_strict_or_boundary_vs_initial_e8",
    "all_registered_metrics_strict_or_boundary_vs_shuffled",
    "strict_four_pp_recall1_gain_vs_official",
}
attribution = report.get("later_stream_attribution_limit", {})
if (
    report.get("status") != "noise_e4_live_shared_hybrid_v3_summary_complete"
    or not isinstance(report.get("strict_full_panel_four_pp_achieved"), bool)
    or not required_gates.issubset(gates)
    or attribution.get("exact_fraction_is_not_targeted_minus_shuffled_fraction")
    is not True
):
    raise SystemExit(1)
PY
}

print_summary_report() {
  local DIRECTORY="$1"
  python - "$DIRECTORY/report.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps(report, indent=2), flush=True)
PY
}

AUTHORITATIVE_SUMMARY="$RUN_ROOT/summary"
if summary_files_complete "$AUTHORITATIVE_SUMMARY" \
   && summary_schema_is_current "$AUTHORITATIVE_SUMMARY"; then
  echo "=== Reusing current strict matched summary from run ${RUN_ID} ==="
else
  if [[ -e "$AUTHORITATIVE_SUMMARY" ]]; then
    echo "=== Preserving legacy/incomplete summary; rebuilding strict summary ==="
    AUTHORITATIVE_SUMMARY="$RUN_ROOT/summary_strict_v3"
  fi
  if summary_files_complete "$AUTHORITATIVE_SUMMARY" \
     && summary_schema_is_current "$AUTHORITATIVE_SUMMARY"; then
    echo "=== Reusing current strict matched summary from run ${RUN_ID} ==="
  elif [[ -e "$AUTHORITATIVE_SUMMARY" ]]; then
    echo "Refusing incomplete or stale strict summary: $AUTHORITATIVE_SUMMARY" >&2
    exit 8
  else
  python -u tasks/summarize_noise_e4_signal_preserving_hybrid_v2.py \
      --hybrid-version v3 \
      --targeted-dir "$(result_dir targeted)" \
      --shuffled-dir "$(result_dir shuffled)" \
      --best-action-dir "$BEST_LEDGER" --e4-base-action-dir "$E4_BASE" \
      --initial-e8-checkpoint "$INITIAL" --output-dir "$AUTHORITATIVE_SUMMARY" \
      --bootstrap-resamples 10000 --seed "$SEED"
  fi
fi
echo "=== Authoritative strict matched summary: $AUTHORITATIVE_SUMMARY ==="
print_summary_report "$AUTHORITATIVE_SUMMARY"

if [[ "$RUN_ROOT" != "$FINAL_ROOT" ]]; then
  [[ ! -e "$FINAL_ROOT" ]] || {
    echo "Final root appeared before atomic promotion: $FINAL_ROOT" >&2
    exit 9
  }
  mv -Tn "$RUN_ROOT" "$FINAL_ROOT"
  [[ ! -e "$RUN_ROOT" && -d "$FINAL_ROOT" ]] || {
    echo "Atomic run-root promotion failed" >&2
    exit 9
  }
fi
echo "=== Live-shared Hybrid V3 result: $FINAL_ROOT ==="
