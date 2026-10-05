#!/bin/bash
#SBATCH --job-name=noise_v3_f0
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=96:00:00
#SBATCH --output=data/validation/noise_v3_f0_%j.out
#SBATCH --error=data/validation/noise_v3_f0_%j.err

set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

[[ -n "${BASE_RUN:-}" ]] || {
  echo "BASE_RUN must name a directory containing completed formal N+P+A4 routes" >&2
  exit 2
}
case "$BASE_RUN" in
  data/validation/noise_corrected_npa4_formal_fold_0_run_*|\
  data/validation/.noise_corrected_npa4_formal_fold_0_run_*.staging) ;;
  *) echo "Unexpected BASE_RUN path: $BASE_RUN" >&2; exit 2 ;;
esac

ALLOCATED_GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -r -a ALLOCATED_GPUS <<< "$ALLOCATED_GPU_LIST"
if (( ${#ALLOCATED_GPUS[@]} < 2 )); then
  echo "Expected two allocated GPUs, got CUDA_VISIBLE_DEVICES=$ALLOCATED_GPU_LIST" >&2
  exit 3
fi
ROUTED_GPU="${ALLOCATED_GPUS[0]}"
CONTROL_GPU="${ALLOCATED_GPUS[1]}"

FOLD=0
GRAPH="data/validation/noise_corrected_candidate_graph_v1_20260906"
EVAL_METADATA="data/validation/chemaware_corrected_candidate_manifest_v1"
DATA="data/models/MassSpecGym_MurckoHist_split.hdf5"
OFFICIAL="data/e1/official_embedding_slim.pt"
ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
INITIAL="data/validation/g8r_noise_final_e8_direct_transfer/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_e8_baseline_symmetric_shared/seed_20260830/fold_${FOLD}/final_shared_encoder.pt"
N_ROUTE="$BASE_RUN/routes/n_route"
P_ROUTE="$BASE_RUN/routes/p_route"
A4_ROUTE="$BASE_RUN/routes/a4_route"

FINAL_ROOT="data/validation/noise_corrected_direct_v3_formal_fold_${FOLD}_run_${SLURM_JOB_ID}"
RUN_ROOT="data/validation/.noise_corrected_direct_v3_formal_fold_${FOLD}_run_${SLURM_JOB_ID}.staging"
LEDGER="$RUN_ROOT/ledger"
PLAN_AUDIT="$RUN_ROOT/plan_audit"
ARMS="$RUN_ROOT/arms"
SUMMARIES="$RUN_ROOT/summaries"

for REQUIRED in \
  "$GRAPH/candidate_graph.npz" "$GRAPH/report.json" \
  "$EVAL_METADATA/manifest.npz" "$DATA" "$OFFICIAL" "$ARCHITECTURE" \
  "$INITIAL" "$(dirname "$INITIAL")/decision.json" \
  "$N_ROUTE/routed_actions.csv.gz" "$N_ROUTE/action_spectra.npz" "$N_ROUTE/report.json" \
  "$P_ROUTE/routed_actions.csv.gz" "$P_ROUTE/action_spectra.npz" "$P_ROUTE/report.json" \
  "$A4_ROUTE/routed_actions.csv.gz" "$A4_ROUTE/action_spectra.npz" "$A4_ROUTE/report.json"; do
  [[ -f "$REQUIRED" ]] || { echo "Missing frozen v3 input: $REQUIRED" >&2; exit 4; }
done
[[ ! -e "$RUN_ROOT" && ! -e "$FINAL_ROOT" ]] || {
  echo "Refusing to overwrite unique atomic run: $RUN_ROOT or $FINAL_ROOT" >&2
  exit 5
}
mkdir -p "$ARMS" "$SUMMARIES"

python -m py_compile \
  tasks/noise_corrected_action_routing.py \
  tasks/noise_corrected_action_routing_v3.py \
  tasks/noise_corrected_action_panel.py \
  tasks/noise_corrected_direct_v3_core.py \
  tasks/noise_corrected_action_expansion_v4.py \
  tasks/noise_corrected_transfer_objective_v4.py \
  tasks/noise_corrected_update_arbitration_v4.py \
  tasks/noise_corrected_directional_update_v4.py \
  tasks/noise_corrected_shuffled_control_v3.py \
  tasks/build_noise_corrected_routed_action_ledger.py \
  tasks/audit_noise_corrected_direct_v3_plan.py \
  tasks/train_noise_corrected_routed_direct.py \
  tasks/summarize_noise_corrected_direct_v3_arms.py \
  tasks/test_noise_corrected_direct_v3_bundle.py \
  tasks/test_noise_corrected_direct_v3_trainer.py \
  tasks/test_noise_corrected_action_panel.py \
  tasks/test_noise_corrected_routed_ledger_streaming.py
python -u tasks/test_noise_corrected_direct_v3_bundle.py
python -u tasks/test_noise_corrected_action_routing.py
python -u tasks/test_noise_corrected_action_routing_v3.py
python -u tasks/test_noise_corrected_action_panel.py
python -u tasks/test_noise_corrected_routed_ledger_streaming.py
python -u tasks/test_noise_final_direct_boundary_v2_core.py
python -u tasks/test_noise_corrected_direct_v3_core.py
python -u tasks/test_noise_corrected_direct_v3_trainer.py
python -u tasks/test_noise_corrected_transfer_objective_v4.py
python -u tasks/test_noise_corrected_update_arbitration_v4.py
python -u tasks/test_noise_corrected_directional_update_v4.py
python -u tasks/test_noise_corrected_fullgraph_evaluation.py
python -u tasks/test_summarize_noise_corrected_direct_v3_arms.py

echo "=== stage 1/4: rebuild the full three-semantics ledger from frozen routes ==="
python -u tasks/build_noise_corrected_routed_action_ledger.py \
  --graph-dir "$GRAPH" --route-dirs "$N_ROUTE" "$P_ROUTE" "$A4_ROUTE" \
  --maximum-corrective-per-query 16 --maximum-risk-per-query 8 \
  --maximum-robust-per-query 8 --output-dir "$LEDGER"

python -u tasks/audit_noise_corrected_direct_v3_plan.py \
  --graph-dir "$GRAPH" --ledger-dir "$LEDGER" --outer-fold "$FOLD" \
  --formula-fold-seed 20260825 --batch-queries 4 --protective-batch-queries 8 \
  --maximum-protective-microbatches-per-step 4 \
  --maximum-auxiliary-microbatches-per-step 4 \
  --maximum-corrective-recycle-factor 4 \
  --maximum-corrective-per-query 16 --maximum-harmful-per-query 8 \
  --maximum-robust-per-query 8 --margin-transfer-fraction 0.50 \
  --margin-transfer-cap 0.10 --output-dir "$PLAN_AUDIT"

python - "$LEDGER/report.json" "$PLAN_AUDIT/report.json" "$GRAPH/candidate_graph.npz" <<'PY'
import hashlib, json, math, numpy as np, pandas as pd, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
audit = json.load(open(sys.argv[2], encoding="utf-8"))
assert report["formal_training_authorized"] is True
assert report["contracts"]["three_semantics_are_separate"] is True
assert report["contracts"]["mechanism_then_source_family_round_robin_selection"] is True
assert report["contracts"]["exact_action_control_candidate_switch_rows_recorded"] is True
assert report["contracts"]["control_semantics_explicit_and_source_validated"] is True
assert report["contracts"]["all_route_formal_configurations_verified"] is True
assert report["contracts"]["all_route_formula_fold_seeds_match"] is True
assert report["contracts"][
    "source_then_family_exposure_balanced_before_recipe_score"
] is True
assert report["selected_corrective_rows"] > 0
assert report["selected_risk_rows"] > 0
assert report["selected_robust_rows"] > 0
actions = pd.read_csv(sys.argv[1].replace("report.json", "training_actions.csv.gz"))
assert set(actions.supervision_kind) == {"corrective", "harmful", "robust"}
assert set(actions.source) >= {
    "N_mature", "P_guided_original", "E10B", "E11", "E12B", "A4_exact"
}
assert {
    "action_positive_row", "action_hard_negative_molecule_index",
    "action_hard_negative_row", "control_positive_row",
    "control_hard_negative_molecule_index", "control_hard_negative_row",
} <= set(actions.columns)
assert "control_semantic" in actions
assert set(actions.loc[actions.source.isin([
    "P_guided_original", "E10B", "E11", "E12B"
]), "control_semantic"]) == {
    "wrong_identity_direction"
}
assert set(actions.loc[actions.source.isin(["N_mature", "A4_exact"]), "control_semantic"]) <= {
    "matched_neutral", "clean_fallback"
}
assert audit["formal"] is True
assert audit["contracts"]["materialized_training_ledger_matches_selector_replay"] is True
assert audit["contracts"]["control_semantics_explicit_and_source_validated"] is True
assert audit["contracts"][
    "dense_auxiliary_panels_cannot_multiply_calibrated_step_mass"
] is True
assert audit["contracts"][
    "source_then_family_exposure_balanced_before_recipe_score"
] is True
assert audit["schedule"]["formal_gate"] == "PASS"
assert audit["schedule"][
    "dense_auxiliary_epoch_mass_capped_at_one_batch_per_step"
] is True
assert audit["schedule"]["auxiliary_microbatches_per_balanced_step"] <= 4
assert audit["direct_action_training_headroom"][
    "fraction_of_initial_E8_errors_with_selected_corrective_action"
] >= 0.80
assert audit["direct_action_training_headroom"][
    "recall1_headroom_uses_actual_action_rank1_not_partial_margin"
] is True
assert audit["contracts"][
    "recall1_headroom_counts_only_selected_actual_top1_corrections"
] is True
assert audit["direct_action_training_headroom"][
    "maximum_training_geometry_recall1_headroom_pp"
] >= 4.0
assert all(
    panel["no_N_P_A4_starvation_under_cap"]
    for panel in audit["hierarchical_selection_coverage"].values()
)
with np.load(sys.argv[3], allow_pickle=False) as body:
    formulas = np.asarray(body["query_formula"], dtype=str)
def fold(value):
    payload = f"20260825|{value}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % 5
protective_queries = sum(fold(value) != 0 for value in formulas)
corrective_queries = actions.loc[
    actions.supervision_kind.eq("corrective"), "query_index"
].nunique()
robust_queries = actions.loc[
    actions.supervision_kind.eq("robust"), "query_index"
].nunique()
harmful_queries = actions.loc[
    actions.supervision_kind.eq("harmful"), "query_index"
].nunique()
protective_batches = math.ceil(protective_queries / 8)
corrective_batches = math.ceil(corrective_queries / 4)
robust_batches = math.ceil(robust_queries / 4)
harmful_batches = math.ceil(harmful_queries / 4)
auxiliary_batches = robust_batches + harmful_batches
steps = max(
    corrective_batches,
    math.ceil(protective_batches / 4),
    math.ceil(auxiliary_batches / 4),
)
ratio = protective_batches / steps
auxiliary_ratio = auxiliary_batches / steps
recycle = steps / corrective_batches
print(json.dumps({
    "protective_queries": protective_queries,
    "corrective_queries": corrective_queries,
    "protective_microbatches_per_action_step": ratio,
    "auxiliary_microbatches_per_action_step": auxiliary_ratio,
    "corrective_batch_recycle_factor": recycle,
}, indent=2))
assert ratio <= 4 and auxiliary_ratio <= 4 and recycle <= 4, (
    "full action ledger violates v3 bounded scheduling"
)
PY

echo "=== stage 1b/4: formal source/family/exact-recipe shuffle preflight ==="
python - "$LEDGER/training_actions.csv.gz" "$RUN_ROOT/shuffled_control_preflight.json" <<'PY'
import json, pathlib, pandas as pd, sys
from noise_corrected_shuffled_control_v3 import source_family_shuffle_strata_report

actions = pd.read_csv(sys.argv[1], low_memory=False)
report = source_family_shuffle_strata_report(actions)
report["status"] = "noise_corrected_shuffled_control_preflight_complete"
report["formal_gate"] = "PASS" if (
    report["matching_columns"]
    == ["supervision_kind", "source", "family", "recipe_id"]
    and report["cross_query_fraction"] >= 0.95
) else "FAIL"
pathlib.Path(sys.argv[2]).write_text(
    json.dumps(report, indent=2, sort_keys=True), encoding="utf-8",
)
print(json.dumps(report, indent=2, sort_keys=True))
assert report["formal_gate"] == "PASS", (
    "formal shuffled control does not retain 95% cross-query coverage"
)
PY

COMMON=(
  --graph-dir "$GRAPH" --source-manifest-dir "$EVAL_METADATA"
  --routed-ledger-dir "$LEDGER" --data "$DATA"
  --official-checkpoint "$OFFICIAL" --architecture-checkpoint "$ARCHITECTURE"
  --initial-student-checkpoint "$INITIAL" --outer-fold "$FOLD"
  --formula-fold-seed 20260825 --corrective-objective-mode v3_direct
  --epochs 4 --batch-queries 4 --protective-batch-queries 8 --positive-references 4
  --action-conditioned-positive-references 32
  --action-conditioned-negative-molecules 32
  --negative-molecules 8 --references-per-negative 2
  --reference-refresh-every-epochs 1 --n-highest-peaks 100
  --unfreeze-blocks 1 --backbone-lr 2e-6 --head-lr 1e-5 --weight-decay 1e-4
  --rank-margin 0.05 --temperature 0.10 --topk-negatives 8
  --action-safety-slack 0.005 --v3-harmful-damage-slack 0.005
  --lambda-margin-transfer 1.0 --lambda-action-rank 0.25 --lambda-action-safety 0.25
  --lambda-v3-corrective-consistency 0.25
  --lambda-v3-robust-rank 0.25 --lambda-v3-robust-floor 0.25
  --lambda-v3-harmful-rank 1.0 --lambda-v3-harmful-floor 1.0
  --margin-transfer-fraction 0.50 --margin-transfer-cap 0.10
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
  --minimum-optimizer-action-attributable-fraction-p10 0.10
  --minimum-corrective-direction-retention-p10 0.10
  --optimizer-attribution-steps-per-epoch 16
  --progress-every-steps 50
  --maximum-clip-event-fraction 0.10 --grad-clip 1.0
  --eval-batch-size 256 --bootstrap-resamples 10000 --device cuda --no-amp
)

echo "=== stage 2/4: one bounded integration smoke on GPU0 ==="
export CUDA_VISIBLE_DEVICES="$ROUTED_GPU"
python -u tasks/train_noise_corrected_routed_direct.py \
  "${COMMON[@]}" --development --epochs 1 --calibration-batches 4 \
  --maximum-corrective-queries 32 --maximum-risk-queries 32 \
  --maximum-robust-queries 16 --maximum-clean-queries 128 \
  --outer-held-eval-queries 32 --seed 20260907 --arm routed_direct \
  --output-dir "$RUN_ROOT/integration_smoke"

echo "=== stage 3/4: nine formal arms in two persistent GPU worker queues ==="
SEEDS=(20260906 20260907 20260908)
WORKER_ZERO_TASKS=(
  "20260906:routed_direct" "20260907:shuffled_action_control"
  "20260908:routed_direct" "20260906:clean_control"
)
WORKER_ONE_TASKS=(
  "20260906:shuffled_action_control" "20260907:routed_direct"
  "20260908:shuffled_action_control" "20260907:clean_control"
  "20260908:clean_control"
)
run_arm_worker() {
  local WORKER_GPU="$1"
  shift
  export CUDA_VISIBLE_DEVICES="$WORKER_GPU" OMP_NUM_THREADS=8
  for ARM_TASK in "$@"; do
    IFS=':' read -r ARM_SEED ARM_NAME <<< "$ARM_TASK"
    python -u tasks/train_noise_corrected_routed_direct.py \
      "${COMMON[@]}" --seed "$ARM_SEED" --arm "$ARM_NAME" \
      --output-dir "$ARMS/seed_${ARM_SEED}/${ARM_NAME}"
  done
}
run_arm_worker "$ROUTED_GPU" "${WORKER_ZERO_TASKS[@]}" &
PID_ZERO=$!
run_arm_worker "$CONTROL_GPU" "${WORKER_ONE_TASKS[@]}" &
PID_ONE=$!
STATUS_ZERO=0
STATUS_ONE=0
wait "$PID_ZERO" || STATUS_ZERO=$?
wait "$PID_ONE" || STATUS_ONE=$?
if (( STATUS_ZERO != 0 || STATUS_ONE != 0 )); then
  echo "Formal arm worker failed: $STATUS_ZERO/$STATUS_ONE" >&2
  exit 6
fi

for SEED in "${SEEDS[@]}"; do
  SEED_ROOT="$ARMS/seed_${SEED}"
    python -u tasks/summarize_noise_corrected_direct_v3_arms.py \
    --routed-dir "$SEED_ROOT/routed_direct" \
    --shuffled-dir "$SEED_ROOT/shuffled_action_control" \
    --clean-dir "$SEED_ROOT/clean_control" \
    --minimum-delta-recall1-pp 4.0 --bootstrap-resamples 10000 \
    --formula-ci-familywise-hypotheses 4 \
    --seed "$SEED" --output-dir "$SUMMARIES/seed_${SEED}"
done

echo "=== stage 4/4: immutable all-seed promotion gate ==="
python - "$SUMMARIES" "$RUN_ROOT/promotion_gate.json" <<'PY'
import json, pathlib, sys
summaries = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
seeds = (20260906, 20260907, 20260908)
records = []
shared_inputs = set()
routed_decisions = set()
routed_model_states = set()
for seed in seeds:
    report = json.loads((summaries / f"seed_{seed}" / "report.json").read_text())
    assert report["training_seed"] == seed
    assert report["formula_ci_familywise_hypotheses"] == 4
    shared_inputs.add((
        report["provenance"]["shared_training_actions_sha256"],
        report["provenance"]["shared_candidate_graph_sha256"],
        report["provenance"]["shared_source_manifest_sha256"],
        report["provenance"]["shared_initial_student_checkpoint_sha256"],
        report["provenance"]["shared_official_checkpoint_sha256"],
    ))
    routed_decisions.add(report["provenance"]["routed_decision_sha256"])
    routed_model_states.add(
        report["provenance"]["routed_final_model_state_sha256"]
    )
    records.append({
        "seed": seed,
        "promote": report["promote"],
        "promotion_checks": report["promotion_checks"],
    })
assert len(shared_inputs) == 1, "three seeds do not share one frozen ledger/graph"
assert len(routed_decisions) == len(seeds), "duplicate routed seed result detected"
assert len(routed_model_states) == len(seeds), "duplicate routed seed model state detected"
decision = {
    "status": "noise_corrected_direct_v3_multiseed_gate_complete",
    "seeds": records,
    "all_three_seeds_passed": all(record["promote"] for record in records),
    "promotion_authorized": all(record["promote"] for record in records),
    "registered_seed_identity_verified": True,
    "shared_frozen_inputs_verified": True,
    "distinct_routed_seed_results_verified": True,
    "distinct_routed_seed_model_states_verified": True,
    "minimum_recall1_gain_pp_each_seed": 4.0,
    "all_required_metric_families_must_improve": True,
    "formula_cluster_ci_must_be_strict_positive": True,
    "routed_must_beat_both_shuffled_and_clean_controls": True,
    "claim_limit": "Fold-0 three-seed gate; no multifold or P3 claim.",
}
output.write_text(json.dumps(decision, indent=2), encoding="utf-8")
print(json.dumps(decision, indent=2))
PY

mv "$RUN_ROOT" "$FINAL_ROOT"
echo "=== direct-v3 formal result preserved at: $FINAL_ROOT ==="
