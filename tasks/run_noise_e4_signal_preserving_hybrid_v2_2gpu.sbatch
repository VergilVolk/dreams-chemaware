#!/bin/bash
#SBATCH --job-name=noise_e4_semantic_v2cb
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=120:00:00
#SBATCH --output=data/validation/noise_e4_semantic_v2cb_%j.out
#SBATCH --error=data/validation/noise_e4_semantic_v2cb_%j.err

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
GPU_ZERO="${ALLOCATED_GPUS[0]}"

FOLD=0
SEED=20260830
BASE_RUN="data/validation/noise_corrected_npa4_formal_fold_0_run_2332161_v3routes"
E4_BASE="$BASE_RUN/routes/n_bank"
BEST_RUN="data/validation/noise_corrected_best_v5_canary_fold_0_run_2332784"
BEST_LEDGER="$BEST_RUN/ledger"
GRAPH_DIR="data/validation/noise_corrected_candidate_graph_v1_20260906"
GRAPH="$GRAPH_DIR/candidate_graph.npz"
EMBEDDING_CACHE="$GRAPH_DIR/official_embeddings.npz"
EVAL_METADATA="data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz"
DATA="data/models/MassSpecGym_MurckoHist_split.hdf5"
OFFICIAL="data/e1/official_embedding_slim.pt"
ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
INITIAL="data/validation/g8r_noise_final_e8_direct_transfer/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_e8_baseline_symmetric_shared/seed_${SEED}/fold_${FOLD}/final_shared_encoder.pt"
INITIAL_DECISION="$(dirname "$INITIAL")/decision.json"
SOURCE_MANIFEST="tasks/noise_e4_signal_preserving_hybrid_v2_source_manifest.sha256"
EXPECTED_SOURCE_MANIFEST_SHA256="14e1a294a8649c968946b2fc48585dc756e176696341fecf48e9cf13c43ee722"
SBATCH_SOURCE="tasks/run_noise_e4_signal_preserving_hybrid_v2_2gpu.sbatch"
SPEC_SOURCE="docs/NOISE_E4_SIGNAL_PRESERVING_HYBRID_V2_SPEC_20260914.md"

FINAL_ROOT="data/validation/noise_e4_signal_preserving_hybrid_v2_clean_boundary_fold_${FOLD}_seed_${SEED}_run_${SLURM_JOB_ID}"
RUN_ROOT="data/validation/.noise_e4_signal_preserving_hybrid_v2_clean_boundary_fold_${FOLD}_seed_${SEED}_run_${SLURM_JOB_ID}.staging"
ARMS="$RUN_ROOT/arms"
SOURCE_SNAPSHOT="$RUN_ROOT/source_snapshot"

for REQUIRED in \
  "$GRAPH" "$GRAPH_DIR/report.json" "$EMBEDDING_CACHE" "$EVAL_METADATA" \
  "$DATA" "$OFFICIAL" "$ARCHITECTURE" "$INITIAL" "$INITIAL_DECISION" \
  "$E4_BASE/report.json" "$E4_BASE/training_actions.csv.gz" \
  "$BEST_LEDGER/report.json" "$BEST_LEDGER/training_actions.csv.gz" \
  "$BEST_LEDGER/action_spectra.npz" \
  "$SOURCE_MANIFEST" "$SBATCH_SOURCE" "$SPEC_SOURCE"; do
  [[ -f "$REQUIRED" ]] || { echo "Missing frozen Hybrid V2 input: $REQUIRED" >&2; exit 4; }
done

printf '%s  %s\n' "$EXPECTED_SOURCE_MANIFEST_SHA256" "$SOURCE_MANIFEST" | sha256sum -c -
sha256sum -c "$SOURCE_MANIFEST"
printf '%s  %s\n' \
  "8047b3f58c6808c86b320ac94b9e610610384040fa3a03e8d70550eb438a24af" \
  "$INITIAL" | sha256sum -c -
printf '%s  %s\n' \
  "9db644dc2592bb6eff2677779b731f6ee1f3f946f7df5868886b31c375cf3dfa" \
  "$E4_BASE/report.json" | sha256sum -c -
printf '%s  %s\n' \
  "1ce8c412f9ee0ef7a4ba4758554313c85f123280d31e73004cb62b76887c7be5" \
  "$E4_BASE/training_actions.csv.gz" | sha256sum -c -
printf '%s  %s\n' \
  "246e7e871e6669fec9f2c62330a69bf9eb563a1b97b72b89732cecd682844349" \
  "$BEST_LEDGER/report.json" | sha256sum -c -
printf '%s  %s\n' \
  "93f0785a69b5e323490a0b543059fa213fabb6ff848697831efba5b3fed667aa" \
  "$BEST_LEDGER/training_actions.csv.gz" | sha256sum -c -
printf '%s  %s\n' \
  "6d57615aebbfb7bd6327bb7edb143c6837d2586221c61aa2ff45e833f5e1512a" \
  "$BEST_LEDGER/action_spectra.npz" | sha256sum -c -
printf '%s  %s\n' \
  "8928f908606c0bd652c5a4107d3c35102f660622958c225a1f625abe4b1ba245" \
  "$OFFICIAL" | sha256sum -c -
printf '%s  %s\n' \
  "8a57bb3a9cccdf69a738f2b093bfad8f6fc4393b23f1342c7a035ac54ce58fa1" \
  "$GRAPH" | sha256sum -c -
printf '%s  %s\n' \
  "18d7632adc67dcda5d650a5c0bc344d8db3878f9e1ff3f8d51a6d4918b96bbda" \
  "$EMBEDDING_CACHE" | sha256sum -c -
printf '%s  %s\n' \
  "504ce0a570ac1bc461e76cad81acb3ca3560e58b008ec2d8ff14036e944599f5" \
  "$EVAL_METADATA" | sha256sum -c -
printf '%s  %s\n' \
  "ccda2c4114d9b21413977df03376ca0fc097956a7fa304b861a3154a2b81e64f" \
  "$DATA" | sha256sum -c -
printf '%s  %s\n' \
  "9884b62ecadf4bd441d22fec79b6787e5ffef168e15e7d8d5804dbdea08b38b2" \
  "$ARCHITECTURE" | sha256sum -c -

mapfile -t SOURCE_FILES < <(awk 'NF == 2 {print $2}' "$SOURCE_MANIFEST")
if (( ${#SOURCE_FILES[@]} < 90 )); then
  echo "Incomplete Hybrid V2 source closure: ${#SOURCE_FILES[@]} files" >&2
  exit 5
fi
[[ ! -e "$RUN_ROOT" && ! -e "$FINAL_ROOT" ]] || {
  echo "Refusing to overwrite unique Hybrid V2 output" >&2
  exit 5
}
mkdir -p "$ARMS" "$SOURCE_SNAPSHOT"

AVAILABLE_KB="$(df -Pk "$PWD" | awk 'NR==2 {print $4}')"
if [[ ! "$AVAILABLE_KB" =~ ^[0-9]+$ ]] || (( AVAILABLE_KB < 8388608 )); then
  echo "Hybrid V2 requires at least 8 GiB filesystem headroom; available_kb=$AVAILABLE_KB" >&2
  exit 5
fi

SNAPSHOT_FILES=("${SOURCE_FILES[@]}" "$SOURCE_MANIFEST" "$SBATCH_SOURCE" "$SPEC_SOURCE")
for SOURCE_FILE in "${SNAPSHOT_FILES[@]}"; do
  SNAPSHOT_TARGET="$SOURCE_SNAPSHOT/$SOURCE_FILE"
  mkdir -p "$(dirname "$SNAPSHOT_TARGET")"
  cp "$SOURCE_FILE" "$SNAPSHOT_TARGET"
done
(
  cd "$SOURCE_SNAPSHOT"
  sha256sum -c "$SOURCE_MANIFEST"
)

SNAPSHOT_ROOT="$PWD/$SOURCE_SNAPSHOT"
cp "$SNAPSHOT_ROOT/dreams/models/dreams/layers_e4_historical.py" \
  "$SNAPSHOT_ROOT/dreams/models/dreams/layers.py"
printf '%s  %s\n' \
  "ab5e44974e7dac2670d687812e2537e8a41c60248512c1e15e72727987536279" \
  "$SNAPSHOT_ROOT/dreams/models/dreams/layers.py" | sha256sum -c -
(
  cd "$SOURCE_SNAPSHOT"
  sha256sum "${SNAPSHOT_FILES[@]}" > executed_source_manifest.sha256
)

export PYTHONPATH="$SNAPSHOT_ROOT:$SNAPSHOT_ROOT/tasks"
PYTHON_SOURCES=()
for SOURCE_FILE in "${SOURCE_FILES[@]}"; do
  [[ "$SOURCE_FILE" == *.py ]] && PYTHON_SOURCES+=("$SNAPSHOT_ROOT/$SOURCE_FILE")
done
python -m py_compile "${PYTHON_SOURCES[@]}"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_final_e4a_direct_augmentation.py"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_corrected_fullgraph_evaluation.py"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_corrected_update_arbitration_v4.py"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_corrected_update_arbitration_v10.py"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_action_injector_v1.py"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_corrected_action_routing_v3.py"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_e4_signal_preserving_hybrid_v2.py"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_e4_signal_preserving_hybrid_v2_sbatch.py"

python - "$E4_BASE/report.json" "$E4_BASE/training_actions.csv.gz" \
  "$BEST_LEDGER/training_actions.csv.gz" "$SNAPSHOT_ROOT/tasks" <<'PY'
import json
import sys
import pandas as pd
sys.path.insert(0, sys.argv[4])
from noise_final_e4_pmt_core import select_materialized_direct_action_panel

base_report = json.load(open(sys.argv[1], encoding="utf-8"))
base = pd.read_csv(sys.argv[2], low_memory=False)
best = pd.read_csv(sys.argv[3], low_memory=False)
selected, _ = select_materialized_direct_action_panel(
    best, mode="multi_action_balanced", margin_floor=5e-6,
)
assert base_report["status"] == "noise_corrected_full_action_bank_complete"
assert base_report["formal"] is True
assert base_report["formal_training_authorized"] is True
assert base_report["configuration"] == {
    "formula_fold_seed": 20260825, "n_highest_peaks": 100,
    "top_k_negatives": 5, "softmax_temperature": 0.1,
    "fragment_tolerance": 0.02, "control_repeats": 2,
    "seed": 20260906, "max_queries": 0, "amp": False,
}
assert base_report["contracts"][
    "registered_formal_action_bank_configuration_verified"
] is True
assert base_report["contracts"]["action_outcomes_computed"] is False
assert base_report["contracts"]["teacher_embedding_or_margin_used"] is False
assert base_report["provenance"]["script_sha256"] == (
    "dafffea08745e59a5af4f37324858bb0472283d9d2b9d30a8bf411d6ee6f7d1d"
)
assert base_report["registered_actions"] == {
    "candidate_gradient": {
        "attenuation": 0.5, "maximum_steps": 6,
        "publish_steps": [3, 4, 5, 6],
    },
    "role_confounder": {
        "attenuation": 1.0, "maximum_steps": 5,
        "publish_steps": [1, 2, 3, 4, 5],
    },
}
assert base_report["source_queries"] == 65286
assert len(base) == 190324
assert base.query_index.nunique() == 44623
assert base.query_ik14.nunique() == 7624
assert base.query_formula.nunique() == 4825
assert not base.formula_fold.astype(int).eq(0).any()
assert len(selected) == 32114
assert selected.query_index.nunique() == 3482
assert selected.query_ik14.nunique() == 1536
assert set(selected.source.astype(str)) == {
    "N_mature", "P_guided_original", "E10B", "E11", "E12B",
    "A4_exact", "V4_gradient_path",
}
print(json.dumps({
    "status": "noise_e4_signal_preserving_hybrid_v2_clean_boundary_preflight_pass",
    "initialization": "frozen_E8",
    "corrected_e4_base_rows": len(base),
    "corrected_e4_base_identities": int(base.query_ik14.nunique()),
    "later_best_action_rows": len(selected),
    "later_best_action_queries": int(selected.query_index.nunique()),
    "later_best_action_identities": int(selected.query_ik14.nunique()),
    "semantic_active_steps_per_epoch": 1536 * 4,
    "e4_base_steps_per_epoch": 7624,
    "zero_semantic_e4_steps_per_epoch": 7624 - 1536 * 4,
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
  --materialized-injection-mode e4_base_semantic_v2
  --optimizer-boundary-mode signal_preserving_v2
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

result_dir() {
  local ARM="$1"
  local SUFFIX="e4_semantic_v2cb_${ARM}_run${SLURM_JOB_ID}"
  local TAG="curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_as_materialized_routed_marm_${ARM}_minj_e4_base_semantic_v2_ob_signal_preserving_v2_warm_${SUFFIX}"
  echo "$ARMS/$ARM/$TAG/seed_${SEED}/fold_${FOLD}"
}

run_arm() {
  local WORKER_GPU="$1"
  local ARM="$2"
  local SUFFIX="e4_semantic_v2cb_${ARM}_run${SLURM_JOB_ID}"
  export CUDA_VISIBLE_DEVICES="$WORKER_GPU" OMP_NUM_THREADS=8
  python -u "$SNAPSHOT_ROOT/tasks/train_noise_final_e4a_direct_augmentation.py" \
    "${COMMON[@]}" --materialized-action-arm "$ARM" \
    --output-root "$ARMS/$ARM" --run-suffix "$SUFFIX"
  python -u "$SNAPSHOT_ROOT/tasks/validate_noise_final_e4a_direct_augmentation.py" \
    --output-dir "$(result_dir "$ARM")"
}

echo "=== E8 + corrected-E4 base + clean-boundary seven-source semantic residual ==="
run_arm "$GPU_ZERO" targeted
run_arm "$GPU_ZERO" shuffled

python -u "$SNAPSHOT_ROOT/tasks/summarize_noise_e4_signal_preserving_hybrid_v2.py" \
  --targeted-dir "$(result_dir targeted)" \
  --shuffled-dir "$(result_dir shuffled)" \
  --best-action-dir "$BEST_LEDGER" --e4-base-action-dir "$E4_BASE" \
  --initial-e8-checkpoint "$INITIAL" --output-dir "$RUN_ROOT/summary" \
  --bootstrap-resamples 10000 --seed "$SEED"

(
  cd "$RUN_ROOT"
  find . -type f \
    ! -name 'artifact_manifest.sha256' \
    ! -name 'artifact_manifest.sha256.tmp' \
    -print0 | sort -z | xargs -0 sha256sum > artifact_manifest.sha256.tmp
  mv artifact_manifest.sha256.tmp artifact_manifest.sha256
)
mv "$RUN_ROOT" "$FINAL_ROOT"
echo "=== Hybrid V2 clean-boundary result: $FINAL_ROOT ==="
