#!/bin/bash
# GLM composition experiment: put the Noise action curriculum on top of the
# ChemAware champion, and test the 1+1 accounting on ONE frozen universe.
#
# Pre-registered design (docs/GLM_INTEGRATION_PREREGISTRATION_20260930.md):
#   base       = Phase-A (chemistry champion), no training
#   arm A      = axis-balanced curriculum on the Phase-A warm start
#   arm N      = un-stratified advantage-max curriculum (axis control)
#   arm C      = matched continuation control (same relations, control spectra)
# Only triplet membership changes; the native DreaMS model, loss, optimizer,
# preprocessor and the exact-Adam continuation are the existing validated code.
# Everything fails closed before training when a floor, a hash or a contract
# is not met, and no held outcome is used for selection.
#SBATCH --job-name=glm_compose_axis
#SBATCH --gpus=3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/glm_compose_axis_%j.out
#SBATCH --error=data/validation/glm_compose_axis_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

IFS=',' read -r -a ALLOCATED_GPUS <<< "${CUDA_VISIBLE_DEVICES:-}"
if (( ${#ALLOCATED_GPUS[@]} != 3 )); then
  echo "Expected exactly three allocated GPUs, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" >&2
  exit 3
fi

# ---- frozen assets -------------------------------------------------------
OFFICIAL="data/e1/official_embedding_slim.pt"
ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
DATA="data/models/MassSpecGym_MurckoHist_split.hdf5"
BASE="data/validation/chemaware_high_coverage_native/run_2340524"
PHASEA_RUN="data/validation/chemaware_online_role_budget_native/run_2345481"
PHASEA="$PHASEA_RUN/phasea_reproduction_gate/chemaware_phasea_max_boundary_step2000.ckpt"
PHASEA_SHA="a8428329ca1d12bfe735f3f8b848ed020a8db18d0a654d4f62cdbeb006e61135"
MANIFEST="data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz"
REUSABLE_CACHE="data/validation/chemaware_fragment_graph_native/run_2346306/phasea_embeddings"
STAGE1="data/validation/noise_dreams_hard_positive_native_fold_0_run_2344820"
NOISE_CKPT="$STAGE1/checkpoint/targeted_final.ckpt"
GRAPH_DIR="data/validation/noise_corrected_candidate_graph_v1_20260906"
GRAPH="$GRAPH_DIR/candidate_graph.npz"
EMBEDDING_CACHE="$GRAPH_DIR/official_embeddings.npz"
MATURE_E8="data/validation/g8r_noise_final_e8_direct_transfer/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_e8_baseline_symmetric_shared/seed_20260830/fold_0/final_shared_encoder.pt"
GNPS_BENCHMARK="data/validation/gnps_gold_silver_10ppm_benchmark_v1"

# Pre-registered floors: an axis claim needs enough taught queries per axis
# (>=20 net corrections on the 1,975-query role-2 panel is the smallest
# interesting effect, and the historical taught-relation transfer is partial).
MIN_AXIS_QUERIES=60
MIN_TOTAL_QUERIES=200

OUT="data/validation/GLM_compose_axis_curriculum/run_${SLURM_JOB_ID}"
BUILD_A="$OUT/build_axis_balanced"
BUILD_N="$OUT/build_advantage_max"
TRAIN="$OUT/training"
SLIM="$OUT/slim"
EVAL="$OUT/evaluation"
GNPS="$OUT/gnps_transfer"
ADD="$OUT/additivity"

[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 4; }
mkdir -p "$OUT" "$TRAIN" "$SLIM" "$EVAL" "$GNPS" "$ADD"

required_files=(
  "$OFFICIAL" "$ARCHITECTURE" "$DATA"
  "$BASE/evidence/selection_triplet_evidence.npz"
  "$BASE/evidence/confirmation_triplet_evidence.npz"
  "$BASE/evidence/report.json"
  "$PHASEA" "$MANIFEST"
  "$REUSABLE_CACHE/rows.npy" "$REUSABLE_CACHE/embeddings_f32.npy"
  "$STAGE1/triplets/train_pool.npz"
  "$STAGE1/triplets/validation_pool.npz"
  "$STAGE1/triplets/action_spectra.npz"
  "$STAGE1/triplets/selected_actions.csv.gz"
  "$STAGE1/triplets/report.json"
  "$NOISE_CKPT" "$GRAPH" "$EMBEDDING_CACHE" "$MATURE_E8"
  "$GNPS_BENCHMARK/report.json"
  tasks/GLM_compose_axis_core.py
  tasks/GLM_build_compose_axis_curriculum.py
  tasks/GLM_summarize_compose_additivity.py
  tasks/train_noise_dreams_native_residual_stage2.py
  tasks/evaluate_chemaware_v2_direct_triplet.py
  tasks/select_chemaware_residual_checkpoint.py
  tasks/prepare_official_embedding_checkpoint.py
  tasks/encode_gnps_gold_silver_10ppm_checkpoint.py
  tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py
  tasks/evaluate_noise_dreams_native.py
)
for path in "${required_files[@]}"; do
  [[ -e "$path" ]] || { echo "Missing composition input: $path" >&2; exit 2; }
done

# ---- contracts before any GPU work --------------------------------------
python -u -m py_compile \
  tasks/GLM_compose_axis_core.py \
  tasks/GLM_build_compose_axis_curriculum.py \
  tasks/GLM_summarize_compose_additivity.py
python -u tasks/test_GLM_build_compose_axis_curriculum.py
python -u tasks/test_GLM_summarize_compose_additivity.py
python -u tasks/test_GLM_compose_axis_curriculum_sbatch.py

# Warm-start pin: the chemistry champion must be the exact checkpoint whose
# hash the protected Phase-A provenance recorded.  A drifted file stops here.
ACTUAL_PHASEA_SHA="$(sha256sum "$PHASEA" | cut -d' ' -f1)"
if [[ "$ACTUAL_PHASEA_SHA" != "$PHASEA_SHA" ]]; then
  echo "GLM_COMPOSE_WARM_START_DRIFT: $ACTUAL_PHASEA_SHA != $PHASEA_SHA" >&2
  exit 2
fi

# ---- build the two curricula in parallel (one GPU each) ------------------
CUDA_VISIBLE_DEVICES="${ALLOCATED_GPUS[0]}" python -u \
  tasks/GLM_build_compose_axis_curriculum.py \
  --stage1-run "$STAGE1" \
  --warm-start-checkpoint "$PHASEA" \
  --expected-warm-start-sha256 "$PHASEA_SHA" \
  --graph "$GRAPH" --embedding-cache "$REUSABLE_CACHE/rows.npy" \
  --data "$DATA" --architecture-checkpoint "$ARCHITECTURE" \
  --output "$BUILD_A" --axis-balance balanced \
  --minimum-axis-queries "$MIN_AXIS_QUERIES" \
  >"$OUT/build_axis_balanced.out" 2>"$OUT/build_axis_balanced.err" &
PID_BUILD_A=$!
CUDA_VISIBLE_DEVICES="${ALLOCATED_GPUS[1]}" python -u \
  tasks/GLM_build_compose_axis_curriculum.py \
  --stage1-run "$STAGE1" \
  --warm-start-checkpoint "$PHASEA" \
  --expected-warm-start-sha256 "$PHASEA_SHA" \
  --graph "$GRAPH" --embedding-cache "$REUSABLE_CACHE/rows.npy" \
  --data "$DATA" --architecture-checkpoint "$ARCHITECTURE" \
  --output "$BUILD_N" --axis-balance advantage-max \
  --minimum-axis-queries "$MIN_AXIS_QUERIES" \
  >"$OUT/build_advantage_max.out" 2>"$OUT/build_advantage_max.err" &
PID_BUILD_N=$!
set +e
wait "$PID_BUILD_A"; STATUS_BUILD_A=$?
wait "$PID_BUILD_N"; STATUS_BUILD_N=$?
set -e
if (( STATUS_BUILD_A != 0 || STATUS_BUILD_N != 0 )); then
  tail -n 80 "$OUT/build_axis_balanced.err" >&2 || true
  tail -n 80 "$OUT/build_advantage_max.err" >&2 || true
  echo "GLM_COMPOSE_BUILD_FAILED: balanced=$STATUS_BUILD_A advantage_max=$STATUS_BUILD_N" >&2
  exit 5
fi

# ---- pre-registered floor gate (no GPU training before it passes) --------
python - "$BUILD_A/report.json" "$BUILD_N/report.json" \
        "$MIN_AXIS_QUERIES" "$MIN_TOTAL_QUERIES" <<'PY'
import json, sys
balanced = json.load(open(sys.argv[1], encoding="utf-8"))
maximum = json.load(open(sys.argv[2], encoding="utf-8"))
min_axis, min_total = int(sys.argv[3]), int(sys.argv[4])
axis = balanced["axis_selected_queries"]
total = int(balanced["selected_action_queries"])
pd_count = int(axis.get("positive_deficit", 0))
ne_count = int(axis.get("negative_excess", 0))
print(f"GLM_COMPOSE_FLOORS: total={total} pd={pd_count} ne={ne_count} "
      f"advantage_max_total={maximum['selected_action_queries']}")
ok = (total >= min_total and pd_count >= min_axis and ne_count >= min_axis
      and int(maximum["selected_action_queries"]) >= min_total)
print("GLM_COMPOSE_FLOOR_GATE:", "PASS" if ok else "STOP")
sys.exit(0 if ok else 6)
PY

# ---- three native continuations from the identical warm start -----------
train_arm() {
  local gpu="$1" triplet_dir="$2" arm="$3" name="$4"
  CUDA_VISIBLE_DEVICES="$gpu" python -u \
    tasks/train_noise_dreams_native_residual_stage2.py \
    --data "$DATA" --triplet-dir "$triplet_dir" --arm "$arm" \
    --warm-start-checkpoint "$PHASEA" \
    --architecture-checkpoint "$ARCHITECTURE" \
    --output "$TRAIN/$name" --seed 3407 --lr 1e-6 --weight-decay 0 \
    --triplet-loss-margin 0.1 --batch-size 4 --num-workers 4 --max-epochs 1 \
    >"$OUT/train_${name}.out" 2>"$OUT/train_${name}.err"
}
train_arm "${ALLOCATED_GPUS[0]}" "$BUILD_A" targeted balanced &
PID_TRAIN_A=$!
train_arm "${ALLOCATED_GPUS[1]}" "$BUILD_A" control control &
PID_TRAIN_C=$!
train_arm "${ALLOCATED_GPUS[2]}" "$BUILD_N" targeted advantage_max &
PID_TRAIN_N=$!
set +e
wait "$PID_TRAIN_A"; STATUS_TRAIN_A=$?
wait "$PID_TRAIN_C"; STATUS_TRAIN_C=$?
wait "$PID_TRAIN_N"; STATUS_TRAIN_N=$?
set -e
if (( STATUS_TRAIN_A != 0 || STATUS_TRAIN_C != 0 || STATUS_TRAIN_N != 0 )); then
  for name in balanced control advantage_max; do
    tail -n 60 "$OUT/train_${name}.err" >&2 || true
  done
  echo "GLM_COMPOSE_TRAINING_FAILED: balanced=$STATUS_TRAIN_A control=$STATUS_TRAIN_C advantage_max=$STATUS_TRAIN_N" >&2
  exit 7
fi
for name in balanced control advantage_max; do
  [[ -f "$TRAIN/$name/final.ckpt" ]] || {
    echo "GLM_COMPOSE_TRAINING_INCOMPLETE: $TRAIN/$name/final.ckpt" >&2
    exit 7
  }
done

# ---- compact weights-only exports for the GNPS stage (secondary) --------
# The primary role-2/role-3 readout consumes the native checkpoints directly,
# so a conversion problem can never invalidate it.

# ---- role-2 readout on ONE universe: chemistry + noise + composition ----
srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$BASE/evidence/selection_triplet_evidence.npz" \
  --role-contract-dir "$BASE/evidence" \
  --checkpoint "official=$OFFICIAL" \
  --checkpoint "phasea_2pp=$PHASEA" \
  --checkpoint "noise_stage1=$NOISE_CKPT" \
  --checkpoint "compose_balanced=$TRAIN/balanced/final.ckpt" \
  --checkpoint "compose_advantage_max=$TRAIN/advantage_max/final.ckpt" \
  --checkpoint "compose_control=$TRAIN/control/final.ckpt" \
  --paired-reference phasea_2pp --formula-role 2 \
  --output "$EVAL/role2_composition.json" \
  --batch-size 64 --bootstrap-draws 10000 --device cuda

# Axis-hypothesis contrast: balanced against the un-stratified arm, paired.
srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$BASE/evidence/selection_triplet_evidence.npz" \
  --role-contract-dir "$BASE/evidence" \
  --checkpoint "compose_advantage_max=$TRAIN/advantage_max/final.ckpt" \
  --checkpoint "compose_balanced=$TRAIN/balanced/final.ckpt" \
  --checkpoint "compose_control=$TRAIN/control/final.ckpt" \
  --paired-reference compose_advantage_max --formula-role 2 \
  --output "$EVAL/role2_axis_contrast.json" \
  --batch-size 64 --bootstrap-draws 10000 --device cuda

python -u tasks/select_chemaware_residual_checkpoint.py \
  --evaluation "$EVAL/role2_composition.json" \
  --base-name phasea_2pp --require-positive-formula-ci \
  --output "$EVAL/checkpoint_selection.json" || true
SELECTED_STEP="0"
[[ -f "$EVAL/checkpoint_selection.step.txt" ]] && \
  SELECTED_STEP="$(tr -d '\r\n' < "$EVAL/checkpoint_selection.step.txt")"
if [[ "$SELECTED_STEP" == "0" ]]; then
  echo "GLM_COMPOSE_ROLE2_STOP: no arm beats the chemistry champion; role 3 stays closed"
fi

# ---- role-3 protection (protected baseline = the chemistry champion) ----
if [[ "$SELECTED_STEP" != "0" ]]; then
  srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
    --policy "$BASE/evidence/confirmation_triplet_evidence.npz" \
    --role-contract-dir "$BASE/evidence" \
    --checkpoint "official=$OFFICIAL" \
    --checkpoint "phasea_2pp=$PHASEA" \
    --checkpoint "compose_balanced=$TRAIN/balanced/final.ckpt" \
    --checkpoint "compose_advantage_max=$TRAIN/advantage_max/final.ckpt" \
    --checkpoint "compose_control=$TRAIN/control/final.ckpt" \
    --paired-reference phasea_2pp --formula-role 3 \
    --output "$EVAL/role3_composition.json" \
    --batch-size 64 --bootstrap-draws 10000 --device cuda
fi

# ---- sealed GNPS Gold/Silver dual panels (secondary, non-blocking) ------
# Wrapped in set +e: conversion, encoding or evaluation trouble only records a
# status; the primary role-2/role-3 result and the additivity ledger stand.
set +e
GNPS_STATUS="ok"
python -u tasks/prepare_official_embedding_checkpoint.py --source "$PHASEA" --output "$SLIM/phasea_2pp.pt" || GNPS_STATUS="slim_failed_phasea"
python -u tasks/prepare_official_embedding_checkpoint.py --source "$NOISE_CKPT" --output "$SLIM/noise_stage1.pt" || GNPS_STATUS="slim_failed_noise"
python -u tasks/prepare_official_embedding_checkpoint.py --source "$TRAIN/balanced/final.ckpt" --output "$SLIM/compose_balanced.pt" || GNPS_STATUS="slim_failed_balanced"
python -u tasks/prepare_official_embedding_checkpoint.py --source "$TRAIN/advantage_max/final.ckpt" --output "$SLIM/compose_advantage_max.pt" || GNPS_STATUS="slim_failed_advantage_max"
python -u tasks/prepare_official_embedding_checkpoint.py --source "$TRAIN/control/final.ckpt" --output "$SLIM/compose_control.pt" || GNPS_STATUS="slim_failed_control"
if [[ "$GNPS_STATUS" == "ok" ]]; then
  srun --export=ALL --preserve-env python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
    --benchmark "$GNPS_BENCHMARK" --checkpoint "$OFFICIAL" \
    --architecture-checkpoint "$ARCHITECTURE" \
    --output "$GNPS/official_embeddings.npz" --batch-size 64 --label official
  for pair in phasea_2pp noise_stage1 compose_balanced compose_advantage_max compose_control; do
    srun --export=ALL --preserve-env python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
      --benchmark "$GNPS_BENCHMARK" --checkpoint "$SLIM/${pair}.pt" \
      --architecture-checkpoint "$ARCHITECTURE" \
      --output "$GNPS/${pair}_embeddings.npz" --batch-size 64 --label "$pair" || GNPS_STATUS="encode_failed_${pair}"
  done
fi
evaluate_gnps() {
  local baseline="$1" candidate="$2" name="$3"
  python -u tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py \
    --benchmark "$GNPS_BENCHMARK" \
    --baseline-embeddings "$GNPS/${baseline}_embeddings.npz" \
    --candidate-embeddings "$GNPS/${candidate}_embeddings.npz" \
    --output "$GNPS/${name}" \
    --bootstrap-resamples 10000 --bootstrap-seed 20260930 || GNPS_STATUS="evaluate_failed_${name}"
}
evaluate_gnps phasea_2pp compose_balanced compose_balanced_vs_phasea
evaluate_gnps phasea_2pp compose_control compose_control_vs_phasea
evaluate_gnps compose_advantage_max compose_balanced compose_balanced_vs_advantage_max
evaluate_gnps official compose_balanced compose_balanced_vs_official
evaluate_gnps official phasea_2pp phasea_vs_official
evaluate_gnps official noise_stage1 noise_stage1_vs_official
if [[ "$GNPS_STATUS" != "ok" ]]; then
  echo "GLM_COMPOSE_GNPS_SECONDARY_STATUS: $GNPS_STATUS" >&2
fi
set -e

# ---- pre-registered additivity verdict ---------------------------------
ROLE3_ARGS=()
if [[ -f "$EVAL/role3_composition.json" ]]; then
  ROLE3_ARGS=(--role3 "$EVAL/role3_composition.json")
fi
python -u tasks/GLM_summarize_compose_additivity.py \
  --role2 "$EVAL/role2_composition.json" \
  --role2-axis "$EVAL/role2_axis_contrast.json" \
  "${ROLE3_ARGS[@]}" \
  --gnps "$GNPS/compose_balanced_vs_phasea" \
         "$GNPS/compose_control_vs_phasea" \
         "$GNPS/compose_balanced_vs_official" \
  --output "$ADD"

# ---- secondary: corrected-graph fold-0 readout (the Noise line's own panel)
set +e
evaluate_fold0() {
  local gpu="$1" name="$2" checkpoint="$3"
  CUDA_VISIBLE_DEVICES="$gpu" python -u tasks/evaluate_noise_dreams_native.py \
    --graph "$GRAPH" --embedding-cache "$EMBEDDING_CACHE" \
    --official-slim-checkpoint "$OFFICIAL" \
    --source-manifest "$MANIFEST" --data "$DATA" \
    --checkpoint "$checkpoint" --architecture-checkpoint "$ARCHITECTURE" \
    --output "$EVAL/fold0_${name}" --arm targeted \
    --outer-fold 0 --formula-fold-seed 20260825 \
    --batch-size 64 --bootstrap-resamples 10000 --bootstrap-seed 20260926 \
    --official-finetuned-checkpoint "$OFFICIAL" \
    --mature-e8-checkpoint "$MATURE_E8" \
    --e8-architecture-checkpoint "$ARCHITECTURE" \
    --no-write-held-metric-evidence \
    >"$OUT/fold0_${name}.out" 2>"$OUT/fold0_${name}.err"
}
evaluate_fold0 "${ALLOCATED_GPUS[0]}" phasea_2pp "$SLIM/phasea_2pp.pt" &
P1=$!
evaluate_fold0 "${ALLOCATED_GPUS[1]}" compose_balanced "$SLIM/compose_balanced.pt" &
P2=$!
evaluate_fold0 "${ALLOCATED_GPUS[2]}" compose_control "$SLIM/compose_control.pt" &
P3=$!
wait "$P1"; S1=$?
wait "$P2"; S2=$?
wait "$P3"; S3=$?
FOLD0_STATUS="ok"
if (( S1 != 0 || S2 != 0 || S3 != 0 )); then
  FOLD0_STATUS="failed: phasea=$S1 balanced=$S2 control=$S3"
  echo "GLM_COMPOSE_FOLD0_SECONDARY_FAILED: $FOLD0_STATUS" >&2
fi
set -e

find "$OUT" -type f -print0 | sort -z | \
  xargs -0 sha256sum > "$OUT/SHA256SUMS"
cat > "$OUT/run_status.json" <<JSON
{
  "job_id": "${SLURM_JOB_ID}",
  "selected_step": "${SELECTED_STEP}",
  "gnps_secondary": "${GNPS_STATUS}",
  "fold0_secondary": "${FOLD0_STATUS}",
  "additivity_report": "$ADD/report.json"
}
JSON
echo "COMPLETE: GLM composition experiment -> $OUT"
