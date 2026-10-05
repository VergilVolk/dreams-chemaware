#!/bin/bash
#SBATCH --job-name=noise_e4_deb
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=30:00:00
#SBATCH --output=data/validation/noise_e4_deb_%j.out
#SBATCH --error=data/validation/noise_e4_deb_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FOLD=0
SEED=20260905
ROUTING_INITIAL="data/validation/g8r_noise_final_e4a_direct/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_highlr_multifold/seed_20260830/fold_${FOLD}/final_shared_encoder.pt"
PRIOR_CLEAN_ROOT="data/validation/g8r_noise_final_e4_pmt/fold_0_run_2331467/arms/clean_duplicate"
SOURCE_RUN="data/validation/g8r_noise_final_dynamic_direct_e4_phase_a_final/fold_0_run_2331284"
LEDGER="$SOURCE_RUN/ledger"
REPLAY="$SOURCE_RUN/replay"
RUN_ROOT="data/validation/g8r_noise_final_e4_deb/fold_${FOLD}_run_${SLURM_JOB_ID}"
MANIFEST="$RUN_ROOT/manifest"
ARMS_ROOT="$RUN_ROOT/arms"
SUMMARY="$RUN_ROOT/summary"

for REQUIRED in \
  "$ROUTING_INITIAL" \
  "data/validation/g8r_noise_final_r0_faithful_s3a/report.json" \
  "data/validation/g8r_noise_final_r0_faithful_s3a/training_actions.csv.gz" \
  "$LEDGER/report.json" "$LEDGER/training_actions.csv.gz" \
  "$REPLAY/report.json" "$REPLAY/current_geometry_outcomes.csv.gz" \
  "data/validation/g8r_error_atlas_listwise_cache.npz" \
  "data/validation/g8r_p2_official_embeddings.npz" \
  "data/models/MassSpecGym_MurckoHist_split.hdf5" \
  "data/e1/official_embedding_slim.pt" \
  "dreams/models/pretrained/ssl_model_server.pt"; do
  [[ -f "$REQUIRED" ]] || { echo "Missing frozen E4-DEB input: $REQUIRED" >&2; exit 3; }
done
PRIOR_CLEAN_COUNT=$(find "$PRIOR_CLEAN_ROOT" -name final_shared_encoder.pt -type f 2>/dev/null | wc -l)
[[ "$PRIOR_CLEAN_COUNT" -eq 1 ]] || {
  echo "Expected exactly one frozen job-2331467 clean-duplicate checkpoint under $PRIOR_CLEAN_ROOT; found $PRIOR_CLEAN_COUNT" >&2
  exit 3
}
INITIAL=$(find "$PRIOR_CLEAN_ROOT" -name final_shared_encoder.pt -type f)
[[ ! -e "$RUN_ROOT" ]] || { echo "Refusing to overwrite unique E4-DEB run: $RUN_ROOT" >&2; exit 4; }

echo "=== E4-DEB direct candidate-boundary phase A started: $(date) ==="
python -m py_compile \
  tasks/noise_final_candidate_boundary_core.py \
  tasks/test_noise_final_candidate_boundary_core.py \
  tasks/build_noise_final_e4_pmt_manifest.py \
  tasks/train_noise_final_e4a_direct_augmentation.py \
  tasks/validate_noise_final_e4a_direct_augmentation.py \
  tasks/summarize_noise_final_e4_deb.py
python -u tasks/audit_noise_final_e4_deb_implementation.py
python -u tasks/test_noise_final_candidate_boundary_core.py
python -u tasks/test_noise_final_e4a_direct_augmentation.py
python -u tasks/test_noise_final_e4_pmt.py

python -u tasks/build_noise_final_e4_pmt_manifest.py \
  --ledger-dir "$LEDGER" --replay-dir "$REPLAY" \
  --initial-checkpoint "$ROUTING_INITIAL" --outer-fold "$FOLD" \
  --advantage-threshold 0.01 --corrective-query-scope errors \
  --output-dir "$MANIFEST"
python -u tasks/validate_noise_final_e4_pmt.py --output-dir "$MANIFEST"

COMMON=(
  --pmt-manifest-dir "$MANIFEST"
  --initial-student-checkpoint "$INITIAL"
  --policy curriculum --action-selection fixed --action-scope all
  --outer-fold "$FOLD" --formula-fold-seed 20260825 --seed "$SEED" --epochs 4
  --batch-actions 4 --views-per-identity 4 --error-views-per-identity 0
  --positive-spectra 4 --negative-molecules 8
  --unfreeze-blocks 1 --backbone-lr 2e-6 --head-lr 1e-5
  --weight-decay 1e-4 --grad-clip 1.0
  --rank-margin 0.05 --temperature 0.10
  --lambda-clean-rank 1.0 --lambda-aug-rank 1.0
  --lambda-consistency 0.25 --lambda-margin-floor 2.0
  --lambda-preserve 5.0 --margin-floor-slack 0.005
  --safety-ratio 1.0 --safety-stream-weight 1.0
  --positive-stream-weight 0.0 --guided-noise-policy none
  --direct-transfer-mode symmetric --rank-reference-mode shared
  --pmt-advantage-cap 0.10 --pmt-preference-gap 0.01
  --lambda-pmt-preference 1.0 --device cuda --no-amp
  --boundary-advantage-temperature 0.02
  --boundary-hard-temperature 0.10 --boundary-topk-negatives 8
  --lambda-boundary-clean 1.0 --lambda-boundary-target 1.0
  --lambda-boundary-counterfactual 1.0 --lambda-boundary-full-clean 0.10
  --boundary-auto-balance --boundary-calibration-formulas 32
  --boundary-calibration-examples 4
  --boundary-action-to-safety-norm-ratio 1.0 --boundary-action-scale-cap 4.0
)

# Exercise the exact treatment path before any formal arm.
python -u tasks/train_noise_final_e4a_direct_augmentation.py \
  --output-root "$RUN_ROOT/smoke" --run-suffix "run_${SLURM_JOB_ID}" \
  --pmt-arm paired_target --pmt-alpha 0 --candidate-boundary-loss \
  --refresh-hard-negatives "${COMMON[@]}" --smoke

LABELS=(clean_duplicate matched_random candidate_boundary)
PMT_ARMS=(clean_duplicate matched_random paired_target)
for INDEX in 0 1 2; do
  LABEL="${LABELS[$INDEX]}"
  ARM="${PMT_ARMS[$INDEX]}"
  EXTRA=(--no-candidate-boundary-loss --no-refresh-hard-negatives)
  if [[ "$LABEL" == "candidate_boundary" ]]; then
    EXTRA=(--candidate-boundary-loss --refresh-hard-negatives)
  fi
  echo "=== E4-DEB arm=$LABEL started: $(date) ==="
  python -u tasks/train_noise_final_e4a_direct_augmentation.py \
    --output-root "$ARMS_ROOT/$LABEL" --run-suffix "run_${SLURM_JOB_ID}" \
    --pmt-arm "$ARM" --pmt-alpha 0 "${EXTRA[@]}" "${COMMON[@]}"
  RESULT_COUNT=$(find "$ARMS_ROOT/$LABEL" -name decision.json -type f | wc -l)
  [[ "$RESULT_COUNT" -eq 1 ]] || { echo "Expected one $LABEL result, found $RESULT_COUNT" >&2; exit 5; }
  RESULT=$(dirname "$(find "$ARMS_ROOT/$LABEL" -name decision.json -type f)")
  python -u tasks/validate_noise_final_e4a_direct_augmentation.py --output-dir "$RESULT"
  echo "=== E4-DEB arm=$LABEL completed: $(date) ==="
done

python -u tasks/summarize_noise_final_e4_deb.py \
  --arms-root "$ARMS_ROOT" \
  --output-dir "$SUMMARY" --bootstrap-resamples 10000 --seed "$SEED"

echo "=== E4-DEB direct candidate-boundary phase A completed: $(date) ==="
