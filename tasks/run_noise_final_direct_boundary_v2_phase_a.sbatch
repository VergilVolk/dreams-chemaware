#!/bin/bash
#SBATCH --job-name=noise_dbv2
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00
#SBATCH --output=data/validation/noise_dbv2_%j.out
#SBATCH --error=data/validation/noise_dbv2_%j.err

set -euo pipefail
echo "BLOCKED: this job uses the superseded 23,876-query restricted graph. Use only as historical reproduction; it cannot support a new formal noise claim." >&2
exit 64
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FOLD=0
INITIAL="data/validation/g8r_noise_final_e4a_direct/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_highlr_multifold/seed_20260830/fold_${FOLD}/final_shared_encoder.pt"
SOURCE_RUN="data/validation/g8r_noise_final_dynamic_direct_e4_phase_a_final/fold_0_run_2331284"
LEDGER="$SOURCE_RUN/ledger"
REPLAY="$SOURCE_RUN/replay"
RUN_ROOT="data/validation/g8r_noise_final_direct_boundary_v2/fold_${FOLD}_run_${SLURM_JOB_ID}"
MANIFEST="$RUN_ROOT/manifest"
RUNS="$RUN_ROOT/runs"
SUMMARY="$RUN_ROOT/summary"

for REQUIRED in \
  "$INITIAL" \
  "data/validation/g8r_error_atlas_listwise_cache.npz" \
  "data/validation/g8r_p2_official_embeddings.npz" \
  "data/validation/g8r_noise_final_r0_faithful_s3a/report.json" \
  "data/validation/g8r_noise_final_r0_faithful_s3a/training_actions.csv.gz" \
  "$LEDGER/report.json" "$LEDGER/training_actions.csv.gz" \
  "$REPLAY/report.json" "$REPLAY/current_geometry_outcomes.csv.gz"; do
  [[ -f "$REQUIRED" ]] || { echo "Missing frozen direct-v2 input: $REQUIRED" >&2; exit 3; }
done
[[ ! -e "$RUN_ROOT" ]] || { echo "Refusing to overwrite unique direct-v2 run: $RUN_ROOT" >&2; exit 4; }

python -m py_compile \
  tasks/noise_final_direct_boundary_v2_core.py \
  tasks/noise_final_e4_pmt_core.py \
  tasks/build_noise_final_e4_pmt_manifest.py \
  tasks/train_noise_final_e4a_direct_augmentation.py \
  tasks/validate_noise_final_e4a_direct_augmentation.py \
  tasks/summarize_noise_direct_boundary_v2.py
python -u tasks/test_noise_final_direct_boundary_v2_core.py
python -u tasks/test_noise_final_e4_pmt.py
python -u tasks/test_noise_final_e4a_candidate_boundary_v2.py
python -u tasks/test_noise_final_e4a_direct_augmentation.py

python -u tasks/build_noise_final_e4_pmt_manifest.py \
  --ledger-dir "$LEDGER" --replay-dir "$REPLAY" \
  --initial-checkpoint "$INITIAL" --outer-fold "$FOLD" \
  --advantage-threshold 0.01 --corrective-query-scope errors \
  --output-dir "$MANIFEST"
python -u tasks/validate_noise_final_e4_pmt.py --output-dir "$MANIFEST"

COMMON=(
  --pmt-manifest-dir "$MANIFEST"
  --initial-student-checkpoint "$INITIAL"
  --policy curriculum --action-selection fixed --action-scope all
  --outer-fold "$FOLD" --formula-fold-seed 20260825 --epochs 4
  --batch-actions 9 --views-per-identity 4 --error-views-per-identity 0
  --positive-spectra 4 --negative-molecules 8
  --unfreeze-blocks 1 --backbone-lr 2e-6 --head-lr 1e-5
  --weight-decay 1e-4 --grad-clip 1.0
  --rank-margin 0.05 --temperature 0.10
  --lambda-preserve 5.0 --margin-floor-slack 0.005
  --safety-ratio 1.0 --safety-stream-weight 1.0
  --positive-stream-weight 0.0 --guided-noise-policy none
  --direct-transfer-mode symmetric --rank-reference-mode shared
  --pmt-arm paired_target --pmt-alpha 0
  --candidate-boundary-loss --candidate-boundary-version v2_molecule_max
  --boundary-topk-negatives 8 --boundary-advantage-temperature 0.02
  --lambda-boundary-clean 1.0 --lambda-boundary-target 0.25
  --lambda-boundary-counterfactual 0.25 --lambda-boundary-full-clean 0.25
  --lambda-boundary-action-safety 0.25
  --boundary-auto-balance --boundary-calibration-formulas 32
  --boundary-calibration-examples 9 --boundary-action-scale-cap 1.0
  --refresh-hard-negatives --bootstrap-resamples 10000
  --device cuda --no-amp
)

SEEDS=(20260906 20260907 20260908)
for SEED in "${SEEDS[@]}"; do
  python -u tasks/train_noise_final_e4a_direct_augmentation.py \
    --output-root "$RUNS" --run-suffix "run_${SLURM_JOB_ID}" \
    --seed "$SEED" "${COMMON[@]}"
  RESULT_COUNT=$(find "$RUNS" -path "*/seed_${SEED}/fold_${FOLD}/decision.json" -type f | wc -l)
  [[ "$RESULT_COUNT" -eq 1 ]] || { echo "Expected one seed $SEED decision, found $RESULT_COUNT" >&2; exit 5; }
  RESULT=$(dirname "$(find "$RUNS" -path "*/seed_${SEED}/fold_${FOLD}/decision.json" -type f)")
  python -u tasks/validate_noise_final_e4a_direct_augmentation.py --output-dir "$RESULT"
done

python -u tasks/summarize_noise_direct_boundary_v2.py \
  --runs-root "$RUNS" --expected-seeds "${SEEDS[@]}" \
  --bootstrap-resamples 10000 --seed 20260906 --output-dir "$SUMMARY"
echo "=== direct-boundary v2 Phase A complete: $RUN_ROOT ==="
