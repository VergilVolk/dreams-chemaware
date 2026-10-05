#!/bin/bash
#SBATCH --job-name=noise_e4_pmt
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/noise_e4_pmt_%j.out
#SBATCH --error=data/validation/noise_e4_pmt_%j.err

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
INITIAL="data/validation/g8r_noise_final_e4a_direct/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_highlr_multifold/seed_20260830/fold_${FOLD}/final_shared_encoder.pt"
SOURCE_RUN="data/validation/g8r_noise_final_dynamic_direct_e4_phase_a_final/fold_0_run_2331284"
LEDGER="$SOURCE_RUN/ledger"
REPLAY="$SOURCE_RUN/replay"
RUN_ROOT="data/validation/g8r_noise_final_e4_pmt/fold_${FOLD}_run_${SLURM_JOB_ID}"
MANIFEST="$RUN_ROOT/manifest"
ARMS_ROOT="$RUN_ROOT/arms"
SUMMARY="$RUN_ROOT/summary"

# Fail before importing the 117M model or spending GPU time.
for REQUIRED in \
  "$INITIAL" \
  "data/validation/g8r_noise_final_r0_faithful_s3a/report.json" \
  "data/validation/g8r_noise_final_r0_faithful_s3a/training_actions.csv.gz" \
  "$LEDGER/report.json" "$LEDGER/training_actions.csv.gz" \
  "$REPLAY/report.json" "$REPLAY/current_geometry_outcomes.csv.gz"; do
  [[ -f "$REQUIRED" ]] || { echo "Missing frozen PMT input: $REQUIRED" >&2; exit 3; }
done
[[ ! -e "$RUN_ROOT" ]] || { echo "Refusing to overwrite unique PMT run: $RUN_ROOT" >&2; exit 4; }

echo "=== E4-PMT minimal paired transfer started: $(date) ==="
python -m py_compile \
  tasks/noise_final_e4_pmt_core.py \
  tasks/build_noise_final_e4_pmt_manifest.py \
  tasks/train_noise_final_e4a_direct_augmentation.py \
  tasks/summarize_noise_final_e4_pmt.py \
  tasks/validate_noise_final_e4_pmt.py
python -u tasks/audit_noise_final_e4_pmt_implementation.py
python -u tasks/test_noise_final_e4_pmt.py
python -u tasks/test_noise_final_e4a_direct_augmentation.py

python -u tasks/build_noise_final_e4_pmt_manifest.py \
  --ledger-dir "$LEDGER" --replay-dir "$REPLAY" \
  --initial-checkpoint "$INITIAL" --outer-fold "$FOLD" \
  --advantage-threshold 0.01 --output-dir "$MANIFEST"
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
)

# One executable-path smoke test before the four formal arms.
python -u tasks/train_noise_final_e4a_direct_augmentation.py \
  --output-root "$RUN_ROOT/smoke" --run-suffix "run_${SLURM_JOB_ID}" \
  --pmt-arm paired_target --pmt-alpha 0.25 "${COMMON[@]}" --smoke

LABELS=(clean_duplicate matched_random alpha025 alpha050)
PMT_ARMS=(clean_duplicate matched_random paired_target paired_target)
ALPHAS=(0 0 0.25 0.50)
for INDEX in 0 1 2 3; do
  LABEL="${LABELS[$INDEX]}"
  ARM="${PMT_ARMS[$INDEX]}"
  ALPHA="${ALPHAS[$INDEX]}"
  echo "=== E4-PMT arm=$LABEL started: $(date) ==="
  python -u tasks/train_noise_final_e4a_direct_augmentation.py \
    --output-root "$ARMS_ROOT/$LABEL" --run-suffix "run_${SLURM_JOB_ID}" \
    --pmt-arm "$ARM" --pmt-alpha "$ALPHA" "${COMMON[@]}"
  RESULT_COUNT=$(find "$ARMS_ROOT/$LABEL" -name decision.json -type f | wc -l)
  [[ "$RESULT_COUNT" -eq 1 ]] || { echo "Expected one $LABEL decision, found $RESULT_COUNT" >&2; exit 5; }
  RESULT=$(dirname "$(find "$ARMS_ROOT/$LABEL" -name decision.json -type f)")
  python -u tasks/validate_noise_final_e4a_direct_augmentation.py --output-dir "$RESULT"
  echo "=== E4-PMT arm=$LABEL completed: $(date) ==="
done

python -u tasks/summarize_noise_final_e4_pmt.py \
  --arms-root "$ARMS_ROOT" --output-dir "$SUMMARY" \
  --bootstrap-resamples 10000 --seed "$SEED"
python -u tasks/validate_noise_final_e4_pmt.py --output-dir "$SUMMARY" --summary
echo "=== E4-PMT minimal paired transfer completed: $(date) ==="
