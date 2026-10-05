#!/bin/bash
#SBATCH --job-name=noise_e4_np_f0
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --output=data/validation/noise_e4_np_f0_%j.out
#SBATCH --error=data/validation/noise_e4_np_f0_%j.err

set -euo pipefail
echo "FAIL-CLOSED: superseded before submission; incomplete N plus fixed-P1 action set." >&2
echo "Use the forthcoming corrected full-action low-loss entrypoint instead." >&2
exit 90

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FOLD=0
SEED=20260906
GRAPH="data/validation/noise_corrected_candidate_graph_v1_20260906"
SOURCE_MANIFEST="data/validation/chemaware_corrected_candidate_manifest_v1"
DATA="data/models/MassSpecGym_MurckoHist_split.hdf5"
OFFICIAL="data/e1/official_embedding_slim.pt"
ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
INITIAL="data/validation/g8r_noise_final_e4a_direct/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_highlr_multifold/seed_20260830/fold_${FOLD}/final_shared_encoder.pt"
FINAL_ROOT="data/validation/noise_corrected_e4_np_fold_${FOLD}_run_${SLURM_JOB_ID}"
RUN_ROOT="data/validation/.noise_corrected_e4_np_fold_${FOLD}_run_${SLURM_JOB_ID}.staging"
N_BANK="$RUN_ROOT/actions_n"
P_BANK="$RUN_ROOT/actions_p"
NP_BANK="$RUN_ROOT/actions_np"
ARMS="$RUN_ROOT/arms"
SUMMARY="$RUN_ROOT/summary"

for REQUIRED in \
  "$GRAPH/candidate_graph.npz" "$GRAPH/official_embeddings.npz" "$GRAPH/report.json" \
  "$SOURCE_MANIFEST/manifest.npz" "$DATA" "$OFFICIAL" "$ARCHITECTURE" \
  "$INITIAL" "$(dirname "$INITIAL")/decision.json"; do
  [[ -f "$REQUIRED" ]] || { echo "Missing frozen corrected-E4 input: $REQUIRED" >&2; exit 3; }
done
[[ ! -e "$RUN_ROOT" && ! -e "$FINAL_ROOT" ]] || {
  echo "Refusing to overwrite unique atomic run: $RUN_ROOT or $FINAL_ROOT" >&2; exit 4;
}
mkdir -p "$RUN_ROOT"

echo "=== corrected E4 N+P fold-0 run started: $(date) ==="
python -m py_compile \
  tasks/noise_corrected_fullgraph_evaluation.py \
  tasks/noise_final_direct_boundary_v2_core.py \
  tasks/build_noise_corrected_full_action_bank.py \
  tasks/build_noise_corrected_fixed_p_action_bank.py \
  tasks/combine_noise_corrected_np_action_banks.py \
  tasks/train_noise_corrected_target_direct.py \
  tasks/summarize_noise_corrected_target_direct_arms.py
python -u tasks/test_noise_corrected_fullgraph_evaluation.py
python -u tasks/test_noise_corrected_full_action_bank.py
python -u tasks/test_noise_corrected_fixed_p_action_bank.py
python -u tasks/test_combine_noise_corrected_np_action_banks.py
python -u tasks/test_noise_final_direct_boundary_v2_core.py

# Build every outcome-free mature N prefix in the current E4 geometry.
python -u tasks/build_noise_corrected_full_action_bank.py \
  --graph-dir "$GRAPH" --data "$DATA" \
  --official-checkpoint "$OFFICIAL" --architecture-checkpoint "$ARCHITECTURE" \
  --initial-student-checkpoint "$INITIAL" --outer-fold "$FOLD" \
  --formula-fold-seed 20260825 --max-queries 0 --device cuda --amp \
  --output-dir "$N_BANK"

# Build the frozen fixed-P1 view for all 65,286 outer-train queries.  The
# corrected source manifest is used only for query adduct evaluation metadata.
python -u tasks/build_noise_corrected_fixed_p_action_bank.py \
  --graph-dir "$GRAPH" --data "$DATA" \
  --official-checkpoint "$OFFICIAL" --architecture-checkpoint "$ARCHITECTURE" \
  --initial-student-checkpoint "$INITIAL" --outer-fold "$FOLD" \
  --formula-fold-seed 20260825 --max-queries 0 --device cuda --no-amp \
  --output-dir "$P_BANK"

python -u tasks/combine_noise_corrected_np_action_banks.py \
  --graph-dir "$GRAPH" --n-action-bank-dir "$N_BANK" \
  --p-action-bank-dir "$P_BANK" --output-dir "$NP_BANK"

COMMON=(
  --graph-dir "$GRAPH" --source-manifest-dir "$SOURCE_MANIFEST" --data "$DATA"
  --official-checkpoint "$OFFICIAL" --architecture-checkpoint "$ARCHITECTURE"
  --initial-student-checkpoint "$INITIAL" --outer-fold "$FOLD"
  --formula-fold-seed 20260825 --seed "$SEED" --epochs 4 --batch-queries 4
  --positive-references 3 --negative-molecules 8 --references-per-negative 2
  --unfreeze-blocks 1 --backbone-lr 2e-6 --head-lr 1e-5 --weight-decay 1e-4
  --rank-margin 0.05 --temperature 0.10 --topk-negatives 8
  --margin-floor-slack 0.005 --lambda-preserve 2.0 --grad-clip 1.0
  --target-preclip-gradient-norm 0.5 --calibration-batches 8
  --transfer-scale-cap 8.0 --eval-batch-size 128 --bootstrap-resamples 5000
  --device cuda --no-amp
)

# Same E4, query set, clean dose, epoch order, optimizer and seed.  P supplies
# complete outer-train query coverage to all three arms; the control has every
# action-derived coefficient exactly zero.
python -u tasks/train_noise_corrected_target_direct.py \
  "${COMMON[@]}" --action-bank-dir "$P_BANK" --output-dir "$ARMS/clean_control" \
  --lambda-clean 1.25 --lambda-action-conditioned-clean 0 \
  --lambda-action-rank 0 --lambda-action-safety 0 --lambda-action-to-clean 0 \
  --target-transfer-to-clean-gradient-ratio 0

python -u tasks/train_noise_corrected_target_direct.py \
  "${COMMON[@]}" --action-bank-dir "$P_BANK" --output-dir "$ARMS/p_direct" \
  --lambda-clean 0.25 --lambda-action-conditioned-clean 1.0 \
  --lambda-action-rank 0.5 --lambda-action-safety 0.25 --lambda-action-to-clean 0.5 \
  --target-transfer-to-clean-gradient-ratio 0.25

python -u tasks/train_noise_corrected_target_direct.py \
  "${COMMON[@]}" --action-bank-dir "$NP_BANK" --output-dir "$ARMS/np_direct" \
  --lambda-clean 0.25 --lambda-action-conditioned-clean 1.0 \
  --lambda-action-rank 0.5 --lambda-action-safety 0.25 --lambda-action-to-clean 0.5 \
  --target-transfer-to-clean-gradient-ratio 0.25

python -u tasks/summarize_noise_corrected_target_direct_arms.py \
  --arms-root "$ARMS" --output-dir "$SUMMARY" \
  --bootstrap-resamples 10000 --seed "$SEED"

mv "$RUN_ROOT" "$FINAL_ROOT"
echo "=== corrected E4 N+P fold-0 run completed: $(date) ==="
