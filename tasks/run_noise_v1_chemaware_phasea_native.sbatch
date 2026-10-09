#!/bin/bash
# Exact ChemAware Phase-A reproduction with one and only one intervention:
# initialize the native DreaMS model from the best Noise V1 shared checkpoint.
#SBATCH --job-name=noise_chem_phasea
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/noise_chem_phasea_%j.out
#SBATCH --error=data/validation/noise_chem_phasea_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

OFFICIAL="data/e1/official_embedding_slim.pt"
NOISE="data/validation/noise_relation_t1_t3_run_2347055/checkpoint/primary_seed_3407_slim.pt"
ARCH="dreams/models/pretrained/ssl_model_server.pt"
BASE="data/validation/chemaware_high_coverage_native/run_2340524"
DATA="data/models/MassSpecGym_MurckoHist_split.hdf5"
MANIFEST="data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz"
TOKEN_ROWS="data/validation/chemaware_corrected_manifest_tokens_v1/rows.npy"
OFFICIAL_EMBEDDINGS="data/validation/chemaware_corrected_manifest_tokens_v1/official_embeddings_f32.npy"
REPLAY="data/e1/e1_train_triplet_pool_10ppm.npz"
BENCHMARK="data/validation/gnps_gold_silver_10ppm_benchmark_v1"
BASE_SCORES="data/validation/noise_gnps_article_benchmark_run_2349091/method_scores.npz"
OUT="data/validation/noise_v1_chemaware_phasea_exact/run_${SLURM_JOB_ID}"
TRIPLETS="$OUT/triplets"
TRAINING="$OUT/training"
SELECTED="$TRAINING/step-002000.ckpt"

[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }
mkdir -p "$OUT"

required_files=(
  "$OFFICIAL" "$NOISE" "$ARCH" "$DATA" "$MANIFEST"
  "$TOKEN_ROWS" "$OFFICIAL_EMBEDDINGS" "$REPLAY"
  "$BASE/triplets/report.json"
  "$BASE/triplets/train_pool.npz"
  "$BASE/triplets/val_pool.npz"
  "$BASE/evidence/train_triplet_evidence.npz"
  "$BASE/evidence/selection_triplet_evidence.npz"
  "$BASE/evidence/confirmation_triplet_evidence.npz"
  "$BENCHMARK/spectra.mgf"
  "$BENCHMARK/manifest.csv.gz"
  "$BENCHMARK/panel_identity_disjoint.npz"
  "$BENCHMARK/panel_formula_disjoint.npz"
  "$BASE_SCORES"
  tasks/build_chemaware_max_boundary_native_triplets.py
  tasks/test_chemaware_max_boundary_native_triplets.py
  tasks/train_chemaware_specific_replay_native.py
  tasks/train_chemaware_dreams_native.py
  tasks/evaluate_chemaware_v2_direct_triplet.py
  tasks/prepare_official_embedding_checkpoint.py
  tasks/encode_gnps_gold_silver_10ppm_checkpoint.py
  tasks/extend_gnps_article_score_bundle.py
  tasks/evaluate_noise_gnps_article_benchmark.py
)
for required_file in "${required_files[@]}"; do
  [[ -f "$required_file" ]] || {
    echo "Missing exact Noise->ChemAware input: $required_file" >&2
    exit 2
  }
done

python -u tasks/test_chemaware_max_boundary_native_triplets.py

# Exact Phase-A triplet construction from the +2.1266 pp run. Inputs,
# parameters, and official-DreaMS mining geometry are unchanged.
python -u tasks/build_chemaware_max_boundary_native_triplets.py \
  --base-bank "$BASE/triplets" \
  --evidence "$BASE/evidence/train_triplet_evidence.npz" \
  --manifest "$MANIFEST" \
  --embedding-rows "$TOKEN_ROWS" \
  --official-embeddings "$OFFICIAL_EMBEDDINGS" \
  --dreams-replay-pool "$REPLAY" \
  --data "$DATA" \
  --output "$TRIPLETS" \
  --margin 0.1 --negative-references-per-error 2 \
  --chemical-candidates-per-error 2 --dreams-replay-events 1024 \
  --seed 3407 --min-error-event-fraction 0.20

# Record the starting point. This evaluation does not select training.
srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$BASE/evidence/selection_triplet_evidence.npz" \
  --checkpoint "official=$OFFICIAL" \
  --checkpoint "noise_v1=$NOISE" \
  --paired-reference noise_v1 --formula-role 2 \
  --output "$OUT/role2_pretraining_baselines.json" \
  --batch-size 64 --bootstrap-draws 2000 --device cuda

# Exact native Phase-A runtime and fixed 2,000-step endpoint. The sole changed
# experimental variable is initialization: Noise V1 replaces official DreaMS.
srun --export=ALL --preserve-env python -u tasks/train_chemaware_specific_replay_native.py \
  --data "$DATA" \
  --train-pool "$TRIPLETS/train_pool.npz" \
  --val-pool "$TRIPLETS/val_pool.npz" \
  --official-checkpoint "$NOISE" \
  --architecture-checkpoint "$ARCH" \
  --output "$TRAINING" \
  --seed 3407 --lr 5e-6 --weight-decay 0 \
  --triplet-loss-margin 0.1 --batch-size 4 --num-workers 0 \
  --max-epochs 3 --max-steps 2000 --checkpoint-mode fixed_steps \
  --save-every-n-steps 2000 --n-highest-peaks 100 --device cuda

test -f "$SELECTED"

# Role 2/3 and GNPS are consumed development evaluations. No checkpoint is
# selected from them: the predeclared Phase-A endpoint is always step 2,000.
srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$BASE/evidence/selection_triplet_evidence.npz" \
  --checkpoint "official=$OFFICIAL" \
  --checkpoint "noise_v1=$NOISE" \
  --checkpoint "noise_chemaware_step2000=$SELECTED" \
  --paired-reference noise_v1 --formula-role 2 \
  --output "$OUT/role2_evaluation.json" \
  --batch-size 64 --bootstrap-draws 5000 --device cuda

srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$BASE/evidence/confirmation_triplet_evidence.npz" \
  --checkpoint "official=$OFFICIAL" \
  --checkpoint "noise_v1=$NOISE" \
  --checkpoint "noise_chemaware_step2000=$SELECTED" \
  --paired-reference noise_v1 --formula-role 3 \
  --output "$OUT/role3_evaluation.json" \
  --batch-size 64 --bootstrap-draws 5000 --device cuda

SELECTED_SLIM="$OUT/noise_chemaware_step2000_slim.pt"
python -u tasks/prepare_official_embedding_checkpoint.py \
  --source "$SELECTED" --output "$SELECTED_SLIM"

# Reuse the frozen Noise/WSE/P2b GNPS bundle unchanged and append only the new
# embedding, so all established downstream scores remain byte-for-byte intact.
srun --export=ALL --preserve-env python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
  --benchmark "$BENCHMARK" \
  --checkpoint "$SELECTED_SLIM" \
  --architecture-checkpoint "$ARCH" \
  --output "$OUT/noise_chemaware_gnps_embeddings.npz" \
  --batch-size 64 --label noise-chemaware-step2000

python -u tasks/extend_gnps_article_score_bundle.py \
  --benchmark "$BENCHMARK" \
  --base-bundle "$BASE_SCORES" \
  --embedding-method "noise_chemaware_step2000=$OUT/noise_chemaware_gnps_embeddings.npz" \
  --output "$OUT/gnps_extended_method_scores.npz"

python -u tasks/evaluate_noise_gnps_article_benchmark.py \
  --benchmark "$BENCHMARK" \
  --score-bundle "$OUT/gnps_extended_method_scores.npz" \
  --output "$OUT/gnps_evaluation" \
  --baseline-method noise_v1 \
  --bootstrap-resamples 10000 \
  --bootstrap-seed 20261009

test -f "$TRIPLETS/report.json"
test -f "$TRAINING/report.json"
test -f "$OUT/role2_evaluation.json"
test -f "$OUT/role3_evaluation.json"
test -f "$OUT/gnps_evaluation/report.json"
echo "NOISE_V1_CHEMAWARE_PHASEA_EXACT_COMPLETE: $OUT"
