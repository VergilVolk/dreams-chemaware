#!/bin/bash
#SBATCH --job-name=bio_b4_direct
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=bioaware_b4_direct_%j.out
#SBATCH --error=bioaware_b4_direct_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FINAL=data/validation/bioaware_b4_direct_shared_embedding_v1_20260905
WORK="${FINAL}.partial.${SLURM_JOB_ID}"
MANIFEST="$WORK/manifest"
RUNS="$WORK/runs"

echo "=== BioAware B4 direct shared-embedding pilot started: $(date) ==="
test ! -e "$FINAL" || { echo "Refusing to overwrite frozen result: $FINAL" >&2; exit 3; }
test ! -e "$WORK" || { echo "Refusing to overwrite partial result: $WORK" >&2; exit 3; }

required=(
  data/models/mona_neg_full.mgf
  data/models/mona_neg_dreams_emb/manifest.csv
  data/models/mona_neg_dreams_emb/embeddings.npy
  data/e1/official_embedding_slim.pt
  dreams/models/pretrained/ssl_model_server.pt
  data/validation/bioaware_metdna3_external_negative_dreams_v2_chemically_filtered/queries.csv.gz
  data/validation/bioaware_metdna3_external_negative_dreams_v2_chemically_filtered/query_embeddings.npz
  data/validation/bioaware_metdna3_external_negative_dreams_v2_chemically_filtered/candidate_scores.csv.gz
  data/validation/bioaware_metdna3_external_negative_dreams_v2_chemically_filtered/candidate_references.csv.gz
  data/validation/bioaware_metdna3_external_negative_loso_ablation_v3_chemically_filtered/full_bioaware__transitions.csv.gz
  data/validation/bioaware_metdna3_external_negative_loso_ablation_v3_chemically_filtered/full_no_edge_gate__transitions.csv.gz
  data/validation/bioaware_metdna3_external_negative_loso_ranker_v4_chemically_filtered/candidate_features.csv.gz
  data/validation/mona_negative_library_chemical_integrity_v1/library_row_integrity.csv.gz
  docs/BIOAWARE_MILESTONE_AND_B4_DIRECT_SHARED_EMBEDDING_20260905.md
  tasks/build_bioaware_b4_direct_manifest.py
  tasks/train_bioaware_b4_direct_shared_embedding.py
  tasks/summarize_bioaware_b4_direct_shared_embedding.py
  tasks/validate_bioaware_b4_direct_manifest.py
)
for unit in \
  BV2cell__hilic BV2cell__rplc Mouse_brain__hilic Mouse_brain__rplc \
  Mouse_liver__hilic Mouse_liver__rplc NIST_plasma__hilic NIST_plasma__rplc; do
  required+=(
    "data/validation/bioaware_metdna3_external_negative_units_v2/$unit/queries.csv.gz"
    "data/validation/bioaware_metdna3_external_negative_units_v2/$unit/query_tensors.npz"
  )
done
for path in "${required[@]}"; do
  test -s "$path" || { echo "Missing required dependency: $path" >&2; exit 2; }
done

python -c "import h5py,numpy,pandas,torch; assert torch.cuda.is_available(); print('[BioAware B4 dependencies] PASS')"
python -m py_compile \
  tasks/build_bioaware_b4_direct_manifest.py \
  tasks/train_bioaware_b4_direct_shared_embedding.py \
  tasks/summarize_bioaware_b4_direct_shared_embedding.py \
  tasks/validate_bioaware_b4_direct_manifest.py \
  tasks/test_bioaware_b4_direct_shared_embedding.py
python -u tasks/test_bioaware_b4_direct_shared_embedding.py

mkdir -p "$WORK" "$RUNS"
python -u tasks/build_bioaware_b4_direct_manifest.py --output-dir "$MANIFEST"
python -u tasks/validate_bioaware_b4_direct_manifest.py --manifest-dir "$MANIFEST"

# End-to-end smoke exercises official replay, one optimizer step, shared
# query/reference evaluation and checkpoint serialization before ten full runs.
python -u tasks/train_bioaware_b4_direct_shared_embedding.py \
  --manifest-dir "$MANIFEST" \
  --output "$WORK/smoke" \
  --arm full_bioaware_safe --outer-fold 0 --folds 5 --seed 20260905 \
  --epochs 1 --steps-per-epoch 1 --batch-queries 4 --references-per-molecule 2 \
  --corrective-fraction 0.50 --unfreeze-blocks 1 \
  --backbone-lr 1e-6 --head-lr 5e-6 --weight-decay 1e-4 \
  --temperature 0.10 --rank-margin 0.05 \
  --lambda-listwise 0.5 --lambda-corrective 1.0 \
  --lambda-safety-floor 2.0 --lambda-preserve 5.0 \
  --harm-safety-weight 4.0 --margin-floor-slack 0.005 \
  --grad-clip 1.0 --batch-size 96 --n-highest-peaks 100 \
  --device cuda --no-amp --torch-threads 8 --smoke
test -s "$WORK/smoke/report.json" || { echo "B4 end-to-end smoke failed" >&2; exit 4; }

for arm in full_bioaware_safe full_no_edge_high_recall; do
  for fold in 0 1 2 3 4; do
    output="$RUNS/$arm/fold_$fold"
    echo "=== B4 arm=$arm fold=$fold: $(date) ==="
    python -u tasks/train_bioaware_b4_direct_shared_embedding.py \
      --manifest-dir "$MANIFEST" \
      --output "$output" \
      --arm "$arm" --outer-fold "$fold" --folds 5 --seed 20260905 \
      --epochs 4 --steps-per-epoch 64 --batch-queries 4 --references-per-molecule 2 \
      --corrective-fraction 0.50 --unfreeze-blocks 1 \
      --backbone-lr 1e-6 --head-lr 5e-6 --weight-decay 1e-4 \
      --temperature 0.10 --rank-margin 0.05 \
      --lambda-listwise 0.5 --lambda-corrective 1.0 \
      --lambda-safety-floor 2.0 --lambda-preserve 5.0 \
      --harm-safety-weight 4.0 --margin-floor-slack 0.005 \
      --grad-clip 1.0 --batch-size 96 --n-highest-peaks 100 \
      --device cuda --no-amp --torch-threads 8
  done
done

python -u tasks/summarize_bioaware_b4_direct_shared_embedding.py \
  --input-root "$RUNS" --manifest-dir "$MANIFEST" \
  --folds 5 --bootstrap-resamples 10000 --seed 20260905
test -s "$RUNS/summary.json" || { echo "Missing pooled B4 summary" >&2; exit 5; }

mv "$WORK" "$FINAL"
echo "=== BioAware B4 direct shared-embedding pilot completed: $(date) ==="
echo "Result: $FINAL/runs/summary.json"
