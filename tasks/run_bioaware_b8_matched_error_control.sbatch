#!/bin/bash
#SBATCH --job-name=bio_b8_match
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=bioaware_b8_matched_%j.out
#SBATCH --error=bioaware_b8_matched_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

B7_ROUTER=data/validation/bioaware_b7_direct_graph_injection_2331964.partial/action_router
MANIFEST=data/validation/bioaware_b4_direct_shared_embedding_v1_20260905/manifest
OUTPUT="data/validation/bioaware_b8_matched_error_control_${SLURM_JOB_ID}"
ROUTER="$OUTPUT/router"
GRAPH="$OUTPUT/matched_graph_fold0"
GENERIC="$OUTPUT/matched_generic_fold0"
COMPARISON="$OUTPUT/comparison.json"

test ! -e "$OUTPUT" || { echo "Refusing to overwrite: $OUTPUT" >&2; exit 3; }
required=(
  "$B7_ROUTER/report.json"
  "$B7_ROUTER/action_routes.csv.gz"
  "$MANIFEST/manifest.npz"
  "$MANIFEST/report.json"
  data/e1/official_embedding_slim.pt
  dreams/models/pretrained/ssl_model_server.pt
  tasks/build_bioaware_b8_matched_error_routers.py
  tasks/train_bioaware_b4_direct_shared_embedding.py
  tasks/compare_bioaware_b8_matched_fold0.py
  tasks/test_bioaware_b8_matched_error_control.py
  tasks/test_bioaware_b7_direct_pipeline.py
)
for path in "${required[@]}"; do
  test -s "$path" || { echo "Missing B8 dependency: $path" >&2; exit 4; }
done

python -m py_compile \
  tasks/build_bioaware_b8_matched_error_routers.py \
  tasks/train_bioaware_b4_direct_shared_embedding.py \
  tasks/compare_bioaware_b8_matched_fold0.py \
  tasks/test_bioaware_b8_matched_error_control.py
python -u tasks/test_bioaware_b7_direct_pipeline.py
python -u tasks/test_bioaware_b8_matched_error_control.py

echo "=== BioAware B8 matched-router construction started: $(date) ==="
python -u tasks/build_bioaware_b8_matched_error_routers.py \
  --b7-router-dir "$B7_ROUTER" \
  --manifest-dir "$MANIFEST" \
  --output-dir "$ROUTER"
if ! python -c "import json,sys; r=json.load(open('$ROUTER/report.json')); ok=bool(r['pass_to_fold0_training']); print('[B8 matched router fold 0]', 'PASS' if ok else 'NOT_IDENTIFIABLE', r['outer_folds']['0']); sys.exit(0 if ok else 2)"; then
  echo "[B8 scientific stop] Fold-0 common support is insufficient; no GPU training was run."
  echo "Result: $ROUTER/report.json"
  exit 0
fi

common=(
  --manifest-dir "$MANIFEST"
  --action-router-dir "$ROUTER"
  --supervision-mode direct_onehot
  --outer-fold 0
  --epochs 4
  --steps-per-epoch 128
  --batch-queries 8
  --references-per-molecule 1
  --corrective-fraction 0.5
  --unfreeze-blocks 1
  --backbone-lr 2e-6
  --head-lr 1e-5
  --temperature 0.10
  --rank-margin 0.05
  --lambda-listwise 0.5
  --lambda-corrective 1.0
  --lambda-safety-floor 2.0
  --lambda-preserve 5.0
  --harm-safety-weight 4.0
  --no-amp
  --device cuda
)

echo "=== BioAware B8 matched graph fold-0 started: $(date) ==="
python -u tasks/train_bioaware_b4_direct_shared_embedding.py \
  "${common[@]}" \
  --output "$GRAPH" \
  --arm matched_graph_direct

echo "=== BioAware B8 matched generic fold-0 started: $(date) ==="
python -u tasks/train_bioaware_b4_direct_shared_embedding.py \
  "${common[@]}" \
  --output "$GENERIC" \
  --arm matched_generic_direct

for path in \
  "$GRAPH/report.json" "$GRAPH/held_per_query.csv.gz" "$GRAPH/final_trainable_state.pt" \
  "$GENERIC/report.json" "$GENERIC/held_per_query.csv.gz" "$GENERIC/final_trainable_state.pt"; do
  test -s "$path" || { echo "Missing B8 result: $path" >&2; exit 5; }
done

echo "=== BioAware B8 matched fold-0 comparison started: $(date) ==="
python -u tasks/compare_bioaware_b8_matched_fold0.py \
  --router-dir "$ROUTER" \
  --graph-dir "$GRAPH" \
  --generic-dir "$GENERIC" \
  --output "$COMPARISON"
python -c "import json; r=json.load(open('$COMPARISON')); print('[B8 final decision]', {'comparison':r['comparison'], 'gates':r['gates'], 'pass_to_remaining_formula_folds':r['pass_to_remaining_formula_folds']})"

echo "=== BioAware B8 matched-error control completed: $(date) ==="
echo "Result: $OUTPUT"
