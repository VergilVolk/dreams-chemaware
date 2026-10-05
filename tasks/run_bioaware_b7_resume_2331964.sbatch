#!/bin/bash
#SBATCH --job-name=bio_b7_resume
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=bioaware_b7_resume_%j.out
#SBATCH --error=bioaware_b7_resume_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SOURCE=data/validation/bioaware_b7_direct_graph_injection_2331964.partial
ROUTER="$SOURCE/action_router"
GRADIENT="$SOURCE/direct_gradient_gate"
MANIFEST_DIR=data/validation/bioaware_b4_direct_shared_embedding_v1_20260905/manifest
FINAL="data/validation/bioaware_b7_direct_graph_injection_2331964_resume_${SLURM_JOB_ID}"

test ! -e "$FINAL" || { echo "Refusing to overwrite: $FINAL" >&2; exit 3; }
required=(
  "$ROUTER/report.json"
  "$ROUTER/action_routes.csv.gz"
  "$ROUTER/candidate_action_scores.csv.gz"
  "$GRADIENT/report.json"
  "$MANIFEST_DIR/manifest.npz"
  "$MANIFEST_DIR/report.json"
  data/e1/official_embedding_slim.pt
  dreams/models/pretrained/ssl_model_server.pt
  tasks/train_bioaware_b4_direct_shared_embedding.py
  tasks/test_bioaware_b7_direct_pipeline.py
)
for path in "${required[@]}"; do
  test -s "$path" || { echo "Missing resume dependency: $path" >&2; exit 4; }
done

python -m py_compile \
  tasks/train_bioaware_b4_direct_shared_embedding.py \
  tasks/test_bioaware_b7_direct_pipeline.py
python -u tasks/test_bioaware_b7_direct_pipeline.py
python -c "import json; r=json.load(open('$ROUTER/report.json')); assert r['pass_to_direct_gradient_gate']; print('[BioAware B7 reused action router] PASS', r['overall_repeated_outer_views'])"
python -c "import json; r=json.load(open('$GRADIENT/report.json')); assert r['pass_to_direct_shared_embedding_pilot']; print('[BioAware B7 reused direct gradient] PASS', r['exact_baseline_wrong_margin_delta'])"

echo "=== BioAware B7 resumed direct shared-embedding fold-0 pilot started: $(date) ==="
python -u tasks/train_bioaware_b4_direct_shared_embedding.py \
  --manifest-dir "$MANIFEST_DIR" \
  --action-router-dir "$ROUTER" \
  --output "$FINAL" \
  --arm graph_prior_direct \
  --supervision-mode direct_onehot \
  --outer-fold 0 \
  --epochs 4 \
  --steps-per-epoch 128 \
  --batch-queries 8 \
  --references-per-molecule 2 \
  --corrective-fraction 0.5 \
  --unfreeze-blocks 1 \
  --backbone-lr 2e-6 \
  --head-lr 1e-5 \
  --temperature 0.10 \
  --rank-margin 0.05 \
  --lambda-listwise 0.5 \
  --lambda-corrective 1.0 \
  --lambda-safety-floor 2.0 \
  --lambda-preserve 5.0 \
  --harm-safety-weight 4.0 \
  --no-amp \
  --device cuda

for path in "$FINAL/report.json" "$FINAL/held_per_query.csv.gz" "$FINAL/final_trainable_state.pt"; do
  test -s "$path" || { echo "Missing resumed B7 result: $path" >&2; exit 5; }
done
cp "$ROUTER/report.json" "$FINAL/action_router_report.json"
cp "$GRADIENT/report.json" "$FINAL/direct_gradient_gate_report.json"
python -c "import json,math; r=json.load(open('$FINAL/report.json')); s=r['shared_embedding']; assert r['arm']=='graph_prior_direct'; assert r['supervision_mode']=='direct_onehot'; assert r['official_replay']['rank_mismatches']==0; assert all(math.isfinite(float(v)) for v in (s['recall1'],s['mrr'],s['preservation'])); print('[BioAware B7 resumed fold-0 result]', s)"
echo "=== BioAware B7 resumed direct graph injection completed: $(date) ==="
echo "Result: $FINAL"
