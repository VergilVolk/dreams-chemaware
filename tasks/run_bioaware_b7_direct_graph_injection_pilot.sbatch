#!/bin/bash
#SBATCH --job-name=bio_b7_direct
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=bioaware_b7_direct_%j.out
#SBATCH --error=bioaware_b7_direct_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MANIFEST_DIR=data/validation/bioaware_b4_direct_shared_embedding_v1_20260905/manifest
CANDIDATES=data/validation/bioaware_b3_reaction_coabundance_local_20260906/candidate_features.csv.gz
RUN_ROOT="data/validation/bioaware_b7_direct_graph_injection_${SLURM_JOB_ID}"
WORK="${RUN_ROOT}.partial"
ROUTER="$WORK/action_router"
GRADIENT="$WORK/direct_gradient_gate"
MODEL="$WORK/fold_0_direct"

test ! -e "$RUN_ROOT" || { echo "Refusing to overwrite: $RUN_ROOT" >&2; exit 3; }
test ! -e "$WORK" || { echo "Refusing to overwrite: $WORK" >&2; exit 3; }
required=(
  "$MANIFEST_DIR/manifest.npz"
  "$MANIFEST_DIR/report.json"
  "$CANDIDATES"
  data/e1/official_embedding_slim.pt
  dreams/models/pretrained/ssl_model_server.pt
  annotation/bioaware_supervision.py
  tasks/build_bioaware_b4_direct_manifest.py
  tasks/develop_bioaware_b1_multisource_action.py
  tasks/build_bioaware_b7_graph_action_router.py
  tasks/audit_bioaware_b7_direct_gradient.py
  tasks/train_bioaware_b4_direct_shared_embedding.py
  tasks/test_bioaware_b7_direct_pipeline.py
)
for path in "${required[@]}"; do
  test -s "$path" || { echo "Missing dependency: $path" >&2; exit 4; }
done

python -c "import numpy,pandas,scipy,sklearn,torch; assert torch.cuda.is_available(); print('[BioAware B7 dependencies] PASS')"
python -c "import tasks.build_bioaware_b7_graph_action_router, tasks.audit_bioaware_b7_direct_gradient, tasks.train_bioaware_b4_direct_shared_embedding; print('[BioAware B7 imports] PASS')"
python -m py_compile \
  tasks/build_bioaware_b7_graph_action_router.py \
  tasks/audit_bioaware_b7_direct_gradient.py \
  tasks/train_bioaware_b4_direct_shared_embedding.py \
  tasks/test_bioaware_b7_direct_pipeline.py
python -u tasks/test_bioaware_b7_direct_pipeline.py

echo "=== BioAware B7 direct graph action freeze started: $(date) ==="
mkdir -p "$WORK"
python -u tasks/build_bioaware_b7_graph_action_router.py \
  --candidate-features "$CANDIDATES" \
  --b4-manifest "$MANIFEST_DIR/manifest.npz" \
  --output-dir "$ROUTER"
python -c "import json; r=json.load(open('$ROUTER/report.json')); assert r['pass_to_direct_gradient_gate']; assert r['overall_repeated_outer_views']['delta_recall1'] >= 0.03; print('[BioAware B7 action headroom] PASS', r['overall_repeated_outer_views'])"

echo "=== BioAware B7 direct gradient gate started: $(date) ==="
python -u tasks/audit_bioaware_b7_direct_gradient.py \
  --b4-manifest "$MANIFEST_DIR/manifest.npz" \
  --action-router-dir "$ROUTER" \
  --output-dir "$GRADIENT" \
  --outer-fold 0 \
  --audit-queries 8 \
  --official-checkpoint data/e1/official_embedding_slim.pt \
  --architecture-checkpoint dreams/models/pretrained/ssl_model_server.pt \
  --backbone-lr 2e-6 \
  --head-lr 1e-5 \
  --device cuda
python -c "import json; r=json.load(open('$GRADIENT/report.json')); assert r['pass_to_direct_shared_embedding_pilot']; print('[BioAware B7 direct gradient] PASS', r['exact_baseline_wrong_margin_delta'])"

echo "=== BioAware B7 direct shared-embedding fold-0 pilot started: $(date) ==="
python -u tasks/train_bioaware_b4_direct_shared_embedding.py \
  --manifest-dir "$MANIFEST_DIR" \
  --action-router-dir "$ROUTER" \
  --output "$MODEL" \
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

for path in "$MODEL/report.json" "$MODEL/held_per_query.csv.gz" "$MODEL/final_trainable_state.pt"; do
  test -s "$path" || { echo "Missing B7 result: $path" >&2; exit 5; }
done
python -c "import json,math; r=json.load(open('$MODEL/report.json')); s=r['shared_embedding']; assert r['arm']=='graph_prior_direct'; assert r['supervision_mode']=='direct_onehot'; assert r['official_replay']['rank_mismatches']==0; assert all(math.isfinite(float(v)) for v in (s['recall1'],s['mrr'],s['preservation'])); print('[BioAware B7 fold-0 result]', s)"

mv "$WORK" "$RUN_ROOT"
echo "=== BioAware B7 direct graph injection completed: $(date) ==="
echo "Result: $RUN_ROOT"
