#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=g8r_eval_only_%j.out
#SBATCH --error=g8r_eval_only_%j.err

# Re-run the (now M1-complete) inner gate evaluator on the EXISTING G8R
# checkpoint.  No training.  This overwrites the old 3-gate g8r_inner_gate.json
# with the six-criteria + bootstrap + full hard-negative result.
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

CKPT="data/validation/g8r_real_condition_head/seed_20260821/best_infonce.pt"
VAL="tasks/massspecgym_isomers/g8r_locked/val.json"
OUT="data/validation/g8r_real_condition_head/seed_20260821/g8r_inner_gate.json"

for f in "$CKPT" "$VAL" tasks/eval_g8r_inner_gate.py \
  data/models/MassSpecGym_MurckoHist_split.hdf5 \
  data/e1/official_embedding_slim.pt dreams/models/pretrained/ssl_model_server.pt; do
  [[ -f "$f" ]] || { echo "MISSING: $f" >&2; exit 2; }
done

python - <<'PY'
import json
v = json.load(open('tasks/massspecgym_isomers/g8r_locked/val.json'))
a = v['meta']['audit']
print('G8R val audit:', a)
assert a['n_ik14'] >= 500
assert a['cross_condition_anchor_fraction'] == 1.0
PY

echo "=== G8R eval-only (M1 gate) on existing checkpoint: $(date) ==="
python -u tasks/eval_g8r_inner_gate.py \
  --device cuda \
  --trained "$CKPT" \
  --val-subset "$VAL" \
  --output "$OUT"

echo "=== G8R eval completed: $(date) ==="
echo "Read $OUT before any further run."
