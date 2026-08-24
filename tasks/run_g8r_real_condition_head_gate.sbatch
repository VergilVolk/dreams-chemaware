#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=g8r_real_condition_head_%j.out
#SBATCH --error=g8r_real_condition_head_%j.err

# G8R is a one-configuration, head-only scientific gate.  It is not a sweep.
# Do not turn this into an array until this exact configuration passes the
# locked inner gate and an external biological annotation gate.
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

TRAIN="tasks/massspecgym_isomers/g8r_locked/train.json"
VAL="tasks/massspecgym_isomers/g8r_locked/val.json"
OUT="data/validation/g8r_real_condition_head"
SEED=20260821

for f in "$TRAIN" "$VAL" data/models/MassSpecGym_MurckoHist_split.hdf5 \
  data/e1/official_embedding_slim.pt dreams/models/pretrained/ssl_model_server.pt \
  tasks/step4_infonce_train.py tasks/eval_g8r_inner_gate.py; do
  [[ -f "$f" ]] || { echo "MISSING: $f" >&2; exit 2; }
done

python - <<'PY'
import json
t=json.load(open('tasks/massspecgym_isomers/g8r_locked/train.json'))
v=json.load(open('tasks/massspecgym_isomers/g8r_locked/val.json'))
ta, va = t['meta']['audit'], v['meta']['audit']
assert ta['n_ik14'] >= 3000 and va['n_ik14'] >= 500
assert ta['cross_condition_anchor_fraction'] == 1.0
assert va['cross_condition_anchor_fraction'] == 1.0
assert {e['ik14'] for e in t['entries']}.isdisjoint({e['ik14'] for e in v['entries']})
print('G8R locked data OK:', ta, va)
PY

echo "=== G8R real-condition head gate: $(date) ==="
python -u tasks/step4_infonce_train.py \
  --device cuda --no-amp --no-resume \
  --seed "$SEED" --epochs 2 --batch-size 32 --lr 1e-5 --unfreeze-layers 0 \
  --train-subset "$TRAIN" \
  --same-mol-as-positive --cross-condition-positive \
  --inbatch-negative-mode hard_only \
  --synthetic-positive-weight 0.10 --real-positive-weight 0.25 \
  --preserve-lambda 5.0 --positive-relation-preserve-lambda 1.0 \
  --protect-main-peaks-above 0.20 --disable-random-added-peaks \
  --output-dir "$OUT"

CKPT="$OUT/seed_${SEED}/best_infonce.pt"
[[ -f "$CKPT" ]] || { echo "Missing checkpoint: $CKPT" >&2; exit 3; }

python -u tasks/eval_g8r_inner_gate.py \
  --device cuda --trained "$CKPT" --val-subset "$VAL" \
  --output "$OUT/seed_${SEED}/g8r_inner_gate.json"

echo "=== G8R completed: $(date) ==="
echo "Read $OUT/seed_${SEED}/g8r_inner_gate.json before any further run."
