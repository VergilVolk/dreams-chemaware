#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=m1b_rank_gate_v2_%j.out
#SBATCH --error=m1b_rank_gate_v2_%j.err

# M1b-v2: single-variable replication of M1b, changing ONLY Issue A
# (hard-stream preservation).  Everything else is locked identical:
#   same data, same seed, same lr, same margin, lam_p=5, 2 epochs, no noise,
#   frozen backbone.  It answers one question: does the missing hard-subspace
#   preservation cause the hard-negative cosine to rise together with the
#   positive?  Do NOT sweep, do NOT unfreeze, until eval_m1b_rank_gate.py gates.pass.
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

TRAIN="tasks/massspecgym_isomers/g8r_locked/train.json"
VAL="tasks/massspecgym_isomers/g8r_locked/val.json"
OUT="data/validation/m1b_rank_v2"
SEED=20260821

for f in "$TRAIN" "$VAL" data/models/MassSpecGym_MurckoHist_split.hdf5 \
  data/e1/official_embedding_slim.pt dreams/models/pretrained/ssl_model_server.pt \
  tasks/step4_m1b_train.py tasks/eval_m1b_rank_gate.py; do
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
assert ta['hard_negative_anchor_fraction'] >= 0.3, ta
assert ta['n_hard_negative_pairs'] >= 3000, ta
print('M1b-v2 locked data OK:', ta, va)
PY

echo "=== M1b-v2 balanced local-ranking train (head-only, hard+global preserve, no noise): $(date) ==="
python -u tasks/step4_m1b_train.py \
  --device cuda --no-resume \
  --seed "$SEED" --epochs 2 --batch-size 32 --lr 1e-5 --unfreeze-layers 0 \
  --margin 0.05 --tau 0.1 --lam-posfloor 1.0 --lam-preserve 5.0 \
  --train-subset "$TRAIN" \
  --output-dir "$OUT"

CKPT="$OUT/seed_${SEED}/final.pt"
[[ -f "$CKPT" ]] || { echo "Missing checkpoint: $CKPT" >&2; exit 3; }

echo "=== M1b-v2 rank gate eval: $(date) ==="
python -u tasks/eval_m1b_rank_gate.py \
  --device cuda --trained "$CKPT" --val-subset "$VAL" \
  --bootstrap 2000 --seed "$SEED" \
  --output "$OUT/seed_${SEED}/m1b_rank_gate.json"

echo "=== M1b-v2 completed: $(date) ==="
echo "Read $OUT/seed_${SEED}/m1b_rank_gate.json (gates.pass) before any noise ablation or unfreeze."
