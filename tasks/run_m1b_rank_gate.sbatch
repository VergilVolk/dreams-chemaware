#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=m1b_rank_gate_%j.out
#SBATCH --error=m1b_rank_gate_%j.err

# M1b is a single locked configuration answering one question: can real
# cross-condition positives + real hard negatives improve LOCAL ranking under a
# head-only linear map?  No synthetic noise, no backbone unfreeze, no sweep.
# Do NOT extend this to an array or change hyperparameters until the locked
# rank gate in eval_m1b_rank_gate.py passes.
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

TRAIN="tasks/massspecgym_isomers/g8r_locked/train.json"
VAL="tasks/massspecgym_isomers/g8r_locked/val.json"
OUT="data/validation/m1b_rank"
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
# M1b hard stream: train must actually carry hard negatives (else nothing to rank).
assert ta['hard_negative_anchor_fraction'] >= 0.3, ta
assert ta['n_hard_negative_pairs'] >= 3000, ta
print('M1b locked data OK:', ta, va)
PY

echo "=== M1b balanced local-ranking train (head-only, no noise): $(date) ==="
python -u tasks/step4_m1b_train.py \
  --device cuda --no-resume \
  --seed "$SEED" --epochs 2 --batch-size 32 --lr 1e-5 --unfreeze-layers 0 \
  --margin 0.05 --tau 0.1 --lam-posfloor 1.0 --lam-preserve 5.0 \
  --train-subset "$TRAIN" \
  --output-dir "$OUT"

CKPT="$OUT/seed_${SEED}/final.pt"
[[ -f "$CKPT" ]] || { echo "Missing checkpoint: $CKPT" >&2; exit 3; }

echo "=== M1b rank gate eval: $(date) ==="
python -u tasks/eval_m1b_rank_gate.py \
  --device cuda --trained "$CKPT" --val-subset "$VAL" \
  --bootstrap 2000 --seed "$SEED" \
  --output "$OUT/seed_${SEED}/m1b_rank_gate.json"

echo "=== M1b completed: $(date) ==="
echo "Read $OUT/seed_${SEED}/m1b_rank_gate.json (gates.pass) before any noise ablation."
