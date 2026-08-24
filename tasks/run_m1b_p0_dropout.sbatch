#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=m1b_p0_dropout_%j.out
#SBATCH --error=m1b_p0_dropout_%j.err

# P0: dropout single-variable control.  Same data / optimizer / batch / lr /
# epochs / margin / lambdas / seed, changing ONLY the frozen-backbone mode:
#   dropout_on  = model.train()  (historical; frozen backbone dropout ON)
#   dropout_off = backbone.eval(), head.train()  (--backbone-eval)
# 3 seeds each.  Read both m1b_rank_gate.json files per seed before any further
# conclusion.  Do NOT sweep other hyperparameters here.
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

TRAIN="tasks/massspecgym_isomers/g8r_locked/train.json"
VAL="tasks/massspecgym_isomers/g8r_locked/val.json"
OUT="data/validation/m1b_p0_dropout"

for f in "$TRAIN" "$VAL" data/models/MassSpecGym_MurckoHist_split.hdf5 \
  data/e1/official_embedding_slim.pt dreams/models/pretrained/ssl_model_server.pt \
  tasks/step4_m1b_train.py tasks/eval_m1b_rank_gate.py; do
  [[ -f "$f" ]] || { echo "MISSING: $f" >&2; exit 2; }
done

for SEED in 20260821 20260822 20260823; do
  for MODE in dropout_on dropout_off; do
    EXTRA=""
    [ "$MODE" = "dropout_off" ] && EXTRA="--backbone-eval"
    D="$OUT/$MODE"
    echo "=== P0 $MODE seed=$SEED: $(date) ==="
    python -u tasks/step4_m1b_train.py \
      --device cuda --no-resume \
      --seed "$SEED" --epochs 2 --batch-size 32 --lr 1e-5 --unfreeze-layers 0 \
      --margin 0.05 --tau 0.1 --lam-posfloor 1.0 --lam-preserve 5.0 \
      --train-subset "$TRAIN" --output-dir "$D" \
      $EXTRA
    CKPT="$D/seed_${SEED}/final.pt"
    [[ -f "$CKPT" ]] || { echo "Missing checkpoint: $CKPT" >&2; exit 3; }
    python -u tasks/eval_m1b_rank_gate.py \
      --device cuda --trained "$CKPT" --val-subset "$VAL" \
      --bootstrap 2000 --seed "$SEED" \
      --output "$D/seed_${SEED}/m1b_rank_gate.json"
  done
done

echo "=== P0 dropout control done: $(date) ==="
echo "Compare $OUT/dropout_on/seed_*/m1b_rank_gate.json vs $OUT/dropout_off/seed_*/m1b_rank_gate.json"
