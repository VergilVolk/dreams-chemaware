#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=g8r_p1_risk_gate_%j.out
#SBATCH --error=g8r_p1_risk_gate_%j.err

# P1: risk-controlled selective ranking switch (learned intervention gate).
# CPU-only (loads caches + trains small sklearn models).  DEV eval on g8r_val
# only; no Test-A/B.  Full-pipeline OOF + frozen RAW-v1 reproduction gate.
# STATUS 2026-08-24: re-run for the THREE final audits
#   (p1_final_audits: all 66 disagreement queries, random-50-subset lambda-net
#   distribution, learned-U correction-vs-introduced discrimination AUC/AUPRC).
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

for f in data/validation/g8r_raw_reranker_cache.npz \
  data/validation/g8r_raw_reranker_cache_val.npz \
  data/validation/g8r_raw_reranker_v1_artifact.json \
  tasks/train_g8r_raw_reranker.py tasks/train_g8r_p1_risk_gate.py; do
  [[ -f "$f" ]] || { echo "MISSING: $f" >&2; exit 2; }
done

python -u tasks/train_g8r_p1_risk_gate.py \
  --output data/validation/g8r_p1_risk_gate.json

echo "=== P1 selective switch done: $(date) ==="
echo "Read data/validation/g8r_p1_risk_gate.json"
