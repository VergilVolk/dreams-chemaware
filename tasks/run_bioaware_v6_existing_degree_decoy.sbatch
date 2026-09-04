#!/bin/bash
#SBATCH --job-name=bio_v6_decoy_eval
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --array=0-99%16
#SBATCH --output=data/validation/bio_v6_decoy_eval_%A_%a.out
#SBATCH --error=data/validation/bio_v6_decoy_eval_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD/tasks:$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

UNITS=(Mouse_brain__rplc Mouse_liver__hilic Mouse_liver__rplc NIST_plasma__hilic NIST_plasma__rplc)
UNIT_INDEX=$((SLURM_ARRAY_TASK_ID / 20))
REPEAT=$((SLURM_ARRAY_TASK_ID % 20))
UNIT=${UNITS[$UNIT_INDEX]}
TAG=$(printf "repeat_%02d" "$REPEAT")
REAL=data/validation/bioaware_metdna3_external_v3_v1/$UNIT
ROOT=$REAL/degree_decoys/$TAG
LEDGER=$ROOT/ledger/candidate_evidence.csv.gz
QUERIES=$REAL/cache/queries.csv.gz
ART=data/validation/bioaware_v6_identifiable_router_frozen_v2_20260830/artifact.json

test -f "$LEDGER" || { echo "missing existing decoy ledger: $LEDGER" >&2; exit 2; }
test -f "$QUERIES" || { echo "missing query cache: $QUERIES" >&2; exit 3; }
test ! -e "$ROOT/result_v6" || { echo "fail-closed: $ROOT/result_v6 exists" >&2; exit 4; }
python -u tasks/evaluate_bioaware_v6_identifiable_router.py \
  --artifact "$ART" --ledger "$LEDGER" --queries "$QUERIES" \
  --panel "$UNIT" --output-dir "$ROOT/result_v6"
