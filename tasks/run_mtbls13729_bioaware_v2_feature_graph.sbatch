#!/bin/bash
#SBATCH --job-name=bioaware_v2_graph
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=data/validation/bioaware_v2_graph_%j.out
#SBATCH --error=data/validation/bioaware_v2_graph_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD/tasks:$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

OUTPUT="data/mtbls13729/bioaware_v2_feature_graph"
EVAL_OUTPUT="data/mtbls13729/bioaware_v2_eval"
if [[ -d "$OUTPUT" ]] && find "$OUTPUT" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing to overwrite non-empty output: $OUTPUT" >&2
  exit 2
fi
if [[ -d "$EVAL_OUTPUT" ]] && find "$EVAL_OUTPUT" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing to overwrite non-empty output: $EVAL_OUTPUT" >&2
  exit 2
fi

python -u tasks/test_mtbls13729_bioaware_v2_feature_graph.py
python -u tasks/build_mtbls13729_bioaware_v2_feature_graph.py \
  --output-dir "$OUTPUT" \
  --device cuda
python -u tasks/validate_mtbls13729_bioaware_v2_feature_graph.py "$OUTPUT"
python -u tasks/evaluate_bioaware_v2_two_layer.py \
  --feature-graph "$OUTPUT" \
  --output-dir "$EVAL_OUTPUT"
python -u tasks/validate_mtbls13729_bioaware_v2_two_layer.py "$EVAL_OUTPUT"
