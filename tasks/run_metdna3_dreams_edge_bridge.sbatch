#!/bin/bash
#SBATCH --job-name=metdna3_dreams_edge
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=data/validation/metdna3_dreams_edge_%j.out
#SBATCH --error=data/validation/metdna3_dreams_edge_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD/tasks:$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

: "${METDNA3_WD:?Set METDNA3_WD to an official MrnAnnoAlgo3 work directory}"
RUN_ID=${SLURM_JOB_ID}
ROOT_OUT="data/validation/metdna3_dreams_edge_bridge_${RUN_ID}"
BRIDGE="$ROOT_OUT/official_bridge"
EMBED="$ROOT_OUT/official_dreams_embeddings"
SCORES="$ROOT_OUT/official_dreams_edge_scores"

command -v Rscript >/dev/null 2>&1 || { echo "Rscript is required for the official MetDNA3 bridge" >&2; exit 2; }
echo "=== MetDNA3 official edge bridge started: $(date) ==="
Rscript tasks/export_metdna3_official_bridge.R "$METDNA3_WD" "$BRIDGE"
python -u tasks/encode_metdna3_feature_dreams_embeddings.py \
  --bridge-dir "$BRIDGE" --output-dir "$EMBED" --device cuda --batch-size 128
python -u tasks/build_metdna3_dreams_ms2_network.py \
  --edges "$BRIDGE/table_ms2_edges.csv" \
  --embeddings "$EMBED/feature_embeddings.npz" \
  --embedding-report "$EMBED/report.json" --output-dir "$SCORES"
echo "=== MetDNA3 official edge bridge completed: $(date) ==="
echo "Raw cosine is intentionally not injected. Calibration is the next locked stage."
