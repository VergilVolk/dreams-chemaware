#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=showspace_index_%j.out
#SBATCH --error=showspace_index_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:$PWD/showspace:$PWD/tasks:${PYTHONPATH:-}"

MODEL="${SHOWSPACE_MODEL:-official_dreams}"
DATA="${SHOWSPACE_DATA:-data/models/MassSpecGym_MurckoHist_split.hdf5}"
OUTPUT_DIR="${SHOWSPACE_INDEX_DIR:-data/showspace/index}"
mkdir -p "$OUTPUT_DIR"

case "$MODEL" in
  official_dreams) OUTPUT="$OUTPUT_DIR/official.npy" ;;
  e4a_shared)
    : "${DREAMS_E4_CKPT:?set DREAMS_E4_CKPT to the frozen deployable shared checkpoint}"
    OUTPUT="$OUTPUT_DIR/e4a.npy"
    ;;
  e8_shared)
    : "${DREAMS_E8_CKPT:?set DREAMS_E8_CKPT to the frozen deployable shared checkpoint}"
    OUTPUT="$OUTPUT_DIR/e8.npy"
    ;;
  *) echo "Unknown SHOWSPACE_MODEL=$MODEL" >&2; exit 2 ;;
esac

for REQUIRED in \
  "$DATA" \
  data/e1/official_embedding_slim.pt \
  dreams/models/pretrained/ssl_model_server.pt \
  showspace/build_embedding_index.py \
  showspace/inference_utils.py \
  tasks/train_e1_identity.py ; do
  [[ -f "$REQUIRED" ]] || { echo "MISSING: $REQUIRED" >&2; exit 2; }
done

export DREAMS_OFFICIAL_SLIM_CKPT="${DREAMS_OFFICIAL_SLIM_CKPT:-data/e1/official_embedding_slim.pt}"
export DREAMS_ARCHITECTURE_CKPT="${DREAMS_ARCHITECTURE_CKPT:-dreams/models/pretrained/ssl_model_server.pt}"

echo "=== Showspace model-aligned index: $MODEL ==="
python -u showspace/build_embedding_index.py \
  --model "$MODEL" --data "$DATA" --output "$OUTPUT" \
  --device cuda --batch-size "${SHOWSPACE_BATCH_SIZE:-128}"
echo "=== completed: $OUTPUT and ${OUTPUT%.npy}.json ==="
