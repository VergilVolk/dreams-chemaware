#!/bin/bash
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --output=mtbls13729_unified_%j.out
#SBATCH --error=mtbls13729_unified_%j.err

# Broad-library, mass-first annotation for MTBLS13729.
# Scientific boundary: this produces library-supported Level-2/3 candidates;
# it does not claim Level 1 or calibrated 1% FDR without standards/valid decoys.

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "===== MTBLS13729 validated unified-v2 mass-first annotation ====="
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

for MODE in pos neg; do
  MGF="data/reference/unified_v2/unified_${MODE}.mgf"
  LIB="data/models/unified_v2_${MODE}_dreams_emb"
  if [ ! -f "$LIB/embeddings.npy" ]; then
    python -u tasks/encode_mona_neg_library.py --mgf "$MGF" --out "$LIB" --device cuda
  else
    echo "[library] reuse $LIB"
  fi
done

for PANEL in pos_rp neg_rp pos_hilic neg_hilic; do
  case "$PANEL" in
    neg_*) MODE=neg ;;
    *) MODE=pos ;;
  esac
  QUERY="data/mtbls13729/embeddings/$PANEL"
  LIB="data/models/unified_v2_${MODE}_dreams_emb"
  OUT="data/mtbls13729/annotation_unified_v2/$PANEL"
  if [ ! -f "$QUERY/embeddings.npy" ]; then
    echo "[error] missing query embeddings: $QUERY" >&2
    exit 2
  fi
  python -u -m annotation.cli annotate \
    --query-dir "$QUERY" --library-dir "$LIB" --out "$OUT" \
    --backend gpu --device cuda
done

echo "Finished: $(date)"
