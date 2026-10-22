#!/bin/bash
#SBATCH --job-name=ca_v8_embed
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-11%4
#SBATCH --output=data/validation/ca_v8_embed_%A_%a.out
#SBATCH --error=data/validation/ca_v8_embed_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

COLLECTION="data/validation/contrast_atlas_open_response_collection_v8_plate3_covid_20261022"
CONTRACT="tasks/contrast_atlas_open_collection_covid_extension_v1.json"
SHARDS="data/validation/contrast_atlas_open_v8_official_embedding_shards_20261022"

STUDY=$(python - "$COLLECTION/source_manifest.tsv" "$SLURM_ARRAY_TASK_ID" <<'PY'
import pandas as pd
import sys
frame = pd.read_csv(sys.argv[1], sep="\t", dtype=str)
index = int(sys.argv[2])
if index < 0 or index >= len(frame):
    raise SystemExit(f"array index {index} outside 0..{len(frame)-1}")
print(str(frame.iloc[index]["study_id"]))
PY
)

OUT="$SHARDS/$STUDY"
if [[ -e "$OUT" ]]; then
  echo "refusing to overwrite frozen shard: $OUT" >&2
  exit 3
fi

python -u tasks/encode_contrast_atlas_open_response_entities.py \
  --contract "$CONTRACT" \
  --collection "$COLLECTION" \
  --model-kind official \
  --encoder-checkpoint data/e1/official_embedding_slim.pt \
  --mgf-override IBD200_GROUP=data/validation/contrast_atlas_ibd200_precursor_repair_v1_20261022/representative_spectra.mgf \
  --study-id "$STUDY" \
  --output-dir "$OUT" \
  --device cuda \
  --batch-size 256
