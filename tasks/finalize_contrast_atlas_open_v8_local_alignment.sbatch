#!/bin/bash
#SBATCH --job-name=ca_v8_local
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/ca_v8_local_%j.out
#SBATCH --error=data/validation/ca_v8_local_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

COLLECTION="data/validation/contrast_atlas_open_response_collection_v8_plate3_covid_20261022"
SHARDS="data/validation/contrast_atlas_open_v8_official_embedding_shards_20261022"
MERGED="data/validation/contrast_atlas_open_v8_official_embeddings_20261022"
IDENTITY_SHARDS="data/validation/contrast_atlas_open_v8_common_identity_shards_20261022"
IDENTITY="data/validation/contrast_atlas_open_v8_common_identity_20261022"
RESULT="data/validation/contrast_atlas_open_v8_official_local_alignment_20261022"

for path in "$MERGED" "$IDENTITY" "$RESULT"; do
  if [[ -e "$path" ]]; then
    echo "refusing to overwrite frozen output: $path" >&2
    exit 3
  fi
done

python -u tasks/merge_contrast_atlas_embedding_shards.py \
  --collection "$COLLECTION" \
  --shards-root "$SHARDS" \
  --output-dir "$MERGED"

python -u tasks/finalize_contrast_atlas_open_identity_partitions.py \
  --collection "$COLLECTION" \
  --partitions-root "$IDENTITY_SHARDS" \
  --output-dir "$IDENTITY"

python -u tasks/score_contrast_atlas_symmetric_nn_alignment.py \
  --collection "$COLLECTION" \
  --embedding-manifest "$MERGED/embedding_manifest.tsv" \
  --identity-partition "$IDENTITY/identity_partition.tsv.gz" \
  --output-dir "$RESULT"
