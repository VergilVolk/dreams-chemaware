#!/bin/bash
#SBATCH --job-name=ca_v7_local
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/ca_v7_local_%j.out
#SBATCH --error=data/validation/ca_v7_local_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

COLLECTION="data/validation/contrast_atlas_open_response_collection_v7_covid_20261022"
SHARDS="data/validation/contrast_atlas_open_v7_official_embedding_shards_20261022"
MERGED="data/validation/contrast_atlas_open_v7_official_embeddings_20261022"
RESULT="data/validation/contrast_atlas_open_v7_official_local_alignment_20261022"

for path in "$MERGED" "$RESULT"; do
  if [[ -e "$path" ]]; then
    echo "refusing to overwrite frozen output: $path" >&2
    exit 3
  fi
done

python -u tasks/merge_contrast_atlas_embedding_shards.py \
  --collection "$COLLECTION" \
  --shards-root "$SHARDS" \
  --output-dir "$MERGED"

python -u tasks/score_contrast_atlas_symmetric_nn_alignment.py \
  --collection "$COLLECTION" \
  --embedding-manifest "$MERGED/embedding_manifest.tsv" \
  --output-dir "$RESULT"
