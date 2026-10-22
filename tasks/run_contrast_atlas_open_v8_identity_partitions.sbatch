#!/bin/bash
#SBATCH --job-name=ca_v8_ident
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --array=0-11%2
#SBATCH --output=data/validation/ca_v8_ident_%A_%a.out
#SBATCH --error=data/validation/ca_v8_ident_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

COLLECTION="data/validation/contrast_atlas_open_response_collection_v8_plate3_covid_20261022"
CONTRACT="tasks/contrast_atlas_open_collection_covid_extension_v1.json"
OUTROOT="data/validation/contrast_atlas_open_v8_common_identity_shards_20261022"
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
MGF_OVERRIDE=()
if [[ "$STUDY" == "IBD200_GROUP" ]]; then
  MGF_OVERRIDE=(
    --mgf-override
    data/validation/contrast_atlas_ibd200_precursor_repair_v1_20261022/representative_spectra.mgf
  )
fi

python -u tasks/partition_contrast_atlas_open_source_identity.py \
  --contract "$CONTRACT" \
  --collection "$COLLECTION" \
  --study-id "$STUDY" \
  "${MGF_OVERRIDE[@]}" \
  --library-manifest data/validation/gnps_gold_silver_10ppm_benchmark_v1/manifest.csv.gz \
  --library-mgf data/validation/gnps_gold_silver_10ppm_benchmark_v1/spectra.mgf \
  --output-dir "$OUTROOT/$STUDY"
