#!/bin/bash
#SBATCH --job-name=panredu_embed
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-22%4
#SBATCH --output=data/validation/pan_redu_embed_%A_%a.out
#SBATCH --error=data/validation/pan_redu_embed_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

JOB_TABLE="data/validation/pan_redu_high_value_raw_panel_jobs_20261014/panel_jobs.tsv"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$JOB_TABLE")
IFS=$'\t' read -r job_index panel_id accession sample_files biological_units expected_bytes manifest panel_object <<< "$LINE"
if [[ -z "$LINE" || "$job_index" != "$SLURM_ARRAY_TASK_ID" ]]; then
  echo "Invalid panel job at array index ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

python -u tasks/encode_pan_redu_panel_spectra.py \
  --panel-object "$panel_object" \
  --panel-id "$panel_id" \
  --output-dir "data/validation/pan_redu_high_value_raw_panel_embeddings_official_20261016/${panel_id}" \
  --encoder-name official_dreams \
  --device cuda \
  --batch-size 128
