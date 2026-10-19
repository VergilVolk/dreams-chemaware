#!/bin/bash
#SBATCH --job-name=panredu_ident
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-21%4
#SBATCH --output=data/validation/pan_redu_ident_%A_%a.out
#SBATCH --error=data/validation/pan_redu_ident_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"

JOB_TABLE="data/validation/pan_redu_positive_panel_jobs_20261018_v2/panel_jobs.tsv"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$JOB_TABLE")
IFS=$'\t' read -r job_index panel_id accession sample_files biological_units expected_bytes manifest panel_object <<< "$LINE"
if [[ -z "$LINE" || "$job_index" != "$SLURM_ARRAY_TASK_ID" ]]; then
  echo "Invalid panel job at array index ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

python -u tasks/partition_pan_redu_panel_identity.py \
  --panel-object "$panel_object" --panel-id "$panel_id" \
  --library-manifest data/validation/gnps_gold_silver_10ppm_benchmark_v1/manifest.csv.gz \
  --library-mgf data/validation/gnps_gold_silver_10ppm_benchmark_v1/spectra.mgf \
  --output-dir "data/validation/pan_redu_high_value_raw_panel_identity_20261018_v2/${panel_id}"
