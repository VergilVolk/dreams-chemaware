#!/bin/bash
#SBATCH --job-name=panredu_effect
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-22%8
#SBATCH --output=data/validation/pan_redu_effect_%A_%a.out
#SBATCH --error=data/validation/pan_redu_effect_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

JOB_TABLE="data/validation/pan_redu_high_value_raw_panel_jobs_20261014/panel_jobs.tsv"
MEMBERSHIPS="data/validation/pan_redu_high_value_raw_panel_jobs_20261014/sealed_contrast_membership/contrast_membership.tsv"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$JOB_TABLE")
if [[ -z "$LINE" ]]; then
  echo "No panel job at array index ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi
IFS=$'\t' read -r job_index panel_id accession sample_files biological_units expected_bytes manifest panel_object <<< "$LINE"
if [[ "$job_index" != "$SLURM_ARRAY_TASK_ID" ]]; then
  echo "Job-table index mismatch: ${job_index} != ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

python -u tasks/estimate_pan_redu_panel_contrasts.py \
  --panel-object "$panel_object" \
  --memberships "$MEMBERSHIPS" \
  --panel-id "$panel_id" \
  --output-dir "data/validation/pan_redu_high_value_raw_panel_effects_20261014/${panel_id}" \
  --minimum-group-measurements 3 \
  --minimum-prevalence 0.20
