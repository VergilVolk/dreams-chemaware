#!/bin/bash
#SBATCH --job-name=panredu_anchor
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --array=0-22%4
#SBATCH --output=data/validation/pan_redu_anchor_%A_%a.out
#SBATCH --error=data/validation/pan_redu_anchor_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

JOB_TABLE="data/validation/pan_redu_high_value_raw_panel_jobs_20261014/panel_jobs.tsv"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$JOB_TABLE")
if [[ -z "$LINE" ]]; then
  echo "No panel job at array index ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi
IFS=$'\t' read -r job_index panel_id accession sample_files biological_units expected_bytes manifest output_dir <<< "$LINE"
if [[ "$job_index" != "$SLURM_ARRAY_TASK_ID" ]]; then
  echo "Job-table index mismatch: ${job_index} != ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

python -u tasks/build_pan_redu_ms2_anchored_panel.py \
  --manifest "$manifest" \
  --output-dir "$output_dir" \
  --workers 8 \
  --minimum-biological-units 3 \
  --minimum-anchors 25 \
  --precursor-ppm 10 \
  --fragment-da 0.02 \
  --cluster-rt-seconds 8 \
  --quant-rt-seconds 15 \
  --resume
