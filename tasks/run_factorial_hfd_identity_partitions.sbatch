#!/bin/bash
#SBATCH --job-name=hfd_identity
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-10%4
#SBATCH --output=data/validation/hfd_identity_%A_%a.out
#SBATCH --error=data/validation/hfd_identity_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"

JOB_TABLE="data/validation/factorial_hfd_panel_jobs_v1_20261016/panel_jobs.tsv"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$JOB_TABLE")
IFS=$'\t' read -r job_index panel_id accession source_family sample_files biological_units expected_bytes manifest panel_object <<< "$LINE"
if [[ -z "$LINE" || "$job_index" != "$SLURM_ARRAY_TASK_ID" ]]; then
  echo "Invalid panel job at array index ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

python -u tasks/partition_factorial_hfd_panel_identity.py \
  --panel-object "$panel_object" \
  --panel-registry data/validation/factorial_hfd_wave_manifest_v5_20261016/panel_registry.tsv \
  --panel-id "$panel_id" \
  --positive-library-manifest data/validation/gnps_gold_silver_10ppm_benchmark_v1/manifest.csv.gz \
  --positive-library-mgf data/validation/gnps_gold_silver_10ppm_benchmark_v1/spectra.mgf \
  --negative-library-mgf data/reference/unified_v3/unified_neg.mgf \
  --output-dir "data/validation/factorial_hfd_panel_identity_v1_20261023/${panel_id}"
