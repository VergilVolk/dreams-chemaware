#!/bin/bash
#SBATCH --job-name=hfd_panel
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --array=0-10%4
#SBATCH --output=data/validation/hfd_panel_%A_%a.out
#SBATCH --error=data/validation/hfd_panel_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

JOB_TABLE="data/validation/factorial_hfd_panel_jobs_v1_20261016/panel_jobs.tsv"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$JOB_TABLE")
IFS=$'\t' read -r job_index panel_id accession source_family sample_files biological_units expected_bytes manifest panel_object <<< "$LINE"
if [[ -z "$LINE" || "$job_index" != "$SLURM_ARRAY_TASK_ID" ]]; then
  echo "Invalid panel job at array index ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

python -u tasks/build_pan_redu_ms2_anchored_panel.py \
  --manifest "$manifest" \
  --output-dir "$panel_object" \
  --workers 8 \
  --minimum-biological-units 3 \
  --minimum-anchors 25 \
  --precursor-ppm 10 \
  --fragment-da 0.02 \
  --cluster-rt-seconds 8 \
  --quant-rt-seconds 15 \
  --resume

FAMILY_DIR="data/validation/factorial_hfd_ion_families_v1_20261016/${panel_id}"
if [[ ! -e "$FAMILY_DIR" ]]; then
  python -u tasks/build_factorial_hfd_ion_families.py \
    --panel-object "$panel_object" \
    --panel-registry data/validation/factorial_hfd_wave_manifest_v5_20261016/panel_registry.tsv \
    --panel-id "$panel_id" \
    --output-dir "$FAMILY_DIR"
fi

EFFECT_DIR="data/validation/factorial_hfd_panel_effects_v1_20261016/${panel_id}"
if [[ ! -e "$EFFECT_DIR" ]]; then
  python -u tasks/estimate_factorial_hfd_panel_contrasts.py \
    --panel-object "$panel_object" \
    --sealed-design data/validation/factorial_hfd_wave_manifest_v5_20261016/sealed_design/factorial_design.tsv \
    --panel-id "$panel_id" \
    --output-dir "$EFFECT_DIR" \
    --minimum-cell-n 4
fi
