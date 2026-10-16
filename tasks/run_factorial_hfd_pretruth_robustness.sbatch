#!/bin/bash
#SBATCH --job-name=hfd_pretruth
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/hfd_pretruth_%j.out
#SBATCH --error=data/validation/hfd_pretruth_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

INDEX="data/validation/factorial_hfd_response_index_official_v1_20261016"
python -u tasks/calibrate_factorial_hfd_response_null.py \
  --study-manifest "$INDEX/study_contrast_manifest.tsv" \
  --panel-registry data/validation/factorial_hfd_wave_manifest_v5_20261016/panel_registry.tsv \
  --effects-root data/validation/factorial_hfd_panel_effects_v1_20261016 \
  --embeddings-root data/validation/factorial_hfd_panel_embeddings_official_v1_20261016 \
  --families-root data/validation/factorial_hfd_ion_families_v1_20261016 \
  --output-dir data/validation/factorial_hfd_response_null_official_v1_20261016 \
  --permutations 1999 \
  --seed 20261016

python -u tasks/audit_factorial_hfd_panel_jackknife.py \
  --panel-manifest "$INDEX/panel_contrast_manifest.tsv" \
  --panel-sketches "$INDEX/panel_contrast_sketches.npz" \
  --output-dir data/validation/factorial_hfd_panel_jackknife_official_v1_20261016 \
  --world all \
  --minimum-entities 12
