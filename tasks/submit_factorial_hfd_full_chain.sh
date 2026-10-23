#!/bin/bash
set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS

download_job=$(sbatch --parsable tasks/run_factorial_hfd_download.sbatch)
verify_job=$(sbatch --parsable --dependency="afterok:${download_job}" tasks/run_factorial_hfd_verify_download.sbatch)
panel_job=$(sbatch --parsable --dependency="afterok:${verify_job}" tasks/run_factorial_hfd_panels_and_effects.sbatch)
embedding_job=$(sbatch --parsable --dependency="afterok:${panel_job}" tasks/run_factorial_hfd_embeddings.sbatch)
index_job=$(sbatch --parsable --dependency="afterok:${embedding_job}" tasks/run_factorial_hfd_pretruth_index.sbatch)
robustness_job=$(sbatch --parsable --dependency="afterok:${index_job}" tasks/run_factorial_hfd_pretruth_robustness.sbatch)
open_job=$(sbatch --parsable --dependency="afterok:${robustness_job}" tasks/run_factorial_hfd_freeze_and_open_primary.sbatch)

printf '%s\n' \
  "FACTORIAL_HFD_CHAIN_SUBMITTED" \
  "download=${download_job}" \
  "verify=${verify_job}" \
  "panels_and_effects=${panel_job}" \
  "embeddings=${embedding_job}" \
  "pretruth_index=${index_job}" \
  "pretruth_robustness=${robustness_job}" \
  "freeze_and_open=${open_job}"
