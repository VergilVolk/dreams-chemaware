#!/bin/bash
""":"
Submit the complete Pan-ReDU biology wave with Slurm dependencies.

Run only from the repository root on the compute cluster:
    bash tasks/submit_pan_redu_biology_wave.sh
": """
set -euo pipefail

ROOT="/data02/run01/scv7tsl/DreaMS"
if [[ "$(pwd -P)" != "$ROOT" ]]; then
  echo "Refusing to submit outside $ROOT" >&2
  exit 2
fi

required=(
  data/validation/pan_redu_high_value_raw_tranche_20261014/raw_acquisition_manifest.tsv
  data/validation/pan_redu_high_value_raw_panel_jobs_20261014/panel_jobs.tsv
  data/validation/pan_redu_high_value_raw_panel_jobs_20261014/sealed_contrast_membership/contrast_membership.tsv
  data/validation/pan_redu_positive_panel_jobs_20261018_v2/panel_jobs.tsv
  data/validation/pan_redu_partial_transport_pairs_20261018_v2/panel_pairs.tsv
  data/validation/pan_redu_high_value_raw_tranche_20261014/selected_contrast_panels.tsv
  data/validation/pan_redu_systematic_wave_acquisition_20261014/ftps_version_map.json
  data/e1/official_embedding_slim.pt
)
for path in "${required[@]}"; do
  if [[ ! -s "$path" ]]; then
    echo "Missing required input: $path" >&2
    exit 2
  fi
done

download_job=$(sbatch --parsable tasks/run_pan_redu_high_value_raw_tranche.sbatch)
download_job=${download_job%%;*}
anchor_job=$(sbatch --parsable --dependency="afterok:${download_job}" tasks/run_pan_redu_ms2_anchored_panels.sbatch)
anchor_job=${anchor_job%%;*}
effect_job=$(sbatch --parsable --dependency="afterok:${anchor_job}" tasks/run_pan_redu_panel_effects.sbatch)
effect_job=${effect_job%%;*}
embedding_job=$(sbatch --parsable --dependency="afterok:${anchor_job}" tasks/run_pan_redu_panel_embeddings.sbatch)
embedding_job=${embedding_job%%;*}
identity_job=$(sbatch --parsable --dependency="afterok:${anchor_job}" tasks/run_pan_redu_panel_identity_partitions.sbatch)
identity_job=${identity_job%%;*}
identity_final_job=$(sbatch --parsable --dependency="afterok:${identity_job}" tasks/finalize_pan_redu_panel_identity.sbatch)
identity_final_job=${identity_final_job%%;*}
family_job=$(sbatch --parsable --dependency="afterok:${embedding_job}" tasks/run_pan_redu_panel_ion_families.sbatch)
family_job=${family_job%%;*}
transport_job=$(sbatch --parsable --dependency="afterok:${effect_job}:${family_job}:${identity_final_job}" tasks/run_pan_redu_partial_transport_pairs.sbatch)
transport_job=${transport_job%%;*}
final_job=$(sbatch --parsable --dependency="afterok:${transport_job}" tasks/finalize_pan_redu_partial_transport_matrix.sbatch)
final_job=${final_job%%;*}

echo "download_job=${download_job}"
echo "anchor_job=${anchor_job}"
echo "effect_job=${effect_job}"
echo "embedding_job=${embedding_job}"
echo "identity_job=${identity_job}"
echo "identity_final_job=${identity_final_job}"
echo "family_job=${family_job}"
echo "transport_job=${transport_job}"
echo "final_job=${final_job}"
echo "Monitor with: squeue -j ${download_job},${anchor_job},${effect_job},${embedding_job},${identity_job},${identity_final_job},${family_job},${transport_job},${final_job}"
