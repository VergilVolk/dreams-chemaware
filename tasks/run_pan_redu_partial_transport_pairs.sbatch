#!/bin/bash
#SBATCH --job-name=panredu_resp_ot
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --array=0-187%8
#SBATCH --output=data/validation/pan_redu_resp_ot_%A_%a.out
#SBATCH --error=data/validation/pan_redu_resp_ot_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

PAIR_ROOT="data/validation/pan_redu_partial_transport_pairs_20261018_v2"
OUT=$(printf "data/validation/pan_redu_fixed_capacity_response_transport_20261019_v5/pair_%04d" "$SLURM_ARRAY_TASK_ID")
python -u tasks/run_pan_redu_panel_pair_partial_transport.py \
  --pair-manifest "$PAIR_ROOT/panel_pairs.tsv" \
  --pair-index "$SLURM_ARRAY_TASK_ID" \
  --topology data/validation/pan_redu_high_value_raw_panel_jobs_20261014/contrast_topology_readiness.tsv \
  --panel-metadata data/validation/pan_redu_high_value_raw_tranche_20261014/selected_contrast_panels.tsv \
  --effects-root data/validation/pan_redu_high_value_raw_panel_effects_20261014 \
  --embeddings-root data/validation/pan_redu_high_value_raw_panel_embeddings_official_20261016 \
  --families-root data/validation/pan_redu_high_value_raw_panel_ion_families_20261018_v3 \
  --identity-table data/validation/pan_redu_high_value_raw_identity_partition_20261018_v2/identity_partition.tsv.gz \
  --output-dir "$OUT" \
  --neighbors 50 --minimum-similarity 0.70 --unmatched-cost 0.20 --minimum-entities 12
