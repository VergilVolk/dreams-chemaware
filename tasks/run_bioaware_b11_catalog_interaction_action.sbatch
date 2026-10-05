#!/usr/bin/env bash
#SBATCH --job-name=bioaware_b11_catalog
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=bioaware_b11_catalog_%j.out
#SBATCH --error=bioaware_b11_catalog_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS

PYTHON=/data02/home/scv7tsl/run/miniconda3/envs/dreams/bin/python
JOB_TAG="${SLURM_JOB_ID:-manual}_$(date +%Y%m%dT%H%M%S)"
OUTPUT="data/validation/bioaware_b11_catalog_interaction_${JOB_TAG}"

test -x "$PYTHON"
test -s data/validation/bioaware_b3_reaction_coabundance_local_20260906/candidate_features.csv.gz
test -s data/validation/bioaware_b4_opportunity_decomposition_local_20260906/mixed_polarity__graph_only__transitions.csv.gz

"$PYTHON" -m py_compile \
  tasks/audit_bioaware_b11_catalog_interaction_action.py \
  tasks/test_bioaware_b11_catalog_interaction_action.py \
  tasks/validate_bioaware_b11_catalog_interaction_action.py
"$PYTHON" tasks/test_bioaware_b11_catalog_interaction_action.py

echo "=== BioAware B11 nested catalog-interaction action started: $(date) ==="
"$PYTHON" -u tasks/audit_bioaware_b11_catalog_interaction_action.py \
  --output-dir "$OUTPUT" \
  --bootstrap-resamples 10000 \
  --seed 20260907
"$PYTHON" tasks/validate_bioaware_b11_catalog_interaction_action.py \
  --output-dir "$OUTPUT"
echo "=== BioAware B11 completed: $(date) ==="
echo "Result: $OUTPUT/report.json"
