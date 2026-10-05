#!/usr/bin/env bash
#SBATCH --job-name=bioaware_b12_multi
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=bioaware_b12_multi_%j.out
#SBATCH --error=bioaware_b12_multi_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS

PYTHON=/data02/home/scv7tsl/run/miniconda3/envs/dreams/bin/python
JOB_TAG="${SLURM_JOB_ID:-manual}_$(date +%Y%m%dT%H%M%S)"
OUTPUT="data/validation/bioaware_b12_multicohort_catalog_${JOB_TAG}"

test -x "$PYTHON"
for path in \
  data/validation/bioaware_b3_reaction_coabundance_local_20260906/candidate_features.csv.gz \
  data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1/candidate_features.csv.gz \
  data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1/per_query.csv.gz \
  data/validation/bioaware_kgmn200std_hidden_seed_v1/candidate_features.csv.gz \
  data/validation/bioaware_kgmn200std_confirmation_manifest_v2/seed_features.csv.gz
do
  test -s "$path"
done

"$PYTHON" -m py_compile \
  tasks/audit_bioaware_b12_multicohort_catalog_action.py \
  tasks/test_bioaware_b12_multicohort_catalog_action.py \
  tasks/validate_bioaware_b12_multicohort_catalog_action.py
"$PYTHON" tasks/test_bioaware_b12_multicohort_catalog_action.py

echo "=== BioAware B12 multicohort catalogue action started: $(date) ==="
"$PYTHON" -u tasks/audit_bioaware_b12_multicohort_catalog_action.py \
  --output-dir "$OUTPUT" \
  --bootstrap-resamples 10000 \
  --seed 20260907
"$PYTHON" tasks/validate_bioaware_b12_multicohort_catalog_action.py \
  --output-dir "$OUTPUT"
echo "=== BioAware B12 completed: $(date) ==="
echo "Result: $OUTPUT/report.json"
