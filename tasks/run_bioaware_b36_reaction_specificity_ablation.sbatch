#!/bin/bash
#SBATCH --job-name=bioaware_b36_specificity
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=bioaware_b36_specificity_%j.out
#SBATCH --error=bioaware_b36_specificity_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS

PYTHON=/data02/home/scv7tsl/run/miniconda3/envs/dreams/bin/python
OUTPUT="data/validation/bioaware_b36_reaction_specificity_${SLURM_JOB_ID}"

for REQUIRED in \
  tasks/audit_bioaware_b36_reaction_specificity_ablation.py \
  tasks/validate_bioaware_b36_reaction_specificity_ablation.py \
  tasks/test_bioaware_b36_reaction_specificity_ablation.py \
  tasks/audit_bioaware_b11_catalog_interaction_action.py \
  tasks/audit_bioaware_b12_multicohort_catalog_action.py \
  tasks/audit_bioaware_b16_pairwise_nonlinear_action.py \
  data/validation/bioaware_b3_reaction_coabundance_local_20260906/candidate_features.csv.gz \
  data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1/candidate_features.csv.gz \
  data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1/per_query.csv.gz \
  data/validation/bioaware_kgmn200std_hidden_seed_v1/candidate_features.csv.gz \
  data/validation/bioaware_kgmn200std_confirmation_manifest_v2/seed_features.csv.gz
do
  test -s "$REQUIRED" || { echo "Missing dependency: $REQUIRED" >&2; exit 2; }
done

test ! -e "$OUTPUT" || { echo "Refusing to overwrite: $OUTPUT" >&2; exit 3; }

"$PYTHON" -u tasks/test_bioaware_b36_reaction_specificity_ablation.py
"$PYTHON" -u tasks/audit_bioaware_b36_reaction_specificity_ablation.py \
  --output-dir "$OUTPUT" \
  --null-repeats 10 \
  --bootstrap-resamples 10000 \
  --seed 20260907
"$PYTHON" -u tasks/validate_bioaware_b36_reaction_specificity_ablation.py "$OUTPUT"

echo "B36 result: $OUTPUT/report.json"
