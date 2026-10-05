#!/bin/bash
#SBATCH --job-name=bioaware_b5_xfer
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=bioaware_b5_external_%j.out
#SBATCH --error=bioaware_b5_external_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
PYTHON_BIN=/data02/home/scv7tsl/run/miniconda3/envs/dreams/bin/python
TRAIN_CANDIDATES="${TRAIN_CANDIDATES:-data/validation/bioaware_b3_reaction_coabundance_local_20260906/candidate_features.csv.gz}"
OUTPUT="data/validation/bioaware_b5_graph_prior_external_${SLURM_JOB_ID}"

[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 2; }
[[ -s "$TRAIN_CANDIDATES" ]] || { echo "Missing Full16 candidate cache: $TRAIN_CANDIDATES" >&2; exit 3; }
[[ ! -e "$OUTPUT" ]] || { echo "Refusing to overwrite: $OUTPUT" >&2; exit 4; }

REQUIRED=(
  data/validation/bioaware_st001154_hilic_frozen_evaluation_v1/candidate_features.csv.gz
  data/validation/bioaware_st001154_hilic_frozen_evaluation_v1/per_query.csv.gz
  data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1/candidate_features.csv.gz
  data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1/per_query.csv.gz
  data/validation/bioaware_kgmn200std_hidden_seed_v1/candidate_features.csv.gz
  data/validation/bioaware_kgmn200std_confirmation_manifest_v2/seed_features.csv.gz
)
for path in "${REQUIRED[@]}"; do
  [[ -s "$path" ]] || { echo "Missing dependency: $path" >&2; exit 5; }
done

"$PYTHON_BIN" tasks/test_bioaware_b5_graph_prior_external_transfer.py
"$PYTHON_BIN" tasks/audit_bioaware_b5_graph_prior_external_transfer.py \
  --train-candidates "$TRAIN_CANDIDATES" \
  --output-dir "$OUTPUT" \
  --bootstrap-resamples 5000 \
  --seed 20260906
"$PYTHON_BIN" tasks/validate_bioaware_b5_graph_prior_external_transfer.py "$OUTPUT"
echo "BioAware B5 result: $OUTPUT/report.json"
