#!/usr/bin/env bash
#SBATCH --job-name=bioaware_b9
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=bioaware_b9_%j.out
#SBATCH --error=bioaware_b9_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS

PYTHON=/data02/home/scv7tsl/run/miniconda3/envs/dreams/bin/python
RUN_ID="${SLURM_JOB_ID:-manual}"
OUTPUT="data/validation/bioaware_b9_reaction_spectral_specificity_${RUN_ID}"

echo "=== BioAware B9 reaction-spectral specificity started: $(date) ==="
"${PYTHON}" -m py_compile \
  tasks/audit_bioaware_b9_reaction_spectral_specificity.py \
  tasks/test_bioaware_b9_reaction_spectral_specificity.py
PYTHONPATH="${PWD}/tasks:${PWD}" "${PYTHON}" tasks/test_bioaware_b9_reaction_spectral_specificity.py

for required in \
  data/validation/bioaware_metdna3_external_v3_v1 \
  data/validation/bioaware_metdna3_external_negative_units_v2 \
  data/validation/bioaware_metdna3_external_negative_loso_ranker_v4_chemically_filtered/candidate_features.csv.gz \
  data/validation/bioaware_metdna3_external_negative_paths_v1 \
  data/validation/bioaware_metdna3_external_negative_edge_step0_v1 \
  data/validation/bioaware_metdna3_external_manifest_v1 \
  data/validation/bioaware_embedding_relation_manifest_v2_20260830/identity_pairs.csv.gz \
  data/reference/metdna2_emrn_network_20260828/metdna2_emrn_edges.csv.gz
do
  test -e "${required}" || { echo "MISSING DEPENDENCY: ${required}" >&2; exit 2; }
done

test ! -e "${OUTPUT}" || { echo "Refusing to overwrite: ${OUTPUT}" >&2; exit 2; }

PYTHONPATH="${PWD}/tasks:${PWD}" "${PYTHON}" -u \
  tasks/audit_bioaware_b9_reaction_spectral_specificity.py \
  --output-dir "${OUTPUT}" \
  --bootstrap-resamples 10000 \
  --seed 20260906

"${PYTHON}" - "${OUTPUT}/report.json" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
report = json.loads(path.read_text())
assert report["status"] == "bioaware_b9_reaction_spectral_specificity_complete"
assert report["formal"] is True
assert len(report["action_cells"]) == 6
assert report["queries"] >= 1000
assert report["contracts"]["P2b_used"] is False
print("[BioAware B9 formal result validation] PASS", {
    "queries": report["queries"],
    "coverage_queries": report["queries_with_any_complete_reaction_control"],
    "advancing_cells": report["advancing_cells"],
})
PY

echo "=== BioAware B9 reaction-spectral specificity completed: $(date) ==="
echo "Result: ${OUTPUT}/report.json"
