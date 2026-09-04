#!/bin/bash
#SBATCH --job-name=bioaware_m3_rplc_ev
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/bioaware_m3_rplc_ev_%j.out
#SBATCH --error=data/validation/bioaware_m3_rplc_ev_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD/tasks:$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

MANIFEST=data/validation/bioaware_metdna3_internal_rplc_manifest_v1
MZML=data/external/metdna3_2025/mzml/internal_rplc
CACHE=data/validation/bioaware_metdna3_internal_rplc_dreams_cache_v1
SCORES=data/validation/bioaware_metdna3_internal_rplc_dreams_official_v1
BASE=data/validation/bioaware_metdna3_internal_rplc_baseline_v1
MS1=data/validation/bioaware_metdna3_internal_rplc_ms1_features_v1
RECURSIVE=data/validation/bioaware_metdna3_internal_rplc_recursive_v1
FEATURE_MS2=data/validation/bioaware_metdna3_internal_rplc_feature_ms2_v1
PATHS=data/validation/bioaware_metdna3_internal_rplc_candidate_paths_v1
EDGE0=data/validation/bioaware_metdna3_internal_rplc_edge_step0_v1
EDGE1=data/validation/bioaware_metdna3_internal_rplc_edge_step1_v1
RULES=data/validation/bioaware_metdna3_internal_rplc_rules_v1
MRN=data/validation/bioaware_metdna3_internal_rplc_depth3_v1
SMN=data/validation/bioaware_metdna3_internal_rplc_smn_v1
RT=data/validation/bioaware_metdna3_internal_rplc_rt_v1

echo "=== BioAware frozen internal RPLC evidence layer started: $(date) ==="
python -u tasks/build_bioaware_metdna3_baseline_transitions.py \
  --scores "$SCORES/candidate_scores.csv.gz" \
  --splits "$MANIFEST/identity_splits.csv.gz" --output-dir "$BASE"
python -u tasks/pilot_bioaware_metdna3_ms1_features.py \
  --mzml-dir "$MZML" --output-dir "$MS1" --workers 8
python -u tasks/audit_bioaware_metdna3_recursive_headroom.py \
  --pilot-dir "$MS1" --development-dir "$MANIFEST" --truth-name internal_level1.csv.gz \
  --query-cache "$CACHE/queries.csv.gz" --baseline-transitions "$BASE/raw_transitions.csv.gz" \
  --output-dir "$RECURSIVE" --scope internal_rplc
python -u tasks/build_bioaware_metdna3_feature_ms2_cache.py \
  --nodes "$RECURSIVE/stable_ms1_feature_nodes.csv.gz" --mzml-dir "$MZML" \
  --output-dir "$FEATURE_MS2" --scope internal_rplc --expected-files 16
python -u tasks/build_bioaware_metdna3_candidate_path_table.py \
  --recursive-dir "$RECURSIVE" --development-dir "$MANIFEST" \
  --truth-name internal_level1.csv.gz --query-cache "$CACHE/queries.csv.gz" \
  --candidate-scores "$SCORES/candidate_scores.csv.gz" \
  --baseline-transitions "$BASE/raw_transitions.csv.gz" --output-dir "$PATHS" \
  --scope internal_rplc
for STEP in 0 1; do
  OUT="$EDGE0"
  if [ "$STEP" = 1 ]; then OUT="$EDGE1"; fi
  python -u tasks/audit_bioaware_metdna3_candidate_edge_ms2.py \
    --recursive-dir "$RECURSIVE" --development-dir "$MANIFEST" \
    --truth-name internal_level1.csv.gz --query-cache "$CACHE" \
    --candidate-scores "$SCORES/candidate_scores.csv.gz" \
    --baseline-transitions "$BASE/raw_transitions.csv.gz" \
    --feature-ms2-dir "$FEATURE_MS2" --output-dir "$OUT" \
    --maximum-network-step "$STEP" --scope internal_rplc
done
python -u tasks/audit_bioaware_candidate_rule_likelihood.py \
  --scores "$SCORES/candidate_scores.csv.gz" --queries "$CACHE/queries.csv.gz" \
  --query-tensors "$CACHE/query_tensors.npz" \
  --references "$CACHE/candidate_references.csv.gz" \
  --all-queries-unresolved --output-dir "$RULES"
python -u tasks/summarize_bioaware_metdna3_candidate_edge_decision.py \
  --paths "$PATHS/candidate_paths.csv.gz" \
  --edge-evidence "$EDGE0/candidate_edge_evidence.csv.gz" \
  --baseline "$BASE/raw_transitions.csv.gz" --output-dir "$MRN" \
  --scope internal_rplc
python -u tasks/audit_bioaware_metdna3_smn_headroom.py \
  --recursive-dir "$RECURSIVE" --development-dir "$MANIFEST" \
  --truth-name internal_level1.csv.gz --query-cache "$CACHE/queries.csv.gz" \
  --candidate-scores "$SCORES/candidate_scores.csv.gz" \
  --baseline-transitions "$BASE/raw_transitions.csv.gz" \
  --mrn-decision-dir "$MRN" --output-dir "$SMN" --scope internal_rplc
python -u tasks/audit_bioaware_metdna3_rt_headroom.py \
  --development-dir "$MANIFEST" --truth-name internal_level1.csv.gz \
  --query-cache "$CACHE/queries.csv.gz" \
  --candidate-scores "$SCORES/candidate_scores.csv.gz" \
  --baseline-transitions "$BASE/raw_transitions.csv.gz" \
  --mrn-decision-dir "$MRN" --smn-dir "$SMN" --output-dir "$RT" \
  --scope internal_rplc
echo "=== BioAware frozen internal RPLC evidence layer completed: $(date) ==="
