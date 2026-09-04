#!/bin/bash
#SBATCH --job-name=bioaware_m3_ext
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --output=data/validation/bioaware_m3_ext_%A_%a.out
#SBATCH --error=data/validation/bioaware_m3_ext_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD/tasks:$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

UNITS=(BV2cell__hilic BV2cell__rplc Mouse_brain__hilic Mouse_brain__rplc Mouse_liver__hilic Mouse_liver__rplc NIST_plasma__hilic NIST_plasma__rplc)
PREFIXES=(M3XBH M3XBR M3XMH M3XMR M3XLH M3XLR M3XPH M3XPR)
FILES=(16 16 16 15 16 16 16 16)
UNIT=${UNITS[$SLURM_ARRAY_TASK_ID]}
PREFIX=${PREFIXES[$SLURM_ARRAY_TASK_ID]}
EXPECTED=${FILES[$SLURM_ARRAY_TASK_ID]}

MANIFEST_ROOT=data/validation/bioaware_metdna3_external_manifest_v1
MANIFEST=$MANIFEST_ROOT/$UNIT
MZML=data/external/metdna3_2025/mzml/external/$UNIT
ROOT=data/validation/bioaware_metdna3_external_v3_v1/$UNIT
ART=data/validation/bioaware_v3_consensus_router_frozen_v2_20260830/artifact.json
ART_V4=data/validation/bioaware_v4_high_precision_router_frozen_v1_20260830/artifact.json
ART_V6=data/validation/bioaware_v6_identifiable_router_frozen_v2_20260830/artifact.json
INTERNAL=data/validation/bioaware_metdna3_internal_rplc_frozen_v3_result_v1/report.json

echo "=== BioAware frozen external unit $UNIT started: $(date) ==="
python -u tasks/preflight_bioaware_metdna3_development_ms2.py \
  --development-dir "$MANIFEST" --truth-name external_level1.csv.gz \
  --manifest-report "$MANIFEST_ROOT/report.json" --mzml-dir "$MZML" \
  --output "$ROOT/preflight.json" --scope external --expected-files "$EXPECTED" \
  --minimum-matched-rows 50 --minimum-matched-identities 50 \
  --minimum-exclusive-identities 50
python -u tasks/build_bioaware_metdna3_dreams_cache.py \
  --development-dir "$MANIFEST" --truth-name external_level1.csv.gz \
  --mzml-dir "$MZML" --preflight "$ROOT/preflight.json" \
  --output-dir "$ROOT/cache" --scope external --query-prefix "$PREFIX" \
  --minimum-identities 50 --minimum-queries 50
python -u tasks/encode_bioaware_metdna3_dreams.py \
  --cache-dir "$ROOT/cache" --output-dir "$ROOT/scores" --device cuda --batch-size 128
python -u tasks/build_bioaware_metdna3_baseline_transitions.py \
  --scores "$ROOT/scores/candidate_scores.csv.gz" \
  --splits "$MANIFEST/identity_splits.csv.gz" --output-dir "$ROOT/baseline"
python -u tasks/pilot_bioaware_metdna3_ms1_features.py \
  --mzml-dir "$MZML" --output-dir "$ROOT/ms1" --workers 8 \
  --expected-files "$EXPECTED" --frozen-noise-threshold 100000
python -u tasks/audit_bioaware_metdna3_recursive_headroom.py \
  --pilot-dir "$ROOT/ms1" --development-dir "$MANIFEST" --truth-name external_level1.csv.gz \
  --query-cache "$ROOT/cache/queries.csv.gz" \
  --baseline-transitions "$ROOT/baseline/raw_transitions.csv.gz" \
  --output-dir "$ROOT/recursive" --scope external
python -u tasks/build_bioaware_metdna3_feature_ms2_cache.py \
  --nodes "$ROOT/recursive/stable_ms1_feature_nodes.csv.gz" --mzml-dir "$MZML" \
  --output-dir "$ROOT/feature_ms2" --scope external --expected-files "$EXPECTED"
python -u tasks/build_bioaware_metdna3_candidate_path_table.py \
  --recursive-dir "$ROOT/recursive" --development-dir "$MANIFEST" \
  --truth-name external_level1.csv.gz --query-cache "$ROOT/cache/queries.csv.gz" \
  --candidate-scores "$ROOT/scores/candidate_scores.csv.gz" \
  --baseline-transitions "$ROOT/baseline/raw_transitions.csv.gz" \
  --output-dir "$ROOT/paths" --scope external
for STEP in 0 1; do
  python -u tasks/audit_bioaware_metdna3_candidate_edge_ms2.py \
    --recursive-dir "$ROOT/recursive" --development-dir "$MANIFEST" \
    --truth-name external_level1.csv.gz --query-cache "$ROOT/cache" \
    --candidate-scores "$ROOT/scores/candidate_scores.csv.gz" \
    --baseline-transitions "$ROOT/baseline/raw_transitions.csv.gz" \
    --feature-ms2-dir "$ROOT/feature_ms2" --output-dir "$ROOT/edge_step$STEP" \
    --maximum-network-step "$STEP" --scope external
done
python -u tasks/audit_bioaware_candidate_rule_likelihood.py \
  --scores "$ROOT/scores/candidate_scores.csv.gz" --queries "$ROOT/cache/queries.csv.gz" \
  --query-tensors "$ROOT/cache/query_tensors.npz" \
  --references "$ROOT/cache/candidate_references.csv.gz" \
  --all-queries-unresolved --output-dir "$ROOT/rules"
python -u tasks/summarize_bioaware_metdna3_candidate_edge_decision.py \
  --paths "$ROOT/paths/candidate_paths.csv.gz" \
  --edge-evidence "$ROOT/edge_step0/candidate_edge_evidence.csv.gz" \
  --baseline "$ROOT/baseline/raw_transitions.csv.gz" \
  --output-dir "$ROOT/depth3" --scope external
python -u tasks/audit_bioaware_metdna3_smn_headroom.py \
  --recursive-dir "$ROOT/recursive" --development-dir "$MANIFEST" \
  --truth-name external_level1.csv.gz --query-cache "$ROOT/cache/queries.csv.gz" \
  --candidate-scores "$ROOT/scores/candidate_scores.csv.gz" \
  --baseline-transitions "$ROOT/baseline/raw_transitions.csv.gz" \
  --mrn-decision-dir "$ROOT/depth3" --output-dir "$ROOT/smn" --scope external
python -u tasks/audit_bioaware_metdna3_rt_headroom.py \
  --development-dir "$MANIFEST" --truth-name external_level1.csv.gz \
  --query-cache "$ROOT/cache/queries.csv.gz" \
  --candidate-scores "$ROOT/scores/candidate_scores.csv.gz" \
  --baseline-transitions "$ROOT/baseline/raw_transitions.csv.gz" \
  --mrn-decision-dir "$ROOT/depth3" --smn-dir "$ROOT/smn" \
  --output-dir "$ROOT/rt" --scope external
python -u tasks/build_bioaware_zero_weight_decoder_evidence.py \
  --scores "$ROOT/scores/candidate_scores.csv.gz" --artifact "$ART" \
  --output-dir "$ROOT/decoder"
python -u tasks/build_bioaware_candidate_evidence_ledger.py \
  --dreams "$ROOT/scores/candidate_scores.csv.gz" \
  --known-edge "$ROOT/edge_step0/candidate_edge_evidence.csv.gz" \
  --predicted-edge "$ROOT/edge_step1/candidate_edge_evidence.csv.gz" \
  --smn "$ROOT/smn/candidate_structural_evidence.csv.gz" \
  --rt "$ROOT/rt/candidate_rt_evidence.csv.gz" \
  --decoder "$ROOT/decoder/candidate_scores.csv.gz" \
  --rules "$ROOT/rules/candidate_rule_scores.csv.gz" --output-dir "$ROOT/ledger"
python -u tasks/evaluate_bioaware_v3_frozen_router.py \
  --artifact "$ART" --ledger "$ROOT/ledger/candidate_evidence.csv.gz" \
  --depth3 "$ROOT/depth3/query_transitions.csv.gz" \
  --queries "$ROOT/cache/queries.csv.gz" --output-dir "$ROOT/result" \
  --scope external_panel --internal-report "$INTERNAL"
python -u tasks/audit_bioaware_v3_result_mechanisms.py \
  --transitions "$ROOT/result/query_transitions.csv.gz" \
  --ledger "$ROOT/ledger/candidate_evidence.csv.gz" --artifact "$ART" \
  --output-dir "$ROOT/audit" --scope "external:$UNIT"
if [[ "$UNIT" != "BV2cell__hilic" ]]; then
  python -u tasks/evaluate_bioaware_v4_high_precision_router.py \
    --artifact "$ART_V4" --ledger "$ROOT/ledger/candidate_evidence.csv.gz" \
    --queries "$ROOT/cache/queries.csv.gz" --panel "$UNIT" \
    --output-dir "$ROOT/result_v4"
fi
if [[ "$UNIT" != "BV2cell__hilic" && "$UNIT" != "BV2cell__rplc" && "$UNIT" != "Mouse_brain__hilic" ]]; then
  python -u tasks/evaluate_bioaware_v6_identifiable_router.py \
    --artifact "$ART_V6" --ledger "$ROOT/ledger/candidate_evidence.csv.gz" \
    --queries "$ROOT/cache/queries.csv.gz" --panel "$UNIT" \
    --output-dir "$ROOT/result_v6"
fi
echo "=== BioAware frozen external unit $UNIT completed: $(date) ==="
