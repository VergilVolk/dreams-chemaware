#!/bin/bash
#SBATCH --job-name=bio_v3_decoy
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-159%16
#SBATCH --output=data/validation/bio_v3_decoy_%A_%a.out
#SBATCH --error=data/validation/bio_v3_decoy_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD/tasks:$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

UNITS=(BV2cell__hilic BV2cell__rplc Mouse_brain__hilic Mouse_brain__rplc Mouse_liver__hilic Mouse_liver__rplc NIST_plasma__hilic NIST_plasma__rplc)
UNIT_INDEX=$((SLURM_ARRAY_TASK_ID / 20))
REPEAT=$((SLURM_ARRAY_TASK_ID % 20))
UNIT=${UNITS[$UNIT_INDEX]}
TAG=$(printf "repeat_%02d" "$REPEAT")

MANIFEST=data/validation/bioaware_metdna3_external_manifest_v1/$UNIT
REAL=data/validation/bioaware_metdna3_external_v3_v1/$UNIT
ROOT=$REAL/degree_decoys/$TAG
NETWORK=data/reference/bioaware_degree_preserving_decoys_v1/$TAG
ART=data/validation/bioaware_v3_consensus_router_frozen_v2_20260830/artifact.json
ART_V4=data/validation/bioaware_v4_high_precision_router_frozen_v1_20260830/artifact.json
ART_V6=data/validation/bioaware_v6_identifiable_router_frozen_v2_20260830/artifact.json
INTERNAL=data/validation/bioaware_metdna3_internal_rplc_frozen_v3_result_v1/report.json

for REQUIRED in \
  "$REAL/recursive/report.json" \
  "$REAL/feature_ms2/report.json" \
  "$REAL/result/report.json" \
  "$NETWORK/metdna2_emrn_edges.csv.gz"; do
  test -f "$REQUIRED" || { echo "Missing prerequisite: $REQUIRED" >&2; exit 2; }
done
test ! -e "$ROOT" || { echo "fail-closed: output exists: $ROOT" >&2; exit 3; }

echo "=== BioAware V3 degree-decoy $UNIT $TAG started: $(date) ==="
python -u tasks/build_bioaware_metdna3_candidate_path_table.py \
  --recursive-dir "$REAL/recursive" --network-dir "$NETWORK" \
  --development-dir "$MANIFEST" --truth-name external_level1.csv.gz \
  --query-cache "$REAL/cache/queries.csv.gz" \
  --candidate-scores "$REAL/scores/candidate_scores.csv.gz" \
  --baseline-transitions "$REAL/baseline/raw_transitions.csv.gz" \
  --output-dir "$ROOT/paths" --scope external
for STEP in 0 1; do
  python -u tasks/audit_bioaware_metdna3_candidate_edge_ms2.py \
    --recursive-dir "$REAL/recursive" --network-dir "$NETWORK" \
    --development-dir "$MANIFEST" --truth-name external_level1.csv.gz \
    --query-cache "$REAL/cache" \
    --candidate-scores "$REAL/scores/candidate_scores.csv.gz" \
    --baseline-transitions "$REAL/baseline/raw_transitions.csv.gz" \
    --feature-ms2-dir "$REAL/feature_ms2" --output-dir "$ROOT/edge_step$STEP" \
    --maximum-network-step "$STEP" --scope external
done
python -u tasks/summarize_bioaware_metdna3_candidate_edge_decision.py \
  --paths "$ROOT/paths/candidate_paths.csv.gz" \
  --edge-evidence "$ROOT/edge_step0/candidate_edge_evidence.csv.gz" \
  --baseline "$REAL/baseline/raw_transitions.csv.gz" \
  --output-dir "$ROOT/depth3" --scope external

# Structure-network, RT, rule and decoder evidence are deliberately held fixed.
# Only evidence derived from the reaction graph is rebuilt from the decoy graph.
python -u tasks/build_bioaware_candidate_evidence_ledger.py \
  --dreams "$REAL/scores/candidate_scores.csv.gz" \
  --known-edge "$ROOT/edge_step0/candidate_edge_evidence.csv.gz" \
  --predicted-edge "$ROOT/edge_step1/candidate_edge_evidence.csv.gz" \
  --smn "$REAL/smn/candidate_structural_evidence.csv.gz" \
  --rt "$REAL/rt/candidate_rt_evidence.csv.gz" \
  --decoder "$REAL/decoder/candidate_scores.csv.gz" \
  --rules "$REAL/rules/candidate_rule_scores.csv.gz" --output-dir "$ROOT/ledger"
python -u tasks/evaluate_bioaware_v3_frozen_router.py \
  --artifact "$ART" --ledger "$ROOT/ledger/candidate_evidence.csv.gz" \
  --depth3 "$ROOT/depth3/query_transitions.csv.gz" \
  --queries "$REAL/cache/queries.csv.gz" --output-dir "$ROOT/result" \
  --scope external_panel --internal-report "$INTERNAL"
if [[ "$UNIT" != "BV2cell__hilic" ]]; then
  python -u tasks/evaluate_bioaware_v4_high_precision_router.py \
    --artifact "$ART_V4" --ledger "$ROOT/ledger/candidate_evidence.csv.gz" \
    --queries "$REAL/cache/queries.csv.gz" --panel "$UNIT" \
    --output-dir "$ROOT/result_v4"
fi
if [[ "$UNIT" != "BV2cell__hilic" && "$UNIT" != "BV2cell__rplc" && "$UNIT" != "Mouse_brain__hilic" ]]; then
  python -u tasks/evaluate_bioaware_v6_identifiable_router.py \
    --artifact "$ART_V6" --ledger "$ROOT/ledger/candidate_evidence.csv.gz" \
    --queries "$REAL/cache/queries.csv.gz" --panel "$UNIT" \
    --output-dir "$ROOT/result_v6"
fi
echo "=== BioAware V3 degree-decoy $UNIT $TAG completed: $(date) ==="
