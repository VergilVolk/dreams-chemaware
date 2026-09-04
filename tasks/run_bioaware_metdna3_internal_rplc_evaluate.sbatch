#!/bin/bash
#SBATCH --job-name=bioaware_m3_rplc_eval
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=data/validation/bioaware_m3_rplc_eval_%j.out
#SBATCH --error=data/validation/bioaware_m3_rplc_eval_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD/tasks:$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

ART=data/validation/bioaware_v3_consensus_router_frozen_v2_20260830/artifact.json
CACHE=data/validation/bioaware_metdna3_internal_rplc_dreams_cache_v1
SCORES=data/validation/bioaware_metdna3_internal_rplc_dreams_official_v1
EDGE0=data/validation/bioaware_metdna3_internal_rplc_edge_step0_v1
EDGE1=data/validation/bioaware_metdna3_internal_rplc_edge_step1_v1
RULES=data/validation/bioaware_metdna3_internal_rplc_rules_v1
MRN=data/validation/bioaware_metdna3_internal_rplc_depth3_v1
SMN=data/validation/bioaware_metdna3_internal_rplc_smn_v1
RT=data/validation/bioaware_metdna3_internal_rplc_rt_v1
DECODER=data/validation/bioaware_metdna3_internal_rplc_zero_decoder_v1
LEDGER=data/validation/bioaware_metdna3_internal_rplc_ledger_v1
RESULT=data/validation/bioaware_metdna3_internal_rplc_frozen_v3_result_v1
REPLAY=data/validation/bioaware_v3_frozen_deploy_replay_v1

echo "=== BioAware frozen V3 internal RPLC evaluation started: $(date) ==="
python -u tasks/evaluate_bioaware_v3_frozen_router.py \
  --artifact "$ART" \
  --ledger data/validation/bioaware_candidate_evidence_ledger_v1/candidate_evidence.csv.gz \
  --depth3 data/validation/bioaware_metdna3_candidate_edge_decision_v1/query_transitions.csv.gz \
  --queries data/validation/bioaware_metdna3_dreams_cache_v2/queries.csv.gz \
  --output-dir "$REPLAY" --scope development_replay
python -u tasks/build_bioaware_zero_weight_decoder_evidence.py \
  --scores "$SCORES/candidate_scores.csv.gz" --artifact "$ART" --output-dir "$DECODER"
python -u tasks/build_bioaware_candidate_evidence_ledger.py \
  --dreams "$SCORES/candidate_scores.csv.gz" \
  --known-edge "$EDGE0/candidate_edge_evidence.csv.gz" \
  --predicted-edge "$EDGE1/candidate_edge_evidence.csv.gz" \
  --smn "$SMN/candidate_structural_evidence.csv.gz" \
  --rt "$RT/candidate_rt_evidence.csv.gz" \
  --decoder "$DECODER/candidate_scores.csv.gz" \
  --rules "$RULES/candidate_rule_scores.csv.gz" --output-dir "$LEDGER"
python -u tasks/evaluate_bioaware_v3_frozen_router.py \
  --artifact "$ART" --ledger "$LEDGER/candidate_evidence.csv.gz" \
  --depth3 "$MRN/query_transitions.csv.gz" --queries "$CACHE/queries.csv.gz" \
  --output-dir "$RESULT" --scope internal_rplc --replay-report "$REPLAY/report.json"
echo "=== BioAware frozen V3 internal RPLC evaluation completed: $(date) ==="
