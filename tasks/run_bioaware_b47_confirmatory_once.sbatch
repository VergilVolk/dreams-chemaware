#!/bin/bash
# One-time Track-C evaluation.  The job exits before touching truth unless the
# repaired R1-R5 gate explicitly authorizes opening.
#SBATCH --job-name=b47_confirm_once
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=data/validation/b47_confirm_once_%j.out
#SBATCH --error=data/validation/b47_confirm_once_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

: "${B47_REPAIR_DIR:?Set B47_REPAIR_DIR}"
: "${B47_CANDIDATE_MANIFEST:?Set B47_CANDIDATE_MANIFEST}"
: "${B47_UNARY_SCORES:?Set B47_UNARY_SCORES}"
: "${B47_CATALOGUE_SCORES:?Set B47_CATALOGUE_SCORES}"
: "${B47_NETWORK_SCORES:?Set B47_NETWORK_SCORES}"
: "${B47_DEGREE_REWIRED_SCORES:?Set B47_DEGREE_REWIRED_SCORES}"
: "${B47_SEED_PERMUTED_SCORES:?Set B47_SEED_PERMUTED_SCORES}"
: "${B47_MATCHED_NONNEIGHBOR_SCORES:?Set B47_MATCHED_NONNEIGHBOR_SCORES}"

SEAL="${B47_CONFIRMATORY_SEAL:-data/validation/bioaware_b47_confirmatory/B47_TRUTH_OPENED.lock}"
OUT="data/validation/bioaware_b47_confirmatory_run_${SLURM_JOB_ID}"
[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 4; }

required=(
  "$B47_REPAIR_DIR/report.json" "$B47_REPAIR_DIR/repaired_action_ledger.csv.gz"
  "$B47_CANDIDATE_MANIFEST" "$B47_UNARY_SCORES" "$B47_CATALOGUE_SCORES"
  "$B47_NETWORK_SCORES" "$B47_DEGREE_REWIRED_SCORES"
  "$B47_SEED_PERMUTED_SCORES" "$B47_MATCHED_NONNEIGHBOR_SCORES"
  tasks/freeze_bioaware_b47_repaired_action.py
  tasks/evaluate_bioaware_b47_repaired_once.py
  tasks/evaluate_bioaware_unified_benchmark.py
  tasks/adjudicate_bioaware_b47_external.py
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || { echo "B47_CONFIRMATORY_MISSING_INPUT: $path" >&2; exit 5; }
done
mkdir -p "$OUT"

python -m py_compile \
  tasks/freeze_bioaware_b47_repaired_action.py \
  tasks/evaluate_bioaware_b47_repaired_once.py \
  tasks/adjudicate_bioaware_b47_external.py

# This command fails closed before reading annotation truth if authorization is false.
python -u tasks/freeze_bioaware_b47_repaired_action.py \
  --repair-dir "$B47_REPAIR_DIR" --unary-scores "$B47_UNARY_SCORES" \
  --output "$OUT/bioaware_exact_event_scores.csv.gz"

python -u tasks/evaluate_bioaware_b47_repaired_once.py \
  --repair-dir "$B47_REPAIR_DIR" \
  --candidate-manifest "$B47_CANDIDATE_MANIFEST" \
  --unary-scores "$B47_UNARY_SCORES" \
  --exact-event-scores "$OUT/bioaware_exact_event_scores.csv.gz" \
  --control "catalogue_degree=$B47_CATALOGUE_SCORES" \
  --control "network=$B47_NETWORK_SCORES" \
  --control "degree_rewired=$B47_DEGREE_REWIRED_SCORES" \
  --control "seed_context_permuted=$B47_SEED_PERMUTED_SCORES" \
  --control "matched_non_neighbor=$B47_MATCHED_NONNEIGHBOR_SCORES" \
  --seal "$SEAL" --output "$OUT/evaluation" --expected-queries 51976

python -u tasks/adjudicate_bioaware_b47_external.py \
  --evaluation "$OUT/evaluation" \
  --candidate-manifest "$B47_CANDIDATE_MANIFEST" \
  --output "$OUT/adjudication.json"

sha256sum \
  "$B47_REPAIR_DIR/report.json" "$B47_REPAIR_DIR/repaired_action_ledger.csv.gz" \
  "$OUT/bioaware_exact_event_scores.csv.gz" "$OUT/evaluation/report.json" \
  "$OUT/adjudication.json" > "$OUT/SHA256SUMS.txt"
echo "B47 one-time confirmatory evaluation complete: $OUT"
