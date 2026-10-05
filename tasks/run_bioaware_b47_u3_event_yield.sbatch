#!/bin/bash
#SBATCH --job-name=bio_b47_u3
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=bioaware_b47_u3_event_yield_%j.out
#SBATCH --error=bioaware_b47_u3_event_yield_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

GRAPH=data/validation/bioaware_b47_truthblind_candidate_graph_20260914_v1
SEEDS=data/validation/bioaware_b47_truthblind_seeds_20260914_v1
U1C=data/validation/bioaware_b47_u1c_seed_denominator_20260921_v1
U2=data/validation/bioaware_b47_u2_catalog_coverage_20260921_v1
RHEA=data/reference/bioaware_rhea_reactome_direction_20260830/rhea_participants.csv.gz
AMENDMENT=docs/BIOAWARE_B47_U3_EVENT_YIELD_PROTOCOL_AMENDMENT_20260921.md
OUTPUT=data/validation/bioaware_b47_u3_event_yield_20260922_v3

echo "=== BioAware B47 U3 truth-blind event yield started: $(date) ==="
required=(
  "$GRAPH/report.json" "$GRAPH/queries.csv.gz" "$GRAPH/candidate_references.csv.gz"
  "$SEEDS/report.json" "$SEEDS/candidate_scores.csv.gz" "$SEEDS/seeds_primary.csv.gz"
  "$U1C/report.json" "$U2/report.json" "$RHEA" "$AMENDMENT"
  "$(dirname "$RHEA")/report.json"
  annotation/bioaware.py annotation/bioaware_relations.py
  tasks/bioaware_b47_u3_core.py
  tasks/audit_bioaware_b47_u3_event_yield.py
  tasks/test_bioaware_b47_u3_event_yield.py
  tasks/validate_bioaware_b47_u3_event_yield.py
)
for path in "${required[@]}"; do
  test -s "$path" || { echo "Missing required dependency: $path" >&2; exit 2; }
done
python -c "import numpy,pandas; print('[BioAware B47 U3 dependencies] PASS')"
python -m py_compile \
  tasks/bioaware_b47_u3_core.py \
  tasks/audit_bioaware_b47_u3_event_yield.py \
  tasks/test_bioaware_b47_u3_event_yield.py \
  tasks/validate_bioaware_b47_u3_event_yield.py
python -u tasks/test_bioaware_b47_u3_event_yield.py
python -u tasks/audit_bioaware_b47_u3_event_yield.py \
  --graph-dir "$GRAPH" --seed-dir "$SEEDS" \
  --u1c-dir "$U1C" --u2-dir "$U2" \
  --participants "$RHEA" --protocol-amendment "$AMENDMENT" \
  --output data/validation/.bioaware_b47_u3_preflight_unused \
  --maximum-degree 250 --null-repeats 20 --seed 20260922 --preflight-only

if test -d "$OUTPUT"; then
  echo "[resume] validating frozen U3 result"
  python -u tasks/validate_bioaware_b47_u3_event_yield.py --input "$OUTPUT"
else
  WORK="${OUTPUT}.partial.${SLURM_JOB_ID}"
  test ! -e "$WORK" || { echo "Refusing stale U3 partial: $WORK" >&2; exit 3; }
  python -u tasks/audit_bioaware_b47_u3_event_yield.py \
    --graph-dir "$GRAPH" --seed-dir "$SEEDS" \
    --u1c-dir "$U1C" --u2-dir "$U2" \
    --participants "$RHEA" --protocol-amendment "$AMENDMENT" \
    --output "$WORK" --maximum-degree 250 --null-repeats 20 --seed 20260922
  python -u tasks/validate_bioaware_b47_u3_event_yield.py --input "$WORK"
  mv "$WORK" "$OUTPUT"
fi
echo "=== BioAware B47 U3 truth-blind event yield completed: $(date) ==="
echo "Result: $OUTPUT/report.json"
