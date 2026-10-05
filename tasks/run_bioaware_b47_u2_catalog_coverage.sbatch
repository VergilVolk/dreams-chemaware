#!/bin/bash
#SBATCH --job-name=bio_b47_u2
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=bioaware_b47_u2_catalog_coverage_%j.out
#SBATCH --error=bioaware_b47_u2_catalog_coverage_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

U1C=data/validation/bioaware_b47_u1c_seed_denominator_20260921_v1
SEEDS=data/validation/bioaware_b47_truthblind_seeds_20260914_v1
RHEA=data/reference/bioaware_rhea_offline_20260827/rhea_participants.csv.gz
KEGG_DIR=data/reference/metdna2_kegg_network_20260828
EMRN_DIR=data/reference/metdna2_emrn_network_20260828
OUTPUT=data/validation/bioaware_b47_u2_catalog_coverage_20260921_v1

echo "=== BioAware B47 U2 catalogue coverage started: $(date) ==="
required=(
  "$U1C/report.json" "$U1C/query_stage_ledger.csv.gz"
  "$SEEDS/candidate_scores.csv.gz" "$RHEA"
  "$KEGG_DIR/metdna2_kegg_compounds.csv.gz"
  "$KEGG_DIR/metdna2_kegg_edges.csv.gz"
  "$EMRN_DIR/metdna2_emrn_compounds.csv.gz"
  "$EMRN_DIR/metdna2_emrn_edges.csv.gz"
  tasks/audit_bioaware_b47_u2_catalog_coverage.py
  tasks/test_bioaware_b47_u2_catalog_coverage.py
  tasks/validate_bioaware_b47_u2_catalog_coverage.py
)
for path in "${required[@]}"; do
  test -s "$path" || { echo "Missing required dependency: $path" >&2; exit 2; }
done
python -c "import numpy,pandas; print('[BioAware B47 U2 dependencies] PASS')"
python -m py_compile \
  tasks/audit_bioaware_b47_u2_catalog_coverage.py \
  tasks/test_bioaware_b47_u2_catalog_coverage.py \
  tasks/validate_bioaware_b47_u2_catalog_coverage.py
python -u tasks/test_bioaware_b47_u2_catalog_coverage.py

if test -d "$OUTPUT"; then
  echo "[resume] validating frozen U2 result"
  python -u tasks/validate_bioaware_b47_u2_catalog_coverage.py --input "$OUTPUT"
else
  WORK="${OUTPUT}.partial.${SLURM_JOB_ID}"
  test ! -e "$WORK" || { echo "Refusing stale U2 partial: $WORK" >&2; exit 3; }
  python -u tasks/audit_bioaware_b47_u2_catalog_coverage.py \
    --u1c-dir "$U1C" --seed-dir "$SEEDS" \
    --rhea-participants "$RHEA" \
    --kegg-compounds "$KEGG_DIR/metdna2_kegg_compounds.csv.gz" \
    --kegg-edges "$KEGG_DIR/metdna2_kegg_edges.csv.gz" \
    --emrn-compounds "$EMRN_DIR/metdna2_emrn_compounds.csv.gz" \
    --emrn-edges "$EMRN_DIR/metdna2_emrn_edges.csv.gz" \
    --output "$WORK" --maximum-degree 250
  python -u tasks/validate_bioaware_b47_u2_catalog_coverage.py --input "$WORK"
  mv "$WORK" "$OUTPUT"
fi
echo "=== BioAware B47 U2 catalogue coverage completed: $(date) ==="
echo "Result: $OUTPUT/report.json"
