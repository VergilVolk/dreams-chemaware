#!/bin/bash
#SBATCH --job-name=chemaware_g0
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/chemaware_g0_%j.out
#SBATCH --error=data/validation/chemaware_g0_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "=== ChemAware G0 full-graph audit started: $(date) ==="

required=(
  data/models/MassSpecGym_MurckoHist_split.hdf5
  data/validation/g8r_error_atlas_listwise_cache.npz
  data/validation/g8r_p2b_rank_fusion.json
  data/validation/g8r_p3_test/p3_lock_summary.json
  dreams/models/chem_aware/chem_rules_data.json
  dreams/models/chem_aware/chem_rules_massbank.json
  tasks/build_chemaware_g0_rule_cache.py
  tasks/audit_chemaware_g0_full_graph.py
)
for file in "${required[@]}"; do
  [[ -f "$file" ]] || { echo "MISSING: $file" >&2; exit 2; }
done

rule_cache=data/validation/g8r_chemaware_g0_rule_cache.npz
if [[ ! -f "$rule_cache" ]]; then
  echo "=== stage 1/2: full 3,486-rule spectrum cache ==="
  python -u tasks/build_chemaware_g0_rule_cache.py \
    --graph data/validation/g8r_error_atlas_listwise_cache.npz \
    --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
    --core-rules dreams/models/chem_aware/chem_rules_data.json \
    --massbank-rules dreams/models/chem_aware/chem_rules_massbank.json \
    --output "$rule_cache" \
    --tolerance 0.02
else
  [[ -f data/validation/g8r_chemaware_g0_rule_cache.json ]] || {
    echo "REFUSE: rule cache exists without its audit JSON" >&2
    exit 3
  }
  echo "=== stage 1/2: reusing existing audited full rule cache ==="
fi

for output in \
  data/validation/g8r_chemaware_g0_full_audit.json \
  data/validation/g8r_chemaware_g0_query_priority.csv.gz \
  data/validation/g8r_chemaware_g0_pair_rule_features.npz \
  data/validation/g8r_chemaware_g0_rule_level.csv.gz; do
  [[ ! -e "$output" ]] || { echo "REFUSE existing formal output: $output" >&2; exit 4; }
done

echo "=== stage 2/2: full candidate-graph/headroom/rule-evidence audit ==="
python -u tasks/audit_chemaware_g0_full_graph.py \
  --graph data/validation/g8r_error_atlas_listwise_cache.npz \
  --rule-cache "$rule_cache" \
  --artifact data/validation/g8r_p2b_rank_fusion.json \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --p3-dir data/validation/g8r_p3_test \
  --target-delta 0.04 \
  --folds 5 \
  --pair-block-size 20000

echo "=== ChemAware G0 full-graph audit completed: $(date) ==="

