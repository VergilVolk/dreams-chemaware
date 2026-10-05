#!/bin/bash
#SBATCH --job-name=bio_b47_seed
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=bioaware_b47_seed_%j.out
#SBATCH --error=bioaware_b47_seed_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

GRAPH=data/validation/bioaware_b47_truthblind_candidate_graph_20260914_v1
EMBED=data/validation/bioaware_b47_truthblind_embeddings_20260914_v1
PARTICIPANTS=data/reference/bioaware_rhea_offline_20260827/rhea_participants.csv.gz
FINAL=data/validation/bioaware_b47_truthblind_seeds_20260914_v1
WORK="${FINAL}.partial.${SLURM_JOB_ID}"

echo "=== BioAware B47 truth-blind seed construction started: $(date) ==="
test ! -e "$FINAL" || { echo "Refusing to overwrite frozen result: $FINAL" >&2; exit 3; }
test ! -e "$WORK" || { echo "Refusing to overwrite partial result: $WORK" >&2; exit 3; }
required=(
  "$GRAPH/report.json" "$GRAPH/queries.csv.gz" "$GRAPH/candidate_references.csv.gz"
  "$EMBED/report.json" "$EMBED/query_embeddings.npy" "$EMBED/reference_embeddings.npy"
  "$EMBED/reference_rows.npy" "$EMBED/query_index.csv" "$PARTICIPANTS"
  tasks/bioaware_b47_seed_core.py tasks/bioaware_b47_truthblind_io.py
  tasks/build_bioaware_b47_truthblind_seeds.py tasks/test_bioaware_b47_seed_core.py
  tasks/validate_bioaware_b47_truthblind_seeds.py
)
for path in "${required[@]}"; do
  test -s "$path" || { echo "Missing required dependency: $path" >&2; exit 2; }
done
python -c "import numpy,pandas,torch; assert torch.cuda.is_available(); print('[B47 seed dependencies] PASS')"
python -m py_compile \
  tasks/bioaware_b47_seed_core.py tasks/build_bioaware_b47_truthblind_seeds.py \
  tasks/test_bioaware_b47_seed_core.py tasks/validate_bioaware_b47_truthblind_seeds.py
python -u tasks/test_bioaware_b47_seed_core.py
python -u tasks/build_bioaware_b47_truthblind_seeds.py \
  --graph-dir "$GRAPH" --embedding-dir "$EMBED" --participants "$PARTICIPANTS" \
  --output "$WORK" --device cuda --score-batch-size 65536 --maximum-seed-degree 250
test -s "$WORK/report.json" || { echo "Missing B47 seed report" >&2; exit 4; }
python -u tasks/validate_bioaware_b47_truthblind_seeds.py --input "$WORK"
mv "$WORK" "$FINAL"
echo "=== BioAware B47 truth-blind seed construction completed: $(date) ==="
echo "Result: $FINAL/report.json"
