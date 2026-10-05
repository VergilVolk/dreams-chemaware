#!/bin/bash
#SBATCH --job-name=bio_b47_u0
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=bioaware_b47_u0_%j.out
#SBATCH --error=bioaware_b47_u0_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

GRAPH=data/validation/bioaware_b47_truthblind_candidate_graph_20260914_v1
EMBED=data/validation/bioaware_b47_truthblind_embeddings_20260914_v1
FINAL=data/validation/bioaware_b47_u0_reference_bias_20260921_v2
WORK="${FINAL}.partial.${SLURM_JOB_ID}"

echo "=== BioAware B47-U0 reference-bias audit started: $(date) ==="
test ! -e "$FINAL" || { echo "Refusing to overwrite frozen result: $FINAL" >&2; exit 3; }
test ! -e "$WORK" || { echo "Refusing to overwrite partial result: $WORK" >&2; exit 3; }
required=(
  "$GRAPH/report.json" "$GRAPH/queries.csv.gz" "$GRAPH/candidate_references.csv.gz"
  "$EMBED/report.json" "$EMBED/query_embeddings.npy" "$EMBED/reference_embeddings.npy"
  "$EMBED/reference_rows.npy" "$EMBED/query_index.csv"
  tasks/bioaware_b47_truthblind_io.py tasks/bioaware_b47_u0_core.py
  tasks/audit_bioaware_b47_u0_reference_bias.py
  tasks/test_bioaware_b47_u0_reference_bias.py
  tasks/validate_bioaware_b47_u0_reference_bias.py
)
for path in "${required[@]}"; do
  test -s "$path" || { echo "Missing required dependency: $path" >&2; exit 2; }
done
python -c "import numpy,pandas,torch; assert torch.cuda.is_available(); print('[B47-U0 dependencies] PASS')"
python -m py_compile \
  tasks/bioaware_b47_u0_core.py \
  tasks/audit_bioaware_b47_u0_reference_bias.py \
  tasks/test_bioaware_b47_u0_reference_bias.py \
  tasks/validate_bioaware_b47_u0_reference_bias.py
python -u tasks/test_bioaware_b47_u0_reference_bias.py
python -u tasks/audit_bioaware_b47_u0_reference_bias.py \
  --graph-dir "$GRAPH" \
  --embedding-dir "$EMBED" \
  --output "$WORK" \
  --device cuda \
  --score-batch-size 65536 \
  --preflight-only
test ! -e "$WORK" || { echo "B47-U0 preflight unexpectedly created output" >&2; exit 4; }

python -u tasks/audit_bioaware_b47_u0_reference_bias.py \
  --graph-dir "$GRAPH" \
  --embedding-dir "$EMBED" \
  --output "$WORK" \
  --device cuda \
  --score-batch-size 65536
test -s "$WORK/report.json" || { echo "Missing B47-U0 report" >&2; exit 4; }
python -u tasks/validate_bioaware_b47_u0_reference_bias.py --input "$WORK"
mv "$WORK" "$FINAL"
echo "=== BioAware B47-U0 reference-bias audit completed: $(date) ==="
echo "Result: $FINAL/report.json"
