#!/bin/bash
#SBATCH --job-name=bio_b47_embed
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=bioaware_b47_embed_%j.out
#SBATCH --error=bioaware_b47_embed_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GRAPH=data/validation/bioaware_b47_truthblind_candidate_graph_20260914_v1
FINAL=data/validation/bioaware_b47_truthblind_embeddings_20260914_v1
WORK="${FINAL}.partial.${SLURM_JOB_ID}"

echo "=== BioAware B47 truth-blind shared embedding started: $(date) ==="
test ! -e "$FINAL" || { echo "Refusing to overwrite frozen result: $FINAL" >&2; exit 3; }
test ! -e "$WORK" || { echo "Refusing to overwrite partial result: $WORK" >&2; exit 3; }
required=(
  "$GRAPH/report.json"
  "$GRAPH/queries.csv.gz"
  "$GRAPH/candidate_references.csv.gz"
  "$GRAPH/queries.mgf"
  data/models/MassSpecGym_MurckoHist_split.hdf5
  data/e1/official_embedding_slim.pt
  dreams/models/pretrained/ssl_model_server.pt
  tasks/bioaware_b47_truthblind_io.py
  tasks/encode_bioaware_b47_truthblind_embeddings.py
  tasks/test_bioaware_b47_truthblind_embedding.py
  tasks/validate_bioaware_b47_truthblind_embeddings.py
)
for path in "${required[@]}"; do
  test -s "$path" || { echo "Missing required dependency: $path" >&2; exit 2; }
done
python -c "import h5py,numpy,torch; assert torch.cuda.is_available(); print('[B47 embedding dependencies] PASS')"
python -m py_compile \
  tasks/bioaware_b47_truthblind_io.py \
  tasks/encode_bioaware_b47_truthblind_embeddings.py \
  tasks/test_bioaware_b47_truthblind_embedding.py \
  tasks/validate_bioaware_b47_truthblind_embeddings.py
python -u tasks/test_bioaware_b47_truthblind_embedding.py

python -u tasks/encode_bioaware_b47_truthblind_embeddings.py \
  --graph-dir "$GRAPH" \
  --reference-hdf5 data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --official-checkpoint data/e1/official_embedding_slim.pt \
  --architecture-checkpoint dreams/models/pretrained/ssl_model_server.pt \
  --output "$WORK" \
  --device cuda --batch-size 128 --n-highest-peaks 100
test -s "$WORK/report.json" || { echo "Missing B47 embedding report" >&2; exit 4; }
test -s "$WORK/query_embeddings.npy" || { echo "Missing B47 query embeddings" >&2; exit 4; }
test -s "$WORK/reference_embeddings.npy" || { echo "Missing B47 reference embeddings" >&2; exit 4; }
python -u tasks/validate_bioaware_b47_truthblind_embeddings.py --input "$WORK"
mv "$WORK" "$FINAL"
echo "=== BioAware B47 truth-blind shared embedding completed: $(date) ==="
echo "Result: $FINAL/report.json"
