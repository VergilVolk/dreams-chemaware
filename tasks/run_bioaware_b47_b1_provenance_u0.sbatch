#!/bin/bash
#SBATCH --job-name=bio_b47_b1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=bioaware_b47_b1_%j.out
#SBATCH --error=bioaware_b47_b1_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

GRAPH=data/validation/bioaware_b47_truthblind_candidate_graph_20260914_v1
EMBED=data/validation/bioaware_b47_truthblind_embeddings_20260914_v1
SEEDS=data/validation/bioaware_b47_truthblind_seeds_20260914_v1
U0=data/validation/bioaware_b47_u0_reference_bias_20260921_v2
REGISTRY=data/validation/bioaware_b47_b1_artifact_registry_20260921_v1
PARTICIPANTS=data/reference/bioaware_rhea_offline_20260827/rhea_participants.csv.gz
SEED_WORK="${SEEDS}.partial.${SLURM_JOB_ID}"
U0_WORK="${U0}.partial.${SLURM_JOB_ID}"
REGISTRY_WORK="${REGISTRY}.partial.${SLURM_JOB_ID}"

echo "=== BioAware B47 B1 provenance + U0 started: $(date) ==="
required=(
  "$GRAPH/report.json" "$GRAPH/queries.csv.gz" "$GRAPH/candidate_references.csv.gz"
  "$GRAPH/queries.mgf" "$EMBED/report.json" "$EMBED/query_embeddings.npy"
  "$EMBED/reference_embeddings.npy" "$EMBED/reference_rows.npy" "$EMBED/query_index.csv"
  "$PARTICIPANTS" tasks/bioaware_b47_truthblind_io.py tasks/bioaware_b47_seed_core.py
  tasks/bioaware_b47_u0_core.py tasks/build_bioaware_b47_truthblind_seeds.py
  tasks/audit_bioaware_b47_u0_reference_bias.py
  tasks/audit_bioaware_b47_b1_artifact_registry.py
  tasks/validate_bioaware_b47_truthblind_embeddings.py
  tasks/validate_bioaware_b47_truthblind_seeds.py
  tasks/validate_bioaware_b47_u0_reference_bias.py
  tasks/validate_bioaware_b47_b1_artifact_registry.py
  tasks/test_bioaware_b47_seed_core.py
  tasks/test_bioaware_b47_u0_reference_bias.py
  tasks/test_bioaware_b47_b1_artifact_registry.py
)
for path in "${required[@]}"; do
  test -s "$path" || { echo "Missing required dependency: $path" >&2; exit 2; }
done
python -c "import numpy,pandas,torch; assert torch.cuda.is_available(); print('[B47 B1 dependencies] PASS')"
python -m py_compile \
  tasks/bioaware_b47_seed_core.py tasks/bioaware_b47_u0_core.py \
  tasks/build_bioaware_b47_truthblind_seeds.py \
  tasks/audit_bioaware_b47_u0_reference_bias.py \
  tasks/audit_bioaware_b47_b1_artifact_registry.py \
  tasks/validate_bioaware_b47_b1_artifact_registry.py
python -u tasks/test_bioaware_b47_seed_core.py
python -u tasks/test_bioaware_b47_u0_reference_bias.py
python -u tasks/test_bioaware_b47_b1_artifact_registry.py
python -u tasks/validate_bioaware_b47_truthblind_embeddings.py --input "$EMBED"

if test -d "$SEEDS"; then
  echo "[resume] validating frozen truth-blind seeds"
  python -u tasks/validate_bioaware_b47_truthblind_seeds.py --input "$SEEDS"
else
  test ! -e "$SEED_WORK" || { echo "Refusing stale seed partial: $SEED_WORK" >&2; exit 3; }
  python -u tasks/build_bioaware_b47_truthblind_seeds.py \
    --graph-dir "$GRAPH" --embedding-dir "$EMBED" --participants "$PARTICIPANTS" \
    --output "$SEED_WORK" --device cuda --score-batch-size 65536 --maximum-seed-degree 250
  python -u tasks/validate_bioaware_b47_truthblind_seeds.py --input "$SEED_WORK"
  mv "$SEED_WORK" "$SEEDS"
fi

if test -d "$U0"; then
  echo "[resume] validating frozen U0 nuisance audit"
  python -u tasks/validate_bioaware_b47_u0_reference_bias.py --input "$U0"
else
  test ! -e "$U0_WORK" || { echo "Refusing stale U0 partial: $U0_WORK" >&2; exit 3; }
  python -u tasks/audit_bioaware_b47_u0_reference_bias.py \
    --graph-dir "$GRAPH" --embedding-dir "$EMBED" --output "$U0_WORK" \
    --device cuda --score-batch-size 65536 --preflight-only
  test ! -e "$U0_WORK" || { echo "U0 preflight unexpectedly created output" >&2; exit 4; }
  python -u tasks/audit_bioaware_b47_u0_reference_bias.py \
    --graph-dir "$GRAPH" --embedding-dir "$EMBED" --output "$U0_WORK" \
    --device cuda --score-batch-size 65536
  python -u tasks/validate_bioaware_b47_u0_reference_bias.py --input "$U0_WORK"
  mv "$U0_WORK" "$U0"
fi

if test -d "$REGISTRY"; then
  echo "[resume] validating frozen B1 artifact registry"
  python -u tasks/validate_bioaware_b47_b1_artifact_registry.py --input "$REGISTRY"
else
  test ! -e "$REGISTRY_WORK" || { echo "Refusing stale registry partial: $REGISTRY_WORK" >&2; exit 3; }
  python -u tasks/audit_bioaware_b47_b1_artifact_registry.py \
    --graph-dir "$GRAPH" --embedding-dir "$EMBED" --seed-dir "$SEEDS" --u0-dir "$U0" \
    --output "$REGISTRY_WORK" --require-complete
  python -u tasks/validate_bioaware_b47_b1_artifact_registry.py --input "$REGISTRY_WORK"
  mv "$REGISTRY_WORK" "$REGISTRY"
fi

echo "=== BioAware B47 B1 provenance + U0 completed: $(date) ==="
echo "Registry: $REGISTRY/report.json"
echo "U0: $U0/report.json"
