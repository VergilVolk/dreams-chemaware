#!/bin/bash
#SBATCH --job-name=bio_full16_m0
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=bioaware_full16_m0_%j.out
#SBATCH --error=bioaware_full16_m0_%j.err

set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

MANIFEST=data/validation/bioaware_metdna3_external_manifest_v1
HDF5=data/models/MassSpecGym_MurckoHist_split.hdf5
NEGATIVE_MANIFEST=data/models/mona_neg_dreams_emb/manifest.csv
NEGATIVE_REPORT=data/validation/bioaware_metdna3_external_negative_dreams_v2_chemically_filtered/report.json
NEGATIVE_INTEGRITY=data/validation/mona_negative_library_chemical_integrity_v1/report.json
NEGATIVE_APPROVED=data/validation/mona_negative_library_chemical_integrity_v1/approved_m_h_library_rows.npy
RHEA=data/validation/bioaware_embedding_relation_manifest_v2_20260830/identity_pairs.csv.gz
KEGG=data/reference/metdna2_kegg_network_20260828/metdna2_kegg_edges.csv.gz
FINAL="data/validation/bioaware_full16_action_support_m0_v7_${SLURM_JOB_ID}"
WORK="${FINAL}.partial"

test ! -e "$FINAL" || { echo "Refusing to overwrite: $FINAL" >&2; exit 3; }
test ! -e "$WORK" || { echo "Refusing to overwrite: $WORK" >&2; exit 3; }

required=(
  "$MANIFEST/report.json"
  "$HDF5"
  "$NEGATIVE_MANIFEST"
  "$NEGATIVE_REPORT"
  "$NEGATIVE_INTEGRITY"
  "$NEGATIVE_APPROVED"
  "$RHEA"
  "$KEGG"
  tasks/audit_bioaware_full16_action_support_m0.py
  tasks/test_bioaware_full16_action_support_m0.py
  tasks/validate_bioaware_full16_action_support_m0.py
)
for source in BV2cell Mouse_brain Mouse_liver NIST_plasma; do
  for separation in hilic rplc; do
    required+=("$MANIFEST/${source}__${separation}/external_level1.csv.gz")
  done
done
for path in "${required[@]}"; do
  test -s "$path" || { echo "Missing dependency: $path" >&2; exit 2; }
done

echo "=== BioAware full16 action-support M0 started: $(date) ==="
echo "[contract] model-free; embedding-free; no P2b; full 16-panel support/power audit"
python -c "import h5py,numpy,pandas; print('[BioAware full16 M0 dependencies] PASS')"
python -m py_compile \
  tasks/audit_bioaware_full16_action_support_m0.py \
  tasks/test_bioaware_full16_action_support_m0.py \
  tasks/validate_bioaware_full16_action_support_m0.py
python -u tasks/test_bioaware_full16_action_support_m0.py

python -u tasks/audit_bioaware_full16_action_support_m0.py \
  --manifest-root "$MANIFEST" \
  --reference-hdf5 "$HDF5" \
  --negative-library-manifest "$NEGATIVE_MANIFEST" \
  --negative-library-report "$NEGATIVE_REPORT" \
  --negative-library-integrity-report "$NEGATIVE_INTEGRITY" \
  --negative-approved-rows "$NEGATIVE_APPROVED" \
  --rhea-relations "$RHEA" \
  --kegg-edges "$KEGG" \
  --output-dir "$WORK" \
  --ppm 10

python -u tasks/validate_bioaware_full16_action_support_m0.py \
  --result-dir "$WORK" \
  --manifest-root "$MANIFEST" \
  --reference-hdf5 "$HDF5" \
  --negative-library-manifest "$NEGATIVE_MANIFEST" \
  --negative-library-report "$NEGATIVE_REPORT" \
  --negative-library-integrity-report "$NEGATIVE_INTEGRITY" \
  --negative-approved-rows "$NEGATIVE_APPROVED" \
  --rhea-relations "$RHEA" \
  --kegg-edges "$KEGG" \
  --audit-script tasks/audit_bioaware_full16_action_support_m0.py \
  --ppm 10
mv "$WORK" "$FINAL"

echo "=== BioAware full16 action-support M0 completed: $(date) ==="
echo "Result: $FINAL"
echo "Scientific decision: $FINAL/report.json -> pass_to_reaction_specific_action_discovery"
