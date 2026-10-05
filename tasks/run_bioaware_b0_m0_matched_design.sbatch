#!/bin/bash
#SBATCH --job-name=bio_b0_m0
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=bioaware_b0_m0_%j.out
#SBATCH --error=bioaware_b0_m0_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

HDF5=data/models/MassSpecGym_MurckoHist_split.hdf5
PARTICIPANTS=data/reference/bioaware_rhea_reactome_direction_20260830/rhea_participants.csv.gz
P3=data/validation/g8r_p3_test
ALLOWED=$P3/p3_p2_allowed_training_ik14.json
OUTPUT=data/validation/bioaware_b0_m0_matched_design_v4_formal_20260905

for path in "$HDF5" "$PARTICIPANTS" "$ALLOWED" "$P3/p3_lock_summary.json" \
  "$P3/p3_main_real_pristine_manifest.json" \
  "$P3/p3_isomer_real_pristine_manifest.json" \
  "$P3/p3_near_core_real_pristine_manifest.json" \
  "$P3/p3_nearmid_real_pristine_manifest.json" \
  "$P3/p3_isomer_real_exposed_extension_manifest.json" \
  "$P3/p3_sim_to_real_secondary_manifest.json" \
  annotation/bioaware_relations.py tasks/audit_bioaware_b0_m0_matched_design.py; do
  test -s "$path" || { echo "Missing required dependency: $path" >&2; exit 2; }
done
test ! -e "$OUTPUT" || { echo "Refusing to overwrite: $OUTPUT" >&2; exit 3; }

python -c "import h5py, numpy, pandas, rdkit, scipy; print('[BioAware B0-M0 dependencies] PASS')"
python tasks/audit_bioaware_b0_m0_matched_design.py --self-test
python -u tasks/audit_bioaware_b0_m0_matched_design.py \
  --hdf5 "$HDF5" \
  --allowed-training "$ALLOWED" \
  --participants "$PARTICIPANTS" \
  --p3-dir "$P3" \
  --output-dir "$OUTPUT" \
  --controls-per-edge 1 \
  --formal

python -m json.tool "$OUTPUT/report.json" >/dev/null
echo "[BioAware B0-M0] completed. Scientific pass is pass_to_signal_test in $OUTPUT/report.json"
