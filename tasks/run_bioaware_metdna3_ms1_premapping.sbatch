#!/bin/bash
#SBATCH --job-name=bioaware_m3_ms1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=data/validation/bioaware_m3_ms1_%j.out
#SBATCH --error=data/validation/bioaware_m3_ms1_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

echo "=== BioAware MetDNA3 MS1 pre-mapping started: $(date) ==="
python -c "import pyopenms, pyreadr, rdata, pandas, numpy"

INDEX=data/reference/metdna2_emrn_mass_adduct_20260828
PILOT=data/validation/bioaware_metdna3_ms1_feature_pilot_v1
PREMAP=data/validation/bioaware_metdna3_ms1_premapping_v1

if [[ ! -s "$INDEX/report.json" ]]; then
  python -u tasks/extract_metdna2_emrn_mass_adduct_index.py
else
  echo "[reuse] frozen eMRN mass/adduct index"
fi
if [[ ! -s "$PREMAP/report.json" ]]; then
  python -u tasks/audit_bioaware_metdna3_ms1_premapping.py
else
  echo "[reuse] frozen mass-only pre-mapping audit"
fi
if [[ ! -s "$PILOT/report.json" ]]; then
  python -u tasks/pilot_bioaware_metdna3_ms1_features.py
else
  echo "[reuse] outcome-blind MS1 detector pilot"
fi
if [[ ! -s data/validation/bioaware_metdna3_recursive_headroom_v1/report.json ]]; then
  python -u tasks/audit_bioaware_metdna3_recursive_headroom.py
else
  echo "[reuse] recursive MS1 feature-graph headroom audit"
fi
for THRESHOLD in 1000 100000; do
  TARGET="data/validation/bioaware_metdna3_recursive_headroom_threshold${THRESHOLD}_v1"
  if [[ ! -s "$TARGET/report.json" ]]; then
    python -u tasks/audit_bioaware_metdna3_recursive_headroom.py \
      --noise-threshold "$THRESHOLD" --output-dir "$TARGET"
  else
    echo "[reuse] recursive sensitivity threshold=$THRESHOLD"
  fi
done
if [[ ! -s data/validation/bioaware_metdna3_recursive_sensitivity_v2.json ]]; then
  python -u tasks/summarize_bioaware_metdna3_recursive_sensitivity.py
else
  echo "[reuse] recursive threshold-sensitivity summary"
fi
if [[ ! -s data/validation/bioaware_metdna3_candidate_paths_v1/report.json ]]; then
  python -u tasks/build_bioaware_metdna3_candidate_path_table.py
else
  echo "[reuse] candidate-specific known-reaction path table"
fi
echo "=== BioAware MetDNA3 MS1 pre-mapping completed: $(date) ==="
