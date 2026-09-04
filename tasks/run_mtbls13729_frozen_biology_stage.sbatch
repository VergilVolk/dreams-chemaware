#!/bin/bash
#SBATCH --job-name=mtbls_biology
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=data/validation/mtbls_biology_%j.out
#SBATCH --error=data/validation/mtbls_biology_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=== MTBLS13729 frozen biology stage started: $(date) ==="

if [[ ! -s data/mtbls13729/frozen_biology_panel_v1/report.json ]]; then
  python -u tasks/build_mtbls13729_frozen_biology_panel.py
fi
python -u tasks/validate_mtbls13729_frozen_biology_panel.py

python -u tasks/requantify_mtbls13729_targeted_eic.py \
  --consensus-dir data/mtbls13729/frozen_biology_panel_v1 \
  --output-dir data/mtbls13729/ms1_eic_requant_threeway_frozen_v1 \
  --panels neg_rp pos_rp \
  --ppm 5.0 \
  --rt-half-window-sec 20.0 \
  --resolve-local-peaks \
  --max-apex-delta-sec 12.0

if [[ ! -s data/mtbls13729/ms1_acquisition_audit/file_audit.csv ]]; then
  echo "[acquisition-audit] file_audit.csv missing; building it from the deposited mzML files"
  python -u tasks/audit_mtbls13729_ms1_acquisition.py \
    --workers 4 \
    --output-dir data/mtbls13729/ms1_acquisition_audit
fi
test -s data/mtbls13729/ms1_acquisition_audit/file_audit.csv

python -u tasks/analyze_mtbls13729_paired_ms1.py \
  --eic-dir data/mtbls13729/ms1_eic_requant_threeway_frozen_v1 \
  --consensus-dir data/mtbls13729/frozen_biology_panel_v1 \
  --audit data/mtbls13729/ms1_acquisition_audit/file_audit.csv \
  --output-dir data/mtbls13729/threeway_biology_v1 \
  --panels neg_rp pos_rp \
  --min-pairs 6

echo "=== MTBLS13729 frozen biology stage completed: $(date) ==="
