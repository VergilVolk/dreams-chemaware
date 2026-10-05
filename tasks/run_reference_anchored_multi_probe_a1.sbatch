#!/bin/bash
#SBATCH --job-name=ref_multi_a1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=data/validation/reference_anchored_multi_probe_a1_%j.out
#SBATCH --error=data/validation/reference_anchored_multi_probe_a1_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

OUT="data/validation/reference_anchored_multi_probe_a1_20260913"
[[ ! -e "$OUT" ]] || { echo "REFUSE: output already exists: $OUT" >&2; exit 3; }

required=(
  tasks/pilot_reference_anchored_multi_probe_a1.py
  tasks/test_reference_anchored_multi_probe_a1.py
  tasks/validate_reference_anchored_multi_probe_a1.py
  data/mtbls13729/reverse_probe_manifest_v4/reference_spectra.csv.gz
  data/mtbls13729/reverse_probe_manifest_v4/report.json
  data/mtbls13729/reverse_probe_mgf_v1/alignment.csv
  data/mtbls13729/reverse_probe_mgf_v1/neg_rp__reverse_probes.mgf
  data/mtbls13729/reverse_probe_mgf_v1/pos_rp__reverse_probes.mgf
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { echo "MISSING: $path" >&2; exit 2; }
done

echo "=== reference-anchored multi-probe A1 started: $(date) ==="
python -u tasks/test_reference_anchored_multi_probe_a1.py
python - <<'PY'
from rdkit import rdBase
print(f"[preflight] RDKit {rdBase.rdkitVersion}")
PY
python -u tasks/pilot_reference_anchored_multi_probe_a1.py --output-dir "$OUT"
python -u tasks/validate_reference_anchored_multi_probe_a1.py --output-dir "$OUT"
echo "=== reference-anchored multi-probe A1 completed: $(date) ==="
