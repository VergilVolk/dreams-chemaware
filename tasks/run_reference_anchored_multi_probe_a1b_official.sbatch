#!/bin/bash
#SBATCH --job-name=ref_multi_a1b
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --output=data/validation/reference_anchored_multi_probe_a1b_%j.out
#SBATCH --error=data/validation/reference_anchored_multi_probe_a1b_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

A1="data/validation/reference_anchored_multi_probe_a1_20260913"
AUDIT="data/validation/reference_anchored_multi_probe_a1_strong_baseline_20260913.json"
MGF="data/mtbls13729/reverse_probe_mgf_v1"
CACHE="data/mtbls13729/reverse_probe_embedding_cache_v1/official"
OUT="data/validation/reference_anchored_multi_probe_a1b_official_20260913"

required=(
  tasks/audit_reference_anchored_multi_probe_a1_strong_baseline.py
  tasks/test_reference_anchored_multi_probe_a1_strong_baseline.py
  tasks/pilot_reference_anchored_multi_probe_a1b_official.py
  tasks/test_reference_anchored_multi_probe_a1b_official.py
  tasks/validate_reference_anchored_multi_probe_a1b_official.py
  tasks/encode_unified_library_for_p2b.py
  data/mtbls13729/reverse_probe_manifest_v4/reference_spectra.csv.gz
  "$MGF/alignment.csv"
  "$MGF/neg_rp__reverse_probes.mgf"
  "$MGF/pos_rp__reverse_probes.mgf"
  data/e1/official_embedding_slim.pt
  dreams/models/pretrained/ssl_model_server.pt
  "$A1/report.json"
  "$A1/per_query_coordinate_recovery.csv.gz"
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { echo "MISSING: $path" >&2; exit 2; }
done
[[ ! -e "$AUDIT" ]] || { echo "REFUSE: strong-baseline audit already exists: $AUDIT" >&2; exit 3; }
[[ ! -e "$OUT" ]] || { echo "REFUSE: A1b output already exists: $OUT" >&2; exit 3; }

echo "=== reference-anchored multi-probe A1b started: $(date) ==="
python -u tasks/test_reference_anchored_multi_probe_a1_strong_baseline.py
python -u tasks/test_reference_anchored_multi_probe_a1b_official.py
python -u tasks/audit_reference_anchored_multi_probe_a1_strong_baseline.py \
  --a1-dir "$A1" --output "$AUDIT"

proceed=$(python - "$AUDIT" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
print("yes" if report["gates"]["worth_one_official_dreams_confirmation"] else "no")
PY
)
if [[ "$proceed" != "yes" ]]; then
  echo "[A1b stop] frozen A1 signal does not beat direct_multichannel; no embedding job run"
  exit 0
fi
echo "[A1b gate] strong direct baseline passed; encoding only the frozen reference spectra"

for panel in neg_rp pos_rp; do
  cache="$CACHE/$panel"
  complete=0
  [[ -s "$cache/report.json" && -s "$cache/manifest.csv" && -s "$cache/embeddings.npy" && -s "$cache/spectra.npy" ]] && complete=1
  if [[ "$complete" -eq 1 ]]; then
    echo "[cache] reusing complete official reference cache: $cache"
  elif [[ -e "$cache" ]]; then
    echo "REFUSE: partial official reference cache exists: $cache" >&2
    exit 4
  else
    python -u tasks/encode_unified_library_for_p2b.py \
      --mgf "$MGF/${panel}__reverse_probes.mgf" --output-dir "$cache" \
      --device cuda --batch-size 128
  fi
done

python -u tasks/pilot_reference_anchored_multi_probe_a1b_official.py \
  --strong-baseline-audit "$AUDIT" --cache-root "$CACHE" --output-dir "$OUT"
python -u tasks/validate_reference_anchored_multi_probe_a1b_official.py --output-dir "$OUT"
echo "=== reference-anchored multi-probe A1b completed: $(date) ==="
