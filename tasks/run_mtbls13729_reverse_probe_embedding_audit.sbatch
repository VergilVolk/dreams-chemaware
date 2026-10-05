#!/bin/bash
#SBATCH --job-name=mtbls_rev_emb
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=data/validation/mtbls13729_reverse_probe_%j.out
#SBATCH --error=data/validation/mtbls13729_reverse_probe_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

MANIFEST="data/mtbls13729/reverse_probe_manifest_v4"
MGF="data/mtbls13729/reverse_probe_mgf_v1"
CACHE="data/mtbls13729/reverse_probe_embedding_cache_v1"
OFFICIAL_OUT="data/mtbls13729/reverse_probe_official_v1"
E6_OUT="data/mtbls13729/reverse_probe_e6_v1"
COMPARE="data/mtbls13729/reverse_probe_embedding_comparison_v1/report.json"
E6_CKPT="data/validation/g8r_noise_final_e6_outcome_mined_direct/curriculum_errors_views2_blocks1_blr_2e-06_hlr_1e-05_sw2_e6_fixed_v2_sw2/seed_20260828/fold_0/final_shared_encoder.pt"

required=(
  tasks/build_mtbls13729_reverse_probe_manifest.py
  tasks/test_mtbls13729_reverse_probe_manifest.py
  tasks/extract_mtbls13729_reverse_probe_mgf.py
  tasks/test_mtbls13729_reverse_probe_mgf.py
  tasks/encode_unified_library_for_p2b.py
  tasks/evaluate_mtbls13729_reverse_probe_embeddings.py
  tasks/compare_mtbls13729_reverse_probe_embeddings.py
  data/reference/unified_v2/unified_neg.mgf
  data/reference/unified_v2/unified_pos.mgf
  data/e1/official_embedding_slim.pt
  dreams/models/pretrained/ssl_model_server.pt
  "$E6_CKPT"
  data/mtbls13729/embeddings/neg_rp/manifest.csv
  data/mtbls13729/embeddings/pos_rp/manifest.csv
  data/mtbls13729/embeddings_e6_fixed_v2_sw2/neg_rp/report.json
  data/mtbls13729/embeddings_e6_fixed_v2_sw2/pos_rp/report.json
  data/mtbls13729/ms1_consensus/neg_rp__requantification_targets.csv.gz
  data/mtbls13729/ms1_consensus/pos_rp__requantification_targets.csv.gz
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { echo "MISSING: $path" >&2; exit 2; }
done
[[ ! -e "$COMPARE" ]] || { echo "REFUSE: frozen comparison already exists: $COMPARE" >&2; exit 3; }

echo "=== MTBLS13729 reverse-probe embedding audit started: $(date) ==="
python -u tasks/test_mtbls13729_reverse_probe_manifest.py
python -u tasks/test_mtbls13729_reverse_probe_mgf.py

if [[ ! -s "$MANIFEST/report.json" ]]; then
  python -u tasks/build_mtbls13729_reverse_probe_manifest.py
else
  echo "[cache] reusing $MANIFEST"
fi
if [[ ! -s "$MGF/report.json" ]]; then
  python -u tasks/extract_mtbls13729_reverse_probe_mgf.py
else
  echo "[cache] reusing $MGF"
fi

for panel in neg_rp pos_rp; do
  probe_mgf="$MGF/${panel}__reverse_probes.mgf"
  official_cache="$CACHE/official/$panel"
  e6_cache="$CACHE/e6_fixed_v2_sw2/$panel"
  if [[ ! -s "$official_cache/report.json" ]]; then
    python -u tasks/encode_unified_library_for_p2b.py \
      --mgf "$probe_mgf" --output-dir "$official_cache" --device cuda --batch-size 128
  fi
  if [[ ! -s "$e6_cache/report.json" ]]; then
    python -u tasks/encode_unified_library_for_p2b.py \
      --mgf "$probe_mgf" --output-dir "$e6_cache" --shared-encoder-checkpoint "$E6_CKPT" \
      --device cuda --batch-size 128
  fi
  python -u tasks/evaluate_mtbls13729_reverse_probe_embeddings.py \
    --panel "$panel" --method official_dreams --reference-cache "$official_cache" \
    --query-embedding-dir "data/mtbls13729/embeddings/$panel" --output-dir "$OFFICIAL_OUT"
  python -u tasks/evaluate_mtbls13729_reverse_probe_embeddings.py \
    --panel "$panel" --method e6_fixed_v2_sw2 --reference-cache "$e6_cache" \
    --query-embedding-dir "data/mtbls13729/embeddings_e6_fixed_v2_sw2/$panel" --output-dir "$E6_OUT"
done

python -u tasks/compare_mtbls13729_reverse_probe_embeddings.py \
  --official-dir "$OFFICIAL_OUT" --experimental-dir "$E6_OUT" \
  --experimental-method e6_fixed_v2_sw2 --output "$COMPARE"
echo "=== MTBLS13729 reverse-probe embedding audit completed: $(date) ==="
