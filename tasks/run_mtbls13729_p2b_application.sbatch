#!/bin/bash
#SBATCH --job-name=mtbls_p2b_app
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=data/validation/mtbls13729_p2b_%j.out
#SBATCH --error=data/validation/mtbls13729_p2b_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

OUT="data/mtbls13729/p2b_application_v1"
E6_OUT="data/mtbls13729/e6_embedding_application_v1"
THREEWAY_OUT="data/mtbls13729/threeway_application_v1"
NEG_CACHE="data/reference/unified_v2_p2b/neg_rp"
POS_CACHE="data/reference/unified_v2_p2b/pos_rp"
E6_NEG_CACHE="data/reference/unified_v2_e6_fixed_v2_sw2/neg_rp"
E6_POS_CACHE="data/reference/unified_v2_e6_fixed_v2_sw2/pos_rp"
E6_NEG_QUERY="data/mtbls13729/embeddings_e6_fixed_v2_sw2/neg_rp"
E6_POS_QUERY="data/mtbls13729/embeddings_e6_fixed_v2_sw2/pos_rp"
E6_CKPT="data/validation/g8r_noise_final_e6_outcome_mined_direct/curriculum_errors_views2_blocks1_blr_2e-06_hlr_1e-05_sw2_e6_fixed_v2_sw2/seed_20260828/fold_0/final_shared_encoder.pt"

required=(
  tasks/encode_unified_library_for_p2b.py
  tasks/infer_mtbls13729_p2b_vs_dreams.py
  tasks/package_mtbls13729_p2b_annotation_comparison.py
  tasks/shared_dreams_inference.py
  tasks/encode_hdf5_shared_dreams.py
  tasks/infer_mtbls13729_embedding_retrieval.py
  tasks/package_mtbls13729_threeway_embedding_comparison.py
  tasks/test_mtbls13729_p2b_application.py
  tasks/preflight_mtbls13729_threeway_application.py
  tasks/g8r_p2_rank_fusion_core.py
  data/reference/unified_v2/unified_neg.mgf
  data/reference/unified_v2/unified_pos.mgf
  data/e1/official_embedding_slim.pt
  dreams/models/pretrained/ssl_model_server.pt
  data/validation/g8r_p2b_rank_fusion.json
  data/validation/g8r_p2b_p3_final.json
  "$E6_CKPT"
  data/mtbls13729/embeddings/neg_rp/manifest.csv
  data/mtbls13729/embeddings/pos_rp/manifest.csv
  data/mtbls13729/ms1_consensus/neg_rp__requantification_targets.csv.gz
  data/mtbls13729/ms1_consensus/pos_rp__requantification_targets.csv.gz
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { echo "MISSING: $path" >&2; exit 2; }
done
[[ ! -e "$OUT/neg_rp__report.json" && ! -e "$OUT/pos_rp__report.json" ]] || {
  echo "REFUSE: application result already exists in $OUT" >&2
  exit 3
}
[[ ! -e "$E6_OUT/neg_rp__e6_fixed_v2_sw2__report.json" && ! -e "$E6_OUT/pos_rp__e6_fixed_v2_sw2__report.json" ]] || {
  echo "REFUSE: E6 application result already exists in $E6_OUT" >&2
  exit 3
}
[[ ! -d "$THREEWAY_OUT" ]] || { echo "REFUSE: $THREEWAY_OUT already exists" >&2; exit 3; }

echo "=== MTBLS13729 frozen P2b application started: $(date) ==="
python -u tasks/test_mtbls13729_p2b_application.py
python -u tasks/preflight_mtbls13729_threeway_application.py --e6-checkpoint "$E6_CKPT"

if [[ ! -s "$NEG_CACHE/report.json" || ! -s "$NEG_CACHE/manifest.csv" || ! -s "$NEG_CACHE/embeddings.npy" || ! -s "$NEG_CACHE/spectra.npy" ]]; then
  python -u tasks/encode_unified_library_for_p2b.py \
    --mgf data/reference/unified_v2/unified_neg.mgf \
    --output-dir "$NEG_CACHE" --device cuda --batch-size 128
else
  echo "[cache] reusing $NEG_CACHE"
fi
if [[ ! -s "$POS_CACHE/report.json" || ! -s "$POS_CACHE/manifest.csv" || ! -s "$POS_CACHE/embeddings.npy" || ! -s "$POS_CACHE/spectra.npy" ]]; then
  python -u tasks/encode_unified_library_for_p2b.py \
    --mgf data/reference/unified_v2/unified_pos.mgf \
    --output-dir "$POS_CACHE" --device cuda --batch-size 128
else
  echo "[cache] reusing $POS_CACHE"
fi

python -u tasks/infer_mtbls13729_p2b_vs_dreams.py \
  --panel neg_rp --library-cache "$NEG_CACHE" --output-dir "$OUT"
python -u tasks/infer_mtbls13729_p2b_vs_dreams.py \
  --panel pos_rp --library-cache "$POS_CACHE" --output-dir "$OUT"
python -u tasks/package_mtbls13729_p2b_annotation_comparison.py \
  --input-dir "$OUT" --output-dir "$OUT/comparison"

# The experimental noise-finetuned embedding is encoded on BOTH sides.  Never
# compare E6 queries against an official-DreaMS library embedding.
if [[ ! -s "$E6_NEG_CACHE/report.json" ]]; then
  python -u tasks/encode_unified_library_for_p2b.py \
    --mgf data/reference/unified_v2/unified_neg.mgf --output-dir "$E6_NEG_CACHE" \
    --shared-encoder-checkpoint "$E6_CKPT" --device cuda --batch-size 128
fi
if [[ ! -s "$E6_POS_CACHE/report.json" ]]; then
  python -u tasks/encode_unified_library_for_p2b.py \
    --mgf data/reference/unified_v2/unified_pos.mgf --output-dir "$E6_POS_CACHE" \
    --shared-encoder-checkpoint "$E6_CKPT" --device cuda --batch-size 128
fi
mapfile -t NEG_HDF5 < <(find data/mtbls13729/mzml/neg_rp -maxdepth 1 -type f -name '*.hdf5' | sort)
mapfile -t POS_HDF5 < <(find data/mtbls13729/mzml/pos_rp -maxdepth 1 -type f -name '*.hdf5' | sort)
[[ "${#NEG_HDF5[@]}" -gt 0 && "${#POS_HDF5[@]}" -gt 0 ]] || {
  echo "Expected non-empty RP HDF5 inputs; got neg=${#NEG_HDF5[@]} pos=${#POS_HDF5[@]}" >&2
  exit 4
}
if [[ ! -s "$E6_NEG_QUERY/report.json" ]]; then
  python -u tasks/encode_hdf5_shared_dreams.py --hdf5 "${NEG_HDF5[@]}" \
    --output-dir "$E6_NEG_QUERY" --shared-encoder-checkpoint "$E6_CKPT" \
    --reference-manifest data/mtbls13729/embeddings/neg_rp/manifest.csv \
    --device cuda --batch-size 128
fi
if [[ ! -s "$E6_POS_QUERY/report.json" ]]; then
  python -u tasks/encode_hdf5_shared_dreams.py --hdf5 "${POS_HDF5[@]}" \
    --output-dir "$E6_POS_QUERY" --shared-encoder-checkpoint "$E6_CKPT" \
    --reference-manifest data/mtbls13729/embeddings/pos_rp/manifest.csv \
    --device cuda --batch-size 128
fi
python -u tasks/infer_mtbls13729_embedding_retrieval.py --panel neg_rp \
  --query-embedding-dir "$E6_NEG_QUERY" --library-cache "$E6_NEG_CACHE" \
  --official-library-cache "$NEG_CACHE" --output-dir "$E6_OUT"
python -u tasks/infer_mtbls13729_embedding_retrieval.py --panel pos_rp \
  --query-embedding-dir "$E6_POS_QUERY" --library-cache "$E6_POS_CACHE" \
  --official-library-cache "$POS_CACHE" --output-dir "$E6_OUT"
python -u tasks/package_mtbls13729_threeway_embedding_comparison.py \
  --p2b-dir "$OUT" --e6-dir "$E6_OUT" --output-dir "$THREEWAY_OUT"

echo "=== MTBLS13729 frozen P2b application completed: $(date) ==="
