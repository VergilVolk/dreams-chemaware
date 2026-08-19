#!/bin/bash
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --output=full_4group_%j.out
#SBATCH --error=full_4group_%j.err

# 全量注释：Met/neg + Met/pos + Lipid/neg + Lipid/pos（4 组全部 mzML）。
# 每组：mzML -> hdf5 -> 嵌入 -> 注释 -> FDR -> 样本级差异。
# 全程断点续传：hdf5 已存在则跳过，诱饵分块已存在则复用。
# 前提：mzML 已按下面目录摆好，mona_neg/pos_full.mgf 在 data/models/ 下。

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "===== 全量注释（4 组全部 mzML）====="
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# (0) 嵌入两个 MONA 库（已存在则跳过；neg/pos 各一次，全组共用）
if [ ! -f data/models/mona_neg_dreams_emb/embeddings.npy ]; then
    python -u tasks/encode_mona_neg_library.py \
        --mgf data/models/mona_neg_full.mgf \
        --out data/models/mona_neg_dreams_emb --device cuda
fi
if [ ! -f data/models/mona_pos_dreams_emb/embeddings.npy ]; then
    python -u tasks/encode_mona_neg_library.py \
        --mgf data/models/mona_pos_full.mgf \
        --out data/models/mona_pos_dreams_emb --device cuda
fi

run_group() {
    # $1=mzml_dir $2=embed_out $3=library $4=mgf $5=ann_dir $6=polarity $7=name
    local mzml_dir="$1" embed_out="$2" library="$3" mgf="$4" ann_dir="$5" polarity="$6" name="$7"
    echo "===== [$name] 开始 $(date) ====="

    python -u tasks/convert_met_neg_hdf5.py --mzml-dir "$mzml_dir"

    python -u tasks/encode_msv100574_spectra.py \
        --hdf5 "$mzml_dir"/*.hdf5 --out "$embed_out" --device cuda

    python -u -m annotation.cli annotate \
        --query-dir "$embed_out" --library-dir "$library" --out "$ann_dir" \
        --backend gpu --device cuda

    # 诱饵缓存按极性共用：neg 两组共用 mona_neg_decoy_*，pos 两组共用 mona_pos_decoy_*
    python -u tasks/run_fdr_met_neg.py \
        --mgf "$mgf" --query-dir "$embed_out" --library-dir "$library" \
        --annotations "$ann_dir/annotations.csv" --out "$ann_dir/annotations_fdr.csv" \
        --decoy-cache "data/models/mona_${polarity}_decoy_emb.npy" \
        --decoy-chunk-dir "data/models/mona_${polarity}_decoy_chunks" \
        --device cuda

    python -u tasks/run_diff.py \
        --annotations "$ann_dir/annotations_fdr.csv" --out "$ann_dir/diff.csv" \
        --group-a PF --group-b HF

    echo "===== [$name] 完成 $(date) ====="
}

run_group data/msv100574/Metabolomics/neg data/msv100574/embeddings/met_neg \
    data/models/mona_neg_dreams_emb data/models/mona_neg_full.mgf \
    data/msv100574/annotation/met_neg neg met_neg

run_group data/msv100574/Metabolomics/pos data/msv100574/embeddings/met_pos \
    data/models/mona_pos_dreams_emb data/models/mona_pos_full.mgf \
    data/msv100574/annotation/met_pos pos met_pos

run_group data/msv100574/Lipidomics/neg data/msv100574/embeddings/lipid_neg \
    data/models/mona_neg_dreams_emb data/models/mona_neg_full.mgf \
    data/msv100574/annotation/lipid_neg neg lipid_neg

run_group data/msv100574/Lipidomics/pos data/msv100574/embeddings/lipid_pos \
    data/models/mona_pos_dreams_emb data/models/mona_pos_full.mgf \
    data/msv100574/annotation/lipid_pos pos lipid_pos

echo "Finished: $(date)"
