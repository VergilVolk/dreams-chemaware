#!/bin/bash
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=mtbls13729_pos_rp_%j.out
#SBATCH --error=mtbls13729_pos_rp_%j.err

# MTBLS13729 正离子反相(RP) 试水：mzML -> hdf5 -> 嵌入 -> MONA 注释 -> FDR -> 差异(Tumor vs Normal)。
# 前提: (1) data/mtbls13729/mzml/pos_rp/ 下已有 60 个按样本名重命名的 .mzML
#        (2) data/models/mona_pos_full.mgf 存在
# 断点续传: hdf5/诱饵分块已存在则跳过。

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "===== MTBLS13729 pos RP 试水 ====="
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

MZML=data/mtbls13729/mzml/pos_rp
EMB=data/mtbls13729/embeddings/pos_rp
ANN=data/mtbls13729/annotation/pos_rp
LIB=data/models/mona_pos_dreams_emb
MGF=data/models/mona_pos_full.mgf

# (0) 嵌入 MONA pos 库（已存在则跳过；pos 全组共用）
if [ ! -f "$LIB/embeddings.npy" ]; then
    python -u tasks/encode_mona_neg_library.py --mgf "$MGF" --out "$LIB" --device cuda
fi

# (1) mzML -> hdf5（MSData.load 只取 MS2 谱，实测每样本 ~5300 张）
python -u tasks/convert_met_neg_hdf5.py --mzml-dir "$MZML"

# (2) 嵌入
python -u tasks/encode_msv100574_spectra.py --hdf5 "$MZML"/*.hdf5 --out "$EMB" --device cuda

# (3) MONA 余弦检索 + m/z 硬约束 + Schymanski 置信度
python -u -m annotation.cli annotate \
    --query-dir "$EMB" --library-dir "$LIB" --out "$ANN" \
    --backend gpu --device cuda

# (4) target-decoy FDR（复用 mona_pos 诱饵缓存）
python -u tasks/run_fdr_met_neg.py \
    --mgf "$MGF" --query-dir "$EMB" --library-dir "$LIB" \
    --annotations "$ANN/annotations.csv" --out "$ANN/annotations_fdr.csv" \
    --decoy-cache data/models/mona_pos_decoy_emb.npy \
    --decoy-chunk-dir data/models/mona_pos_decoy_chunks \
    --device cuda

# (5) 差异: Tumor vs Normal（样本级 Fisher + BH）。
#     如需更严可加 --fdr-pass；如需其它比较改 --group-col/--group-a/--group-b。
python -u tasks/run_diff_mtbls13729.py \
    --annotations "$ANN/annotations_fdr.csv" --out "$ANN/diff_tumor_vs_normal.csv" \
    --group-col tissue --group-a Tumor --group-b Normal

echo "Finished: $(date)"
