#!/bin/bash
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --output=mtbls13729_all_panels_%j.out
#SBATCH --error=mtbls13729_all_panels_%j.err

# MTBLS13729 全 4 板（pos_rp / neg_rp / pos_hilic / neg_hilic）用我们自己的流程跑一遍注释，
# 并把输出导出成与原生 DreaMS API 相同的 TSV schema。
#
# 每板: mzML -> hdf5 -> 嵌入 -> MONA 余弦检索 + m/z 硬约束 -> target-decoy FDR -> 导出原生 schema。
# 库映射: pos 板 -> mona_pos_full.mgf / mona_pos_dreams_emb（pos 两板共用）
#         neg 板 -> mona_neg_full.mgf / mona_neg_dreams_emb（neg 两板共用）
#
# 前提 (data/ 上传清单见文末注释):
#   data/mtbls13729/mzml/{pos_rp,neg_rp,pos_hilic,neg_hilic}/  各板 .mzML（按样本名重命名）
#   data/models/mona_pos_full.mgf, mona_neg_full.mgf
#   dreams/models/pretrained/ssl_model_server.pt, data/e1/official_embedding_slim.pt
# 断点续传: hdf5 / 库嵌入 / 诱饵分块已存在则跳过。

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "===== MTBLS13729 全 4 板（我们自己的流程 + 原生 schema 导出）====="
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

for PANEL in pos_rp neg_rp pos_hilic neg_hilic; do
  case "$PANEL" in
    neg_*) MODE=neg; MGF=data/models/mona_neg_full.mgf; LIB=data/models/mona_neg_dreams_emb ;;
    *)     MODE=pos; MGF=data/models/mona_pos_full.mgf; LIB=data/models/mona_pos_dreams_emb ;;
  esac
  MZML="data/mtbls13729/mzml/$PANEL"
  EMB="data/mtbls13729/embeddings/$PANEL"
  ANN="data/mtbls13729/annotation/$PANEL"
  NATIVE="data/mtbls13729/annotation_native/$PANEL"
  DECOY_CACHE="data/models/mona_${MODE}_decoy_emb.npy"
  DECOY_CHUNK="data/models/mona_${MODE}_decoy_chunks"

  echo ""
  echo "===== [$PANEL] mode=$MODE ====="

  # (0) 嵌入 MONA ${MODE} 库（该 mode 两板共用，已存在则跳过）
  if [ ! -f "$LIB/embeddings.npy" ]; then
    python -u tasks/encode_mona_neg_library.py --mgf "$MGF" --out "$LIB" --device cuda
  else
    echo "[$PANEL] 库嵌入已存在，跳过 $LIB"
  fi

  # (1) mzML -> hdf5（MSData.load 只取 MS2 谱）
  python -u tasks/convert_met_neg_hdf5.py --mzml-dir "$MZML"

  # (2) query 嵌入
  python -u tasks/encode_msv100574_spectra.py --hdf5 "$MZML"/*.hdf5 --out "$EMB" --device cuda

  # (3) MONA 余弦检索 + m/z 硬约束
  python -u -m annotation.cli annotate \
      --query-dir "$EMB" --library-dir "$LIB" --out "$ANN" \
      --backend gpu --device cuda

  # (4) target-decoy FDR（${MODE} 诱饵缓存复用）
  python -u tasks/run_fdr_met_neg.py \
      --mgf "$MGF" --query-dir "$EMB" --library-dir "$LIB" \
      --annotations "$ANN/annotations.csv" --out "$ANN/annotations_fdr.csv" \
      --decoy-cache "$DECOY_CACHE" --decoy-chunk-dir "$DECOY_CHUNK" \
      --device cuda

  # (5) 导出原生 DreaMS TSV schema
  python -u tasks/export_native_schema.py \
      --annotations "$ANN/annotations_fdr.csv" \
      --query-manifest "$EMB/manifest.csv" \
      --out-dir "$NATIVE"
done

# (6) 差异分析：只对 pos_rp 做 Tumor vs Normal（headline 生物问题）。
#     其它板要跑，把 --annotations 换成对应板的 annotations_fdr.csv 即可。
python -u tasks/run_diff_mtbls13729.py \
    --annotations data/mtbls13729/annotation/pos_rp/annotations_fdr.csv \
    --out data/mtbls13729/annotation/pos_rp/diff_tumor_vs_normal.csv \
    --group-col tissue --group-a Tumor --group-b Normal

echo ""
echo "Finished: $(date)"

# ---- 数据上传清单（本地 -> 服务器 /data02/run01/scv7tsl/DreaMS）----
#   data/mtbls13729/mzml/{pos_rp,neg_rp,pos_hilic,neg_hilic}/*.mzML   (240 个)
#   data/models/mona_pos_full.mgf, data/models/mona_neg_full.mgf
#   dreams/models/pretrained/ssl_model_server.pt                      (443MB)
#   data/e1/official_embedding_slim.pt                                (447MB)
#   tasks/export_native_schema.py, tasks/convert_met_neg_hdf5.py,
#   tasks/encode_msv100574_spectra.py, tasks/encode_mona_neg_library.py,
#   tasks/run_fdr_met_neg.py, tasks/run_diff_mtbls13729.py
#   以及 annotation/ 与 dreams/ 源码目录（git pull 或整包 rsync）。
