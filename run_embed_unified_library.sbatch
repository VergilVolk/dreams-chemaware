#!/bin/bash
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=embed_unified_library_%j.out
#SBATCH --error=embed_unified_library_%j.err

# Step 2（服务器侧）—— 把本地建好的统一参考库（unified_pos/neg.mgf）嵌入成检索库。
# 复用 tasks/encode_mona_neg_library.py（它其实是通用 MGF -> 嵌入器，只是名字还叫 mona）。
# 统一库只有 ~3-4 万张谱（按 InChIKey 去重后），单卡几分钟就能嵌完。
#
# 前提（上传清单见文末注释）:
#   data/reference/unified/unified_pos.mgf, unified_neg.mgf
#   dreams/models/pretrained/ssl_model_server.pt, data/e1/official_embedding_slim.pt
#   tasks/encode_mona_neg_library.py + 其 import 的 tasks/*.py（git pull 或整包 rsync）

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "===== 嵌入统一参考库（pos + neg）====="
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

python -u tasks/encode_mona_neg_library.py \
    --mgf data/reference/unified_v2/unified_pos.mgf \
    --out data/models/unified_v2_pos_dreams_emb \
    --device cuda

python -u tasks/encode_mona_neg_library.py \
    --mgf data/reference/unified_v2/unified_neg.mgf \
    --out data/models/unified_v2_neg_dreams_emb \
    --device cuda

echo "Finished: $(date)"

# ---- 数据上传清单（本地 -> 服务器 /data02/run01/scv7tsl/DreaMS）----
#   data/reference/unified/unified_pos.mgf, unified_neg.mgf   (本地建库产物，各 ~几十 MB)
#   tasks/encode_mona_neg_library.py 及其 import（tasks/e1_checkpoint_io.py,
#   tasks/pilot_paired_layer_cka.py）—— 若服务器已有这些文件则跳过
#   （dreams/models/pretrained/ssl_model_server.pt 与 data/e1/official_embedding_slim.pt
#     服务器应已存在，见 run_mtbls13729_all_panels.sbatch 的上传清单）
#
# 嵌完后，把 run_mtbls13729_all_panels.sbatch 里的库路径换成统一库即可重新注释：
#   pos 板 -> MGF=data/reference/unified/unified_pos.mgf
#             LIB=data/models/unified_pos_dreams_emb
#   neg 板 -> MGF=data/reference/unified/unified_neg.mgf
#             LIB=data/models/unified_neg_dreams_emb
