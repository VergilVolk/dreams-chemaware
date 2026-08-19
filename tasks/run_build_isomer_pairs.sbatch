#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=build_isomer_pairs_%j.out
#SBATCH --error=build_isomer_pairs_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# ============================================================================
# Step 1: MassSpecGym train fold 同 formula 异构体对（MCES 分级）构建。
#   纯 CPU 任务（myopic_mces 是 MILP / RDKit 计算），但集群强制 --gpus=1 才放行，
#   申请 1 卡只占卡不调用（16 核 CPU 做 MCES 并行）。
#
# DEPENDENCIES（须已同步/安装到服务器，否则 exit 2）：
#   1. data/models/MassSpecGym_MurckoHist_split.hdf5   # 谱+smiles+INCHIKEY（旧文件，应在）
#   2. tasks/build_massspecgym_isomer_pairs.py         # 本脚本（最近新写，须同步）
#   3. 运行库：myopic_mces、rdkit、h5py、numpy        # myopic_mces 最关键，缺则脚本退出
#
# GENERATED（本 job 产出，不须预先存在）：
#   tasks/massspecgym_isomers/pairs.json        # 异构体对 near/mid/far/uncomputed
#   tasks/massspecgym_isomers/ik_to_rows.json   # ik14 -> HDF5 谱行索引（Step 3 用）
#   tasks/massspecgym_isomers/ik_to_smiles.json # ik14 -> 完整 SMILES（接口②用）
#   tasks/massspecgym_isomers/stats.json        # 统计
# ============================================================================

echo "===== Step 1: 构建 MassSpecGym 异构体对（MCES 分级，纯 CPU）====="
echo "Started: $(date)"

for REQUIRED in \
    data/models/MassSpecGym_MurckoHist_split.hdf5 \
    tasks/build_massspecgym_isomer_pairs.py ; do
    if [[ ! -f "$REQUIRED" ]]; then
        echo "MISSING DEPENDENCY: $REQUIRED  (see DEPENDENCIES block above)" >&2
        exit 2
    fi
done
echo "输入依赖齐。"
echo ""

python -u tasks/build_massspecgym_isomer_pairs.py --workers 16

echo ""
echo "===== DONE $(date) ====="
echo "结果目录: tasks/massspecgym_isomers/"
echo "  stats.json 看 near/mid/far/uncomputed 分级数量。"
