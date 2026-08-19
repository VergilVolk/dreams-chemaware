#!/bin/bash
# 下载 MTBLS13729 正离子反相(RP) 60 个 mzML，并按真实样本名重命名。
#
# 映射来源: data/mtbls13729/pos_rp_map.tsv（mzml_file -> sample_name）。
#   注意: 映射用的是 assay 表的 "MS Assay Name" 列（第29列），不是 "Sample Name" 列。
#   原 assay 表 Sample Name 列被对调过（P01-LN 在前），实测 1.mzML = P01-Ltu(肿瘤)。
#   验证结论: 奇数 mzML = 肿瘤(30)，偶数 = 正常(30)。
#
# 断点续传：已下载完成的文件（非空）跳过；中断后重跑继续。
# 用法（登录节点或 sbatch 均可）:
#     bash tasks/download_mtbls13729_pos_rp.sh
set -euo pipefail

BASE="https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS13729/FILES/DERIVED_FILES/REVERSE/POS"
MAP="data/mtbls13729/pos_rp_map.tsv"
OUT="data/mtbls13729/mzml/pos_rp"

cd "$(dirname "$0")/.."   # 切到 DreaMS 根目录
mkdir -p "$OUT"

if [ ! -f "$MAP" ]; then
    echo "缺少映射表 $MAP（请先 git pull / scp data/mtbls13729/pos_rp_map.tsv）" >&2
    exit 1
fi

dl_file() {  # $1=url  $2=out_path
    if command -v aria2c >/dev/null 2>&1; then
        aria2c -x 8 -s 8 -c --allow-overwrite=true --file-allocation=none -q \
            -d "$(dirname "$2")" -o "$(basename "$2")" "$1"
    elif command -v wget >/dev/null 2>&1; then
        wget -c -q -O "$2" "$1"
    else
        curl -sS -C - -o "$2" "$1"
    fi
}

total=$(tail -n +2 "$MAP" | grep -c .)

is_complete() {  # $1=path — 非空且末尾有 mzML 闭合标签才算完整（避免把中断的半截文件当"已存在"跳过）
    [ -s "$1" ] || return 1
    tail -c 64 "$1" 2>/dev/null | grep -qE '/(indexedmzML|mzML)>'
}

JOBS="${JOBS:-1}"                     # JOBS>1 时用 xargs 并行下载（本地 curl 单线程慢，建议 JOBS=6）
LIST="$(mktemp)"
trap 'rm -f "$LIST"' EXIT

# 第一阶段：扫一遍，把还没完整的文件写进待办清单（mzml<TAB>sample）
: > "$LIST"
i=0
while IFS=$'\t' read -r mzml sample; do
    # 剥掉 Windows CRLF 的 \r，避免样本名/文件名里混入回车符
    mzml="${mzml%$'\r'}"
    sample="${sample%$'\r'}"
    [ -z "$mzml" ] && continue
    i=$((i+1))
    target="$OUT/${sample}.mzML"
    if is_complete "$target"; then
        echo "[$i/$total] skip  ${sample}.mzML (已完整)"
        continue
    fi
    echo "[$i/$total] 下载  ${mzml} -> ${sample}.mzML$([ -s "$target" ] && echo ' (续传)')"
    printf '%s\t%s\n' "$mzml" "$sample" >> "$LIST"
done < <(tail -n +2 "$MAP")

[ -s "$LIST" ] || { echo "全部 ${total} 个 mzML 已就位，无需下载。"; exit 0; }
echo "待下载 $(wc -l < "$LIST") 个（JOBS=$JOBS）"

# 第二阶段：下载（续传由 curl -C - 保证；awk 生成命令避免 CRLF/转义坑）
if [ "$JOBS" -gt 1 ]; then
    awk -F'\t' -v b="$BASE" -v o="$OUT" '
        { printf "curl -sS -C - -o \"%s/%s.mzML\" \"%s/%s\" && echo OK:%s || echo FAIL:%s\n", o, $2, b, $1, $2, $2 }
    ' "$LIST" | xargs -P "$JOBS" -I {} bash -c "{}"
else
    while IFS=$'\t' read -r mzml sample; do
        dl_file "$BASE/$mzml" "$OUT/${sample}.mzML"
    done < "$LIST"
fi

have=$(ls -1 "$OUT"/*.mzML 2>/dev/null | wc -l)
echo "完成: ${have}/${total} 个 mzML 就位（目录 $OUT）"
