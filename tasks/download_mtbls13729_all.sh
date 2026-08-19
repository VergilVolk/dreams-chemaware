#!/bin/bash
# 下载 MTBLS13729 全部 4 个面板（反相/亲水 × 正/负离子），按真实样本名重命名。
#
# 关键事实（已用 assay 原文逐行核验，见 data/mtbls13729/sample_map.tsv）:
#   - 4 个面板的 mzML 顺序完全一致: 1.mzML→P01-Ltu(肿瘤), 2→P01-LN(正常), ...
#     奇数 mzML = 肿瘤(30), 偶数 = 正常(30)。
#   - 映射用 assay 的 "MS Assay Name"(第30列,0-based 29) + "Derived Spectral Data File"(第34列)。
#     "Sample Name"(第1列) 被对调过，不可用（上次的对调陷阱）。
#
# 断点续传(curl -C -) + 完整性校验(末尾闭合标签)，JOBS>1 时 xargs 并行。
# 用法:
#   JOBS=6 bash tasks/download_mtbls13729_all.sh          # 下全部 4 面板
#   PANELS="neg_rp pos_hilic neg_hilic" JOBS=6 bash tasks/download_mtbls13729_all.sh
set -euo pipefail

cd "$(dirname "$0")/.."

BASE="https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS13729/FILES/DERIVED_FILES"
MAP="data/mtbls13729/sample_map.tsv"

declare -A SUB=(
    [pos_rp]="REVERSE/POS"
    [neg_rp]="REVERSE/NEG"
    [pos_hilic]="HILIC/POS"
    [neg_hilic]="HILIC/NEG"
)
PANELS="${PANELS:-pos_rp neg_rp pos_hilic neg_hilic}"
JOBS="${JOBS:-1}"

[ -f "$MAP" ] || { echo "缺少映射表 $MAP" >&2; exit 1; }

is_complete() {  # $1=path — 非空且末尾有 mzML 闭合标签才算完整
    [ -s "$1" ] || return 1
    tail -c 64 "$1" 2>/dev/null | grep -qE '/(indexedmzML|mzML)>'
}

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

LIST="$(mktemp)"
trap 'rm -f "$LIST"' EXIT

for panel in $PANELS; do
    sub="${SUB[$panel]:?未知面板 $panel}"
    OUT="data/mtbls13729/mzml/$panel"
    mkdir -p "$OUT"
    URL="$BASE/$sub"

    # 第一阶段：扫一遍，把该面板还没完整的文件写进待办清单
    : > "$LIST"
    i=0
    while IFS=$'\t' read -r mzml sample; do
        mzml="${mzml%$'\r'}"
        sample="${sample%$'\r'}"
        [ -z "$mzml" ] && continue
        i=$((i+1))
        target="$OUT/${sample}.mzML"
        is_complete "$target" && { echo "[$panel $i/60] skip ${sample}"; continue; }
        printf '%s\t%s\n' "$mzml" "$sample" >> "$LIST"
    done < "$MAP"
    todo=$(wc -l < "$LIST")
    echo "=== [$panel] 待下载 $todo/60（$URL）==="
    [ "$todo" -eq 0 ] && { echo "[$panel] 全部就位"; continue; }

    # 第二阶段：下载（awk 生成命令避免 CRLF/转义坑；curl -C - 续传）
    if [ "$JOBS" -gt 1 ]; then
        awk -F'\t' -v u="$URL" -v o="$OUT" '
            { printf "curl -sS -C - -o \"%s/%s.mzML\" \"%s/%s\" && echo OK:%s || echo FAIL:%s\n", o, $2, u, $1, $2, $2 }
        ' "$LIST" | xargs -P "$JOBS" -I {} bash -c "{}"
    else
        while IFS=$'\t' read -r mzml sample; do
            dl_file "$URL/$mzml" "$OUT/${sample}.mzML"
        done < "$LIST"
    fi
    have=$(ls -1 "$OUT"/*.mzML 2>/dev/null | wc -l)
    echo "[$panel] 就位 $have/60"
done

echo "=== 全部面板完成，最终校验 ==="
for panel in $PANELS; do
    OUT="data/mtbls13729/mzml/$panel"
    c=0; p=0; n=0
    for f in "$OUT"/*.mzML; do
        [ -e "$f" ] || continue
        n=$((n+1))
        is_complete "$f" && c=$((c+1)) || p=$((p+1))
    done
    echo "$panel: 完整 $c / 残缺 $p / 总数 $n"
done
