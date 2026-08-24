#!/usr/bin/env bash
set -euo pipefail

manifest="${1:-/mnt/d/DreaMS/data/external/MTBLS1905/metadata/positive_ms1_processing_manifest.tsv}"
out_dir="${2:-/mnt/d/DreaMS/data/external/MTBLS1905/positive_patients}"
mkdir -p "$out_dir"

total=0
downloaded=0
skipped=0
failed=0

while IFS=$'\t' read -r file_name url; do
  file_name="${file_name%$'\r'}"
  url="${url%$'\r'}"

  total=$((total + 1))
  dest="$out_dir/$file_name"
  part="$dest.part"

  if [[ -s "$dest" ]]; then
    echo "[$total] skip existing: $file_name"
    skipped=$((skipped + 1))
    continue
  fi

  echo "[$total] download: $file_name"
  if wget --continue --tries=5 --timeout=60 --retry-connrefused --output-document="$part" "$url"; then
    mv -f "$part" "$dest"
    downloaded=$((downloaded + 1))
  else
    echo "FAILED: $file_name" >&2
    failed=$((failed + 1))
  fi
done < <(awk -F '\t' 'NR > 1 && ($3 == "blank" || $3 == "pooled_qc") && $5 == "True" { gsub(/\r/, "", $4); gsub(/\r/, "", $6); print $4 "\t" $6 }' "$manifest")

echo "complete total=$total downloaded=$downloaded skipped=$skipped failed=$failed"
[[ "$failed" -eq 0 ]]
