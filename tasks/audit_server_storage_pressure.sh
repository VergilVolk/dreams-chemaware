#!/usr/bin/env bash
set -euo pipefail

# Compact, read-only snapshot for a user-quota incident.
root="/data02/run01/scv7tsl"
repo="$root/DreaMS"
[[ -d "$repo/data/validation" ]] || { echo "missing repository: $repo" >&2; exit 2; }

echo '[QUOTA]'
quota -w 2>&1 || true
echo '[JOBS]'
squeue -u "$USER" -o '%.18i %.32j %.10T %.10M %R' 2>&1 || true
echo '[RECENT_LARGE_OUTPUTS]'
find "$repo/data/validation" -xdev -type f -mmin -1440 -size +64M \
  -printf '%s\t%TY-%Tm-%TdT%TH:%TM\t%p\n' 2>/dev/null | sort -nr | head -n 40
echo '[TOP_VALIDATION_DIRECTORIES]'
du -x -B1 --max-depth=2 "$repo/data/validation" 2>/dev/null | sort -nr | head -n 40
echo '[LARGE_CHECKPOINTS]'
find "$repo/data/validation" -xdev -type f \
  \( -name '*.ckpt' -o -name '*.pt' \) -size +256M \
  -printf '%s\t%TY-%Tm-%TdT%TH:%TM\t%p\n' 2>/dev/null | sort -nr | head -n 60
echo '[PARTIAL_AND_STAGING]'
find "$repo/data/validation" -xdev \
  \( -type f \( -name '*.filepart' -o -name '*.partial' -o -name '*.part' \) \
     -o -type d -name '*.staging' \) \
  -printf '%y\t%s\t%TY-%Tm-%TdT%TH:%TM\t%p\n' 2>/dev/null | sort -k4
echo '[OPEN_DELETED]'
if command -v lsof >/dev/null 2>&1; then
  lsof -nP +L1 2>/dev/null | awk '$9 ~ /^\/data02\/run01\/scv7tsl\// {sum += $7; print} END {printf "bytes=%0.f\n", sum}'
else
  echo 'lsof unavailable'
fi
