#!/usr/bin/env bash
set -euo pipefail

# Read-only storage inventory. This script never deletes, moves, compresses, or
# modifies an artifact. Run it from the repository root on the server:
#   bash tasks/audit_server_storage_readonly.sh data/validation > server_storage_inventory.tsv

target_root="${1:-data/validation}"

if [[ ! -d "$target_root" ]]; then
  echo "ERROR: directory does not exist: $target_root" >&2
  exit 2
fi

resolved_root="$(realpath "$target_root")"
case "$resolved_root" in
  /|/home|/scratch|/data)
    echo "ERROR: refusing broad inventory root: $resolved_root" >&2
    exit 3
    ;;
esac

printf 'SECTION\tFILESYSTEM\n'
df -h "$resolved_root"

printf '\nSECTION\tTOP_LEVEL_BYTES\n'
printf 'bytes\tpath\n'
du -x -B1 --max-depth=1 "$resolved_root" 2>/dev/null | sort -nr

printf '\nSECTION\tFILES_OVER_256_MIB\n'
printf 'bytes\tmtime\tpath\n'
find "$resolved_root" -xdev -type f -size +256M \
  -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS\t%p\n' 2>/dev/null | sort -nr

printf '\nSECTION\tCHECKPOINT_TOTALS_BY_TOP_LEVEL_DIR\n'
printf 'bytes\tfiles\tpath\n'
while IFS= read -r -d '' child; do
  checkpoint_bytes="$({ find "$child" -xdev -type f \
    \( -name '*.pt' -o -name '*.ckpt' -o -iname '*optimizer*' -o -iname '*optim_state*' \) \
    -printf '%s\n' 2>/dev/null || true; } | awk '{s+=$1} END {print s+0}')"
  checkpoint_files="$({ find "$child" -xdev -type f \
    \( -name '*.pt' -o -name '*.ckpt' -o -iname '*optimizer*' -o -iname '*optim_state*' \) \
    -print 2>/dev/null || true; } | wc -l)"
  if [[ "$checkpoint_files" -gt 0 ]]; then
    printf '%s\t%s\t%s\n' "$checkpoint_bytes" "$checkpoint_files" "$child"
  fi
done < <(find "$resolved_root" -mindepth 1 -maxdepth 1 -type d -print0) | sort -nr

printf '\nSECTION\tPARTIAL_TEMP_AND_STAGING_FILES\n'
printf 'bytes\tmtime\tpath\n'
find "$resolved_root" -xdev -type f \
  \( -name '*.filepart' -o -name '*.tmp' -o -name '*.partial' -o -name '*.part' \) \
  -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS\t%p\n' 2>/dev/null | sort -nr

printf '\nSECTION\tEMPTY_DIRECTORIES\n'
find "$resolved_root" -xdev -type d -empty -print 2>/dev/null | sort

printf '\nSECTION\tRUN_DIRECTORIES_WITHOUT_REPORT_OR_DECISION\n'
while IFS= read -r -d '' run_dir; do
  if [[ ! -f "$run_dir/report.json" && ! -f "$run_dir/decision.json" ]]; then
    printf '%s\n' "$run_dir"
  fi
done < <(find "$resolved_root" -xdev -type d -name 'run_*' -print0 2>/dev/null) | sort

