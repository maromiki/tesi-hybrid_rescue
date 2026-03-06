#!/bin/bash
set -euo pipefail

ARCH_ROOT="/ext/${USER}/archive_finalissima_20260226"
LOG_FILE="${ARCH_ROOT}/archive.log"
MANIFEST="${ARCH_ROOT}/manifest.tsv"

mkdir -p "${ARCH_ROOT}"

echo -e "timestamp\tsource\tarchive\tstatus\tsize_before" > "${MANIFEST}"

aarchive() {
  local src="$1"
  if [[ ! -d "$src" ]]; then
    echo "SKIP missing: $src" | tee -a "$LOG_FILE"
    return 0
  fi

  local base
  base="$(basename "$src")"
  local parent
  parent="$(basename "$(dirname "$src")")"
  local arc="${ARCH_ROOT}/${parent}__${base}.tar.zst"
  local ts
  ts="$(date '+%F %T')"
  local size_before
  size_before="$(du -sh "$src" | awk '{print $1}')"

  echo "[$ts] ARCHIVE START: $src -> $arc (size=$size_before)" | tee -a "$LOG_FILE"

  # Archive with compression and delete source only on successful completion
  tar --use-compress-program='zstd -T0 -3' -cf "$arc" "$src"

  # Sanity check
  if [[ ! -s "$arc" ]]; then
    echo "[$(date '+%F %T')] ARCHIVE ERROR: empty archive $arc" | tee -a "$LOG_FILE"
    echo -e "${ts}\t${src}\t${arc}\tERROR_EMPTY_ARCHIVE\t${size_before}" >> "$MANIFEST"
    return 1
  fi

  rm -rf "$src"
  mkdir -p "$src"

  echo "[$(date '+%F %T')] ARCHIVE DONE: $src" | tee -a "$LOG_FILE"
  echo -e "${ts}\t${src}\t${arc}\tOK\t${size_before}" >> "$MANIFEST"
}

# Priority: biggest and less critical for the current new run
aarchive "data/dataset/camisim/simulation_short_biased"
aarchive "data/dataset/camisim/simulation_long_biased"
aarchive "data/output/minimap2/camisim"
aarchive "data/output/sga/camisim"
aarchive "data/output/metaspades/camisim"
aarchive "data/output/flye/camisim"
aarchive "data/output/ground_truth/camisim"
aarchive "data/output/4cac/camisim"
aarchive "data/output/dmc/camisim"
aarchive "data/output/hybrid/camisim"

echo "All archive tasks completed at $(date '+%F %T')" | tee -a "$LOG_FILE"
