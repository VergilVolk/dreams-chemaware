#!/bin/bash
#SBATCH --job-name=chem_sirius_src
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/chem_sirius_src_%j.out
#SBATCH --error=data/validation/chem_sirius_src_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

SIRIUS_BIN="${SIRIUS_BIN:-sirius}"
OUT="data/validation/chemaware_sirius_source/run_${SLURM_JOB_ID}"
PANEL="$OUT/source_panel"
DATABASE="$OUT/ChemCand.siriusdb"

[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }
command -v "$SIRIUS_BIN" >/dev/null 2>&1 || {
  echo "SIRIUS 6 launcher not found. Set SIRIUS_BIN to the installed launcher." >&2
  exit 2
}
mkdir -p "$OUT"

required_files=(
  data/models/MassSpecGym_MurckoHist_split.hdf5
  data/validation/chemaware_high_coverage_native/run_2340524/evidence/train_triplet_evidence.npz
  data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz
  data/validation/chemaware_msfinder_rule_tables_v1_20260928/msfinder_rules.json
  tasks/export_chemaware_sirius_source_panel.py
  tasks/build_chemaware_layered_rule_corpus.py
  tasks/build_chemaware_layered_rule_source_ledger.py
  tasks/import_chemaware_sirius_scores.py
  tasks/import_chemaware_sirius_controlled_source_ledger.py
  tasks/qualify_chemaware_candidate_source_ledger.py
  tasks/test_export_chemaware_sirius_source_panel.py
  tasks/test_chemaware_layered_rule_corpus.py
  tasks/test_chemaware_layered_rule_source_ledger.py
  tasks/test_import_chemaware_sirius_scores.py
  tasks/test_import_chemaware_sirius_controlled_source_ledger.py
)
for required_file in "${required_files[@]}"; do
  [[ -f "$required_file" ]] || {
    echo "Missing ChemAware SIRIUS source file: $required_file" >&2
    exit 2
  }
done

python -u tasks/test_export_chemaware_sirius_source_panel.py
python -u tasks/test_chemaware_layered_rule_corpus.py
python -u tasks/test_chemaware_layered_rule_source_ledger.py
python -u tasks/test_import_chemaware_sirius_scores.py
python -u tasks/test_import_chemaware_sirius_controlled_source_ledger.py
"$SIRIUS_BIN" --version 2>&1 | tee "$OUT/sirius_version.txt"
grep -Eq '(^|[^0-9])6\.[0-9]+' "$OUT/sirius_version.txt" || {
  echo "ChemAware controlled import requires SIRIUS major version 6" >&2
  exit 2
}

python -u tasks/export_chemaware_sirius_source_panel.py --output "$PANEL"

# Build and qualify the versioned rule-table source independently of SIRIUS.
# These scores use the real HDF5 spectrum_row from query_registry.tsv; manifest
# query ordinals must never index the spectrum table directly.
python -u tasks/build_chemaware_layered_rule_corpus.py \
  --output "$OUT/layered_rule_corpus"
python -u tasks/build_chemaware_layered_rule_source_ledger.py \
  --corpus "$OUT/layered_rule_corpus/rule_corpus.json" \
  --candidate-ledger "$PANEL/candidate_ledger.tsv" \
  --query-registry "$PANEL/query_registry.tsv" \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --output "$OUT/unqualified_static_source"
python -u tasks/qualify_chemaware_candidate_source_ledger.py \
  --ledger "$OUT/unqualified_static_source" \
  --manifest data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz \
  --output "$OUT/qualified_static_source" \
  --bootstrap-draws 5000 --minimum-applicable-queries 100

# The database contains exactly the structures already present in the formal
# ChemAware candidate graph.  Each query-formula replica fixes one candidate
# formula, guaranteeing complete fragmentation-tree/fingerprint coverage.
"$SIRIUS_BIN" custom-db --input "$PANEL/candidate_structures.tsv" \
  --name ChemCand \
  --location "$DATABASE"

run_sirius_arm() {
  local arm="$1"
  local input_root="$2"
  local profile input project summary
  local -a profile_args
  for profile in orbitrap qtof default; do
    input="$input_root/$profile"
    [[ -d "$input" ]] || continue
    project="$OUT/projects/${arm}_${profile}.sirius"
    summary="$OUT/summaries/$arm/$profile"
    mkdir -p "$summary"
    if [[ "$profile" == "orbitrap" ]]; then
      profile_args=(-p orbitrap --ppm-max 10 --ppm-max-ms2 5)
    elif [[ "$profile" == "qtof" ]]; then
      profile_args=(-p qtof --ppm-max 10 --ppm-max-ms2 10)
    else
      # qtof is the documented SIRIUS default for unknown instruments.
      profile_args=(--ppm-max 10 --ppm-max-ms2 10)
    fi
    "$SIRIUS_BIN" \
      --input "$input" \
      --project "$project" \
      formulas "${profile_args[@]}" \
      fingerprints \
      structures --database ChemCand \
      write-summaries --top-k-summary 50 --output "$summary"
  done
}

# Correct chemistry and two capacity-matched spectral controls are processed
# with the identical SIRIUS command.  No truth label is available to SIRIUS.
run_sirius_arm correct "$PANEL/query_formula_ms"
run_sirius_arm intensity_rank_permuted \
  "$PANEL/query_formula_ms_intensity_rank_permuted"
run_sirius_arm mass_shifted "$PANEL/query_formula_ms_mass_shifted"

python -u tasks/import_chemaware_sirius_controlled_source_ledger.py \
  --source-panel "$PANEL" \
  --correct-summary "$OUT/summaries/correct" \
  --intensity-summary "$OUT/summaries/intensity_rank_permuted" \
  --mass-summary "$OUT/summaries/mass_shifted" \
  --output "$OUT/unqualified_source"

python -u tasks/qualify_chemaware_candidate_source_ledger.py \
  --ledger "$OUT/unqualified_source" \
  --manifest data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz \
  --output "$OUT/qualified_source" \
  --bootstrap-draws 5000 --minimum-applicable-queries 100

SIRIUS_AUTHORIZED="$(python -c "import json; print(int(json.load(open('$OUT/qualified_source/report.json'))['formal_triplet_mining_authorized']))")"
STATIC_AUTHORIZED="$(python -c "import json; print(int(json.load(open('$OUT/qualified_static_source/report.json'))['formal_triplet_mining_authorized']))")"
if [[ "$SIRIUS_AUTHORIZED" != "1" && "$STATIC_AUTHORIZED" != "1" ]]; then
  echo "CHEMAWARE_SOURCE_STOP: neither SIRIUS nor static rules passed every matched-control gate"
  echo "COMPLETE: controlled negative evidence -> $OUT"
  exit 0
fi

echo "CHEMAWARE_SOURCE_AUTHORIZATION static=$STATIC_AUTHORIZED sirius=$SIRIUS_AUTHORIZED"
echo "COMPLETE: qualified ChemAware source evidence -> $OUT"
