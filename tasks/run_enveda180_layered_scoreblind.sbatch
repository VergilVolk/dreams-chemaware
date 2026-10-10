#!/bin/bash
# Generate frozen component scores without loading Enveda labels or metrics.
# This is not the one-time opening job.
#SBATCH --job-name=e180_scoreblind
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --output=data/validation/enveda180_scoreblind_%j.out
#SBATCH --error=data/validation/enveda180_scoreblind_%j.err

set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
  export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
fi

BENCHMARK="${ENVEDA180_BENCHMARK:-data/validation/enveda180_cross_condition_benchmark_v2_20261010}"
AUDIT="${ENVEDA180_AUDIT:-data/validation/enveda180_scoreblind_manifest_v2_20261010}"
FREEZE="tasks/enveda180_layered_freeze_contract_v1.json"
OFFICIAL="data/e1/official_embedding_slim.pt"
NOISE="data/validation/noise_relation_t1_t3_run_2347055/checkpoint/primary_seed_3407_slim.pt"
ARCH="dreams/models/pretrained/ssl_model_server.pt"
OUT="data/validation/enveda180_layered_scoreblind_run_${SLURM_JOB_ID}"

required=(
  "$BENCHMARK/report.json"
  "$BENCHMARK/checksums.sha256"
  "$BENCHMARK/spectra.mgf"
  "$BENCHMARK/manifest.csv.gz"
  "$BENCHMARK/panel_identity_disjoint.npz"
  "$BENCHMARK/panel_formula_disjoint.npz"
  "$AUDIT/report.json"
  "$FREEZE"
  "$OFFICIAL"
  "$NOISE"
  "$ARCH"
  tasks/encode_gnps_gold_silver_10ppm_checkpoint.py
  tasks/build_noise_gnps_article_score_bundle.py
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { echo "ENVEDA180_SCOREBLIND_MISSING_INPUT: $path" >&2; exit 5; }
done
[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 4; }

python - "$BENCHMARK" "$AUDIT" "$FREEZE" "$OFFICIAL" "$NOISE" <<'PY'
import hashlib, json, pathlib, sys
benchmark, audit, freeze_path, official, noise = map(pathlib.Path, sys.argv[1:])
freeze = json.loads(freeze_path.read_text())
report = json.loads((benchmark / "report.json").read_text())
audit_report = json.loads((audit / "report.json").read_text())
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8<<20),b""): h.update(block)
    return h.hexdigest()
checks = {
    "freeze": freeze.get("freeze_status") == "FROZEN_BEFORE_ENVEDA_SCORING",
    "benchmark_path": freeze.get("benchmark_directory") == str(benchmark),
    "benchmark_status": report.get("status") == "ENVEDA180_CROSS_CONDITION_BENCHMARK_FROZEN",
    "unopened": report.get("performance_scores_opened") is False,
    "role": report.get("evaluation_role") == "sealed_external_test",
    "exclusions": report.get("leakage_guard",{}).get("exclusion_policy_complete") is True,
    "identity_overlap": report.get("leakage_guard",{}).get("selected_consumed_identity_overlap") == 0,
    "spectrum_overlap": report.get("leakage_guard",{}).get("selected_consumed_spectrum_overlap") == 0,
    "audit_sources": audit_report.get("exclusion_policy_complete") is True and all(
        row.get("status") == "loaded" for row in audit_report.get("exclusion_sources",[])
        if row.get("required",True)),
    "official_checkpoint": sha(official) == freeze["checkpoints"]["official_dreams"]["sha256"],
    "noise_checkpoint": sha(noise) == freeze["checkpoints"]["noise_v1"]["sha256"],
}
failed=[key for key,value in checks.items() if not value]
if failed: raise SystemExit(f"ENVEDA180_SCOREBLIND_PREFLIGHT_FAIL: {failed}")
print(json.dumps({"status":"ENVEDA180_SCOREBLIND_PREFLIGHT_PASS","checks":checks},indent=2))
PY

mkdir -p "$OUT"

LOCAL_BASE="${SLURM_TMPDIR:-/tmp}"
LOCAL_DEPS="$LOCAL_BASE/enveda180_score_deps_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_DEPS"
python -m pip install --no-deps --target "$LOCAL_DEPS" \
  third_party/noise_article_wheels/ms_entropy-1.5.2-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl
export PYTHONPATH="$LOCAL_DEPS:$PYTHONPATH"

python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
  --benchmark "$BENCHMARK" --checkpoint "$OFFICIAL" --architecture-checkpoint "$ARCH" \
  --output "$OUT/official_embeddings.npz" --label enveda180_official \
  --batch-size 64 --expected-selected-rows 0 --expected-records 0

python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
  --benchmark "$BENCHMARK" --checkpoint "$NOISE" --architecture-checkpoint "$ARCH" \
  --output "$OUT/noise_v1_embeddings.npz" --label enveda180_noise_v1 \
  --batch-size 64 --expected-selected-rows 0 --expected-records 0

WORKERS="${SLURM_CPUS_PER_TASK:-10}"
python -u tasks/build_noise_gnps_article_score_bundle.py \
  --benchmark "$BENCHMARK" \
  --official-embeddings "$OUT/official_embeddings.npz" \
  --noise-v1-embeddings "$OUT/noise_v1_embeddings.npz" \
  --output "$OUT/method_scores_unopened.npz" \
  --workers "$WORKERS" --fragment-tolerance 0.02 --n-highest-peaks 100

cp "$FREEZE" "$OUT/freeze_contract.json"
python - "$OUT" <<'PY'
import hashlib, json, pathlib, sys
root=pathlib.Path(sys.argv[1])
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8<<20),b""): h.update(block)
    return h.hexdigest()
meta=json.loads((root/"method_scores_unopened.npz.json").read_text())
if meta.get("labels_opened") is not False: raise SystemExit("score generation opened labels")
report={
 "status":"ENVEDA180_LAYERED_COMPONENT_SCORES_READY_UNOPENED",
 "performance_metrics_computed":False,
 "labels_opened":False,
 "score_bundle":"method_scores_unopened.npz",
 "score_bundle_sha256":sha(root/"method_scores_unopened.npz"),
 "freeze_contract_sha256":sha(root/"freeze_contract.json"),
}
(root/"scoreblind_report.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
PY

echo "Enveda-180 score-blind components ready; external labels remain unopened: $OUT"
