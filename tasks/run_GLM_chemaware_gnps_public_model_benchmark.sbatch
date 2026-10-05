#!/bin/bash
# Public pretrained models (Spec2Vec GNPS-2019, MS2DeepScore dual-mode 2024)
# enter the unified GNPS article benchmark through the pinned pair-score
# cache contract, then the frozen base recipe is extended and evaluated.
# Model files are pinned by Zenodo record and md5; nothing is tuned here.
#SBATCH --job-name=chem_gnps_public
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/chem_gnps_public_%j.out
#SBATCH --error=data/validation/chem_gnps_public_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

BENCHMARK="data/validation/gnps_gold_silver_10ppm_benchmark_v1"
BASELINE_RUN="data/validation/gnps_absolute_baselines_run_2347472"
OFFICIAL_EMB="$BASELINE_RUN/gnps/official_embeddings.npz"
NOISE_V1_EMB="$BASELINE_RUN/gnps/v1_embeddings.npz"
MS_ENTROPY_WHEEL="third_party/noise_article_wheels/ms_entropy-1.5.2-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl"
# Compute nodes have no direct egress (zenodo unreachable; pip goes through
# the site mirror only), so the pinned public models are vendored once on the
# shared filesystem and verified by md5 here.  No downloads in this job.
MODELS="data/reference/gnps_public_models_v1"
OUT="data/validation/GLM_chemaware_gnps_public_model_benchmark/run_${SLURM_JOB_ID}"
SCRATCH_BASE="${SLURM_TMPDIR:-/tmp}"
DEPS="$SCRATCH_BASE/chem_gnps_public_deps_${SLURM_JOB_ID}"

[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }
mkdir -p "$OUT" "$DEPS"
cleanup() { rm -rf -- "$DEPS"; }
trap cleanup EXIT

required=(
  "$BENCHMARK/spectra.mgf"
  "$BENCHMARK/manifest.csv.gz"
  "$BENCHMARK/panel_identity_disjoint.npz"
  "$BENCHMARK/panel_formula_disjoint.npz"
  "$OFFICIAL_EMB" "$NOISE_V1_EMB" "$MS_ENTROPY_WHEEL"
  "$MODELS/spec2vec_UniqueInchikeys_ratio05_filtered_iter_50.model"
  "$MODELS/spec2vec_UniqueInchikeys_ratio05_filtered_iter_50.model.trainables.syn1neg.npy"
  "$MODELS/spec2vec_UniqueInchikeys_ratio05_filtered_iter_50.model.wv.vectors.npy"
  "$MODELS/ms2deepscore_model.pt"
  "$MODELS/settings.json"
  tasks/GLM_build_gnps_public_model_pair_scores.py
  tasks/noise_gnps_article_spectral_scores.py
  tasks/build_noise_gnps_article_score_bundle.py
  tasks/evaluate_noise_gnps_article_benchmark.py
  tasks/extend_gnps_article_score_bundle.py
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || { echo "GLM_GNPS_PUBLIC_MISSING_INPUT: $path" >&2; exit 2; }
done

python -m py_compile \
  tasks/GLM_build_gnps_public_model_pair_scores.py \
  tasks/build_noise_gnps_article_score_bundle.py \
  tasks/evaluate_noise_gnps_article_benchmark.py \
  tasks/extend_gnps_article_score_bundle.py

LOCAL_FREE_KB=$(df -Pk "$SCRATCH_BASE" | awk 'NR==2 {print $4}')
if [[ -z "$LOCAL_FREE_KB" ]] || (( LOCAL_FREE_KB < 8388608 )); then
  echo "Node-local space below 8 GiB: base=$SCRATCH_BASE free_kb=${LOCAL_FREE_KB:-unknown}" >&2
  exit 5
fi

# Coherent 2024-era inference environment for the glibc-2.17 nodes.  The
# scientific ABI is installed into node-local scratch as one closed stack:
# numpy 1.26 / scipy 1.12 simultaneously satisfy gensim 4.3.3, matchms 0.27
# and MS2DeepScore 2.4.0.  SciPy 1.13 cannot be used because it removed
# scipy.linalg.triu while gensim 4.3.3 still imports that symbol.  This also
# avoids the impossible combination introduced
# by matchms 0.28 (scipy>=1.14.1) plus gensim 4.3.3 (scipy<1.14).
# --upgrade is REQUIRED, not cosmetic: several wheels ship colliding bin/
# script directories and pip --target aborts with exit 1 on the conflict
# (verified locally; the silent failure mode kills the job at this line).
python -m pip install --no-deps --upgrade --quiet --target "$DEPS" \
  numpy==1.26.4 \
  scipy==1.12.0 \
  pandas==2.2.3 \
  scikit-learn==1.5.2 \
  matplotlib==3.7.2 \
  numba==0.59.1 \
  llvmlite==0.42.0 \
  matchms==0.27.0 \
  sparsestack==0.5.0 \
  spec2vec==0.8.0 \
  gensim==4.3.3 \
  smart_open==7.1.0 \
  ms2deepscore==2.4.0 \
  pickydict==0.5.0 \
  deprecated==1.3.1 \
  wrapt==2.5.0 \
  pubchempy==1.0.5 \
  pyteomics==5.0.1 \
  networkx==3.6.1 \
  pyyaml==6.0.3 \
  requests==2.34.2 \
  tqdm==4.70.1 \
  tensorboard==2.18.0 \
  absl-py==2.5.0 \
  markdown==3.11 \
  werkzeug==3.1.9 \
  protobuf==4.25.8
# Some pinned wheels bundle a top-level tests/ package (spec2vec does, and a
# second one collides on this node).  pip leaves that as a warning and the
# directory is dead weight shadowing on PYTHONPATH, so remove it outright.
rm -rf -- "$DEPS/tests"
export PYTHONPATH="$DEPS:$PYTHONPATH"
python -m pip install --no-deps --upgrade --quiet --target "$DEPS" "$MS_ENTROPY_WHEEL"

# Fail before model loading if the isolated stack cannot host the adapters.
# Exact versions are checked after PYTHONPATH activation, so a dependency
# accidentally imported from the parent conda environment cannot pass.
python - <<'PYEOF'
from importlib.metadata import version
import numpy
import scipy
import torch
import matchms
from spec2vec import Spec2Vec
from gensim.models import Word2Vec
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model
probe = Word2Vec([["a", "b"], ["b", "c"]], vector_size=8, min_count=1, epochs=1)
probe.wv.most_similar("a")
expected = {
    "numpy": "1.26.4",
    "scipy": "1.12.0",
    "pandas": "2.2.3",
    "scikit-learn": "1.5.2",
    "matplotlib": "3.7.2",
    "matchms": "0.27.0",
    "spec2vec": "0.8.0",
    "gensim": "4.3.3",
    "ms2deepscore": "2.4.0",
    "sparsestack": "0.5.0",
    "numba": "0.59.1",
}
observed = {name: version(name) for name in expected}
if observed != expected:
    raise RuntimeError(f"public-model dependency drift: {observed} != {expected}")
if not torch.cuda.is_available():
    raise RuntimeError("allocated GPU is not visible to MS2DeepScore")
print("PUBLIC_MODEL_IMPORT_PREFLIGHT_PASS", observed, "torch", torch.__version__)
PYEOF

# Pinned model integrity over the vendored shared-FS copies (no downloads:
# compute nodes cannot reach zenodo; only the site pip mirror works).
printf '%s  %s\n' \
  a7333c19ace863c5a0a04abc89600f49 "$MODELS/spec2vec_UniqueInchikeys_ratio05_filtered_iter_50.model" \
  cd774bf75d00b1715334a4fbfe2b711a "$MODELS/spec2vec_UniqueInchikeys_ratio05_filtered_iter_50.model.trainables.syn1neg.npy" \
  0962044c1020073d487ae2e07d195f0b "$MODELS/spec2vec_UniqueInchikeys_ratio05_filtered_iter_50.model.wv.vectors.npy" \
  d5cbf4694a1c476ae59e0c810f56c320 "$MODELS/ms2deepscore_model.pt" \
  5d6007c453692b44cd73941a9a434d84 "$MODELS/settings.json" \
  | md5sum --check --strict

python -u tasks/GLM_build_gnps_public_model_pair_scores.py \
  --benchmark "$BENCHMARK" \
  --method spec2vec_gnps_2019 \
  --method ms2deepscore_dual_2024 \
  --spec2vec-model-dir "$MODELS" \
  --ms2deepscore-model-dir "$MODELS" \
  --embedding-batch-size 256 \
  --output-root "$OUT/public_model_pair_caches"

python -u tasks/build_noise_gnps_article_score_bundle.py \
  --benchmark "$BENCHMARK" \
  --official-embeddings "$OFFICIAL_EMB" \
  --noise-v1-embeddings "$NOISE_V1_EMB" \
  --output "$OUT/base_method_scores.npz" \
  --workers 8 \
  --fragment-tolerance 0.02 \
  --n-highest-peaks 100

python -u tasks/extend_gnps_article_score_bundle.py \
  --benchmark "$BENCHMARK" \
  --base-bundle "$OUT/base_method_scores.npz" \
  --pair-score-method "spec2vec_gnps_2019=$OUT/public_model_pair_caches/spec2vec_gnps_2019_pair_cache" \
  --pair-score-method "ms2deepscore_dual_2024=$OUT/public_model_pair_caches/ms2deepscore_dual_2024_pair_cache" \
  --output "$OUT/extended_method_scores.npz"

python -u tasks/evaluate_noise_gnps_article_benchmark.py \
  --benchmark "$BENCHMARK" \
  --score-bundle "$OUT/extended_method_scores.npz" \
  --output "$OUT/evaluation" \
  --baseline-method official_dreams \
  --bootstrap-resamples 10000 \
  --bootstrap-seed 20261003

sha256sum "$OUT"/public_model_pair_caches/*/report.json \
  "$OUT/base_method_scores.npz" \
  "$OUT/extended_method_scores.npz" \
  "$OUT"/evaluation/report.json > "$OUT/SHA256SUMS.txt"

echo "GLM_GNPS_PUBLIC_MODEL_BENCHMARK_COMPLETE: $OUT"
