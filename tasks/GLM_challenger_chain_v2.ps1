# GLM challenger chain v2: restartable, skips completed stages.
# Reads staging dir to determine what's already done.
$ErrorActionPreference = "Continue"
Set-Location D:\DreaMS
$bench = "data/validation/gnps_gold_silver_10ppm_benchmark_v1"
$frozen = "data/validation/noise_gnps_article_benchmark_run_2349091"
$models = "third_party\public_models"
$staging = "data\validation\GLM_challenger_scores\staging"
$panels = "data\validation\GLM_gnps_identity_panel_reconstruction"
$run15 = "data\validation\GLM_gnps_article_benchmark_challengers\run_local"
$bundle12 = "data\validation\GLM_gnps_article_benchmark_public_models\run_local\bundle"
$bundle15 = "$run15\bundle"
$log = "data\validation\GLM_challenger_scores\challenger_v2.log"
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
function Step($msg) { "$(Get-Date -Format 'HH:mm:ss')  $msg" | Tee-Object -FilePath $log -Append }
function Die($msg) { Step "FATAL: $msg"; exit 1 }
Step "challenger chain v2 started"

$env:PYTHONPATH = "D:\DreaMS;D:\DreaMS\tasks"
$env:PYTHONUTF8 = "1"

# ---- determine what still needs doing ----
$need_entropy = -not (Test-Path "$staging\scores_identity_disjoint_entropy_raw_public.npy")
$need_denoise = -not (Test-Path "$staging\scores_identity_disjoint_denoising_search_public.npy")
$need_s2v26 = -not (Test-Path "$staging\scores_identity_disjoint_spec2vec_2026_retrained.npy")

# ---- 1. entropy_raw (skip if staged) ----
if ($need_entropy) {
    Step "computing entropy_raw..."
    cmd /c "python -X utf8 tasks/GLM_score_challenger_models_on_gnps.py --benchmark $bench --frozen-run $frozen --models-dir $models --staging $staging --methods entropy_raw_public >> `"$log`" 2>&1"
    if ($LASTEXITCODE -ne 0) { Die "entropy_raw failed ($LASTEXITCODE)" }
    Step "entropy_raw staged"
} else { Step "entropy_raw already staged, skipping" }

# ---- 2. denoising_search (skip if staged) ----
if ($need_denoise) {
    Step "computing denoising_search (heavy, ~40 min)..."
    cmd /c "python -X utf8 tasks/GLM_score_challenger_models_on_gnps.py --benchmark $bench --frozen-run $frozen --models-dir $models --staging $staging --methods denoising_search_public >> `"$log`" 2>&1"
    if ($LASTEXITCODE -ne 0) { Die "denoising_search failed ($LASTEXITCODE)" }
    Step "denoising_search staged"
} else { Step "denoising_search already staged, skipping" }

# ---- 3. wait for zip then score spec2vec_2026 (skip if staged) ----
if ($need_s2v26) {
    $zip = "$models\Spec2Vec_2026.zip"
    $target = 1992840216
    Step "waiting for Spec2Vec zip ($((Get-Item $zip -ErrorAction SilentlyContinue).Length) / $target bytes)..."
    $deadline = (Get-Date).AddHours(8)
    while (((Get-Item $zip -ErrorAction SilentlyContinue).Length) -lt $target -and (Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 120
    }
    $zlen = (Get-Item $zip -ErrorAction SilentlyContinue).Length
    if ($zlen -ne $target) { Die "zip incomplete: $zlen / $target" }
    Step "zip complete, extracting..."
    $extract = "$models\spec2vec_2026"
    New-Item -ItemType Directory -Force $extract | Out-Null
    cmd /c "powershell -Command Expand-Archive -Path '$zip' -DestinationPath '$extract' -Force >> `"$log`" 2>&1"
    if ($LASTEXITCODE -ne 0) { Die "unzip failed" }
    $model_dir = (Get-ChildItem $extract -Recurse -Filter "*150225*model" | Select-Object -First 1).DirectoryName
    if (-not $model_dir) { Die "model file not found" }
    Step "scoring spec2vec_2026 from $model_dir..."
    cmd /c "python -X utf8 tasks/GLM_score_challenger_models_on_gnps.py --benchmark $bench --frozen-run $frozen --models-dir '$model_dir' --staging $staging --methods spec2vec_2026_retrained >> `"$log`" 2>&1"
    if ($LASTEXITCODE -ne 0) { Die "spec2vec_2026 failed ($LASTEXITCODE)" }
    Step "spec2vec_2026 staged"
} else { Step "spec2vec_2026 already staged, skipping" }

# ---- 4. merge into 15-method bundle (skip if already done) ----
if (-not (Test-Path "$bundle15\method_scores.npz")) {
    Step "merging 15-method bundle..."
    New-Item -ItemType Directory -Force (Split-Path $bundle15) | Out-Null
    python -X utf8 -c @"
import numpy as np, json
from pathlib import Path
staging = Path('$staging')
bundle12 = Path('$bundle12')
bundle15 = Path('$bundle15')
with np.load(bundle12 / 'method_scores.npz') as z:
    names = [str(v) for v in z['method_names']]
    old = {p: np.asarray(z[f'scores_{p}']) for p in ('identity_disjoint','formula_disjoint')}
new_names = [n for n in ('entropy_raw_public','denoising_search_public','spec2vec_2026_retrained')
             if staging.joinpath(f'scores_identity_disjoint_{n}.npy').is_file()]
all_names = names + new_names
matrices = {}
for panel in ('identity_disjoint','formula_disjoint'):
    blocks = [old[panel].astype(np.float32)]
    for nm in new_names:
        p = staging / f'scores_{panel}_{nm}.npy'
        blocks.append(np.load(p).astype(np.float32))
    matrices[panel] = np.vstack(blocks)
np.savez_compressed(bundle15 / 'method_scores.npz',
    method_names=np.asarray(all_names),
    **{f'scores_{p}': matrices[p] for p in matrices})
meta = json.loads((bundle12 / 'method_scores.npz.json').read_text(encoding='utf-8'))
meta.update({'status': 'challenger_extended_bundle', 'methods': all_names,
             'information_levels': {
                 'spectrum_only': meta.get('information_levels',{}).get('spectrum_only',[]) + new_names,
                 'frozen_candidate_reranker': meta.get('information_levels',{}).get('frozen_candidate_reranker',[]),
             }})
(bundle15 / 'method_scores.npz.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
print(f'bundle written: {len(all_names)} methods')
"@ 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { Die "merge failed" }
    Step "15-method bundle written"
} else { Step "bundle already exists, skipping" }

# ---- 5. evaluate (skip if already done) ----
if (-not (Test-Path "$run15\evaluation\report.json")) {
    Step "running frozen evaluator..."
    cmd /c "python -X utf8 tasks/evaluate_noise_gnps_article_benchmark.py --score-bundle `"$bundle15\method_scores.npz`" --benchmark $panels --output `"$run15\evaluation`" --baseline-method official_dreams --bootstrap-resamples 10000 --bootstrap-seed 20261006 >> `"$log`" 2>&1"
    if ($LASTEXITCODE -ne 0) { Die "evaluator failed ($LASTEXITCODE)" }
    Step "evaluation complete"
} else { Step "evaluation already done, skipping" }

# ---- 6. ladder (skip if already done) ----
if (-not (Test-Path "deliverables\GLM_gnps_article_ladder\run_local\ladder_full.csv") -or
    -not (Test-Path "$run15\evaluation\report.json") -or
    (Get-Item "deliverables\GLM_gnps_article_ladder\run_local\ladder_full.csv").LastWriteTime -lt
    (Get-Item "$run15\evaluation\report.json" -ErrorAction SilentlyContinue).LastWriteTime) {
    Step "assembling verified ladder..."
    cmd /c "python -X utf8 tasks/GLM_assemble_gnps_article_ladder.py --run $run15 >> `"$log`" 2>&1"
    if ($LASTEXITCODE -ne 0) { Die "ladder failed ($LASTEXITCODE)" }
    Step "ladder written"
} else { Step "ladder already current, skipping" }

# ---- 7. commit + push ----
git add docs/GLM_GNPS_ARTICLE_LADDER_20261007.md tasks/GLM_score_challenger_models_on_gnps.py 2>$null
git add -f deliverables/figures/GLM_gnps_article_ladder.png 2>$null
git commit -q -m "15-method GNPS ladder with challengers (restartable chain)" 2>$null
git push origin module1-chem-attn-v6-0717 2>$null
Step "challenger chain v2 complete"
