param(
    [string]$BasePython = 'D:\dreams_env\python.exe',
    [switch]$Recreate
)
$ErrorActionPreference = 'Stop'

$ShowspaceDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = Split-Path -Parent $ShowspaceDir
$VenvDir = Join-Path $RepoDir '.venv_showspace'
$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $BasePython)) { throw "Base Python not found: $BasePython" }
if ($Recreate -and (Test-Path -LiteralPath $VenvDir)) {
    $repoFull = [IO.Path]::GetFullPath($RepoDir).TrimEnd('\')
    $venvFull = [IO.Path]::GetFullPath($VenvDir).TrimEnd('\')
    if (-not $venvFull.StartsWith($repoFull + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a virtual environment outside the repository: $venvFull"
    }
    Remove-Item -LiteralPath $venvFull -Recurse -Force
}
if (-not (Test-Path -LiteralPath $VenvPython)) {
    & $BasePython -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create Showspace virtual environment.' }
}

& $VenvPython -m pip install --upgrade pip wheel
& $VenvPython -m pip install `
    'gradio==4.44.1' 'gradio-client==1.3.0' 'huggingface_hub==0.33.5' `
    'fastapi==0.115.2' 'starlette==0.40.0' 'pydantic==2.9.2' `
    'matplotlib==3.8.4' 'numpy==1.26.4' 'pandas==2.2.1' `
    'h5py==3.11.0' 'pyteomics==4.7.5' 'websockets==12.0'
if ($LASTEXITCODE -ne 0) { throw 'Failed to install Showspace web dependencies.' }

$baseSite = & $BasePython -c "import site; print(next(p for p in site.getsitepackages() if p.lower().endswith('site-packages')))"
$pth = Join-Path $VenvDir 'Lib\site-packages\dreams_base_environment.pth'
Set-Content -LiteralPath $pth -Value $baseSite -Encoding ascii

& $VenvPython -c "import gradio, fastapi, torch, h5py, rdkit; print('[setup] PASS', gradio.__version__, fastapi.__version__, torch.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Final Showspace environment import check failed.' }
Write-Host "Showspace environment ready: $VenvPython"
