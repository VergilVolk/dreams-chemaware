$ErrorActionPreference = 'Stop'

$ShowspaceDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = Split-Path -Parent $ShowspaceDir
$EnvFile = Join-Path $ShowspaceDir '.env'
$RunDir = Join-Path $ShowspaceDir 'run'
$LogDir = Join-Path $ShowspaceDir 'logs'
$PidFile = Join-Path $RunDir 'showspace.pid'

if (-not (Test-Path -LiteralPath $EnvFile)) {
    throw "Missing $EnvFile. Copy .env.example to .env and configure local assets."
}

Get-Content -LiteralPath $EnvFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith('#') -and $line.Contains('=')) {
        $name, $value = $line.Split('=', 2)
        [Environment]::SetEnvironmentVariable($name.Trim(), $value.Trim(), 'Process')
    }
}

$configuredPython = [Environment]::GetEnvironmentVariable('SHOWSPACE_PYTHON', 'Process')
$pythonCandidates = @(
    $configuredPython,
    (Join-Path $RepoDir '.venv_showspace\Scripts\python.exe'),
    (Join-Path $RepoDir '.venv\Scripts\python.exe')
) | Where-Object { $_ }
$VenvPython = $pythonCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $VenvPython) {
    throw "No Showspace Python found. Run showspace\setup_showspace_env.ps1 first."
}

foreach ($name in @('DREAMS_OFFICIAL_SLIM_CKPT', 'DREAMS_ARCHITECTURE_CKPT', 'DREAMS_MOLECULE_DB', 'DREAMS_OFFICIAL_EMBEDDING_INDEX', 'DREAMS_OFFICIAL_EMBEDDING_INDEX_MANIFEST')) {
    $value = [Environment]::GetEnvironmentVariable($name, 'Process')
    if (-not $value -or -not (Test-Path -LiteralPath $value)) {
        throw "$name is missing or does not exist: $value"
    }
}

if ([Environment]::GetEnvironmentVariable('GRADIO_SHARE', 'Process') -ne 'false') {
    throw 'GRADIO_SHARE must remain false for the local-compute deployment.'
}
if ([Environment]::GetEnvironmentVariable('GRADIO_SERVER_NAME', 'Process') -ne '127.0.0.1') {
    throw 'GRADIO_SERVER_NAME must be 127.0.0.1; publish through the reverse tunnel, not an open listening socket.'
}

New-Item -ItemType Directory -Force -Path $RunDir, $LogDir | Out-Null
if (Test-Path -LiteralPath $PidFile) {
    $existingPid = [int](Get-Content -LiteralPath $PidFile | Select-Object -First 1)
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId=$existingPid" -ErrorAction SilentlyContinue
    if ($existing -and $existing.Name -match '^python(w)?\.exe$' -and $existing.CommandLine -match 'showspace[\\/]+app\.py') {
        throw "showspace is already running with PID $existingPid"
    }
    Remove-Item -LiteralPath $PidFile -Force
}

& $VenvPython -c "import gradio, torch, h5py, numpy; print('[preflight] imports PASS', gradio.__version__, torch.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Showspace Python dependency preflight failed.' }
& $VenvPython (Join-Path $ShowspaceDir 'preflight_deployment.py')
if ($LASTEXITCODE -ne 0) { throw 'Showspace model/index deployment preflight failed.' }

$stdout = Join-Path $LogDir 'showspace.out.log'
$stderr = Join-Path $LogDir 'showspace.err.log'
$process = Start-Process -FilePath $VenvPython -WorkingDirectory $RepoDir -WindowStyle Hidden `
    -ArgumentList @('-u', 'showspace\app.py') -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr -PassThru
$process.Id | Set-Content -LiteralPath $PidFile
Write-Host "showspace started with PID $($process.Id)"
Write-Host "Local URL: http://127.0.0.1:$([Environment]::GetEnvironmentVariable('GRADIO_SERVER_PORT','Process'))"
Write-Host "Logs: $LogDir"
