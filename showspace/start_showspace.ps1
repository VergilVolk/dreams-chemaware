$ErrorActionPreference = 'Stop'

$ShowspaceDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = Split-Path -Parent $ShowspaceDir
$VenvPython = Join-Path $RepoDir '.venv\Scripts\python.exe'
$EnvFile = Join-Path $ShowspaceDir '.env'
$RunDir = Join-Path $ShowspaceDir 'run'
$LogDir = Join-Path $ShowspaceDir 'logs'
$PidFile = Join-Path $RunDir 'showspace.pid'

if (-not (Test-Path $VenvPython)) {
    throw "Python virtual environment not found: $VenvPython`nCreate it with: py -3.11 -m venv .venv"
}
if (-not (Test-Path $EnvFile)) {
    throw "Missing $EnvFile. Copy .env.example to .env and edit the model/database paths."
}

New-Item -ItemType Directory -Force -Path $RunDir, $LogDir | Out-Null
Get-Content $EnvFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith('#') -and $line.Contains('=')) {
        $name, $value = $line.Split('=', 2)
        [Environment]::SetEnvironmentVariable($name.Trim(), $value.Trim(), 'Process')
    }
}

foreach ($name in @('DREAMS_EMBEDDING_CKPT', 'DREAMS_SSL_CKPT', 'DREAMS_MOLECULE_DB')) {
    $value = [Environment]::GetEnvironmentVariable($name, 'Process')
    if (-not $value -or -not (Test-Path $value)) {
        throw "$name is missing or does not exist: $value"
    }
}

$existingPid = $null
if (Test-Path $PidFile) { $existingPid = Get-Content $PidFile | Select-Object -First 1 }
if ($existingPid -and (Get-Process -Id $existingPid -ErrorAction SilentlyContinue)) {
    throw "showspace is already running with PID $existingPid"
}

$stdout = Join-Path $LogDir 'showspace.out.log'
$stderr = Join-Path $LogDir 'showspace.err.log'
$process = Start-Process -FilePath $VenvPython -WorkingDirectory $RepoDir `
    -ArgumentList @('showspace\app.py') -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr -PassThru
$process.Id | Set-Content $PidFile
Write-Host "showspace started with PID $($process.Id)"
Write-Host "URL: http://$([Environment]::GetEnvironmentVariable('GRADIO_SERVER_NAME','Process')):$([Environment]::GetEnvironmentVariable('GRADIO_SERVER_PORT','Process'))"
if ([Environment]::GetEnvironmentVariable('GRADIO_SHARE','Process') -eq 'true') {
    Write-Host 'Gradio share link is enabled; check the output log for the public URL.'
}
Write-Host "Logs: $LogDir"
