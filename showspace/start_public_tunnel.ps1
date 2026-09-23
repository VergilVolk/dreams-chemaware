$ErrorActionPreference = 'Stop'

$ShowspaceDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvFile = Join-Path $ShowspaceDir '.env'
$RunDir = Join-Path $ShowspaceDir 'run'
$LogDir = Join-Path $ShowspaceDir 'logs'
$PidFile = Join-Path $RunDir 'cloudflared.pid'

if (-not (Test-Path -LiteralPath $EnvFile)) { throw "Missing $EnvFile" }
Get-Content -LiteralPath $EnvFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith('#') -and $line.Contains('=')) {
        $name, $value = $line.Split('=', 2)
        [Environment]::SetEnvironmentVariable($name.Trim(), $value.Trim(), 'Process')
    }
}

if ([Environment]::GetEnvironmentVariable('SHOWSPACE_PUBLIC_MODE','Process') -ne 'true') {
    throw 'Set SHOWSPACE_PUBLIC_MODE=true only after local healthcheck passes.'
}
$tokenFile = [Environment]::GetEnvironmentVariable('CLOUDFLARE_TUNNEL_TOKEN_FILE','Process')
if (-not $tokenFile -or -not (Test-Path -LiteralPath $tokenFile)) {
    throw 'Missing CLOUDFLARE_TUNNEL_TOKEN_FILE. Put the named-tunnel token in the ignored secrets directory.'
}
$cloudflared = [Environment]::GetEnvironmentVariable('CLOUDFLARED_EXE','Process')
if (-not $cloudflared) { $cloudflared = 'cloudflared.exe' }
if (-not (Get-Command $cloudflared -ErrorAction SilentlyContinue)) { throw "cloudflared executable not found: $cloudflared" }

& (Join-Path $ShowspaceDir 'healthcheck_showspace.ps1')
New-Item -ItemType Directory -Force -Path $RunDir, $LogDir | Out-Null
if (Test-Path -LiteralPath $PidFile) {
    $existingPid = [int](Get-Content -LiteralPath $PidFile | Select-Object -First 1)
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId=$existingPid" -ErrorAction SilentlyContinue
    if ($existing -and $existing.Name -eq 'cloudflared.exe') { throw "cloudflared already running with PID $existingPid" }
    Remove-Item -LiteralPath $PidFile -Force
}

$stdout = Join-Path $LogDir 'cloudflared.out.log'
$stderr = Join-Path $LogDir 'cloudflared.err.log'
$process = Start-Process -FilePath $cloudflared -WindowStyle Hidden `
    -ArgumentList @('tunnel','--no-autoupdate','run','--token-file', $tokenFile) `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$process.Id | Set-Content -LiteralPath $PidFile
Write-Host "Cloudflare named tunnel started with PID $($process.Id)"
Write-Host "Logs: $LogDir"
