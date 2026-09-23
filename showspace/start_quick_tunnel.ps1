$ErrorActionPreference = 'Stop'

$ShowspaceDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvFile = Join-Path $ShowspaceDir '.env'
$RunDir = Join-Path $ShowspaceDir 'run'
$LogDir = Join-Path $ShowspaceDir 'logs'
$PidFile = Join-Path $RunDir 'cloudflared-quick.pid'
$UrlFile = Join-Path $RunDir 'cloudflared-quick.url'

if (-not (Test-Path -LiteralPath $EnvFile)) { throw "Missing $EnvFile" }
Get-Content -LiteralPath $EnvFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith('#') -and $line.Contains('=')) {
        $name, $value = $line.Split('=', 2)
        [Environment]::SetEnvironmentVariable($name.Trim(), $value.Trim(), 'Process')
    }
}

$port = [Environment]::GetEnvironmentVariable('GRADIO_SERVER_PORT', 'Process')
if (-not $port) { $port = '7860' }
$headers = @{}
$username = [Environment]::GetEnvironmentVariable('SHOWSPACE_USERNAME', 'Process')
$password = [Environment]::GetEnvironmentVariable('SHOWSPACE_PASSWORD', 'Process')
if ($username -and $password) {
    $bytes = [Text.Encoding]::ASCII.GetBytes("$username`:$password")
    $headers.Authorization = 'Basic ' + [Convert]::ToBase64String($bytes)
}
try {
    $health = Invoke-WebRequest -UseBasicParsing -Headers $headers -Uri "http://127.0.0.1:$port/" -TimeoutSec 5
    if ($health.StatusCode -ne 200) { throw "HTTP $($health.StatusCode)" }
} catch {
    throw "Showspace is not healthy on 127.0.0.1:$port. Start it before the tunnel."
}

$cloudflared = [Environment]::GetEnvironmentVariable('CLOUDFLARED_EXE', 'Process')
if (-not $cloudflared) { $cloudflared = 'cloudflared.exe' }
if (-not (Test-Path -LiteralPath $cloudflared)) {
    $resolved = Get-Command $cloudflared -ErrorAction SilentlyContinue
    if (-not $resolved) { throw "cloudflared executable not found: $cloudflared" }
    $cloudflared = $resolved.Source
}

New-Item -ItemType Directory -Force -Path $RunDir, $LogDir | Out-Null
if (Test-Path -LiteralPath $PidFile) {
    $existingPid = [int](Get-Content -LiteralPath $PidFile | Select-Object -First 1)
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId=$existingPid" -ErrorAction SilentlyContinue
    if ($existing -and $existing.Name -eq 'cloudflared.exe' -and $existing.CommandLine -match 'tunnel.+--url') {
        throw "Quick tunnel is already running with PID $existingPid"
    }
    Remove-Item -LiteralPath $PidFile -Force
}

$stdout = Join-Path $LogDir 'cloudflared-quick.out.log'
$stderr = Join-Path $LogDir 'cloudflared-quick.err.log'
Remove-Item -LiteralPath $stdout, $stderr, $UrlFile -Force -ErrorAction SilentlyContinue
$process = Start-Process -FilePath $cloudflared -WindowStyle Hidden -PassThru `
    -ArgumentList @('tunnel', '--no-autoupdate', '--url', "http://127.0.0.1:$port") `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$process.Id | Set-Content -LiteralPath $PidFile

$publicUrl = $null
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    Start-Sleep -Seconds 1
    if ($process.HasExited) {
        $detail = Get-Content -LiteralPath $stderr -Raw -ErrorAction SilentlyContinue
        throw "cloudflared exited before publishing the URL: $detail"
    }
    $text = ((Get-Content -LiteralPath $stdout -Raw -ErrorAction SilentlyContinue) + "`n" +
             (Get-Content -LiteralPath $stderr -Raw -ErrorAction SilentlyContinue))
    $match = [regex]::Match($text, 'https://[a-z0-9-]+\.trycloudflare\.com')
    if ($match.Success) { $publicUrl = $match.Value; break }
}
if (-not $publicUrl) {
    throw "Quick tunnel started but no public URL appeared within 30 seconds. Inspect $stderr"
}
$publicUrl | Set-Content -LiteralPath $UrlFile
Write-Host "Quick public URL: $publicUrl"
Write-Host 'Engineering test only. The random hostname changes after restart.'
