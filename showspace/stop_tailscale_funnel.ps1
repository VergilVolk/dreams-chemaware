$ErrorActionPreference = 'Stop'
$tailscale = 'C:\Program Files\Tailscale\tailscale.exe'
if (-not (Test-Path -LiteralPath $tailscale)) { throw 'Tailscale CLI is not installed.' }
& $tailscale funnel reset
if ($LASTEXITCODE -ne 0) { throw 'Failed to reset Tailscale Funnel.' }
Write-Host 'Tailscale Funnel stopped and its serve configuration was reset.'
