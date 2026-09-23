$ErrorActionPreference = 'Stop'
$ShowspaceDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvFile = Join-Path $ShowspaceDir '.env'
$port = 7860
if (Test-Path -LiteralPath $EnvFile) {
    Get-Content -LiteralPath $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith('#') -and $line.Contains('=')) {
            $name, $value = $line.Split('=', 2)
            if ($name.Trim() -eq 'GRADIO_SERVER_PORT') { $port = [int]$value.Trim() }
        }
    }
}
$tailscale = 'C:\Program Files\Tailscale\tailscale.exe'
if (-not (Test-Path -LiteralPath $tailscale)) { throw 'Tailscale CLI is not installed.' }

& (Join-Path $ShowspaceDir 'healthcheck_showspace.ps1')
& $tailscale funnel --bg --yes $port
if ($LASTEXITCODE -ne 0) { throw 'Tailscale Funnel could not be enabled.' }
& $tailscale funnel status
