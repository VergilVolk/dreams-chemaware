param(
    [string]$OutputDirectory = "data/external/GSE178341_mucinous_secretory_audit",
    [int]$Connections = 8
)

$ErrorActionPreference = "Stop"

$sourceUrl = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE178nnn/GSE178341/suppl/GSE178341_crc10x_full_c295v4_submit.h5"
$expectedBytes = [int64]1203550558
$expectedSha256 = "f435bb2651ff5297d0c24a99daf58850ed67ae1ed6c5ef05fad48fa3f0186670"
$fileName = "GSE178341_crc10x_full_c295v4_submit.h5"
$resolvedOutput = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $OutputDirectory))
$workspaceRoot = [System.IO.Path]::GetFullPath((Get-Location).Path)
if (-not $resolvedOutput.StartsWith($workspaceRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Output directory must remain inside the workspace: $resolvedOutput"
}

New-Item -ItemType Directory -Force -Path $resolvedOutput | Out-Null
$finalPath = Join-Path $resolvedOutput $fileName
$partsDir = Join-Path $resolvedOutput "$fileName.parts"
$logPath = Join-Path $resolvedOutput "gse178341_h5_download.log"

if (Test-Path -LiteralPath $finalPath) {
    $size = (Get-Item -LiteralPath $finalPath).Length
    if ($size -eq $expectedBytes) {
        $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $finalPath).Hash.ToLowerInvariant()
        if ($hash -eq $expectedSha256) {
            "[download] existing complete file: $finalPath bytes=$size sha256=$hash" | Tee-Object -FilePath $logPath -Append
            exit 0
        }
        throw "Existing full-length file has wrong SHA256: expected=$expectedSha256 actual=$hash"
    }
    throw "Refusing to overwrite incomplete final file: $finalPath bytes=$size"
}

New-Item -ItemType Directory -Force -Path $partsDir | Out-Null
$chunkSize = [int64][math]::Ceiling($expectedBytes / [double]$Connections)
$processes = @()
$parts = @()

for ($index = 0; $index -lt $Connections; $index++) {
    $start = [int64]$index * $chunkSize
    $end = [math]::Min($expectedBytes - 1, $start + $chunkSize - 1)
    if ($start -gt $end) { continue }
    $partPath = Join-Path $partsDir ("part_{0:D2}.bin" -f $index)
    $expectedPartBytes = $end - $start + 1
    $parts += [pscustomobject]@{
        Index = $index
        Path = $partPath
        Start = $start
        End = $end
        ExpectedBytes = $expectedPartBytes
    }
    if ((Test-Path -LiteralPath $partPath) -and ((Get-Item -LiteralPath $partPath).Length -eq $expectedPartBytes)) {
        "[download] part $index already complete" | Tee-Object -FilePath $logPath -Append
        continue
    }
    $stdout = Join-Path $partsDir ("part_{0:D2}.out" -f $index)
    $stderr = Join-Path $partsDir ("part_{0:D2}.err" -f $index)
    $arguments = @(
        "-L", "--fail", "--retry", "5", "--retry-delay", "2",
        "--connect-timeout", "30", "--range", "$start-$end",
        "-o", $partPath, $sourceUrl
    )
    $processes += Start-Process -FilePath "curl.exe" -ArgumentList $arguments -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    "[download] started part $index range=$start-$end pid=$($processes[-1].Id)" | Tee-Object -FilePath $logPath -Append
}

foreach ($process in $processes) {
    $process.WaitForExit()
    if ($process.ExitCode -ne 0) {
        throw "curl process failed: pid=$($process.Id) exit=$($process.ExitCode)"
    }
}

foreach ($part in $parts) {
    if (-not (Test-Path -LiteralPath $part.Path)) {
        throw "Missing part: $($part.Path)"
    }
    $actual = (Get-Item -LiteralPath $part.Path).Length
    if ($actual -ne $part.ExpectedBytes) {
        throw "Part length mismatch: $($part.Path) expected=$($part.ExpectedBytes) actual=$actual"
    }
}

$destination = [System.IO.File]::Open($finalPath, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
try {
    foreach ($part in ($parts | Sort-Object Index)) {
        $source = [System.IO.File]::OpenRead($part.Path)
        try { $source.CopyTo($destination) } finally { $source.Dispose() }
    }
} finally {
    $destination.Dispose()
}

$finalBytes = (Get-Item -LiteralPath $finalPath).Length
if ($finalBytes -ne $expectedBytes) {
    throw "Final length mismatch: expected=$expectedBytes actual=$finalBytes"
}
$finalHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $finalPath).Hash.ToLowerInvariant()
if ($finalHash -ne $expectedSha256) {
    throw "Final SHA256 mismatch: expected=$expectedSha256 actual=$finalHash"
}

"[download] complete: $finalPath bytes=$finalBytes sha256=$finalHash" | Tee-Object -FilePath $logPath -Append
