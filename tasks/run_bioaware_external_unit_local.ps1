param(
    [Parameter(Mandatory=$true)][string]$Unit,
    [Parameter(Mandatory=$true)][string]$Prefix,
    [int]$ExpectedFiles = 16
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))

$manifestRoot = 'data/validation/bioaware_metdna3_external_manifest_v1'
$manifest = Join-Path $manifestRoot $Unit
$mzml = Join-Path 'data/external/metdna3_2025/mzml/external' $Unit
$root = Join-Path 'data/validation/bioaware_metdna3_external_v3_v1' $Unit
$artV3 = 'data/validation/bioaware_v3_consensus_router_frozen_v2_20260830/artifact.json'
$artV4 = 'data/validation/bioaware_v4_high_precision_router_frozen_v1_20260830/artifact.json'
$artV6 = 'data/validation/bioaware_v6_identifiable_router_frozen_v2_20260830/artifact.json'
$internal = 'data/validation/bioaware_metdna3_internal_rplc_frozen_v3_result_v1/report.json'

function Invoke-Python([string[]]$Arguments) {
    & python -u @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "python stage failed with exit code $LASTEXITCODE`: $($Arguments -join ' ')"
    }
}

function Has-Report([string]$Directory) {
    return Test-Path (Join-Path $Directory 'report.json')
}

if (-not (Test-Path (Join-Path $root 'preflight.json'))) {
    Invoke-Python @('tasks/preflight_bioaware_metdna3_development_ms2.py',
        '--development-dir', $manifest, '--truth-name', 'external_level1.csv.gz',
        '--manifest-report', (Join-Path $manifestRoot 'report.json'), '--mzml-dir', $mzml,
        '--output', (Join-Path $root 'preflight.json'), '--scope', 'external',
        '--expected-files', [string]$ExpectedFiles, '--minimum-matched-rows', '50',
        '--minimum-matched-identities', '50', '--minimum-exclusive-identities', '50')
}
if (-not (Has-Report (Join-Path $root 'cache'))) {
    Invoke-Python @('tasks/build_bioaware_metdna3_dreams_cache.py',
        '--development-dir', $manifest, '--truth-name', 'external_level1.csv.gz',
        '--mzml-dir', $mzml, '--preflight', (Join-Path $root 'preflight.json'),
        '--output-dir', (Join-Path $root 'cache'), '--scope', 'external',
        '--query-prefix', $Prefix, '--minimum-identities', '50', '--minimum-queries', '50')
}
if (-not (Has-Report (Join-Path $root 'scores'))) {
    Invoke-Python @('tasks/encode_bioaware_metdna3_dreams.py', '--cache-dir', (Join-Path $root 'cache'),
        '--output-dir', (Join-Path $root 'scores'), '--device', 'cpu', '--batch-size', '64')
}
if (-not (Has-Report (Join-Path $root 'baseline'))) {
    Invoke-Python @('tasks/build_bioaware_metdna3_baseline_transitions.py',
        '--scores', (Join-Path $root 'scores/candidate_scores.csv.gz'),
        '--splits', (Join-Path $manifest 'identity_splits.csv.gz'),
        '--output-dir', (Join-Path $root 'baseline'))
}
if (-not (Has-Report (Join-Path $root 'ms1'))) {
    Invoke-Python @('tasks/pilot_bioaware_metdna3_ms1_features.py', '--mzml-dir', $mzml,
        '--output-dir', (Join-Path $root 'ms1'), '--workers', '8',
        '--expected-files', [string]$ExpectedFiles, '--frozen-noise-threshold', '100000')
}
if (-not (Has-Report (Join-Path $root 'recursive'))) {
    Invoke-Python @('tasks/audit_bioaware_metdna3_recursive_headroom.py',
        '--pilot-dir', (Join-Path $root 'ms1'), '--development-dir', $manifest,
        '--truth-name', 'external_level1.csv.gz', '--query-cache', (Join-Path $root 'cache/queries.csv.gz'),
        '--baseline-transitions', (Join-Path $root 'baseline/raw_transitions.csv.gz'),
        '--output-dir', (Join-Path $root 'recursive'), '--scope', 'external')
}
if (-not (Has-Report (Join-Path $root 'feature_ms2'))) {
    Invoke-Python @('tasks/build_bioaware_metdna3_feature_ms2_cache.py',
        '--nodes', (Join-Path $root 'recursive/stable_ms1_feature_nodes.csv.gz'),
        '--mzml-dir', $mzml, '--output-dir', (Join-Path $root 'feature_ms2'),
        '--scope', 'external', '--expected-files', [string]$ExpectedFiles)
}
if (-not (Has-Report (Join-Path $root 'paths'))) {
    Invoke-Python @('tasks/build_bioaware_metdna3_candidate_path_table.py',
        '--recursive-dir', (Join-Path $root 'recursive'), '--development-dir', $manifest,
        '--truth-name', 'external_level1.csv.gz', '--query-cache', (Join-Path $root 'cache/queries.csv.gz'),
        '--candidate-scores', (Join-Path $root 'scores/candidate_scores.csv.gz'),
        '--baseline-transitions', (Join-Path $root 'baseline/raw_transitions.csv.gz'),
        '--output-dir', (Join-Path $root 'paths'), '--scope', 'external')
}
foreach ($step in 0,1) {
    $edge = Join-Path $root "edge_step$step"
    if (-not (Has-Report $edge)) {
        Invoke-Python @('tasks/audit_bioaware_metdna3_candidate_edge_ms2.py',
            '--recursive-dir', (Join-Path $root 'recursive'), '--development-dir', $manifest,
            '--truth-name', 'external_level1.csv.gz', '--query-cache', (Join-Path $root 'cache'),
            '--candidate-scores', (Join-Path $root 'scores/candidate_scores.csv.gz'),
            '--baseline-transitions', (Join-Path $root 'baseline/raw_transitions.csv.gz'),
            '--feature-ms2-dir', (Join-Path $root 'feature_ms2'), '--output-dir', $edge,
            '--maximum-network-step', [string]$step, '--scope', 'external')
    }
}
if (-not (Has-Report (Join-Path $root 'rules'))) {
    Invoke-Python @('tasks/audit_bioaware_candidate_rule_likelihood.py',
        '--scores', (Join-Path $root 'scores/candidate_scores.csv.gz'),
        '--queries', (Join-Path $root 'cache/queries.csv.gz'),
        '--query-tensors', (Join-Path $root 'cache/query_tensors.npz'),
        '--references', (Join-Path $root 'cache/candidate_references.csv.gz'),
        '--all-queries-unresolved', '--output-dir', (Join-Path $root 'rules'))
}
if (-not (Has-Report (Join-Path $root 'depth3'))) {
    Invoke-Python @('tasks/summarize_bioaware_metdna3_candidate_edge_decision.py',
        '--paths', (Join-Path $root 'paths/candidate_paths.csv.gz'),
        '--edge-evidence', (Join-Path $root 'edge_step0/candidate_edge_evidence.csv.gz'),
        '--baseline', (Join-Path $root 'baseline/raw_transitions.csv.gz'),
        '--output-dir', (Join-Path $root 'depth3'), '--scope', 'external')
}
if (-not (Has-Report (Join-Path $root 'smn'))) {
    Invoke-Python @('tasks/audit_bioaware_metdna3_smn_headroom.py',
        '--recursive-dir', (Join-Path $root 'recursive'), '--development-dir', $manifest,
        '--truth-name', 'external_level1.csv.gz', '--query-cache', (Join-Path $root 'cache/queries.csv.gz'),
        '--candidate-scores', (Join-Path $root 'scores/candidate_scores.csv.gz'),
        '--baseline-transitions', (Join-Path $root 'baseline/raw_transitions.csv.gz'),
        '--mrn-decision-dir', (Join-Path $root 'depth3'), '--output-dir', (Join-Path $root 'smn'),
        '--scope', 'external')
}
if (-not (Has-Report (Join-Path $root 'rt'))) {
    Invoke-Python @('tasks/audit_bioaware_metdna3_rt_headroom.py',
        '--development-dir', $manifest, '--truth-name', 'external_level1.csv.gz',
        '--query-cache', (Join-Path $root 'cache/queries.csv.gz'),
        '--candidate-scores', (Join-Path $root 'scores/candidate_scores.csv.gz'),
        '--baseline-transitions', (Join-Path $root 'baseline/raw_transitions.csv.gz'),
        '--mrn-decision-dir', (Join-Path $root 'depth3'), '--smn-dir', (Join-Path $root 'smn'),
        '--output-dir', (Join-Path $root 'rt'), '--scope', 'external')
}
if (-not (Has-Report (Join-Path $root 'decoder'))) {
    Invoke-Python @('tasks/build_bioaware_zero_weight_decoder_evidence.py',
        '--scores', (Join-Path $root 'scores/candidate_scores.csv.gz'), '--artifact', $artV3,
        '--output-dir', (Join-Path $root 'decoder'))
}
if (-not (Has-Report (Join-Path $root 'ledger'))) {
    Invoke-Python @('tasks/build_bioaware_candidate_evidence_ledger.py',
        '--dreams', (Join-Path $root 'scores/candidate_scores.csv.gz'),
        '--known-edge', (Join-Path $root 'edge_step0/candidate_edge_evidence.csv.gz'),
        '--predicted-edge', (Join-Path $root 'edge_step1/candidate_edge_evidence.csv.gz'),
        '--smn', (Join-Path $root 'smn/candidate_structural_evidence.csv.gz'),
        '--rt', (Join-Path $root 'rt/candidate_rt_evidence.csv.gz'),
        '--decoder', (Join-Path $root 'decoder/candidate_scores.csv.gz'),
        '--rules', (Join-Path $root 'rules/candidate_rule_scores.csv.gz'),
        '--output-dir', (Join-Path $root 'ledger'))
}
if (-not (Has-Report (Join-Path $root 'result'))) {
    Invoke-Python @('tasks/evaluate_bioaware_v3_frozen_router.py', '--artifact', $artV3,
        '--ledger', (Join-Path $root 'ledger/candidate_evidence.csv.gz'),
        '--depth3', (Join-Path $root 'depth3/query_transitions.csv.gz'),
        '--queries', (Join-Path $root 'cache/queries.csv.gz'), '--output-dir', (Join-Path $root 'result'),
        '--scope', 'external_panel', '--internal-report', $internal)
}
if (-not (Has-Report (Join-Path $root 'audit'))) {
    Invoke-Python @('tasks/audit_bioaware_v3_result_mechanisms.py',
        '--transitions', (Join-Path $root 'result/query_transitions.csv.gz'),
        '--ledger', (Join-Path $root 'ledger/candidate_evidence.csv.gz'), '--artifact', $artV3,
        '--output-dir', (Join-Path $root 'audit'), '--scope', "external:$Unit")
}
if ($Unit -ne 'BV2cell__hilic' -and -not (Has-Report (Join-Path $root 'result_v4'))) {
    Invoke-Python @('tasks/evaluate_bioaware_v4_high_precision_router.py', '--artifact', $artV4,
        '--ledger', (Join-Path $root 'ledger/candidate_evidence.csv.gz'),
        '--queries', (Join-Path $root 'cache/queries.csv.gz'), '--panel', $Unit,
        '--output-dir', (Join-Path $root 'result_v4'))
}
$opened = @('BV2cell__hilic','BV2cell__rplc','Mouse_brain__hilic')
if ($opened -notcontains $Unit -and -not (Has-Report (Join-Path $root 'result_v6'))) {
    Invoke-Python @('tasks/evaluate_bioaware_v6_identifiable_router.py', '--artifact', $artV6,
        '--ledger', (Join-Path $root 'ledger/candidate_evidence.csv.gz'),
        '--queries', (Join-Path $root 'cache/queries.csv.gz'), '--panel', $Unit,
        '--output-dir', (Join-Path $root 'result_v6'))
}
Write-Host "[BioAware local unit complete] $Unit"
