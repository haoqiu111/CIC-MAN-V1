param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,
    [string]$PythonExe = 'python'
)

$ErrorActionPreference = 'Stop'
$pilot = Join-Path $PSScriptRoot 'run_v2_pilot.py'
$shortcut = Join-Path $PSScriptRoot 'run_shortcut_v2.py'
$outputRoot = Join-Path $ProjectRoot 'outputs'

function Invoke-Checked([string[]]$Arguments) {
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment command failed with exit code $LASTEXITCODE."
    }
}

# R6 matched v6ic ablation (three seeds).
foreach ($seed in 42, 2025, 2026) {
    Invoke-Checked @(
        '-u', $pilot,
        '--project-root', $ProjectRoot,
        '--output-root', $outputRoot,
        '--epochs', '40',
        '--seed', "$seed",
        '--models', 'v6ic_no_prior,v6ic_no_consensus,v6ic_no_disentangle,v6ic_no_view_dropout,v6ic_router_no_evidence',
        '--isolate'
    )
}

# Equal-backbone CCN baseline (five seeds).
foreach ($seed in 42, 2025, 2026, 7, 123) {
    Invoke-Checked @(
        '-u', $pilot,
        '--project-root', $ProjectRoot,
        '--output-root', $outputRoot,
        '--epochs', '40',
        '--seed', "$seed",
        '--models', 'dg_ccn',
        '--isolate'
    )
}

# Five-seed shortcut-reversal audit.
foreach ($seed in 42, 2025, 2026, 7, 123) {
    Invoke-Checked @(
        '-u', $shortcut,
        '--project-root', $ProjectRoot,
        '--epochs', '40',
        '--seed', "$seed",
        '--models', 'single_env_order,cicman_v4,cicman_v6ic'
    )
}

Write-Host 'R6 Paper 1 revision queue completed.'
