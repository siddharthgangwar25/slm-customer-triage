param(
    [ValidateSet('All', 'Prompted', 'Train', 'Select', 'Report')]
    [string]$Stage = 'All'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$python = Join-Path $projectRoot 'environments/training/.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Install the locked training environment first; see docs/finetuning.md.' }
$env:HF_HUB_OFFLINE = '1'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'

function Invoke-Triage {
    param([string[]]$Arguments)
    & $python -m triage.cli @Arguments
    if ($LASTEXITCODE -ne 0) { throw "triage failed (exit $LASTEXITCODE); inspect output before retrying." }
}

$prompted = 'artifacts/prompted-qwen3-06b-nf4-v1-val'
$training = 'artifacts/finetuned-qwen3-06b-qlora-v1'
$selection = 'artifacts/finetuned-qwen3-06b-qlora-v1-selection'
$baseline = 'reports/baseline-c1-v1-val/predictions.jsonl'

if ($Stage -in @('All', 'Prompted')) {
    if (-not (Test-Path "$prompted/prediction_manifest.json")) {
        $triageArguments = @('predict', '--config', 'configs/prompted-small.yaml', '--split', 'val')
        if (Test-Path "$prompted/run.json") { $triageArguments += '--resume' }
        Invoke-Triage $triageArguments
    }
    if (-not (Test-Path 'reports/prompted-qwen3-06b-nf4-v1-val')) {
        Invoke-Triage @('evaluate', '--predictions', "$prompted/predictions.jsonl", '--output', 'reports/prompted-qwen3-06b-nf4-v1-val')
    }
    if (-not (Test-Path 'reports/prompted-qwen3-06b-nf4-v1-policy')) {
        Invoke-Triage @('policy', 'select', '--predictions', "$prompted/predictions.jsonl", '--split', 'val', '--output', 'reports/prompted-qwen3-06b-nf4-v1-policy')
    }
}

if ($Stage -in @('All', 'Train')) {
    $complete = $false
    if (Test-Path "$training/result.json") {
        $complete = (Get-Content -Raw "$training/result.json" | ConvertFrom-Json).status -eq 'complete'
    }
    if (-not $complete) {
        $triageArguments = @('train', 'slm', '--config', 'configs/finetune.yaml', '--smoke-evidence', "$training-smoke")
        if (Test-Path "$training/run.json") {
            $checkpoint = Get-ChildItem -LiteralPath $training -Directory -Filter 'checkpoint-*' |
                Where-Object { Test-Path (Join-Path $_.FullName 'bundle.json') } |
                Sort-Object { [int]($_.Name -replace '^checkpoint-', '') } -Descending |
                Select-Object -First 1
            if (-not $checkpoint) { throw 'No sealed checkpoint to resume. Keep the partial run for inspection; do not overwrite it.' }
            $triageArguments += @('--resume-checkpoint', $checkpoint.FullName)
        }
        Invoke-Triage $triageArguments
        $result = Get-Content -Raw "$training/result.json" | ConvertFrom-Json
        if ($result.status -ne 'complete') { throw "Training status: $($result.status). Stop here and review the saved evidence." }
    }
}

if ($Stage -in @('All', 'Select')) {
    $triageArguments = @('checkpoint', 'select', '--config', 'configs/finetune.yaml', '--output', $selection)
    if (Test-Path "$selection/run.json") { $triageArguments += '--resume' }
    Invoke-Triage $triageArguments
}

if ($Stage -in @('All', 'Report')) {
    $selected = (Get-Content -Raw "$selection/selection.json" | ConvertFrom-Json).selected.predictions
    if (-not (Test-Path 'reports/finetuned-qwen3-06b-qlora-v1-val')) {
        Invoke-Triage @('evaluate', '--predictions', $selected, '--output', 'reports/finetuned-qwen3-06b-qlora-v1-val')
    }
    if (-not (Test-Path 'reports/finetuned-qwen3-06b-qlora-v1-policy')) {
        Invoke-Triage @('policy', 'select', '--predictions', $selected, '--split', 'val', '--output', 'reports/finetuned-qwen3-06b-qlora-v1-policy')
    }
    if (-not (Test-Path 'reports/three-way-qwen3-06b-v1')) {
        Invoke-Triage @('compare', '--baseline', $baseline, '--candidate', "$prompted/predictions.jsonl", '--finetuned', $selected, '--output', 'reports/three-way-qwen3-06b-v1')
    }
}

Write-Host "Milestone 4 stage '$Stage' finished. Ask Codex to inspect the artifacts and update implementation_status.md."
