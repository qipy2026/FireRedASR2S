# Requires: repo root as cwd, .venv, pretrained_models/FireRedASR2-AED
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
$outDir = Join-Path $root "output"
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }
$outPt = Join-Path $outDir "aed_dynamic_int8_cpu.pt"

& $py scripts\quantize_aed_int8.py --model_dir pretrained_models\FireRedASR2-AED --out_path $outPt
Write-Host "Done. Use --aed_dynamic_int8_pt $outPt with --asr_use_gpu 0 --asr_use_half 0"
