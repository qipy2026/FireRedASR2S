# Example: AED + Intel XPU + bfloat16 (see torch_device.resolve_compute_dtype)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root

$env:PYTORCH_ENABLE_XPU_FALLBACK = "1"
$py = Join-Path $root ".venv\Scripts\python.exe"
$wav = if ($args[0]) { $args[0] } else { throw "Usage: .\run_cli_aed_half_xpu.ps1 <path-to-16k-mono.wav>" }

& $py -m fireredasr2s.fireredasr2s_cli `
  --asr_type aed --asr_device xpu --asr_use_gpu 1 --asr_use_half 1 `
  --wav_path $wav --outdir (Join-Path $root "output\cli_aed_half_xpu")
