# Switch the active Python environment to Intel XPU PyTorch (hello-agents style by default).
#
# Default (recommended, matches hello-agents .venv-intel-xpu / PyTorch XPU docs):
#   torch 2.11.x + torchvision + torchaudio from https://download.pytorch.org/whl/xpu
#   (true +xpu wheels + Intel runtime deps). Do NOT use PyPI/Intel find-links alone — pip
#   cache often resolves the wrong CPU wheel for "torch==2.11.0".
#
# Legacy (IPEX 2.8 + PyTorch 2.8 CPU from PyPI + IPEX from Intel index):
#   .\scripts\install_intel_xpu_pytorch.ps1 -LegacyIpex28
#
#   .\scripts\install_intel_xpu_pytorch.ps1
#   pip install -r requirements-asr-no-torch.txt
#
# https://intel.github.io/intel-extension-for-pytorch/index.html#installation
#
# Prerequisites: Intel GPU driver; Windows: VC++ runtime / oneAPI per Intel docs if needed.

param(
    [switch]$LegacyIpex28
)

$ErrorActionPreference = "Continue"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$DefaultPy = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if ($args.Count -ge 1 -and $args[0] -and ($args[0] -as [string]) -notmatch '^-') {
    $Py = $args[0]
} elseif (Test-Path $DefaultPy) {
    $Py = $DefaultPy
    Write-Host "Using project venv: $Py"
} else {
    $Py = "python"
    Write-Host "Using PATH python: $Py"
}

$PytorchXpuIndex = "https://download.pytorch.org/whl/xpu"
$IntelExtraIndex = "https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"

Write-Host ""
Write-Host "=== Step 1: Remove existing PyTorch / IPEX wheels ===" -ForegroundColor Cyan
& $Py -m pip uninstall -y torch torchvision torchaudio intel-extension-for-pytorch

Write-Host ""
if ($LegacyIpex28) {
    Write-Host "=== Step 2 (Legacy): PyTorch 2.8 + IPEX 2.8 from Intel extra-index ===" -ForegroundColor Cyan
    $TorchVer = "2.8.0"
    $TorchVisionVer = "0.23.0"
    $TorchAudioVer = "2.8.0"
    $IpexVer = "2.8.10+xpu"
    Write-Host "extra-index-url: $IntelExtraIndex"
    Write-Host "pins: torch==$TorchVer torchvision==$TorchVisionVer torchaudio==$TorchAudioVer intel-extension-for-pytorch==$IpexVer"
    & $Py -m pip install --upgrade pip
    & $Py -m pip install `
        "torch==$TorchVer" `
        "torchvision==$TorchVisionVer" `
        "torchaudio==$TorchAudioVer" `
        "intel-extension-for-pytorch==$IpexVer" `
        --extra-index-url $IntelExtraIndex
    if ($LASTEXITCODE -ne 0) {
        throw "pip install legacy Intel XPU stack failed (exit $LASTEXITCODE)."
    }
} else {
    Write-Host "=== Step 2 (Default): PyTorch XPU from pytorch.org (hello-agents style +xpu) ===" -ForegroundColor Cyan
    Write-Host "index-url: $PytorchXpuIndex"
    Write-Host "pins: torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0"
    & $Py -m pip install --upgrade pip
    # --no-cache-dir avoids reusing a stale PyPI CPU wheel from pip cache when version is also 2.11.0.
    & $Py -m pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url $PytorchXpuIndex
    if ($LASTEXITCODE -ne 0) {
        throw "pip install PyTorch XPU stack failed (exit $LASTEXITCODE). Try -LegacyIpex28 or https://pytorch.org/get-started/locally/"
    }
}

Write-Host ""
Write-Host "=== Step 3: Verify XPU ===" -ForegroundColor Cyan
& $Py -c @"
import torch
v = torch.__version__
print('torch:', v)
x = getattr(torch, 'xpu', None)
if x is None:
    print('torch.xpu: missing')
else:
    print('torch.xpu.is_available():', x.is_available())
    if x.is_available() and x.device_count() > 0:
        print('torch.xpu.get_device_name(0):', x.get_device_name(0))
print('torch.cuda.is_available():', torch.cuda.is_available())
if '+xpu' not in v.lower() and x is not None and x.is_available():
    print('Note: XPU is available; version string may omit +xpu on some builds.')
elif '+cpu' in v.lower() and (x is None or not x.is_available()):
    print('WARNING: CPU-only torch. Re-run this script (default uses https://download.pytorch.org/whl/xpu) or -LegacyIpex28.')
"@

Write-Host ""
Write-Host "=== Next: ASR dependencies without torch ===" -ForegroundColor Green
Write-Host "  pip install -r requirements-asr-no-torch.txt"
Write-Host "Then run: python -m unittest tests.test_xpu_device -v"
Write-Host ""
