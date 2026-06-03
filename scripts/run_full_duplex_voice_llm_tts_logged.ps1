# 以 UTF-8 记录 full_duplex_voice_llm_tts.py（调用 scripts/logged_duplex_voice_llm_tts.py）。
# 用法:
#   .\scripts\run_full_duplex_voice_llm_tts_logged.ps1 -Device xpu -SessionSeconds 120
param(
    [string] $Device = "xpu",
    [double] $SessionSeconds = 120,
    [string] $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$ErrorActionPreference = "Stop"
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $RepoRoot "output\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logPath = Join-Path $logDir "full_duplex_voice_llm_tts_${Device}_${SessionSeconds}s_$ts.log"

$py = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$runner = Join-Path $RepoRoot "scripts\logged_duplex_voice_llm_tts.py"
if (-not (Test-Path $py)) { throw "missing venv python: $py" }
if (-not (Test-Path $runner)) { throw "missing $runner" }

$env:PYTORCH_ENABLE_XPU_FALLBACK = "1"

& $py $runner --repo $RepoRoot --log $logPath --device $Device --session-seconds $SessionSeconds
$exit = $LASTEXITCODE
Write-Host ""
Write-Host "LOG: $logPath"
exit $exit
