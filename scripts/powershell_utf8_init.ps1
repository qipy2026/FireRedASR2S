# Cursor/VS Code 集成终端或手动 dot-source：统一 UTF-8 代码页与控制台编码。
# 工作区终端配置见 .vscode/settings.json（默认使用本脚本）。
try {
    chcp 65001 *> $null
} catch {}
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
if (-not $env:PYTHONUTF8) { $env:PYTHONUTF8 = '1' }
if (-not $env:PYTHONIOENCODING) { $env:PYTHONIOENCODING = 'utf-8' }
