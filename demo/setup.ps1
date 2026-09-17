param([switch]$Cuda)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
uv venv --python 3.12.12 .venv
if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed' }
$lock = if ($Cuda) { 'requirements-cuda128.lock' } else { 'requirements-cpu.lock' }
uv pip sync --python .venv/Scripts/python.exe $lock
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
Write-Host 'Environment ready. Run .venv/Scripts/python.exe demo.py prepare'
