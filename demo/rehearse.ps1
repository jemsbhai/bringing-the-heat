param([switch]$Cpu)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
foreach ($command in @('prepare', 'preflight')) {
    & $py demo.py $command
    if ($LASTEXITCODE -ne 0) { throw "$command failed" }
}
if ($Cpu) { & $py demo.py train --cpu } else { & $py demo.py train }
if ($LASTEXITCODE -ne 0) { throw 'Training failed' }
foreach ($command in @('export', 'evaluate', 'benchmark', 'gate')) {
    & $py demo.py $command
    if ($LASTEXITCODE -ne 0) { throw "$command failed (a failed gate is a valid block; inspect results)" }
}
& $py demo.py gate --inject-failure
if ($LASTEXITCODE -ne 2) { throw 'Expected the simulated bad candidate to exit 2' }
& $py -m pytest -q test_gate.py
if ($LASTEXITCODE -ne 0) { throw 'Policy tests failed' }
Write-Host 'Rehearsal complete. Launch .venv/Scripts/python.exe app.py'
