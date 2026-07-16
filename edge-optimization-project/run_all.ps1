param(
    [string]$PythonExe = "C:\Users\emanu\anaconda3\python.exe"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Using Python: $PythonExe"
Write-Host "Project root: $root"

& $PythonExe -m pip install -r (Join-Path $root "requirements.txt")

& $PythonExe (Join-Path $root "scripts\make_synthetic_sensor_dataset.py") --out (Join-Path $root "data\sensor_dataset.csv")
& $PythonExe (Join-Path $root "scripts\train_baseline.py") --data (Join-Path $root "data\sensor_dataset.csv") --artifacts (Join-Path $root "artifacts")
& $PythonExe (Join-Path $root "scripts\optimize_tflite.py") --artifacts (Join-Path $root "artifacts")
& $PythonExe (Join-Path $root "scripts\benchmark_models.py") --artifacts (Join-Path $root "artifacts") --reports (Join-Path $root "reports")

Write-Host ""
Write-Host "Done."
Write-Host "Check:"
Write-Host " - $root\reports\benchmark_summary.md"
Write-Host " - $root\reports\benchmark_results.csv"
