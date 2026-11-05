$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $venvPython)) {
    throw 'Chưa có .venv. Hãy chạy run.ps1 một lần trước.'
}

$phoenixData = Join-Path $projectRoot 'data\phoenix'
New-Item -ItemType Directory -Force -Path $phoenixData | Out-Null
$env:PHOENIX_WORKING_DIR = $phoenixData
Set-Location -LiteralPath $projectRoot
& $venvPython -m phoenix.server.main serve
