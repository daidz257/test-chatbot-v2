$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $venvPython)) {
    $pythonVersion = $null
    foreach ($candidate in @('3.13', '3.12')) {
        & py "-$candidate" --version *> $null
        if ($LASTEXITCODE -eq 0) {
            $pythonVersion = $candidate
            break
        }
    }
    if (-not $pythonVersion) {
        throw 'Không tìm thấy Python 3.13 hoặc 3.12. Cài một trong hai bản, mở PowerShell mới rồi chạy lại. Kiểm tra bản đã cài bằng: py -0p'
    }

    & py "-$pythonVersion" -m venv (Join-Path $projectRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw "Không tạo được môi trường bằng Python $pythonVersion." }
}

& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Không cập nhật được pip.' }
& $venvPython -m pip install -r (Join-Path $projectRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Cài thư viện thất bại. Xem lỗi ở trên để biết thư viện nào chưa cài được.' }

Set-Location -LiteralPath $projectRoot
& $venvPython -m streamlit run app.py
