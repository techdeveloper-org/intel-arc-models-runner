# Intel Arc Models Runner - Automated Windows Installer
# Usage: powershell -ExecutionPolicy Bypass -File .\install.ps1

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "       Intel Arc Models Runner - Automated Windows Setup         " -ForegroundColor White
Write-Host "       Target Silicon: Intel Arc GPUs & Intel AI Boost NPUs      " -ForegroundColor DarkCyan
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Verify Python
Write-Host "[1/4] Checking Python runtime..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Python is not installed or not available in PATH." -ForegroundColor Red
    Write-Host "  Please install Python 3.10+ from https://www.python.org/" -ForegroundColor Yellow
    exit 1
}

# 2. Check Pip
Write-Host "[2/4] Checking pip package manager..." -ForegroundColor Yellow
try {
    $pipVersion = python -m pip --version 2>&1
    Write-Host "  Found: $pipVersion" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: pip is not available." -ForegroundColor Red
    exit 1
}

# 3. Install Package
Write-Host "[3/4] Installing Intel Arc Models Runner..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
python -m pip install -e . --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Package installation failed." -ForegroundColor Red
    exit 1
}
Write-Host "  Package installed successfully in editable mode." -ForegroundColor Green

# 4. Hardware Self-Test
Write-Host "[4/4] Executing hardware discovery self-test..." -ForegroundColor Yellow
Write-Host ""
python -m arc_runner.cli.main devices

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=================================================================" -ForegroundColor Green
    Write-Host "  Setup Complete! Intel Arc Models Runner is ready for local AI.  " -ForegroundColor White
    Write-Host "  Run 'python -m arc_runner.cli.main devices' anytime to test.    " -ForegroundColor Cyan
    Write-Host "=================================================================" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "  Warning: Hardware probe returned an exit code of $LASTEXITCODE." -ForegroundColor Yellow
}
