# Intel Arc Models Runner - Automated Windows Installer
# Usage: .\install.bat (works when PowerShell script execution is restricted)
#        .\install.ps1 (when the current PowerShell policy allows scripts)

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
$pythonScriptsDir = (python -c "import sysconfig; print(sysconfig.get_path('scripts'))").Trim()
if (-not [string]::IsNullOrWhiteSpace($pythonScriptsDir)) {
    $pythonScriptsDir = [System.IO.Path]::GetFullPath($pythonScriptsDir)
    $processHasScripts = @($env:Path -split ';' | Where-Object {
        [string]::Equals($_.TrimEnd('\'), $pythonScriptsDir.TrimEnd('\'), [StringComparison]::OrdinalIgnoreCase)
    })
    if ($processHasScripts.Count -eq 0) {
        $env:Path = "$env:Path;$pythonScriptsDir"
    }
}
python -m pip install --upgrade pip --quiet
python -m pip install -e . --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Package installation failed." -ForegroundColor Red
    exit 1
}
Write-Host "  Package installed successfully in editable mode." -ForegroundColor Green

# 3b. Make installed CLI commands available permanently on Windows.
# Resolve the actual Scripts directory used by this Python interpreter so this
# works for both virtual environments and global Python installations.
Write-Host "  Configuring command-line PATH..." -ForegroundColor Yellow
$scriptsDir = (python -c "import sysconfig; print(sysconfig.get_path('scripts'))").Trim()
if ([string]::IsNullOrWhiteSpace($scriptsDir)) {
    Write-Host "  WARNING: Could not determine Python Scripts directory." -ForegroundColor Yellow
} else {
    $scriptsDir = [System.IO.Path]::GetFullPath($scriptsDir)
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $pathEntries = @()
    if (-not [string]::IsNullOrWhiteSpace($userPath)) {
        $pathEntries = $userPath -split ';' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    }

    $alreadyConfigured = $pathEntries | Where-Object {
        [string]::Equals($_.TrimEnd('\'), $scriptsDir.TrimEnd('\'), [StringComparison]::OrdinalIgnoreCase)
    }
    if (-not $alreadyConfigured) {
        $newUserPath = (($pathEntries + $scriptsDir) -join ';')
        [Environment]::SetEnvironmentVariable("Path", $newUserPath, "User")
        Write-Host "  Added '$scriptsDir' to your permanent User PATH." -ForegroundColor Green
    } else {
        Write-Host "  Python Scripts directory is already in your User PATH." -ForegroundColor Green
    }

    # Also update this installer process so the command can be used immediately.
    $currentPathEntries = $env:Path -split ';'
    $currentPathHasScripts = $currentPathEntries | Where-Object {
        [string]::Equals($_.TrimEnd('\'), $scriptsDir.TrimEnd('\'), [StringComparison]::OrdinalIgnoreCase)
    }
    if (-not $currentPathHasScripts) {
        $env:Path = "$env:Path;$scriptsDir"
    }

    # Notify Windows apps (including Explorer) that the persistent User PATH
    # changed, so terminals opened afterward inherit the updated value.
    if (-not ([System.Management.Automation.PSTypeName]'ArcRunner.EnvironmentBroadcast').Type) {
        Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
namespace ArcRunner {
    public static class EnvironmentBroadcast {
        [DllImport("user32.dll", CharSet = CharSet.Auto, SetLastError = true)]
        public static extern IntPtr SendMessageTimeout(IntPtr hWnd, uint Msg, UIntPtr wParam,
            string lParam, uint fuFlags, uint uTimeout, out UIntPtr lpdwResult);
    }
}
"@
    }
    $broadcastResult = [UIntPtr]::Zero
    [void][ArcRunner.EnvironmentBroadcast]::SendMessageTimeout(
        [IntPtr]0xffff, 0x001A, [UIntPtr]::Zero, "Environment", 0x0002, 5000,
        [ref]$broadcastResult
    )
}

# 4. Hardware Self-Test
Write-Host "[4/4] Executing hardware discovery self-test..." -ForegroundColor Yellow
Write-Host ""
python -m arc_runner.cli.main devices

if ($LASTEXITCODE -eq 0) {
    Write-Host "  Verifying installed CLI command..." -ForegroundColor Yellow
    arc-runner --version
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: arc-runner CLI verification failed." -ForegroundColor Red
        exit 1
    }
    Write-Host ""
    Write-Host "=================================================================" -ForegroundColor Green
    Write-Host "  Setup Complete! Intel Arc Models Runner is ready for local AI.  " -ForegroundColor White
    Write-Host "  Run 'python -m arc_runner.cli.main devices' anytime to test.    " -ForegroundColor Cyan
    Write-Host "=================================================================" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "  Warning: Hardware probe returned an exit code of $LASTEXITCODE." -ForegroundColor Yellow
}
