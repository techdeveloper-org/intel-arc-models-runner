@echo off
setlocal
title Intel Arc Models Runner - Installer

echo Checking PowerShell script policy and starting installer...
powershell.exe -NoProfile -Command "$policy = Get-ExecutionPolicy; if ($policy -in @('Restricted','AllSigned','RemoteSigned')) { $env:ARC_RUNNER_INSTALLER_BYPASS='1'; & (Get-Process -Id $PID).Path -NoProfile -ExecutionPolicy Bypass -File '%~dp0install.ps1' } else { & '%~dp0install.ps1' }; exit $LASTEXITCODE"

if %ERRORLEVEL% neq 0 (
    echo.
    echo Installation failed with error code %ERRORLEVEL%.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Press any key to exit.
pause >nul
endlocal
