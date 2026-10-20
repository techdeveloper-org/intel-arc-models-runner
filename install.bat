@echo off
setlocal
title Intel Arc Models Runner - Installer

echo Launching PowerShell Installer with ExecutionPolicy Bypass...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"

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
