#!/bin/sh
set -e

echo "================================================================================"
echo " Intel Arc Models Runner (Containerized Ubuntu 24.04 Environment)"
echo "================================================================================"

if [ -d "/dev/dri" ]; then
    echo "[INFO] Direct Rendering Infrastructure (/dev/dri) detected."
    if [ -e "/dev/dri/renderD128" ]; then
        if [ -r "/dev/dri/renderD128" ] && [ -w "/dev/dri/renderD128" ]; then
            echo "[INFO] Intel GPU render node (/dev/dri/renderD128) is read/write accessible."
        else
            echo "[WARNING] /dev/dri/renderD128 found but not writable by UID $(id -u). Verify host video/render GIDs."
        fi
    fi
else
    echo "[NOTICE] /dev/dri not mapped into container. Heterogeneous GPU offload unavailable; using CPU."
fi

if [ -f "/opt/venv/bin/arc-runner" ]; then
    echo "[INFO] Python virtual environment verified: $(python3 --version)"
fi

if [ "$1" = "server" ] || [ "$1" = "devices" ] || [ "$1" = "run" ] || [ "$1" = "hub" ] || [ "$1" = "benchmark" ]; then
    exec arc-runner "$@"
fi

exec "$@"
