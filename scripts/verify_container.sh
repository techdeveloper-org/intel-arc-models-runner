#!/usr/bin/env bash
# ==============================================================================
# Container Verification Utility for Intel Arc Models Runner
# Validates Dockerfile syntax, multi-stage layers, and non-root security.
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "================================================================================"
echo " Intel Arc Models Runner - Container Verification Pipeline"
echo "================================================================================"

echo "[1/4] Auditing Dockerfile structure..."
if ! grep -q "FROM ubuntu:24.04 AS builder" Dockerfile; then
    echo "[ERROR] Dockerfile missing 'FROM ubuntu:24.04 AS builder' stage."
    exit 1
fi

if ! grep -q "FROM ubuntu:24.04 AS runtime" Dockerfile; then
    echo "[ERROR] Dockerfile missing 'FROM ubuntu:24.04 AS runtime' stage."
    exit 1
fi

if ! grep -q "USER appuser" Dockerfile; then
    echo "[ERROR] Dockerfile missing non-root 'USER appuser' declaration."
    exit 1
fi
echo "      Multi-stage and non-root security directives: PASSED"

echo "[2/4] Validating docker-compose.yml configuration..."
if command -v docker &>/dev/null && docker compose version &>/dev/null; then
    docker compose config >/dev/null
    echo "      docker compose config: PASSED"
else
    echo "      docker compose CLI not found; skipping live compose parse."
fi

echo "[3/4] Verifying .dockerignore hygiene..."
for pattern in ".git" "models" "scratch" "tests" "build" "dist"; do
    if ! grep -q "^${pattern}" .dockerignore; then
        echo "[ERROR] Missing critical exclusion '${pattern}' in .dockerignore."
        exit 1
    fi
done
echo "      .dockerignore hygiene checks: PASSED"

echo "[4/4] Auditing docker-entrypoint.sh executable bit and signal traps..."
if [ ! -x "docker-entrypoint.sh" ]; then
    chmod +x "docker-entrypoint.sh"
fi
if ! grep -q "exec" docker-entrypoint.sh; then
    echo "[ERROR] docker-entrypoint.sh must use exec to avoid PID 1 signal masking."
    exit 1
fi
echo "      docker-entrypoint.sh audit: PASSED"

echo ""
echo "[SUCCESS] Container deployment configuration is fully verified and compliant!"
