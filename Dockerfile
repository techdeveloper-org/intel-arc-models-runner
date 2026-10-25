# ==============================================================================
# Intel Arc Models Runner - Production Docker Image
# Ubuntu 24.04 LTS (Noble Numbat) with Pre-Baked Intel Level Zero & OpenVINO Runtime
# Complies with Rule 29-container-deployment.md: Multi-Stage, Non-Root (UID 1000)
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Builder
# ------------------------------------------------------------------------------
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

COPY pyproject.toml README.md MANIFEST.in ./
COPY src/ ./src/

RUN pip install --upgrade pip setuptools wheel build && \
    pip install .

# ------------------------------------------------------------------------------
# Stage 2: Final Minimal Runtime
# ------------------------------------------------------------------------------
FROM ubuntu:24.04 AS runtime

LABEL org.opencontainers.image.title="Intel Arc Models Runner" \
      org.opencontainers.image.description="High-performance local LLM runner and MultiModal Studio optimized for Intel Arc GPUs and NPUs" \
      org.opencontainers.image.authors="techdeveloper-org <info@techdeveloper.org>" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.source="https://github.com/techdeveloper-org/intel-arc-models-runner"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/lib" \
    LIBVA_DRIVERS_PATH="/usr/lib/x86_64-linux-gnu/dri" \
    LIBVA_DRIVER_NAME="iHD" \
    ONEAPI_DEVICE_SELECTOR="level_zero:*" \
    HOME="/home/appuser"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    python3 \
    python3-minimal \
    ffmpeg \
    libva2 \
    libva-drm2 \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://repositories.intel.com/gpu/intel-graphics.key | gpg --dearmor -o /etc/apt/keyrings/intel-graphics.gpg \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble client" > /etc/apt/sources.list.d/intel-graphics.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    intel-opencl-icd \
    intel-level-zero-gpu \
    level-zero \
    level-zero-dev \
    intel-media-va-driver-non-free \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

RUN touch /var/mail/ubuntu || true \
    && (id -u ubuntu >/dev/null 2>&1 && userdel -r ubuntu || true) \
    && (getent group ubuntu >/dev/null 2>&1 && groupdel ubuntu || true) \
    && (getent group 1000 >/dev/null 2>&1 && groupdel $(getent group 1000 | cut -d: -f1) || true) \
    && groupadd --system --gid 1000 appgroup \
    && useradd --system --uid 1000 --gid appgroup --create-home --home-dir /home/appuser --shell /bin/bash appuser \
    && groupadd -f -g 44 video \
    && groupadd -f -g 107 render \
    && usermod -aG video,render appuser \
    && mkdir -p /home/appuser/.cache/arc-runner/models \
    && mkdir -p /home/appuser/.cache/arc-runner/aot \
    && mkdir -p /models \
    && chown -R appuser:appgroup /home/appuser /models

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

USER appuser
WORKDIR /home/appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["server", "start", "--host", "0.0.0.0", "--port", "8000"]
