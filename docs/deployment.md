# 🚀 Deployment Guide: Standalone PyPI Wheel & Docker Containerization

[← Back to README](../README.md) · [Roadmap](roadmap.md) · [Architecture](architecture.md) · [Mathematical Proofs](research/06_packaging_and_containerization_mathematical_proofs.md)

---

## 1. Overview & Packaging Deliverables

`intel-arc-models-runner` provides two official production deployment targets designed for high-performance silicon execution across heterogeneous Intel Arc GPUs (discrete and integrated) and Intel AI Boost NPUs:

1. **Standalone PyPI Wheel (`pip install arc-runner`)**:
   - Universal Python wheel (`arc_runner-0.1.0-py3-none-any.whl`).
   - Dual CLI binary commands: `arc-runner` and `intel-arc-models-runner`.
   - Complete bundling of all 7 MultiModal Studio UI tabs, CSS stylesheets, client-side JS, and PEP 561 `py.typed` markers.
   - Cross-platform dynamic driver loading across Windows (`ze_loader.dll`) and Linux (`libze_loader.so.1`).
2. **Production Docker Container (`ubuntu:24.04`)**:
   - Multi-stage image adhering strictly to **Rule 29-container-deployment.md**.
   - Pre-baked with official Intel Graphics compute packages: `intel-level-zero-gpu`, `level-zero`, `intel-opencl-icd`, `intel-media-va-driver-non-free`.
   - Hardened non-root user execution (`appuser`, UID 1000, GID 1000).
   - Direct kernel DRI pass-through (`/dev/dri`) for zero-copy GPU inference.

---

## 2. Option 1: Standalone PyPI Wheel Distribution

### 2.1 Installation
Install directly from PyPI (or from pre-compiled local wheel artifact):

```bash
# Production install from PyPI
pip install arc-runner

# Or install from local distribution build
pip install dist/arc_runner-0.1.0-py3-none-any.whl
```

### 2.2 CLI Verification & Silicon Discovery
Verify the installation and discover all heterogeneous Intel accelerators:

```bash
# Check version
arc-runner --version

# Run automated silicon self-test
arc-runner devices discover

# Launch the OpenAI & Ollama compatible API server with Studio UI
arc-runner server start --host 0.0.0.0 --port 8000
```

Both `arc-runner` and `intel-arc-models-runner` entrypoints are fully supported and alias to the identical command pipeline.

---

## 3. Option 2: Docker Container Deployment (Ubuntu 24.04)

### 3.1 Single-Command Docker Compose (Recommended)
The repository includes a production-ready `docker-compose.yml` pre-configured with GPU pass-through, health monitoring, and persistent caching volumes:

```bash
# Launch the service in the background
docker compose up -d

# Follow real-time service logs
docker compose logs -f arc-runner

# Verify container health status
docker compose ps
```

The Arc MultiModal Studio Web UI is immediately accessible at:
👉 **`http://localhost:8000`**

### 3.2 Direct Docker CLI Execution
To run the container manually with full Intel Arc GPU acceleration:

```bash
docker run -d \
  --name intel-arc-models-runner \
  --restart unless-stopped \
  --device /dev/dri:/dev/dri \
  -p 8000:8000 \
  -v arc-models-cache:/home/appuser/.cache/arc-runner/models \
  -v arc-aot-cache:/home/appuser/.cache/arc-runner/aot \
  ghcr.io/techdeveloper-org/intel-arc-models-runner:latest
```

> [!IMPORTANT]
> The `--device /dev/dri:/dev/dri` flag grants the container direct access to Intel Level Zero render nodes (`/dev/dri/renderD128`).
> If deployed on a system without Intel silicon, the entrypoint automatically logs a notice and operates in CPU mode.

---

## 4. Kubernetes Production Deployment

For orchestrating `arc-runner` on Kubernetes clusters with Intel GPU Device Plugins:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: arc-runner-deployment
  labels:
    app: arc-runner
spec:
  replicas: 1
  selector:
    matchLabels:
      app: arc-runner
  template:
    metadata:
      labels:
        app: arc-runner
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
      containers:
        - name: arc-runner
          image: ghcr.io/techdeveloper-org/intel-arc-models-runner:latest
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8000
              name: http
          resources:
            limits:
              gpu.intel.com/i915: 1
              memory: "16Gi"
              cpu: "4"
            requests:
              gpu.intel.com/i915: 1
              memory: "8Gi"
              cpu: "2"
          livenessProbe:
            httpGet:
              path: /v1/health
              port: 8000
            initialDelaySeconds: 15
            periodSeconds: 30
          volumeMounts:
            - name: model-cache
              mountPath: /home/appuser/.cache/arc-runner/models
      volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: arc-runner-models-pvc
```

---

## 5. Security & Isolation Matrix

| Capability | Container Implementation | Standard Specification |
|:---|:---|:---|
| **Non-Root Execution** | `USER appuser` (UID 1000, GID 1000) | Rule `29-container-deployment.md` |
| **GPU Permissions** | Supplementary groups `video` (44), `render` (107) | Linux DRI spec |
| **Attack Surface** | Multi-stage builder pruning, no compiler toolchains in runtime | Proof M117 ($\ge 59.9\%$ footprint reduction) |
| **Memory Boundaries** | cgroup v2 ceiling with $P(\text{OOM}) = 0$ margin | Proof M118 |
| **Health Auditing** | `HEALTHCHECK` probing `/v1/health` | Docker & K8s standard |
