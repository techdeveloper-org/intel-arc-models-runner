# Deployment Guide: PyPI Wheel and Docker

[Back to README](../README.md) | [Roadmap](roadmap.md) | [Architecture](architecture.md)

Every command, route and environment variable on this page was checked against `src/arc_runner/cli/main.py`, `src/arc_runner/server/` and `src/arc_runner/core/path_policy.py`. Nothing here has been run on a published artifact: no release has been tagged and neither the PyPI package nor a container image exists yet (see section 7).

---

## 1. Overview

Two deployment targets are prepared:

1. **Python wheel** (`arc_runner-<version>-py3-none-any.whl`): console commands `arc-runner` and `intel-arc-models-runner` (same entry point). The Level Zero loader is found at run time (`ze_loader.dll` on Windows, `libze_loader.so.1` on Linux).
2. **Docker image** (`ubuntu:24.04` base, multi-stage, non-root `appuser` with UID 1000): installs the Intel compute runtime packages from the Intel apt repository and expects `/dev/dri` to be passed through for GPU access.

Both are built from this repository. Neither is published; build them locally (sections 2.1 and 3.1).

---

## 2. Python wheel

### 2.1 Install

```bash
# From a checkout (works today)
pip install .

# Build the wheel and sdist locally
python scripts/build_wheel.py        # writes dist/
pip install dist/arc_runner-<version>-py3-none-any.whl

# From PyPI: only after the first release is published (not the case yet)
pip install arc-runner
```

### 2.2 Verify and start

```bash
arc-runner --version
arc-runner devices            # lists CPU, GPU and NPU found by the detector; add --json for JSON
arc-runner serve --host 127.0.0.1 --port 11434
```

`arc-runner serve` defaults to `127.0.0.1:11434`. Useful options: `--model-dir <dir>` (repeatable), `--default-model <name>`, `--debug` (enables `/docs`), `--log-file <path>`. Other subcommands: `pull`, `models`, `rm`, `benchmark`, `inspect-gguf`, `import-ollama`, `aot-compile`, `embed`, `convert`, `adapter`, `batch`; run `arc-runner --help` for the full list.

### 2.3 HTTP routes (as implemented)

| Route | Purpose |
|:---|:---|
| `GET /` | Liveness probe: plain text `Ollama is running` (the HTML UI when the client asks for `text/html`) |
| `GET /ui` | Web UI |
| `GET /api/version`, `/api/tags`, `POST /api/show`, `/api/generate`, `/api/chat`, `/api/pull`, `DELETE /api/delete`, `POST /api/copy` | Ollama-compatible API |
| `GET /v1/models`, `GET /v1/models/{model}`, `POST /v1/chat/completions` | OpenAI-compatible API |
| `POST /v1/studio/...` | Multimodal studio endpoints |
| `GET /api/devices`, `/api/batch/status`, `POST /api/batch/preempt`, `/api/adapters/*` | Devices, batch and LoRA adapters |

There is no `/v1/health` route; use `GET /` for health checks.

---

## 3. Docker

### 3.1 Docker Compose

`docker-compose.yml` builds the image from the `Dockerfile` (tag `intel-arc-models-runner:local`), passes `/dev/dri` through, mounts two cache volumes and `./models` read-only at `/models`, and health-checks `GET /`.

```bash
docker compose up -d --build
docker compose logs -f arc-runner
docker compose ps
```

The server listens on `http://localhost:8000` (the image command is `arc-runner serve --host 0.0.0.0 --port 8000`). Because it binds to all interfaces inside the container, publish the port only to networks you trust, or set `ARC_API_KEY` (section 5).

### 3.2 Direct `docker run`

After a release image exists, use a version tag, not `latest`:

```bash
docker run -d \
  --name intel-arc-models-runner \
  --restart unless-stopped \
  --device /dev/dri:/dev/dri \
  -p 8000:8000 \
  -e ARC_API_KEY="$ARC_API_KEY" \
  -v arc-models-cache:/home/appuser/.cache/arc-runner/models \
  -v arc-aot-cache:/home/appuser/.cache/arc-runner/aot \
  ghcr.io/techdeveloper-org/intel-arc-models-runner:<version>
```

`--device /dev/dri:/dev/dri` gives the container the render nodes. Without `/dev/dri` the entrypoint prints a notice and the runner uses the CPU. Nothing about GPU performance in the container has been measured here.

### 3.3 Image tags

The `docker-build` workflow pushes images only when a GitHub release for a tag `vX.Y.Z` is published (the tag must equal the `VERSION` file). The tags are the version number and a short commit SHA; `latest` is added only for a release tag without a pre-release suffix. Pushes to `main` and manual runs build the image without pushing it.

The base image is pinned by tag (`ubuntu:24.04`). Pinning its digest is a maintainer step before a release (see the comment at the top of the `Dockerfile`).

### 3.4 Native acceleration library (optional)

The package works without the native library (pure-Python path). To use it, build it on the target machine with GCC (MinGW-w64 on Windows):

```bash
python src/arc_runner/native/build_native.py
```

The build writes `libarc_native_sys.<dll|so>` and a `.build.json` sidecar (sha256, compiler, flags, ABI version). The Python bridge refuses a library with another ABI version or a different hash and falls back to the Python path. The Linux container image and the published wheel contain no Linux `.so`, so they run the Python path unless one is built into the image. Related variables: `ARC_NATIVE_DISABLE=1` forces the fallback, `ARC_ZE_LOADER_PATH` overrides the Level Zero loader, `ARC_ZE_SYNC_TIMEOUT_MS` sets the GPU wait limit (clamped to 1900), `ARC_PIN_RING=1` enables best-effort layer pinning, `ARC_KV_CACHE_PRECISION=u8` opts in to the u8 KV cache (not recommended: on the Arc GPU with default settings the one committed quality run, `docs/benchmarks/quality/kv_u8_qwen-2.5-1.5b.json`, failed with a paged-attention block-size error and 0 passing prompts).

Further variables, as the code reads them:

| Variable | Effect |
|:---|:---|
| `ARC_CONTEXT_WINDOW` | Fixes the context window for every loaded model. Only ASCII digits, at most 7 characters (leading zeros count, so `00000008` is rejected), value 1 to 2097152. Unset or empty: `max_position_embeddings` from the model's `config.json`, then 4096. An invalid value makes `arc-runner serve` print one line on stderr and exit with code 2 before any model is loaded. See docs/claude-code.md. |
| `ARC_CACHE_DIR` | Root of the OpenVINO compile cache; a sub-folder per device is created below it. Unset or empty: `~/.arc-runner/cache/ov_cache`. If the folder cannot be created or written, the cache is switched off with a WARNING and the model still loads. |
| `ARC_WARMUP_ON_LOAD` | `on` runs one short fixed generation (4 new tokens) right after a model loads; `off` or empty (default) skips it. Case-insensitive; any other value is logged at WARNING and treated as off. |
| `ARC_NATIVE_ALLOW_MISSING_SIDECAR` | Exactly `1` lets the native library load when its `.build.json` sidecar is missing (a WARNING is logged and integrity is not checked). It never excuses a sidecar that is present but has a wrong hash. |
| `ARC_ZE_LOADER_ALLOW_ANY` | Exactly `1` lets `ARC_ZE_LOADER_PATH` point at a file whose name does not look like a Level Zero loader (`ze_loader`, `libze_loader.so*`). Without it, an override must be an existing regular file with such a name, and a refused override yields no loader (the default names are not tried instead). The override is code loaded into the process; use only for a file you trust. |

---

## 4. Kubernetes (example, untested)

This manifest has not been applied to a cluster. It assumes the Intel GPU device plugin exposes `gpu.intel.com/i915`. Replace `<version>` with a released tag.

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
          image: ghcr.io/techdeveloper-org/intel-arc-models-runner:<version>
          imagePullPolicy: IfNotPresent
          env:
            - name: ARC_API_KEY
              valueFrom:
                secretKeyRef:
                  name: arc-runner-secrets
                  key: api-key
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
              path: /
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

The probe path `/` (and `/health`) stays open when `ARC_API_KEY` is set; every other route then needs the key. The memory and CPU figures are placeholders to adjust, not measured requirements.

---

## 5. Security settings

| Variable | Effect |
|:---|:---|
| `ARC_API_KEY` | When set and non-empty, every route except `/`, `/health`, `/docs`, `/openapi.json` and `/static/*` requires the key, sent as `Authorization: Bearer <key>` or `X-API-Key: <key>`. Missing or wrong key returns HTTP 401; `/v1/messages*` answer 401 in the Anthropic error envelope (`authentication_error`); `HEAD /api/hello` also needs the key. `GET /health` (200, `{"status": "ok"}`, no model access) is exempt from the key too. Unset means no authentication. |
| `ARC_CORS_ORIGINS` | Comma-separated extra origins allowed by CORS. Local origins (`localhost`, `127.0.0.1`, `[::1]`, any port) are always allowed. Credentials are not allowed. |
| `ARC_ALLOWED_PATHS` | Extra directories (separated by the OS path separator: `:` on Linux, `;` on Windows) that request-supplied paths may point into. Model names and paths outside the models directories and this list are refused with HTTP 403 `PATH_NOT_ALLOWED`. |

In the container, `./models` is mounted at `/models`. To let the server read models there, start it with `--model-dir /models` or add `/models` to `ARC_ALLOWED_PATHS`.

Container properties taken from the `Dockerfile`: non-root `appuser` (UID 1000), supplementary groups `video` (44) and `render` (107), a `HEALTHCHECK` on `GET /`, and a runtime stage without the compiler toolchain. The Intel apt repository key is downloaded during the build without a fingerprint check; that is an open item for the maintainer.

---

## 6. Release automation

- `pypi-publish.yml` runs on a published GitHub release only. It checks that the tag equals `vVERSION`, runs `check_docs_sync.py` and the non-hardware tests, builds the wheel and sdist, and uploads from a separate job that uses the GitHub environment `pypi` (configure required reviewers on it) with PyPI trusted publishing. Branch pushes and manual runs never upload.
- `docker-build.yml` pushes to GHCR only for a published release (section 3.3).

---

## 7. Status and open items

- No release is tagged or published; `pip install arc-runner` and the GHCR image do not work yet.
- **LICENSE: pending.** The repository has no `LICENSE` file yet. `pyproject.toml` and the README name Apache-2.0, but the license text must be added by the maintainer (holder and year) before the first public upload; wheel and sdist currently contain no license text.
- Maintainer setup before the first release: create the PyPI project and trusted publisher for `arc-runner` (repository `techdeveloper-org/intel-arc-models-runner`, workflow `pypi-publish.yml`, environment `pypi`), create the `pypi` environment with a required reviewer, and pin the base image digest.
