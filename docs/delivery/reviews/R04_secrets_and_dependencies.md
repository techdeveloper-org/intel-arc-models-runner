---
doc_id: DOC-R04-REVIEW
title: R04 - Secrets scan and dependency audit
stage: S-G
version: 1.0.0
status: draft
created: 2026-10-29
updated: 2026-10-29
authors: [library agent via main session]
reviewers: [user]
---

# R04 - Secrets scan and dependency audit

Branch `claude/nice-volta-bs7mnf` (85 commits, 2 local branches: this and `main`). Run date 2026-10-29.
Agents played: `secrets-detection-specialist` (Task A), then `dependency-vulnerability-analyst` (Task B).
Repo visibility: **public** (`gh api repos/techdeveloper-org/intel-arc-models-runner` returned `public`, `private=false`), so any finding would have been escalated one severity tier.

## 0. Skills read (proof)

Read in part only (first 100-120 lines of each; the files are 495-1937 lines). Full files were NOT read.
- devsecops-core (L1-120): "# DevSecOps Core Skill Instructions"
- application-security-core (L1-100): "### M1: CVSS Scoring and Risk Quantification"
- data-security-privacy-core (L1-100): "### 1.1 Four-Tier Classification Model"
- india-cyber-regulatory-core (L1-100): "### 1.18 Section 70B -- CERT-In Functions and Powers"
- vulnerability-management-core (L1-100): "### 1.1 Six-Phase Lifecycle"
- security-testing-ci-core (L1-100): "**Evidence retention:** All scan artifacts in SARIF format, tagged with release version and NTP-sourced timestamp. Minimum 180 days per CERT-In Directions 2022."

Not read: the remainder of all six skills, plus the Optional skill `cyber-risk-quantification-core` and the `cyber-mathematics-expert` delegate. No Bayesian/entropy derivation was delegated; see section A.3 for how I limited the claims accordingly.

## A. Task A - Secrets detection

### A.1 Method (all reproducible)
- Scanner: `/tmp/claude-0/-home-user/9f5d9574-39ea-55de-8475-bf2e506385a2/scratchpad/r04/scan.py`; raw output `.../r04/scan.out`. It is a custom regex + Shannon-entropy scanner (not TruffleHog/Gitleaks/detect-secrets, which are not installed; no `gitleaks` run was possible).
- Scope WT: `git ls-files -co --exclude-standard` (tracked + untracked-not-ignored, so the uncommitted edits and `tests/test_tier_failures.py` are included), text files only.
- Scope history: `git log --all -p`, every added line of every commit on both branches (85 commits).
- Patterns: AWS AKIA/ASIA, GitHub PAT forms, HF `hf_`, OpenAI/Anthropic `sk-`, Slack, Stripe, Google `AIza`, PEM private-key headers, JWT, URL-embedded credentials, `password|secret|api_key|token = "..."`, e-mails, Windows/POSIX home paths, RFC1918 IPs, internal hostnames, plus tokens >=32 chars with H(X) >= 4.5 bits/char (hex-only strings excluded).
- Binary files were not text-scanned; the DLL was checked with `strings` (see A.4).
- Active provider verification: not performed (no candidates survived triage).

### A.2 Result
**Zero credential hits.** None of: AWS, GitHub, HF, OpenAI, Anthropic, Slack, Stripe, Google keys, PEM private keys, JWTs, URL credentials, or `key = "value"` secrets matched in the working tree or in any historical added line. No `.env`, `*.pem`, `*.key`, `*.pfx`, `credentials*` file is tracked or present (`git ls-files` + `ls -a`), and none ever appeared in history (`git log --all --name-only`).
CI uses only `${{ secrets.GITHUB_TOKEN }}` (docker-build.yml:37) and PyPI trusted publishing (`id-token: write`); no static token in workflows.
Gate verdict (zero-tolerance): **PASS, 0 blocking findings.**

### A.3 Entropy candidates (1862 raw, all triaged as false positives)
Raw count is inflated because the same line recurs across many commits and historical versions of README.md. Categories, each sampled and verified by reading the matching text:
| Category | Where | Why not a secret |
|---|---|---|
| draw.io compressed-diagram payloads inside `app.diagrams.net/?url=...#R7V...` links | README.md / docs/diagrams/README.md / docs/architecture.md (WT and history, ~1830 hits) | Diagram data embedded in a URL fragment |
| snake_case test function names >32 chars | tests/test_nextgen_orchestration_triad.py (e.g. `test_qsv_encoder_should_selectH264Profile_when_invoked`) | Identifiers |
| `/usr/lib/x86_64-linux-gnu/dri` | docker-compose.yml:16 (`LIBVA_DRIVERS_PATH`) | Filesystem path |
I did not compute Bayesian posteriors (the agent delegates that to `cyber-mathematics-expert`, not invoked); the classification above rests on inspecting content, not on a probability. Because every candidate is explained by content, no suppression-list entry was needed.

### A.4 Personal data and metadata
| Item | Location | Assessment |
|---|---|---|
| Commit identity `Piyush Makhija <piyushmakhija028@gmail.com>` (56 commits, author=committer) | git metadata | Expected (maintainer). Note: differs from the session user's address and from the package address below; informational only. |
| `Claude <noreply@anthropic.com>` (30 commits) | git metadata | Expected tooling attribution |
| `info@techdeveloper.org` | pyproject.toml:13, Dockerfile:45 (+ history) | Intentional project contact; published in PyPI metadata |
| Local Windows user + workspace path `C:\Users\techd\...` | docs/plans/MASTER_PLAN_v4.md:4-5, scratch/generate_all_synchronized_drawio.py:10,512, scratch/sync_all_diagrams_video.py:15,18, scratch/update_diagrams_readme.py:33 (also in history: uml/*.md in a6d76af2) | **LOW**: leaks the maintainer's local username, workspace directory (`workspace-spring-tool-suite-4-4.27.0-new`) and a sibling private-looking project name (`claude-workflow-engine`). Not a credential. Fix is to replace with a placeholder; history still holds it. |
| `/home/appuser/`, `/home/user/` | Dockerfile, compose, docs | Container user / generic; no issue |
| `scratch/*.py` reference local paths | tracked | See above |
No RFC1918 addresses, internal hostnames (the only `internal_host` hit is `.gitignore` `*.local` text) or phone/ID-number patterns were found.

### A.5 Tracked-binary / ignored-artifact review (.gitignore vs `git ls-files`)
| File | Tracked? | Verdict |
|---|---|---|
| `src/arc_runner/native/libarc_native_sys.dll` (56 KB) + `.dll.build.json` | yes | **Intentional** (CLAUDE.md: library + sidecar travel in the commit; packaged via MANIFEST.in:5 and pyproject.toml:83). `.gitignore` ignores `*.so` and `*.so.build.json` but not `*.dll`, so the asymmetry is a deliberate Windows-only shipping choice. Verified: `sha256sum` of the DLL = `68d64d75...cbb4`, equal to the sidecar's `sha256`; sidecar `git_sha` 0fe6981b exists as a commit. `strings` on the DLL shows no user paths, tokens or e-mails (MinGW-built, no embedded `C:\Users`). Provenance note: a committed prebuilt binary cannot be rebuilt-and-compared by a reviewer without GCC 15.2/MinGW; MED-LOW supply-chain note, not a leak. |
| `probe.json` | yes | Measured hardware probe, keys: schema, platform, python, cpu, memory, bandwidth, level_zero_devices, sysman, openvino. Grep for host/user/serial/uuid/mac/path found nothing. Platform string `Windows-11-10.0.26220-SP0` only. OK per project honesty rules (real result). |
| `docs/benchmarks/results/*.json` (3 files) | yes | Intentional published results; no host/user/path strings. |
| `scratch/live_real_models_benchmark_results.json` | yes | Result-like JSON under `scratch/` (not the schema-v1 results dir). Contains model completions (`completion_preview`) only; no secrets. LOW hygiene: provenance unclear vs METHOD.md rule "published numbers come from result files". |
| `.benchmarks/`, `build/`, `dist/`, `*.so`, `__pycache__`, `*.egg-info` | no (ignored, present locally) | Correctly ignored |
| `models/`, `*.bin|*.pt|*.onnx|*.safetensors|*.gguf` | no tracked match | Correct. Largest blob anywhere in history is 286 KB (`scratch/chat-deepseek-live-final.png`), so no model weights were ever committed. |
| Deleted-from-HEAD binaries still in history | in history | 5 `.mp4` under `.benchmarks/studio_exports/` (largest 273 KB, deleted 94257bd, 2026-10-24) and 8 `scratch/chat-*.png` screenshots (PNG strings check on one: no user/e-mail/URL text). Not sensitive; ~1.2 MiB pack total. LOW: they contradict `.benchmarks/` being gitignored (were added before the ignore rule or forced). |
Also deleted from HEAD but in history: `scratch/*benchmark_results.json` (5 files) and `scratch/run_full_catalog_benchmark_audit.py`; the full-added-line scan covered them (no hits beyond the paths above).

### A.6 Residual caveats
- Custom scanner, not the agent's named tools; entropy and regex lists are mine, so provider-specific formats I did not include (e.g. Azure SAS, Twilio `AC...`, SendGrid `SG.`) were only covered by the generic `key=value` and entropy rules. Absence is therefore "no hit by these rules", not a proof.
- Git objects unreachable from any ref (dangling) and the GitHub-side PR/issue/Actions-log history were not scanned. `mcp__github__run_secret_scanning` was not run.

## B. Task B - Dependency review

### B.1 Method
- Installed set from the scratchpad venv (Python 3.13.17, `arc-runner 0.2.0` installed editable). Runtime closure computed from `arc-runner`'s declared `Requires-Dist` by walking installed metadata (34 distributions: 11 direct + 23 transitive, `uvicorn[standard]` extra included, markers evaluated on Linux).
- Vulnerability check: `pip-audit 2.10.1` installed in a **separate** scratch venv (`.../r04/audvenv`) against `pip freeze` of the full venv (44 packages incl. dev tools): **"No known vulnerabilities found"**. The advisory DB was reachable: control run with `requests==2.19.0` returned 5 PYSEC advisories, so the clean result is not vacuous. Source is the default PyPI-backed service (OSV/PYSEC IDs). KEV/EPSS were not queried; no CVE exists to score, so CVSS x EPSS x KEV arithmetic was not needed and not delegated.
- Reachability: no call graph supplied; nothing to classify (zero findings).
- Repo not modified; pip-audit installed only under the scratchpad.

### B.2 SBOM-style table (runtime closure, Linux resolution)
Spec = declared in pyproject.toml. All lower bounds are open-ended (no `<` caps, no hashes, no lockfile).
| Name | Installed | Specifier | License (from metadata) | Risk note |
|---|---|---|---|---|
| openvino | 2026.4.1 | >=2026.1.0 | Apache-2.0 | Native binary wheel; floats to newest; itself caps `numpy<2.6.0` |
| openvino-genai | 2026.4.1.0 | >=2026.0.0.0 | Apache | Metadata pins `openvino_tokenizers~=2026.4.1.dev` (lockstep) |
| openvino-tokenizers | 2026.4.1.0 | >=2026.0.0.0 | Apache-2.0 | Pins `openvino~=2026.4.1.dev` |
| openvino-telemetry | 2025.2.0 | transitive | Apache | Telemetry package pulled by openvino; privacy/offline note |
| numpy | 2.5.4 | >=1.24.0 | BSD-3-Clause AND 0BS | Window is 1.24 .. <2.6 only via openvino's cap |
| pydantic / pydantic-core | 2.14.1 / 2.50.1 | >=2.5.0 | MIT | Floor 2.5 vs code using `typing_extensions.Self` etc.; floors untested |
| fastapi | 0.162.0 | >=0.110.0 | MIT | 0.x API; floor 0.110 untested with starlette 1.x |
| starlette | 1.10.0 | transitive | BSD-3-Clause | Major version moved past 0.x; imported directly in server/app.py but not declared |
| uvicorn | 0.54.0 | uvicorn[standard]>=0.28.0 | BSD-3-Clause | Server binds 0.0.0.0 in Docker CMD |
| httpx | 0.28.1 | >=0.27.0 | BSD-3-Clause | |
| httpcore, h11, anyio, idna, certifi, click | 1.0.9, 0.16.0, 4.15.1, 3.20, 2026.10.23, 8.5.0 | transitive | BSD/MIT; certifi MPL-2.0 | certifi MPL-2.0 is file-level weak copyleft, unmodified redistribution fine |
| httptools, uvloop, watchfiles, websockets, python-dotenv, pyyaml | 0.9.0, 0.23.0, 1.3.0, 17.3, 1.2.4, 6.0.3 | transitive via `[standard]` | MIT/BSD | `uvloop` is not installed on Windows (marker), so the Windows set differs from this table; native C extensions |
| rich, markdown-it-py, mdurl, pygments | 15.0.0, 4.2.0, 0.1.2, 2.21.0 | >=13.7.0 | MIT / BSD-2 | |
| Pillow | 13.0.0 | >=10.0.0 | MIT-CMU (HPND) | Image parsing of user input; bundles C libs; floor 10.0 is old for decoder CVE history, but note this is my background knowledge, UNVERIFIED for 10.x |
| opencv-python-headless | 5.0.0.93 | >=4.9.0 | Apache-2.0 | Wheel bundles FFmpeg (LGPL) and other libs; major jump 4.x -> 5.x within one open specifier |
| annotated-doc, annotated-types, opentelemetry-api, typing-extensions, typing-inspection | 0.0.5, 0.8.0, 1.45.1, 4.16.0, 0.4.4 | transitive | MIT/Apache/PSF | |
Dev-only (not in runtime closure): pytest 9.2.0, pytest-asyncio 1.4.1, pytest-mock 3.17.0, ruff 0.17.3, plus coverage/pytest-cov/psutil present in the venv (psutil and pytest-cov are not declared anywhere in pyproject.toml; `mypy`, `build`, `twine` declared in `[dev]` but not installed here).

### B.3 Findings
| ID | Sev | Finding | Evidence |
|---|---|---|---|
| D-1 | MED | **No reproducibility anchor.** Every dependency is a `>=` lower bound; no lockfile, no hashes, no constraints file (`ls` found no requirements*/uv.lock/poetry.lock). Fresh installs today get openvino 2026.4.1 / numpy 2.5.4 / starlette 1.10.0; the recorded benchmark numbers depend on openvino version, which is not pinned. Failure scenario: a 2026.5 OpenVINO release changes GPU kernels; the same repo commit then yields different throughput and different `openvino_genai` behaviour with no change in git. Mitigated partly: openvino-genai -> tokenizers -> openvino are mutually `~=2026.4.1` in their own metadata (verified via `importlib.metadata`), so the trio moves together within a minor. | pyproject.toml:46-56; metadata dump |
| D-2 | MED | **Declared floors are untested.** The lowest versions (openvino 2026.1.0, genai 2026.0.0.0, fastapi 0.110, pydantic 2.5, numpy 1.24, Pillow 10, opencv 4.9) are never installed in CI (CI does `pip install -e ".[dev]"`, resolves to newest). `openvino>=2026.1.0` with `openvino-genai>=2026.0.0.0` are also inconsistent floors (the real constraint comes from genai's own `~=` pin). Claimed compatibility range is unverified. | ci.yml:34,92,126; pyproject.toml:46-48 |
| D-3 | LOW | **Undeclared imports.** `starlette` (server/app.py) and `typing_extensions` (engine/audio_engine.py:22, unguarded top-level import) are used directly but only arrive transitively (fastapi/pydantic). `huggingface_hub` (hub/downloader.py:102, benchmark/harness_common.py:69) and `imageio` (video_engine.py:319, muxer.py:492, wrapped in try/except) are optional and absent from the venv and from pyproject, so model download via HF fails/falls back unless the user installs it. Not verified at run time (I did not execute the download path). | grep/AST import listing |
| D-4 | LOW | **No LICENSE/NOTICE file in the repo** (`ls LICENSE* NOTICE*` empty; `git ls-files | grep -i licen` empty) although pyproject declares Apache-2.0 and README shows the badge. Apache-2.0 asks for the license text to travel with redistribution; sdist/wheel and the GHCR image ship without it. | repo root |
| D-5 | LOW | **License notes (not legal advice).** All runtime Python dependencies are permissive (MIT/BSD/Apache/PSF/HPND) except certifi (MPL-2.0, file-level). Bundled native code inside opencv-python-headless wheels includes LGPL FFmpeg (dynamic libs; compliance is the wheel publisher's); Docker image additionally installs Ubuntu `ffmpeg` and Intel `intel-media-va-driver-non-free` (non-free licence) and is pushed publicly to ghcr.io by docker-build.yml:50, so redistribution terms of that driver should be checked by the maintainer. Compatible with Apache-2.0 for the project's own code. | B.2, Dockerfile:62-72 |
| D-6 | INFO | No known vulnerabilities in 44 installed packages as of 2026-10-29 (pip-audit, control-verified). This is a point-in-time result; with floating specifiers it says nothing about versions a future install resolves. | B.1 |
| D-7 | INFO | Prebuilt `libarc_native_sys.dll` is a binary dependency with no SLSA/provenance attestation; integrity is a sidecar SHA-256 checked at load. SLSA verification of PyPI wheels (`cosign verify-attestation`) was not attempted; levels unknown. | A.5 |

### B.4 Dockerfile and CI pins
| Item | Observation | Sev |
|---|---|---|
| Base image | `FROM ubuntu:24.04` twice (Dockerfile:10, :39): floating tag, no `@sha256` digest | MED (reproducibility/supply chain) |
| Python deps in image | `pip install --upgrade pip setuptools wheel build` then `pip install .` (Dockerfile:34-35): unpinned, so each rebuild resolves current latest; no `--require-hashes` | MED (same as D-1) |
| Intel apt repo | `curl -fsSL https://repositories.intel.com/gpu/intel-graphics.key | gpg --dearmor` (Dockerfile:62-63): key fetched over TLS without fingerprint verification; packages (`intel-opencl-icd`, `level-zero`, etc.) unpinned | MED |
| Non-root | `USER appuser` UID 1000, multi-stage, `--no-install-recommends`: good | positive |
| Exposure | `EXPOSE 8000`, CMD `--host 0.0.0.0`, compose maps `8000:8000`, and server/app.py:107 sets CORS `allow_origins=["*"]`; I did not find an auth layer in app.py/cli main (grep for api_key/Authorization empty). Out of scope here (belongs to SAST/infra review) but relevant to deployment risk | note |
| GitHub Actions | All third-party actions referenced by tag (`actions/checkout@v4`, `docker/build-push-action@v5`) and `pypa/gh-action-pypi-publish@release/v1` is a moving branch, none SHA-pinned; the PyPI publisher is the highest-impact one (trusted publishing, `id-token: write` set workflow-wide, pypi-publish.yml:16). CI/publish run `pip install --upgrade pip` unpinned. Permissions default to `contents: read` (good). | LOW-MED |
| Dependabot / audit in CI | No `.github/dependabot.yml`, no `pip-audit`/gitleaks job in any workflow (grep found none) | MED: nothing automatically re-runs this review |

### B.5 Recommendations (not applied; audit only)
1. Add a constraints/lock file (`uv pip compile --generate-hashes` or pip-tools) for the benchmark/Docker path and record `openvino` version in result JSON (if not already); keep `pyproject` bounds as ranges, add upper bounds on `openvino*` (`<2027` or `~=2026.4`).
2. Add a CI job testing the lowest declared versions (`uv pip install --resolution lowest-direct`) or raise the floors to what is tested.
3. Declare `starlette`, `typing_extensions`; add an optional extra `[hub] = huggingface_hub`, `[video] = imageio`; declare `pytest-cov`/`psutil` if used.
4. Add LICENSE (Apache-2.0 text) and, since the image includes non-free Intel and FFmpeg components, a THIRD_PARTY note.
5. Pin base image by digest, verify the Intel key fingerprint, pin Intel packages.
6. Add `.github/dependabot.yml` (pip, docker, github-actions) and a scheduled `pip-audit` + `gitleaks` workflow; SHA-pin the publish action.
7. Replace `C:\Users\techd\...` in 4 tracked files with placeholders (history keeps it; rewriting history is not recommended for LOW data).

## C. Verdicts
- Secrets-detection-specialist: **PASS / APPROVED**, 0 verified secrets, 0 findings with P>=0.85 (no Bayesian scoring performed; zero candidates remained after content triage). 2 LOW hygiene notes (local path leak, historic binaries).
- Dependency-vulnerability-analyst: **PASS / APPROVED on CVE gate** (0 known vulnerabilities, 0 KEV). Not a clean bill for reproducibility: D-1, D-2 and the Docker/CI pinning items are MED. SBOM was produced as the table above only (no CycloneDX/SPDX files generated, since the instruction was a table and the repo must not be modified); SBOM coverage metric not computed.

## Revision history
| Version | Date | Author | What changed | Why | Commit |
|:--|:--|:--|:--|:--|:--|
| 1.0.0 | 2026-10-29 | library agent via main session | first version | independent review wave A | pending |
