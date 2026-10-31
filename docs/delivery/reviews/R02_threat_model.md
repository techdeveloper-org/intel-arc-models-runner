---
doc_id: DOC-R02-REVIEW
title: R02 - STRIDE threat model with attack trees
stage: S-G
version: 1.0.0
status: draft
created: 2026-10-29
updated: 2026-10-29
authors: [library agent via main session]
reviewers: [user]
---

# R02 - STRIDE threat model with attack trees

Branch `claude/nice-volta-bs7mnf`, run date 2026-10-29. Agent played: `threat-modeling-specialist` (Phase F.1 style).
Verdict under the agent's own gate (any open HIGH/CRITICAL blocks): **REJECTED** (2 CRITICAL, 2 HIGH open).
Method: read the code, then ran proof-of-concept inputs in temp dirs under `/tmp/claude-0/.../scratchpad/r02/` (scripts `poc_*.py`; nothing in the repo was modified). Every finding is tagged VERIFIED (reproduced by running) or TRACED (read, not executed).

## 0. Skills read (proof) and limits

Read in part (first 100-200 lines of each; files are 492-1600 lines):
- threat-modeling-core (L1-200): "### M1: Attack-Tree / Attack-Graph Formalism"
- application-security-core (L1-120): "### M1: CVSS Scoring and Risk Quantification"
- cyber-risk-quantification-core (L1-100): "### 1.1 Qualitative vs Quantitative Risk Assessment"
- india-cyber-regulatory-core (L1-100): "### 1.18 Section 70B -- CERT-In Functions and Powers"

Not read: M2-M6 of threat-modeling-core and the remainder of all four skills; optional `security-testing-ci-core`; `cyber-mathematics-expert` was NOT invoked. Consequence: **no numeric leaf probabilities, no RAM triple, no CVSS/FAIR numbers** are given (none could be sourced, and the task said qualitative ranking). Attack trees below are structural AND/OR with qualitative likelihood only. Ranking = likelihood x impact, qualitative. R04 (secrets, dependency CVEs) is not repeated here.

## 1. Scope, DFD and trust boundaries

External entities: E1 any web page in the user's browser; E2 any LAN host (when bound off loopback / Docker port published); E3 Hugging Face hub; E4 supplier of a `.gguf`/IR/LoRA file; E5 whoever controls env vars or the install directory; E6 PyPI/GHCR consumers.
Processes: P1 FastAPI daemon (`server/app.py`); P2 `GGUFParser`; P3 `libarc_native_sys` via ctypes; P4 OpenVINO engines; P5 `GGUFTranspiler`; P6 `HubManager`/`ModelDownloader`.
Stores: D1 models dir; D2 AOT cache; D3 adapters dir; D4 arbitrary output dirs; D5 package dir holding DLL/SO plus `.build.json`.
Trust boundaries: TB1 network/browser -> P1; TB2 file bytes -> P2/P5; TB3 environment/filesystem -> loader (P3, Sysman); TB4 HF -> D1; TB5 build/release -> user.
Counts: 6 processes, 5 stores, 5 boundaries. Assets: host files (delete/overwrite/read), process integrity (code exec in the daemon, which holds GPU access), availability of the Arc device and RAM/disk, integrity of models and of published artifacts, honesty of reported results.
Default bind is loopback (`core/constants.py:14`), but Docker CMD binds 0.0.0.0 (`Dockerfile:106`) and compose publishes `8000:8000` on all interfaces (`docker-compose.yml:9-10`).

## 2. Findings summary (ranked)

| ID | Sev | Surface | STRIDE | Status | One line |
|---|---|---|---|---|---|
| TM-01 | CRITICAL | Server | S, E | VERIFIED | No authentication; CORS reflects any origin with credentials; no Host check |
| TM-02 | CRITICAL | Server/hub | T, D, E | VERIFIED | `DELETE /api/delete` removes any directory (absolute path or `..`) |
| TM-03 | HIGH | Server/hub | T, E | VERIFIED | `POST /api/copy` writes a model tree to any path |
| TM-04 | HIGH | Server/convert | T, D, E | VERIFIED | GGUF convert (GET and POST) reads any path and writes to any `output_dir` |
| TM-05 | MED | Server | I, E | VERIFIED | Arbitrary path probing: existence oracle plus first 4 bytes of any file; engines load any directory |
| TM-06 | MED | Loader | E | VERIFIED | `ARC_ZE_LOADER_PATH` loads any library; constructor ran |
| TM-07 | MED | Loader | T | VERIFIED | Sidecar check fails open and is writable by the same actor it guards against |
| TM-08 | MED | GGUF parser | D | VERIFIED | Untyped exceptions, ~9x memory amplification, slow arrays |
| TM-09 | MED | Convert | T, R | VERIFIED | Unescaped names injected into IR XML; output labelled INT4 is a raw copy |
| TM-10 | MED | Download | T, S, R | VERIFIED | Unpinned, unverified HF pulls; "verifying checksum" does not verify |
| TM-11 | MED | Server | D | VERIFIED/TRACED | Unbounded tokens, images, caches, background tasks, downloads |
| TM-12 | LOW | Native C | T, E | TRACED | `arc_ze_dma_copy_async` fallback checks only the start pointer |
| TM-13 | MED | Packaging | T, S | VERIFIED/TRACED | Unreproducible shipped binary, stale-dist upload, mutable CI refs, unauthenticated apt key |

India tags (agent rule 5): `certIn_reportable` would apply to TM-01..04 (unauthorised access, data destruction) only if the deployment is by an Indian body corporate; the project's India scope is not stated, so this is conditional, not asserted. No DPDP personal-data store is evident in this repo (prompts are not persisted; not exhaustively checked). No CII/SEBI/RBI attestation is required.

## 3. Surface 1 - GGUF parser (`quant/gguf_parser.py`), hostile model file

Assets: server availability, RAM, correctness of tensor offsets given to the mmap layer. Boundary TB2.

STRIDE:
- S: not applicable (no identity in a file). Magic and version are checked (`:123`, `:131`), nothing authenticates the file.
- T: duplicate metadata keys silently overwrite (`:255`; PoC `dup_keys` OK); invalid UTF-8 is replaced not rejected (`:155`). Tensor `offset` is never checked against file size at parse time (checked later in `weight_streamer.py:296` and C `arc_flash_stream_mmap.c:260`).
- R: no integrity record of what was parsed.
- I: parse errors echo declared sizes only, no file content (`:57`). OK.
- D: see TM-08.
- E: pure Python, no memory-safety exposure here; the C layer receives only validated values (`bridge.py:70-87`, C `:245-270`).

Existing mitigations (verified by running): declared string/dimension lengths bounded by remaining bytes (`gguf_parser.py:38-58`; PoCs `string_len_2^64-1`, `array_len_2^63_no_body`, `tensor_count_2^63`, `kv_count_2^63` all raise `GGUFParseError`); `n_dims <= 4` (`:275`); offsets out of file raise `ValueError` in the streamer (PoC `offset beyond EOF`); native range checks (`arc_range_ok`, tests `tests/test_native_bounds.py`).

**TM-08 (MED, VERIFIED)** residual defects, each reproduced with a hostile file via `GGUFParser(...).parse_all()` / `get_summary()`:
| Input | Result | Cause |
|---|---|---|
| unknown value type 99 (also as array elem type) | `ValueError` | `GGUFValueType(int)` at `:230`, `:253` |
| unknown tensor type 9999 | `ValueError` | `GGMLType(int)` at `:285` |
| 5000 nested array headers (60 KB file) | `RecursionError` | recursion at `:232` |
| `general.alignment` = 0 | `ZeroDivisionError` | `current_pos % alignment` at `:311`; string alignment gives `TypeError` |
| `<arch>.context_length` string, array, NaN, inf | `ValueError`/`TypeError`/`OverflowError` | `int(...)` at `:326-329` |
| u8 array of 20 MB | 182 MB peak RSS growth, 3.3 s | one Python int slot per byte, `:232` |
| 1M nested empty arrays (12 MB file) | 69 MB, 2.0 s | per-element list objects |
Likelihood: HIGH for any user who loads a downloaded GGUF; impact: one request or CLI call fails or stalls. The server's inspect route survives because it catches `Exception` (`routes_ui.py:146`), so this is a contract violation (CLAUDE.md: raise the specific `ArcRunnerError` subclass) plus a resource risk, not a crash. Extrapolation, not measured: memory grows linearly, so a multi-GB hostile file would need roughly 9x its size in RAM (the project machine has 15 GB). Also `FlashStreamWeightManager.index_layers` raises a raw `ValueError` for a tensor named `blk.<5000 digits>.w` (`weight_streamer.py:207`, PoC) because of Python's 4300-digit int limit.
Fix (TM-08): convert enum errors to `GGUFParseError`; cap array length by `remaining / min_element_size` and nesting depth (1 or 2); validate `alignment` is a power of two in [1, 2^20]; wrap the `int()` conversions in `get_summary` in a typed error; cap `\d{1,9}` in the layer regex; stream-skip large arrays that are not needed (tokenizer arrays) instead of materialising them; validate tensor offsets against file size in `parse_all`.

Attack tree A (goal: make the host unusable or corrupt state via a `.gguf`), qualitative:
```
A  Exhaust/disable runner via hostile GGUF                       [OR]
|- A1 Crash the parse path                                      [OR]
|   |- A1a unknown enum id (ValueError)                  likelihood high, cost trivial
|   |- A1b alignment 0 / non-numeric metadata            likelihood high, cost trivial
|   '- A1c nesting depth > ~1000 (RecursionError)       likelihood high, cost trivial
|- A2 Exhaust RAM/CPU                                           [AND]
|   |- A2a victim opens a file >= several GB (user action)  likelihood medium
|   '- A2b huge fixed-width array declared and present   cost = file size only
'- A3 Memory-unsafe read via offsets (needs native bug)         [AND]
    |- A3a offsets/sizes beyond EOF
    '- A3b native bounds check missing                  BLOCKED: C arc_range_ok + Python check hold
```
Minimal cut sets: {A1a,A1b,A1c fixes} for A1; {array/nesting caps} for A2; A3 already cut by existing checks.

## 4. Surface 2 - native loader, sidecar, env vars

Assets: code integrity of the daemon process. Boundary TB3. Components: `bridge.py:135-191`, `devices/sysman.py:189-218`, C `arc_level_zero_bridge.c:179-196`.

STRIDE:
- S: a library named `ze_loader`/`libarc_native_sys` is trusted by name or path only.
- T: TM-06, TM-07.
- R: load failures log `path` and `reason` (`bridge.py:173,190`); a successful load of an override path is not logged at all.
- I: n/a. D: `ARC_NATIVE_DISABLE` is fail-safe (falls back to the Python read path, `bridge.py:161-163`); ABI mismatch returns None (`:181-186`). E: TM-06.

**TM-06 (MED, VERIFIED)** `ARC_ZE_LOADER_PATH` is used verbatim by Python (`sysman.py:195-197`, `zes_preinit` `:212-214`, `LevelZeroSysman._load_library` `:240-242`) and by C (`arc_level_zero_bridge.c:180-182`, `LoadLibraryA`/`dlopen`). PoC: `gcc -shared` library with a constructor, `ARC_ZE_LOADER_PATH=<it>`, then `NativeLevelZeroBridge()`: the constructor ran (marker file written) before any symbol lookup failed. No absolute-path requirement, owner check, hash, or logging. Likelihood LOW (attacker must control the process environment: a `.env`/compose file, a launcher, a service definition), impact CRITICAL (code execution with the daemon's rights and GPU access) -> MED overall. TRACED, not run (no Windows host here): C `arc_level_zero_bridge.c:185,190` loads the bare name `ze_loader.dll` with `LoadLibraryA`, which uses the legacy search order (application directory, system dirs, current directory, `PATH`), unlike Python's `CDLL`; `utils/dll_loader.py:117-121` additionally registers any `PATH` entry containing "intel", "openvino" or "oneapi" as a DLL directory.
Fix: accept the override only as an absolute path to an existing regular file, log it at WARNING with its sha256, refuse if the file or its directory is world/group-writable (POSIX) or not under an admin-owned directory (Windows), or gate it behind an explicit `ARC_ALLOW_LOADER_OVERRIDE=1`; on Windows load with `LoadLibraryExW(full_path, NULL, LOAD_LIBRARY_SEARCH_SYSTEM32)` and resolve `ze_loader.dll` from System32 only.

**TM-07 (MED, VERIFIED)** `_sidecar_matches` (`bridge.py:135-151`) gives only integrity against accident:
| Case (PoC `poc_sidecar.py`) | Result |
|---|---|
| tampered library, original sidecar | rejected (good) |
| tampered library, sidecar deleted | **accepted** (`:145-146` returns True) |
| tampered library, sidecar rewritten with new hash | **accepted** |
| sidecar is JSON `[1,2]` or `null` | uncaught `AttributeError` (`:148` catches only OSError/ValueError; the call at `:172` is outside the `try`), so `NativeMemoryEngine()` raises instead of falling back |
Also: the file is hashed (`:151`) and later loaded (`:177`) as two separate opens (TOCTOU, TRACED); the test `test_native_bounds.py:235` pins "accepts missing" as intended. The Windows DLL in the stale `dist/arc_runner-0.2.0` wheel has no sidecar at all (verified by `unzip -l`), so the check would be skipped there.
Residual risk: anyone who can write the package directory can already replace Python files, so the sidecar adds little against a local attacker; its value is detecting a stale or partial build. Say so in `docs/` rather than present it as tamper protection ("refusing mismatched or tampered builds", `bridge.py:155`).
Fix: catch `(OSError, ValueError, AttributeError, TypeError)`; treat a missing sidecar as failure when `ARC_NATIVE_REQUIRE_SIDECAR=1`, and ship that as default in wheels; hash from one open handle (read bytes once, write to a private temp copy, `CDLL` that copy); for real tamper evidence compare against the wheel's `RECORD` hash or a signed manifest.

Attack tree B (goal: code execution in the daemon via the native layer):
```
B  Run attacker code in the daemon                                [OR]
|- B1 Library substitution                                        [OR]
|   |- B1a set ARC_ZE_LOADER_PATH (needs env control)     likelihood low, VERIFIED
|   |- B1b replace libarc_native_sys + delete sidecar (needs dir write)   likelihood low, VERIFIED
|   '- B1c plant ze_loader.dll in cwd/PATH on Windows (loader missing)    likelihood low, UNVERIFIED
|- B2 Memory corruption in own C code                             [AND]
|   |- B2a reach arc_ze_dma_copy_async with bad size (TM-12)  needs a Python-side caller bug
|   '- B2b fallback path taken (driver rejects the copy)
'- B3 Remote trigger of B1 through the daemon                    [AND]
    |- B3a TM-01 (no auth) plus an API that writes files (TM-03/TM-04: output_dir)
    '- B3b a path that reaches the package dir or an env-controlled location   not demonstrated
```
Minimal cuts: B1 {override validation, sidecar required + same-handle load}; B3 {TM-01 fix}. B3 is why TM-01 matters for the loader: file-write primitives exist, but no route sets environment variables, and I did not find one that writes into the package directory (not demonstrated, not ruled out for a user running the daemon from an editable install).

## 5. Surface 3 - FastAPI routes (`server/`)

Assets: host files, GPU, RAM, disk, model integrity. Boundary TB1.

**TM-01 (CRITICAL, VERIFIED)** `app.py:105-111`: `CORSMiddleware(allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])`. PoC via `TestClient`: a preflight from `Origin: https://evil.example` for `DELETE` with `content-type` returned 200 with `access-control-allow-origin: https://evil.example`, `allow-credentials: true`; the real response echoed the same origin. There is no authentication, no `TrustedHostMiddleware`, no token (grep for api_key/Authorization/Bearer found none). Consequence: any web page the user visits can call every route on `127.0.0.1:11434` (including state-changing GETs that need no preflight, see TM-04); any LAN host can when the Docker port is published (`docker-compose.yml:9-10` binds all interfaces; `Dockerfile:106` binds 0.0.0.0; `serve` accepts any `--host` with no warning). `/openapi.json` is still served although `docs_url` is disabled (`app.py:95`; route listed in PoC).
Likelihood HIGH (drive-by needs only a visited page), impact = TM-02..05.
Fix (recommend; design belongs to solution-architect): default CORS allowlist = `http://localhost:*`/`127.0.0.1` only, `allow_credentials=False`; `TrustedHostMiddleware` with localhost names (also stops DNS rebinding); require `Authorization: Bearer <token>` (generated at first start, stored 0600) on all non-health routes and refuse a non-loopback bind without a token; compose `127.0.0.1:8000:8000`; require `Content-Type: application/json` and reject cross-site `Sec-Fetch-Site` on mutating routes; move state-changing `GET /api/gguf/convert/stream` to POST.

**TM-02 (CRITICAL, VERIFIED)** `HubManager.resolve_model_dir` (`hub/manager.py:64-66`): `clean_id = model_id.strip().split(":")[0]`; `(base / clean_id).resolve()` with an absolute `clean_id` discards `base`, and `..` escapes it; any existing directory is accepted; `delete` then runs `shutil.rmtree(target_dir)` (`:157`). PoC (unit and HTTP): `DELETE /api/delete {"name": "<abs dir>"}` -> 200 and the directory was gone; `"../other"` removed a sibling; `".."` removed the parent of the models dir. Reached from `routes_ollama.py:545-550`. The existing tests `test_hub.py:223` and `test_ui_endpoints.py:330` are named as traversal protection, pass today, and prove nothing: both only assert that a non-existent target is "not found" (I ran both: pass, while the PoC deletes).
Fix: accept names matching `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$` (no separators, no colon beyond one tag split); after resolve require `target.parent == base_dir`; refuse to delete unless the directory contains `*.xml` + `*.bin`; log the deletion; write the test red-first with an existing out-of-tree directory.

**TM-03 (HIGH, VERIFIED)** `HubManager.copy` (`manager.py:210-217`): `dest_dir = (base / destination.strip()).resolve()` is unchecked, `copytree(src, dest)` writes anywhere the process may write (PoC: absolute destination created `outside_copy/a.xml`); the source may be any directory via TM-02's resolver. Same fix as TM-02 for both arguments.

**TM-04 (HIGH, VERIFIED)** `routes_ui.py:219-224` (POST) and `:241-277` (GET `/api/gguf/convert/stream`) take `path` and `output_dir` from the request unconfined. `GGUFTranspiler.transpile_stream` does `mkdir(parents=True)` (`transpiler.py:234`) and writes `openvino_model.bin/.xml` and `config.json` (`:268`, `:341`, `:353`), overwriting same-named files. PoC: a GET carrying `Origin: https://evil.example` returned 200 and created the three files in an arbitrary new directory. Because it is a GET it needs no preflight, so TM-01's CORS is not even required. The bin output is a full copy of the tensor data (disk-fill amplification), worker tasks are not tracked or cancelled on disconnect (`routes_ui.py:289-290`), and there is no concurrency limit.
Fix: resolve `path` and `output_dir` under configured roots (`model_dirs`, an `output_root`), POST only, one conversion at a time per server, refuse to overwrite a non-empty directory, size check vs free disk.

**TM-05 (MED, VERIFIED)** Path-taking surface beyond TM-02..04:
- `POST /api/ollama/inspect {"path": ...}` (`routes_ui.py:120-148`): any file's existence is an oracle (404 vs 400), and the error echoes the first four bytes: PoC on `/etc/passwd` returned `got b'root'`. Any real GGUF anywhere is parsed and summarised.
- `ModelManager.resolve_model_path` (`model_manager.py:86-98`) accepts any absolute directory for `model` in chat/generate/embeddings; the engine then hands that directory to OpenVINO (native IR/tokenizer parsers) - an attacker needs a planted IR directory, so likelihood LOW, but it widens TM-10.
- LoRA: `_validate_safe_path` (`routes_lora.py:63`) rejects only `..` segments, allows absolute paths and symlinks; `adapter` in `/api/generate` and `/api/chat` is not checked at all (`routes_ollama.py:180-181`); safetensors header parse is capped at 100 MB (`lora_manager.py:159`), so `json.loads` of 100 MB is a CPU/RAM hit.
- `GET /api/tags` returns absolute server paths (`model_manager.py:266`); several routes return `str(exc)` with local paths (`routes_ui.py:148,206,238`).
Fix: one shared `confine(path, roots)` helper used by every route; return generic errors with an id and log detail server-side; drop `path` from `/api/tags`.

**TM-11 (MED)** Resource exhaustion (all unauthenticated): `options.num_predict`/`max_tokens` taken as `int(...)` with no cap (`routes_ollama.py:163-166`; OpenAI `max_tokens` only `ge=1`, `schemas.py:145`); base64 images decoded with Pillow and no size/pixel/count limit (`vision_engine.py:155-167`, TRACED); no request body limit; `_PROJECT_CACHE` grows forever (`routes_studio.py:35,72,102`); `/api/pull` has no concurrency or size cap (TM-10); no check that generation stops on client disconnect (not examined). Fix: cap tokens/images/body size, bound caches (LRU), one generation per device, cap concurrent downloads and free-disk precheck.

Attack tree C (goal: delete or overwrite user data, or run code, from a web page):
```
C  Destroy/overwrite host data or execute code from a web page     [OR]
|- C1 Delete directory                                              [AND]
|   |- C1a victim has the daemon running (loopback)           likelihood medium-high
|   |- C1b victim visits attacker page (preflight allowed, TM-01)  likelihood medium
|   '- C1c DELETE /api/delete with absolute path (TM-02)       cost trivial, VERIFIED
|- C2 Write files into chosen directory                             [AND]
|   |- C1a, then GET convert/stream with output_dir (TM-04), no preflight needed
|   '- needs a readable .gguf on the host (any small one, attacker cannot supply it remotely: partial)
|- C3 Plant tree via /api/copy (TM-03)                               [AND] C1a + source dir exists
'- C4 Execute code                                                   [AND]
    |- C2/C3 write into a location that gets executed (startup folder, site-packages .pth)
    '- target path known/writable                              likelihood low-medium, not demonstrated
```
Minimal cut sets: {TM-01 fix} severs C1-C4 from browsers; {TM-02/03/04 confinement} severs them from anyone who can reach the port. Both are needed (defence in depth).

## 6. Surface 4 - model download and convert paths

Assets: integrity of weights, disk, honesty of reported verification. Boundary TB4.

**TM-10 (MED, VERIFIED)**
- `ModelRegistry.resolve` (`hub/registry.py:474-481`) turns any unknown string into `repo_id`, so `/api/pull` downloads any HF repo; `alias = alias_or_repo` unvalidated feeds `target_dir` (`manager.py:121`). `dest.mkdir(parents=True)` runs before any validation (`downloader.py:84`); HF's repo-id check later blocks `..`-style repo ids, so this is directory creation only (TRACED).
- `snapshot_download(repo_id=..., local_dir=...)` (`downloader.py:103-107`) has no `revision`, no allowlist of orgs, no hash comparison (grep for `revision` in `src/`: none). A moved branch or a compromised repo changes the model silently.
- Honesty defect: the stream emits `"verifying checksum"` at 99.9 % (`downloader.py:157-166`) but only sums directory size; `digest` is `sha256(repo_id)` (`:85`), shown to users as a model digest. Per the project honesty rule these should be removed or implemented. `HubManager.verify_integrity` (`manager.py:76-97`) only checks that `*.xml` and `*.bin` exist.
- Ollama import: `_resolve_digest_path` (`ollama_importer.py:112-115`) does not require 64 hex chars; a manifest value like `sha256:../x` yields `blobs/sha256-../x`, which does not resolve on common filesystems (TRACED); blobs are never re-hashed against their digest.
- Scratch scripts (`scratch/run_live_video_benchmarks.py:55-62`, `run_live_real_models_benchmark_and_compare.py:269-270`): repo ids come from hard-coded specs, `subprocess.run` uses a list (no shell), `--trust-remote-code` is not passed (grep: no `trust_remote_code` anywhere). LOW; the same no-revision gap applies.
Fix: curated alias registry with pinned commit SHAs plus per-file sha256 (HF exposes LFS hashes) verified after download; `/api/pull` accepts curated aliases or an explicit allowlist, `revision` required for others; remove the fake "verifying checksum" step or make it real; require `^[0-9a-f]{64}$` and re-hash Ollama blobs on import; cap concurrent pulls and check free space.

**TM-09 (MED, VERIFIED)** `GGUFTranspiler` writes the IR from file-controlled strings without escaping: `<net name="{model_name}">` (`transpiler.py:423`, `model_name` = `general.architecture`) and `<layer ... name="{t.name}">` (`:432`). PoC: a tensor name `w"/><layer id="99" name="EVIL" type="Parameter"/><layer name="` and an architecture `x"><injected_net .../>` appeared as live elements in `openvino_model.xml` (my payload was not well-formed XML; a crafted one can be). Impact: graph tampering or parser errors inside OpenVINO's XML reader. Fix: `xml.sax.saxutils.quoteattr`/ElementTree for all attributes. Related integrity facts found while reading (not exploit-driven): `transpiler.py:271` seeks `tensor.offset` without adding `parser.data_offset` (the streamer does add it, `weight_streamer.py:237,292`), so tensor bytes are read from the wrong place; types other than F32/F16/Q4_0 are sized as 1 byte per element (`:249-252`); no quantization happens (raw bytes are copied, `:272-279`) yet the output and `config.json` say `INT4`/`quantization`; `sqnr_val` is a constant, 34.6 for Q4_0 else 99.0 (`:317`), not a measurement. This conflicts with the project's honesty rule and should be routed to the maintainer. Hostile dims also surface as `MemoryError`/`OverflowError` (not `TranspilationError`; PoC dims 2^40 and 2^62) instead of a typed error.

## 7. Surface 5 - Docker and packaging

Assets: users' machines, trust in the PyPI/GHCR artifacts. Boundary TB5.

Mitigations in place (verified): container runs as non-root UID 1000 (`Dockerfile:97`); multi-stage build; `.dockerignore` excludes `.git`, tests, docs, `scratch`, `.env`; PyPI publishing uses OIDC trusted publishing, no static token (`pypi-publish.yml:16`); `publish_pypi.py` defaults to dry run (`publish_pypi.py:119-120`, `--upload` required); Actions token is `contents: read` for CI.

**TM-13 (MED)**
- Network exposure: see TM-01 (`Dockerfile:106`, `docker-compose.yml:9-10`); compose has no `read_only`, `cap_drop`, `security_opt: no-new-privileges`.
- Unauthenticated trust anchor: `curl -fsSL https://repositories.intel.com/gpu/intel-graphics.key | gpg --dearmor` (`Dockerfile:68`) with no fingerprint check; base image `ubuntu:24.04` by tag (`:10,:41`); `pip install .` (`:36`) resolves unpinned, unhashed dependencies (`pyproject.toml:43-53` uses `>=`).
- Binary provenance (also noted in R04): the wheel is tagged `py3-none-any` but ships a Windows DLL and a Linux SO (`MANIFEST.in:5`, `pyproject.toml:83`). The publish workflow never builds the native library (no build step in `pypi-publish.yml`), so the published DLL is whatever binary is committed; the `.so` is gitignored so a CI-built wheel has no Linux library (falls back to Python path, fail-safe). A reviewer cannot rebuild-and-compare the DLL without GCC 15.2/MinGW.
- Stale artifacts: local `dist/` currently holds `0.1.0` and `0.2.0` wheels/sdists; `publish_pypi.py:56,105` uploads every `*.whl`/`*.tar.gz` in `dist/` with no clean step, version-equals-`VERSION` check or "built from HEAD" check. `build_wheel.py` cleans `dist/` first, so the risk is running `publish_pypi.py --upload` alone. The 0.2.0 wheel there contains a DLL that differs from the committed one (sha256 prefix `cbfce3df...` vs committed `68d64d75...`) and has no DLL sidecar.
- `build_wheel.py:125-139` "verifies" RECORD hashes against files from the same archive: it detects corruption, not tampering, and does not check the native binary or sidecar; `:63` takes `wheels[0]` of an unordered glob.
- CI: actions pinned to mutable tags/branch (`actions/checkout@v4`, `pypa/gh-action-pypi-publish@release/v1`, `pypi-publish.yml:25,28,47`); the publish job runs no tests and has no protected `environment:`; `docker-build.yml` pushes `latest` and sha tags on every push to main touching `src/**`, with no image scan, SBOM or signature.
Fix: bind compose to loopback; add `cap_drop: [ALL]`, `no-new-privileges`, `read_only` with tmpfs; verify the Intel key fingerprint in the Dockerfile and pin the base image digest; build native libraries in the publish job per platform (platform-tagged wheels) and add `pip install --require-hashes` lockfile for the image; make `publish_pypi.py` refuse unless `dist/` holds exactly the artifacts for `VERSION` and a clean tree at the release tag; pin Actions by commit SHA; add a protected `pypi` environment, a test gate, and `actions/attest-build-provenance` plus image signing.

**TM-12 (LOW, TRACED)** `arc_find_alloc` (`arc_level_zero_bridge.c:426-440`) checks only that `ptr` lies inside a tracked allocation; `arc_ze_dma_copy_async` then does `memcpy(dst, src, size_bytes)` in the fallback (`:499-502`) without checking `ptr + size_bytes` stays inside the record. Reachable only from Python callers (`device_stager.py:125` passes sizes derived from arrays), and only when the driver rejects the copy, so no remote path; but the function is exported API. Fix: return the record, require `[ptr, ptr+size)` inside it for both pointers, add a case to the C sanitizer harness.

## 8. STRIDE coverage matrix (threat count per surface; "-" = not applicable, rationale in text)

| Surface | S | T | R | I | D | E |
|---|---|---|---|---|---|---|
| 1 GGUF parser | - | 2 | - | 0 | 1 (TM-08) | 0 |
| 2 Loader/sidecar | 1 | 2 (TM-06, 07) | 1 | - | 0 | 1 |
| 3 Server | 1 (TM-01) | 3 (02, 03, 04) | 0 | 1 (TM-05) | 1 (TM-11) | 2 |
| 4 Download/convert | 1 | 2 (09, 10) | 2 (fake verify, constants) | 0 | 1 | 0 |
| 5 Docker/packaging | 1 | 2 | 0 | 0 | 0 | 0 |
Repudiation is mostly absent: no audit log for destructive routes (delete, copy, convert write only generic request lines; request id is taken from the client header, `app.py:55`). Add an audit line (route, resolved path, client ip) for TM-02/03/04 actions.

## 9. Recommended order of work

1. TM-01 + TM-02/03/04/05 together: auth token, origin/host allowlist, one `confine()` helper, tests red-first with existing out-of-tree targets (replace the two vacuous traversal tests).
2. TM-06/07 loader hardening (small, local, testable).
3. TM-08 parser caps and typed errors (extend `tests/test_gguf_untrusted_input.py`).
4. TM-10 honest verification and pinned revisions; TM-09 XML escaping and the offset bug.
5. TM-13 packaging/CI pinning; TM-11 limits; TM-12 range check.

## 10. Limits of this review

Windows loader behaviour (TM-06 bare-name search, `dll_loader.py` PATH registration) is read-only analysis; no Windows host was available. The OpenVINO/openvino_tokenizers native parsers were not fuzzed (out of scope). Generation cancellation on client disconnect, `routes_batch`/`batch_pipeline` internals, and the `StudioPipeline` subprocess (`orchestrator/muxer.py:325,456,550`, ffmpeg argument construction) were not examined; treat them as unreviewed attack surface. No numeric probabilities, CVSS, FAIR or RAM values were computed (see section 0).

## Revision history
| Version | Date | Author | What changed | Why | Commit |
|:--|:--|:--|:--|:--|:--|
| 1.0.0 | 2026-10-29 | library agent via main session | first version | independent review wave A | pending |
