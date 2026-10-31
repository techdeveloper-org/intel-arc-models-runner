---
doc_id: DOC-R03-REVIEW
title: R03 SAST review: src/arc_runner (Python) and src/arc_runner/native (C)
stage: S-G
version: 1.0.0
status: draft
created: 2026-10-29
updated: 2026-10-29
authors: [library agent via main session]
reviewers: [user]
---

# R03 SAST review: src/arc_runner (Python) and src/arc_runner/native (C)

Agent: sast-engineer v1.0.0 (library agent definition read in full). Scope: `src/arc_runner` (22,456 Python lines), `src/arc_runner/native/*.c|h` (1,578 lines), plus `scripts/`, `scratch/`, Dockerfile and compose as requested ("scratch scripts", ffmpeg, build scripts).
Severity is a qualitative rating (CRITICAL/HIGH/MED/LOW). No CVSS v4.0 vectors were scored (the delegate `cyber-mathematics-expert` is not available here), so no finding is rated CRITICAL.
India context: NO (DL-039 records no Indian personal data; no CERT-In, DPDP or IT Act 43A tags applied).

## Skills read (proof)
Only the first 120 lines of each Mandatory skill were read (files are 1937, 917, 495, 1388 lines). The rest was not read.
- devsecops-core: `**Architecture:** Pattern-based static analysis using lightweight rules written in YAML.`
- application-security-core: `  False positive rate: FP / (TP + FP) × 100%`
- security-testing-ci-core: `SAST analyzes source code without executing it, finding vulnerabilities through control flow and data flow analysis.`
- india-cyber-regulatory-core: `- **Penalty**: Compensation up to Rs.5 crore (civil). No imprisonment.`
- Not read: threat-modeling-core (Optional). No `threat_list.json` was supplied, so threat-model correlation was not done.

## Tools run (versions from the scratchpad venv)
| Tool | Result |
|---|---|
| ruff `--select S` (src, scripts) | ran. 54 S-rule hits plus 10 `invalid-syntax` hits (a real finding, F9). |
| bandit (src, scripts) | ran. 69 hits. scratch/ separately: 27 hits. |
| semgrep | installed, but the rule registry is blocked. The proxy refused `semgrep.dev:443` and the `p/python`, `p/c` and `p/owasp-top-ten` packs never loaded. **Semgrep produced no findings and was not run.** I did not write substitute rules. |
| pip-audit | ran against the scratch venv, not the project lockfile. 5 hits, see F11. |
| gcc `-fanalyzer -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wformat=2 -Wcast-qual -Wnull-dereference -O2` | 0 warnings on both C files. |
| clang-tidy 18.1.3 (`clang-analyzer-*,bugprone-*,cert-*`) | 34 hits, all false positives (see C-FP). |
| cppcheck | not installed. |
| AddressSanitizer PoC (own harness, scratchpad) | built the bridge with ASan+UBSan against the repo's `tests/native_c/fake_ze_loader.c`. Confirmed F4. |

Counts. Raw tool rows: ruff 54 (+10 syntax), bandit 69 (src+scripts) + 27 (scratch), clang-tidy 34, gcc-analyzer 0, pip-audit 5, semgrep 0 (blocked). That is 199 rows, but ruff and bandit overlap heavily on the same lines (S603/S607 vs B603/B607, and so on), so this is not a distinct-issue count.
**Confirmed security defects: 6 (F1, F2, F3, F4, F5, F6). Confirmed non-security or integrity defects: 3 (F8, F9, F11 note). Suspected, not reproduced: 3 (F7a, F7b, F5b).** Everything else is a triaged false positive or informational.

## Confirmed findings

### F1. HIGH. Path traversal in model delete and copy (CWE-22, CWE-73). VERIFIED end to end
- `src/arc_runner/hub/manager.py:60-61`: `resolve_model_dir` does `(self._base_dir / clean_id).resolve()` and only checks that the result is an existing directory. It never checks that it stays under `_base_dir`. `Path(base) / "/abs"` also discards the base.
- `hub/manager.py:157`: `delete()` calls `shutil.rmtree(target_dir)` on that path.
- `hub/manager.py:216-217`: `copy()` uses `destination` the same way. `shutil.copytree` can write a model directory anywhere the process can write.
- Route: `server/routes_ollama.py:528-555` (`DELETE /api/delete`) and `558-580` (`POST /api/copy`). No authentication.
- Failure scenario (reproduced). With a models dir `M` and a sibling directory `victim/`, I ran `HubManager(M).delete("../victim")`. `victim/` was removed. I then ran the same through the real FastAPI app with `TestClient`: `DELETE /api/delete {"name":"../victim"}` returned 200 `{"status":"success"}` and `victim/` was gone. `copy("../victim","../copied")` created `copied/` outside the models dir.
  Any directory the daemon user can write, such as Documents, can be wiped by one JSON request.
- Why the traversal check is missing: `routes_lora.py:55-68` has a `..` filter, but the Ollama routes do not.
- Fix, as a description only (the agent does not patch). Resolve the candidate and require `candidate.is_relative_to(base_dir)`. Reject separators and absolute paths in the model name. Apply the same rule to `copy` destination and `pull` alias.

### F2. MED standalone, makes F1 HIGH in practice. Wildcard CORS with credentials and no authentication (CWE-942, CWE-306). VERIFIED
- `server/app.py:105-111`: `allow_origins=["*"]`, `allow_credentials=True`, `allow_methods=["*"]`, `allow_headers=["*"]`.
- Reproduced: a preflight `OPTIONS /api/delete` with `Origin: http://evil.example` returned 200 with `access-control-allow-origin: http://evil.example` and `access-control-allow-methods: DELETE, ...`. The following `DELETE` with that Origin succeeded (F1).
- Scenario: the default bind is `127.0.0.1` (`core/constants.py:14`), but a web page the user visits can issue cross-origin requests to `http://localhost:11434`. That page could delete or copy directories using F1, or unload models and run generation. The Docker image serves `--host 0.0.0.0` (Dockerfile CMD) and compose publishes `8000:8000` on all host interfaces, so the unauthenticated API is also LAN-reachable.
- Fix, as a description only: restrict origins to an explicit allow-list (the Ollama default is localhost origins), drop `allow_credentials`, and add an optional bearer token. Consider a Host-header check against DNS rebinding.

### F3. MED. API accepts arbitrary filesystem paths (CWE-22, CWE-200). VERIFIED by reading, not exercised
- `server/model_manager.py:85-86`: `Path(model_id).resolve()` is used as a model directory whenever it exists. An HTTP `model` field can point at any directory. This acts as a directory-existence oracle, and `ModelNotFoundError` echoes the search paths. Loading is delegated to OpenVINO, which parses `*.xml` and `*.bin` from that directory.
- `server/routes_lora.py:55-68`: the `..` filter permits absolute paths. `/api/adapters/load` and `/api/adapters/inspect` therefore read `adapter_config.json` or a safetensors header from anywhere readable. The error text echoes the resolved path. `engine/lora_manager.py:154-160` allows a header read of up to 100 MB per call (`header_len > 100*1024*1024`), so the memory cost is attacker-sized.
- Scenario: a remote or cross-origin caller probes the host filesystem layout and reads JSON fields of files in arbitrary directories. Combined with F2 this is browser-reachable. Impact is information disclosure and resource use, not code execution; I did not find a path that executes loaded content.
- Not a defect: `engine/vision_engine.py:148-165`. The HTTP `images` field is `str`, and `str` is only base64-decoded, never opened as a path. File reads happen only for `Path` objects from in-process callers.

### F4. MED (latent, exported C ABI). Unbounded `memcpy` in the DMA fallback (CWE-787, CWE-125). VERIFIED with AddressSanitizer
- `src/arc_runner/native/arc_level_zero_bridge.c:497-506`. When `zeCommandListAppendMemoryCopy` fails and both pointers fall inside tracked host or shared allocations, the code runs `memcpy(dst, src, size_bytes)`. `arc_find_alloc` (426-440) only checks that the start pointer lies inside an allocation. It never checks that `[ptr, ptr+size_bytes)` fits the tracked `size`. The normal driver path (492) also forwards unvalidated length and pointers.
- Reproduction (scratch harness, own files): allocate two 64-byte USM buffers with the fake loader, set `FAKE_ZE_COPY_FAIL=1`, call `arc_ze_dma_copy_async(ctx, shared, host, 4096)`. ASan reports the access at `arc_level_zero_bridge.c:502` and states the destination is "0 bytes after 64-byte region". The report is labeled `memcpy-param-overlap` because the two allocations were adjacent; the 4096-byte length overruns a 64-byte region either way.
- Reachability: the only in-tree caller, `engine/flash_stream/device_stager.py:125`, derives `size` from `array.nbytes` and sizes the allocation as the sum, so it cannot trigger this today. The defect is in the exported function contract (`bridge.py:732-747` passes raw ints through). It is exposed to any future caller or any ctypes user of the library.
- Fix, as a description only: when the copy is tracked, require `dst`/`src` ranges to lie wholly inside one tracked allocation each. Do the same check before the driver call, not only in the fallback.

### F5. LOW. GGUF parser fails with non-project exceptions on hostile input (CWE-20, CWE-248). VERIFIED
`quant/gguf_parser.py` is documented as treating input as hostile. Crafted files I built in the scratchpad each escape the `GGUFParseError` taxonomy:
- `general.alignment` = 0: `ZeroDivisionError` at line 317 (`current_pos % alignment`). A string value gives `TypeError`.
- Unknown value type or element type (`GGUFValueType(99)`, line 233 and line 248): raw `ValueError`. Same for `GGMLType` at line 283.
- Deeply nested arrays (5000 levels, about 60 KB): `RecursionError` from `_read_value` (line ~230). The recursive array decode has no depth cap.
- Impact: crashes the load request with a 500 and a leaked message. It does not corrupt memory and the length checks (`_read_declared`) held in my tests. This violates the project rule "raise the specific ArcRunnerError subclass".
- F5b, SUSPECTED, not measured: array values decode to Python lists (`gguf_parser.py:~230`), so a large `UINT8` array can use several times its file size in RAM. I did not measure it.

### F6. LOW. Native library and Level Zero loader trust (CWE-427, CWE-494)
- `arc_level_zero_bridge.c:179-196`: the loader path comes from env `ARC_ZE_LOADER_PATH`, and the default is the bare name `ze_loader.dll` via `LoadLibraryA` (Windows). The search order includes the application directory first, so a planted `ze_loader.dll` beside `python.exe` or in the working set would be loaded. The env override is attacker-useful only if the attacker already controls the environment.
- `native/bridge.py:143-151,177`: the sidecar SHA-256 is skipped when the sidecar is missing (`_sidecar_matches` returns True). The hash is read and the library loaded in two separate steps (TOCTOU). The sidecar sits in the same directory as the library, so it detects accidental corruption and stale builds, not tampering by someone who can write that directory. This matches the stated purpose, but docs call it "tampered builds" (`bridge.py:155`).
- Not a defect: ctypes `argtypes` and `restype` are set for every exported function (`bridge.py:194-266`). The pointer-to-array path is bounds-checked at `bridge.py:569-585` (size multiple of itemsize, read-only view, guard keeps the mapping alive).

### F8. LOW (supply chain and honesty). Unpinned model download, and a claim with no check behind it
- `hub/downloader.py:103` and `benchmark/harness_common.py:72`: `snapshot_download(repo_id=...)` without `revision` (bandit B615 true positive). The API-supplied repo id pulls whatever HEAD is, with no digest check. Model files then go to OpenVINO parsing. No `trust_remote_code` anywhere (grep confirmed).
- `hub/downloader.py:157-165` yields `status="verifying checksum"` and then `success` without verifying any checksum. This breaks the project's honesty rule: the status text asserts a check that never ran.

### F9. MED, non-security but verified. `cli/main.py` is a SyntaxError on Python 3.10 and 3.11
- `pyproject.toml:10` says `requires-python >=3.10` and the classifiers list 3.10 and 3.11. `cli/main.py:1060, 1062, 1172, 1173, 1174` reuse the outer quote inside f-string expressions (`f"...{_measured(x, ".2f", " it/s")}..."`), which is valid only from Python 3.12.
- Reproduced: `python3.11 -c "import ast;ast.parse(open('src/arc_runner/cli/main.py').read())"` gives `SyntaxError: invalid decimal literal`. I parsed every other file in `src` and `scripts` with python3.11 and found no other syntax errors. Python 3.12 parses `main.py`.
- Impact: the console entry point cannot start on 3.10 and 3.11, which the package metadata claims to support. (Ruff flagged it; `ast.parse(feature_version=(3,10))` on 3.13 wrongly says OK.)

### F11. Dependency findings routed (Operating Rule 8, not counted as SAST)
pip-audit on the scratch venv listed `mcp 1.29.0` (4 CVEs) and `opentelemetry-api 1.37.0` (1 CVE). `mcp` is only required by semgrep, which I installed for this review, so it is not an arc-runner dependency. `opentelemetry-api` is also pulled in by semgrep in this venv, and the declared project dependencies (`pyproject.toml`) do not list it. Route to `dependency-vulnerability-analyst` against the real lockfile; I did not audit that.

## Suspected, not reproduced
- F7a. `arc_flash_stream_mmap.c:217` maps with `MAP_SHARED` and no write protection on the file. If another local process truncates the file after `fstat`, reads of the mapping raise SIGBUS and kill the process. The Windows branch (165) uses `FILE_SHARE_READ | FILE_SHARE_DELETE` without `FILE_SHARE_WRITE`, which blocks writers. I did not test truncation. Local denial of service only.
- F7b. `arc_mmap_register_layer` (272-299) checks `sealed` and then mutates the table with no lock; `arc_mmap_get_tensor_ptr` re-validates the range (334) but reads `t->size_bytes` again at 339 after that check. Safe only if the caller serializes register against seal, which the Python layer seems to do (`_guard` and `seal()` before `get_tensor_ptr`). I did not run a race.
- Non-security: `orchestrator/muxer.py:456-470` writes frames to ffmpeg stdin with `stderr=PIPE` never drained; a large stderr could block ffmpeg and stall the writer until the 30 s wait. Not reproduced.

## Triaged false positives and informational (not defects)
| Rule and location | Verdict and reason |
|---|---|
| S603/S607/B603/B607/B404 (about 29 sites: `muxer.py:326,456,550,569`, `build_native.py:85,130,145`, `results.py:41`, scripts) | FP. All use list arguments without `shell=True`. Executables are fixed (`ffmpeg`, `git`, compiler). The ffmpeg paths are built from the studio `output_filename` after `Path(...).name` (`orchestrator/pipeline.py:171`) joined under the export dir, so no user text becomes an option and no shell sees it. Partial-path executables (`ffmpeg`, `git`) rely on PATH, an accepted dev-tool trade-off. |
| S310/B310 `cli/main.py:747,748,799,805` | FP. URL is `daemon_url` from a CLI argument or the localhost default, used by the user's own CLI. Not server-side, so not SSRF. Scheme is not checked (`file:` would work) but the operator is the trust boundary. Same for `scratch/run_live_real_models_benchmark_and_compare.py:220`. |
| S311/B311 `bench_regression.py:119`, `benchmark/stats.py:194,224` | FP. Seeded `random` is for bootstrap resampling and run-order shuffling, not security. |
| S105/B105 `orchestrator/director.py:30-31` | FP. `PLAN_START_TOKEN = '<<<STUDIO_PLAN_START>>>'` is a text delimiter, not a credential. A repo-wide grep for key and token patterns found no secrets. |
| S101 (9) `gguf_parser.py:81,89,97,321-323`, `check_ring_model.py` | Informational. Asserts after parse guard `Optional` narrowing. Removed under `python -O`, harmless because parse_all runs first. |
| S110/S112 (7) `detector.py`, `sysman.py`, `lora_manager.py`, `downloader.py:60`, `ollama_importer.py` | Informational. Swallowed exceptions in best-effort probes. They conflict with the project's "do not swallow errors" rule but are not vulnerabilities. |
| B406/B405/B314 `scripts/generate_diagrams.py:23`, `scratch/*drawio*.py`, `scratch/verify_all_diagrams.py` | FP. `xml.sax.saxutils.escape` is output escaping; the ElementTree parses read the repo's own generated `.drawio` files, not untrusted input. |
| B610 `benchmark/workloads.py:104` | FP. `extra` is a callable argument, not a SQL query. |
| `hub/ollama_importer.py:113-120` digest to blob path | Informational. A manifest digest `sha256:../x` becomes `blobs/sha256-../x`. The first component is `sha256-..`, a name no filesystem treats as a parent reference, so it does not escape. Manifests come from the user's own Ollama directory. |
| `routes_studio.py:167-189` video download | Not a defect. The path comes from the in-memory `_PROJECT_CACHE` populated by the server, never from the URL. Project ids are `uuid4`. |
| `hub/manager.py:60`, `registry.py:474-480` pull alias | Not exploitable by me. An unknown alias is used as both repo id and directory name, so `huggingface_hub` repo-id validation should reject `..`; `huggingface_hub` is not in this venv, so I could not confirm that, and F1's fix should cover it regardless. |
| C-FP: clang-tidy 22x `bugprone-multi-level-implicit-pointer-conversion` (`arc_level_zero_bridge.c:140-171,262,264`) | FP. Deliberate `dlsym` function-pointer copy idiom (documented at 119-142); sizes are checked (`out_size != sizeof symbol`). |
| C-FP: clang-tidy 12x `insecureAPI.DeprecatedOrUnsafeBufferHandling` (`memcpy`/`memset`, `arc_flash_stream_mmap.c:281,289,296`, `arc_level_zero_bridge.c:140,251,276,284,294`) | FP. Annex K advice. Every length is `sizeof` a fixed struct or already validated. `arc_mmap_register_layer` validates the private copy (`arc_validate_layer`, 245-270) before copying into the table. |

## C memory-safety review, conclusion
`arc_flash_stream_mmap.c` (487 lines): no defect found beyond F7a/F7b. Overflow-safe range check (`arc_range_ok`, header 102-104, no addition of two untrusted values), name termination, `n_dims` and tensor-count caps, saturating unlock counter, `O_CLOEXEC`, size_t fit check for 32-bit. `arc_level_zero_bridge.c` (600 lines): F4 and F6; the allocation table, lock and teardown are otherwise consistent. `arc_ze_free_usm` frees a buffer that a recorded but unexecuted command list may still reference; a caller-ordering contract, not exploitable without a misusing caller (not tested).
The repo's own sanitizer harnesses in `tests/native_c/` did not catch F4 because no test passes a length larger than the allocation to the fallback path (`ze_harness.c:173-200` only uses 64 into 64).

## Gate
- Raw findings: 199 tool rows (overlapping). After dedupe and triage: 6 security defects, 3 non-security defects, 3 suspected.
- Confirmed by severity: CRITICAL 0, HIGH 1 (F1), MED 3 security (F2, F3, F4) and 1 non-security (F9), LOW 2 security (F5, F6) and 1 (F8), plus F11 routed.
- Threat-model correlated: not assessed (no `threat_list.json`).
- Status: PASS (zero CRITICAL). Verdict: APPROVED_WITH_CONDITIONS per Operating Rule 2. Condition: F1 (HIGH) needs a named owner and remediation or accepted risk before APPROVED. F2 should be fixed together with F1, because the traversal is browser-reachable only through F2.
- Limits of this review: semgrep not run (registry blocked); cppcheck absent; no dynamic testing of the live server beyond the in-process `TestClient`; Windows-only code paths (`#if defined(_WIN32)`) read, not compiled or run; dependency audit done on the scratch venv, not the project lockfile.
- PoC files live only in the scratchpad (`.../scratchpad/sast_poc`, `sast_poc2`, `sast_c`, `sast_gguf`); no repository file was modified.

## Revision history
| Version | Date | Author | What changed | Why | Commit |
|:--|:--|:--|:--|:--|:--|
| 1.0.0 | 2026-10-29 | library agent via main session | first version | independent review wave A | pending |
