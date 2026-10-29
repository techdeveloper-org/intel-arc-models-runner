---
doc_id: DOC-R10-REVIEW
title: R10 Release-readiness review for v0.2.0 (nothing tagged, published or pushed)
stage: S-G
version: 1.0.0
status: draft
created: 2026-10-29
updated: 2026-10-29
authors: [library agent via main session]
reviewers: [user]
---

# R10 Release-readiness review for v0.2.0 (nothing tagged, published or pushed)

Reviewed at HEAD `f25e8b3` (branch claude/nice-volta-bs7mnf). Agents played: `release-engineering-specialist` (release engineering) and `technical-writer-agent` (README / deployment docs).
Note on the master plan: it also lists `supply-chain-engineer`. That library agent is a warehouse/logistics agent (wrong domain), so it was NOT used.

## 0. Skill proof

Read (first lines only, not the whole file), one exact line each:
- release-versioning-management-core (lines 1-120 of 547): `bump(commits) = max_i( bump(commit_i) )   where MAJOR > MINOR > PATCH > none`
- github-actions-ci (1-120 of 1090): `Total jobs = Π(dimension sizes) = 3 × 3 = 9 concurrent jobs`
- linux-shell-scripting (1-80 of 866): `fork(): ~0.1-1ms (copy-on-write, fast)`
- testing-core (1-80 of 808): `Statement coverage:    covered_statements / total_statements`
- technical-writing-core (1-100 of 298): `FKGL = 0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59`
- procedural-documentation-core (1-100 of 321): `A **migration guide** is a specialized, higher-stakes variant for a ONE-TIME, often irreversible transition`

Not read: jenkins-pipeline, docker, devsecops-core, powershell-scripting (optional), requirements-traceability-core (optional), and the remainder of every file above (including all M1-M6 math). No readability formula (FKGL/Fog) was computed; the docs review below is about correctness of commands and claims, not grade level.

## 1. Verdict

NOT ready to tag. No CRITICAL defect, but 4 HIGH items must be closed or consciously accepted first. The code, native layer and wheel build are sound; the gaps are in release automation guardrails, a CHANGELOG that is not in a taggable state, and documentation that tells users commands that do not exist.

## 2. What was run (all in the scratchpad, none in the repo tree)

Base dir `/tmp/claude-0/-home-user/9f5d9574-39ea-55de-8475-bf2e506385a2/scratchpad/r10/` (git clone of HEAD = what a clean runner sees; fresh venv Python 3.13.17, gcc 13.3.0).
- `pip install -e ".[dev]"`: OK (40 s; resolved openvino 2026.x from PyPI).
- `python src/arc_runner/native/build_native.py` (-Werror): OK, 21,992-byte .so + sidecar.
- `pytest -m "not hardware"`: **878 passed, 8 skipped, 23 deselected** (45 s). Skips: Level Zero driver absent (6), Windows-only `add_dll_directory` (1), no QSV device (1). All 8 are environment skips, none counted as passed.
- `check_ring_model.py 2 3 2 2`, `validate_bench_json.py` (3 files, 0 invalid), `render_bench_docs.py --check`, `check_docs_sync.py` (0 problems), `check_delivery_docs.py` (0 problems): all OK, i.e. every step of ci.yml `linux` passes on a clean Linux runner at Python 3.13.
- NOT run: Python 3.10 leg (only 3.13 available here), the Windows job, the mermaid job (needs npm), the manual `redfirst` job, Docker build (docker CLI present, daemon absent).
- `pytest -m hardware` on this Xeon host: 15 passed, 7 skipped, **1 failed** (`tests/integration/test_real_hardware.py::test_numerical_stability_cpu_vs_gpu_vs_npu`, OpenVINO RuntimeError at core.cpp:117). Suspected cause: the test requests a device that is absent instead of skipping (not traced further). Not a CI gate (CI excludes hardware), but CLAUDE.md wants unavailable hardware reported as "not run", and this reports a failure. LOW.
- Artifact check: a copy of the tree through `git archive` fails `test_repository_passes_docs_sync_and_delivery_checks` because it has no `.git`; in a real clone it passes. Expected, not a defect.

## 3. Version consistency

Verified: `VERSION` = `0.2.0`; `src/arc_runner/_version.py:3` = `0.2.0`; `pyproject.toml:7` = `0.2.0`; CHANGELOG top released section `[0.2.0] - 2026-10-29`; `arc-runner --version` prints `arc-runner 0.2.0`; built wheel/sdist are 0.2.0. `scripts/check_docs_sync.py` enforces the four and passes.

Findings:
- **HIGH R10-1. CHANGELOG is not in a taggable state.** `CHANGELOG.md:7` has `## [UNRELEASED]` ABOVE `## [0.2.0]` (line 25) and it holds substantial feature work (Sysman power/frequency, schema-v1 workload/metrics, pin_resident_layers, USM staging, KV quality gate, Added entries for 5 runners). Tagging 0.2.0 now ships code whose notes sit under "UNRELEASED". Decide: fold UNRELEASED into 0.2.0 (nothing is published, so allowed) or cut 0.3.0 for the Added items (SemVer MINOR, `release-versioning-management-core` s2.1). The checker skips UNRELEASED by design (`check_docs_sync.py` "top released"), so CI will not catch it.
- **MED R10-2. 0.2.0 section is stale and self-contradicting.** `CHANGELOG.md:27` says "Not published yet ... items under 'Needs a real machine' are not done"; lines 56-57 list "Rebuild and commit the Windows DLL, run probe_hardware, run the benchmark matrix". The DLL was rebuilt (commit 9cfb3ab), `probe.json` and 3 result JSONs exist (`docs/benchmarks/results/`). Still open per RESUME.md: TLC run, full `pytest -m hardware` pass, regression benchmark. Rewrite the section to the true remaining list before tagging.
- LOW R10-3. Format drift from Keep a Changelog: `[UNRELEASED]` (spec: `[Unreleased]`), no compare-link footer, `[0.1.0]` has no date, an orphan bullet after a blank line (line 22), and a 0.2.0 entry that bundles breaking-looking changes (u8 KV default off, single `generate()` call form, shrunken sync timeout) with no "Breaking/Behavior change" marker. Pre-1.0, so MINOR is legitimate, but the 0.y.z notice (skill s1.3) should be explicit.
- LOW R10-4. Version is stated in 4 hand-edited places (NFR-103 style duplication); `dynamic = ["version"]` reading `_version.py` would make it one. Optional.
- **No git tag exists** (`git tag` is empty), including `v0.1.0`. README badge/roadmap say v0.1.0 "Wheel Available" and `pip install arc-runner` is "Delivered". `https://pypi.org/pypi/arc-runner/json` returned **HTTP 404** during this review, i.e. the name is not on PyPI (or not reachable via the proxy; the 404 is from pypi.org). So the first-ever publish is 0.2.0, README Option 1 does not work today, and the PyPI project/trusted publisher must be created first (see s9).

## 4. Workflows

### 4.1 Triggers, secrets, permissions (verified by reading)
| Workflow | Triggers | Secrets | Permissions |
|:--|:--|:--|:--|
| ci.yml | push main, every PR, dispatch | none | `contents: read` (good) |
| docker-build.yml | release published, **push to main on Dockerfile/entrypoint/compose/src/**/pyproject paths**, dispatch | `GITHUB_TOKEN` only (GHCR) | `contents: read, packages: write` at workflow level |
| pypi-publish.yml | release published, dispatch (`dry_run` input) | none (OIDC trusted publishing) | `contents: read, id-token: write` at workflow level |

- **HIGH R10-5. docker-build is not tag-only.** `docker-build.yml:6-13`: every merge to main touching `src/**` builds and pushes `:latest` and `:sha-<short>` to GHCR (`push: ${{ github.event_name != 'pull_request' }}` at line 54 is always true because no PR trigger exists). Scenario: merging an unrelated `src/` fix on main publishes an image labelled `latest` with no version, no test run, no gate on CI. Also `workflow_dispatch` from any ref pushes. Fix: restrict push to tags `v*` (+ dispatch dry-run), or keep branch builds but never tag `latest` from them.
- **HIGH R10-6. pypi-publish has no release gate.** On `release: published` it builds and uploads immediately: no `needs:` on CI, no test run, no check that the tag equals `VERSION`, no `environment:` with required reviewers (so no approval pause). Scenario: a release created from the wrong commit or with tag `v0.2.1` while the tree says 0.2.0 uploads 0.2.0 (or fails after partial work; PyPI uploads are irreversible and a version number cannot be reused). Fix: add `environment: pypi` with a required reviewer, a step `test "$GITHUB_REF_NAME" = "v$(cat VERSION)"`, and run tests before publish.
- MED R10-7. `id-token: write` is workflow-wide (`pypi-publish.yml:14-16`), so the build steps (pip install of build/twine, arbitrary build backend code) also hold an OIDC-capable token. Split into `build` (contents: read) and `publish` (id-token: write) jobs, publish downloading the artifact. The skill-recommended and PyPA-recommended layout.
- MED R10-8. `workflow_dispatch` input `dry_run` is dead: the publish step already requires `github.event_name == 'release'` (line 46), so dispatch can never upload regardless of the input (input default is the string `'false'` on a boolean). Harmless but misleading; also on a `release` event `inputs.dry_run` is empty so `!inputs.dry_run` is true (works).
- LOW R10-9. `pypa/gh-action-pypi-publish@release/v1` is a moving branch.

### 4.2 Action pinning
All third-party actions are pinned by mutable tag, none by SHA: checkout@v4, setup-python@v5, setup-node@v4, upload-artifact@v4, docker/setup-buildx@v3, login@v3, metadata@v5, build-push@v5, pypa `@release/v1`. `release-engineering-specialist` rule: pin release-critical actions. MED for the two release workflows (publish + GHCR login hold write credentials), LOW for ci.yml. Runners are floating `ubuntu-latest`/`windows-latest`; Dockerfile base is `ubuntu:24.04` without digest. `npm install @mermaid-js/mermaid-cli@10.9.1` is pinned (good); `pip install build twine` in publish is unpinned.

### 4.3 Matrix, caching, hygiene
- Matrix: Linux 3.10 and 3.13 only; `pyproject` classifiers claim 3.11 and 3.12 (untested in CI, though the publish job builds on 3.12). Windows: 3.13 only. LOW/MED: add 3.11/3.12 or drop the classifiers.
- No caching anywhere except Docker `type=gha`: no `cache: pip` on setup-python (the `[dev]` install pulls openvino, ~40 s here, more on a cold runner), no npm cache. No `concurrency:` group (PR pushes pile up), no `timeout-minutes` on any job (default 360). LOW.
- ruff and mypy are dev dependencies but never run in CI; `ruff check src scripts` with no config reports 1217 findings (default rule set), so adding it needs a config first. LOW.
- The Linux job installs `gcc-mingw-w64-x86-64`, used by two mingw compile tests (`tests/test_native_bounds.py:297`, `tests/test_level_zero_bridge.py:198`): those compile only, they do not produce or compare the shipped DLL.
- `windows` job: `gcc --version` assumes MinGW gcc is on the windows-latest image. Not verified here (cannot run Windows). If absent the job fails at step 3.
- MED R10-10. No CI check ties the committed DLL to the sources: the Windows job rebuilds the DLL in the workspace and uploads it as an artifact but never compares it with the committed binary or its sidecar. Suggest a cheap test: sidecar sha256 equals DLL bytes, `abi_version` equals `ARC_NATIVE_ABI_VERSION`, and `git diff --quiet <sidecar git_sha> HEAD -- src/arc_runner/native/*.c *.h`.
- ci.yml `redfirst` job (manual) depends on baseline commit `c7312e0` and three test files existing; not run here.
- Docker image is never built in CI before it is pushed (Dockerfile changes are first exercised by the push itself). `scripts/verify_container.sh` is not called by any workflow, and it only greps the Dockerfile; its step 4 silently `chmod +x`es the entrypoint (it modified the tracked file mode in my clone: `M docker-entrypoint.sh`), so it repairs instead of failing. LOW.

## 5. Packaging: pyproject, MANIFEST.in, build scripts

Verified by running `scripts/build_wheel.py` and `publish_pypi.py --dry-run` in the clone: wheel `arc_runner-0.2.0-py3-none-any.whl` 339,335 bytes + sdist build, 104 wheel entries, entry points `arc-runner` and `intel-arc-models-runner`, RECORD hashes pass, `twine check` PASSED on both. `publish_pypi.py` defaults to dry-run and only uploads with `--upload` (it is not used by the workflow, which uses the PyPA action).
- **HIGH R10-11. No LICENSE file exists in the repository** (`git ls-files | grep LICENSE` empty; `ls LICENSE*` fails). `MANIFEST.in:2` says `include LICENSE`, `pyproject.toml:11` says `license = { text = "Apache-2.0" }`, README links the OSI page. Neither wheel nor sdist contains a license text (checked both). Apache-2.0 requires the text to travel with redistributed copies, and PyPI will show no license file. Scenario: first public upload ships without the license. Fix: commit LICENSE (maintainer decision: holder/year) and use `license-files`.
- MED R10-12. `requires = ["setuptools>=61.0", "wheel"]` is unbounded; fine today, but builds are not reproducible across setuptools majors. LOW.
- LOW R10-13. `scripts/build_wheel.py:63` returns `wheels[0]`, and `clean_build_artifacts` deletes `dist/` and `build/` and `*.egg-info` of the working tree (destructive when run locally; fine in CI). The local `dist/` in the repo holds a STALE `arc_runner-0.2.0-*` built at 08:39 containing the OLD 49,530-byte DLL with no DLL sidecar (not the committed 54,979-byte one). `dist/` is git-ignored, so it cannot be shipped by CI, but a maintainer running `publish_pypi.py --upload` locally without rebuilding would upload it. Delete it before release.
- Wheel tag is `py3-none-any` while it carries a Windows PE DLL (and possibly a Linux .so, see s6). pip installs it on every platform; the loader keys on `sys.platform`, so it is inert elsewhere. Acceptable, but undocumented in the metadata. LOW.

## 6. Native binaries: provenance, reproducibility, wheel, loader

Windows DLL `src/arc_runner/native/libarc_native_sys.dll`:
- Verified: `sha256sum` = `68d64d75901ced1841f143823d4d6d5881c62189f1b8074cdbdb4d61eb38cbb4` = the `sha256` in `libarc_native_sys.dll.build.json`. Sidecar says ABI 3, compiler `gcc.EXE (x86_64-posix-seh-rev0, Built by MinGW-Builds project) 15.2.0`, flags `-O2 -std=c17 -Wall -Wextra -Werror -Wpedantic` (+ `-shared -Wl,--no-undefined`), host Windows-11-10.0.26220, `git_sha` `0fe6981...`. The `strings` of the DLL show the same GCC 15.2.0 banner, consistent with the sidecar.
- Verified: both files were added together in commit 9cfb3ab, whose parent is 0fe6981 (= sidecar `git_sha`), and `git diff 0fe6981 HEAD -- src/arc_runner/native/*.c *.h` is empty. So the C sources used equal the sources at HEAD. Not verifiable: whether the maintainer's tree was clean when building (the sidecar has no "dirty" flag). LOW R10-14: add `dirty` to the sidecar.
- **Not bit-reproducible, and cannot be reproduced here.** With the Ubuntu cross compiler (`x86_64-w64-mingw32-gcc 13-win32`, a different toolchain and size, 104,389 bytes) two consecutive builds from identical objects gave different SHA-256 (308afd84..., 889f6d37...), and still differed with `-Wl,--no-insert-timestamp` (a00540fa..., 0dee9980...). Cause not identified (suspected PE header timestamp/linker nondeterminism). Consequence: nobody can independently confirm the committed DLL from source; trust rests on the maintainer plus the sidecar. MED R10-15 for a first public release shipping a committed binary. Mitigation: build the release DLL in CI (the Windows job already does) and publish its sidecar, or at least record the Windows job's artifact hash in the release notes.
- Linux `.so`: **reproducible**. Two builds in different directories gave identical SHA-256 `63e783f4...b0f4` (21,992 bytes), equal to the sidecar of the local ignored file.
- **Wheel contents.** Wheel built from a clean clone after running `build_native.py` contains `.dll`, `.dll.build.json`, `.so`, `.so.build.json`, and the DLL's sha equals its sidecar (verified in the zip). BUT `*.so` and `*.so.build.json` are git-ignored (`.gitignore:128-129`; `git ls-files '*.so'` is empty) and the publish job runs no `build_native.py`, so the PyPI wheel built by the workflow will contain the Windows DLL and sidecar and NO Linux .so (derived from those facts, not run in Actions). Linux users therefore get the pure-Python path, which `docs/deployment.md` s3b states for containers. Decide explicitly: either document that the native layer is Windows-only in the PyPI wheel, or add a Linux build step (or manylinux wheels). MED R10-16.
- Docker: the `Dockerfile` builder stage does `pip install .` with no `build_native.py`, so the image has the Windows DLL (unused) and no .so: pure-Python path (consistent with the docs).
- **Loader behaviour (`bridge.py:135-195`, exercised by me on a copy in the clone):** (A) pristine .so + sidecar: loads. (B) .so changed by one byte, sidecar unchanged: `WARNING Native library hash mismatch`, returns None, falls back to Python. (C) changed .so with sidecar deleted: **loads** (the check passes when no sidecar exists, `bridge.py:145-146`); only the ABI-version gate (`arc_native_abi_version() != 3`, lines 181-187) remains, and an old DLL without that export raises AttributeError, caught, fallback. (D) pristine .so without sidecar: loads. So a stale same-ABI binary without its sidecar loads silently. MED R10-17: acceptable for dev trees, but in a wheel the sidecar is the only integrity signal; consider refusing a missing sidecar when the package is installed (not editable). The sidecar is next to the binary, so it protects against accidents, not against an attacker who can write the directory (stated, not a defect).
- `ARC_NATIVE_DISABLE=1` path is covered by tests (suite passes); not separately exercised by me.

## 7. Dockerfile, compose, entrypoint

Verified by reading + `docker compose config` + `scripts/verify_container.sh` (all 4 steps PASSED). Image NOT built (no Docker daemon in this sandbox), so runtime behaviour is unverified.
- Multi-stage, non-root `appuser` (uid 1000), HEALTHCHECK `curl /` which exists (`routes_ollama.py:77`). Good.
- MED R10-18: `curl ... intel-graphics.key | gpg --dearmor` (`Dockerfile:68`) trusts whatever the URL serves, with no fingerprint pin; apt packages are unpinned, so rebuilding the same tag gives different images. Base `ubuntu:24.04` not digest-pinned. No `--provenance`/SBOM in the build step.
- LOW: `docker-entrypoint.sh` in git is mode 644 (the Dockerfile chmods it, so the image is fine). `docker-compose.yml` mounts `./models:/models:ro` and tags `intel-arc-models-runner:latest`.
- LOW: the docker-build workflow tags `type=semver` (needs tag `vX.Y.Z`) but nothing documents the tag format; the first tag decides the image tag.

## 8. Documentation review (README quick start, docs/deployment.md)

Commands tried in the clone venv with `HOME` redirected:
- `arc-runner --version` -> `arc-runner 0.2.0` OK; `intel-arc-models-runner --version` OK; `arc-runner devices` OK (reports CPU-only here, correct); `python -m arc_runner.cli.main devices` (CONTRIBUTING) is the same entry.
- **HIGH R10-19. `docs/deployment.md:45` `arc-runner devices discover` fails**: `error: unrecognized arguments: discover`.
- **HIGH R10-20. `docs/deployment.md:48` `arc-runner server start --host 0.0.0.0 --port 8000` fails**: `invalid choice: 'server'`. The real command is `arc-runner serve --host ... --port ...` (README line ~157 has it right). These two are the guide's verification steps for the PyPI install, so a user following it fails at step 2.
- MED R10-21. `docs/deployment.md:150,173` K8s liveness probe and "Health Auditing" row use `/v1/health`; no such route exists (grep of `src/arc_runner/server`: only `/` in `routes_ollama.py:77` and `/api/devices`). A pod using that probe is killed by the kubelet (404). The Dockerfile HEALTHCHECK correctly uses `/`. Also the K8s example pins `:latest`.
- MED R10-22. README Option 1 (`pip install arc-runner`) and the badge `Release v0.1.0 (Wheel Available)` (README line 8) describe a PyPI release that does not exist (HTTP 404 above, no tags). Option 2 pulls `ghcr.io/.../intel-arc-models-runner:latest`, which exists only after docker-build has run (not verified; GHCR not queried).
- MED R10-23. Stale test counts, all disagreeing with the measured 878 passed / 8 skipped / 23 deselected: README badge "569 Passing (100%)" (line 7), README line 29 "741 ... 733 passed", README line 373-383 "533 Tests / 33 Modules ... # -> 533 passed in ~24.5s", `docs/testing.md:9` 741/733, `docs/testing.md:13,53` 569/36 modules, `docs/roadmap.md:22` 533. Per the honesty rule, replace with one measured number (or none) taken from a run, and make `check_docs_sync.py` check it.
- MED R10-24. README "Building the Native Library" (line ~307): "The prebuilt `libarc_native_sys.dll` in the repository history predates the current native ABI (version 3) and is refused" is now false: the committed DLL is ABI 3 with a matching sidecar (commit 9cfb3ab). The roadmap also names `arc_runner-0.1.0-py3-none-any.whl` (README ~394, roadmap line 115).
- LOW: README Option 3 / `install.ps1` install with `-e .` (editable) from a checkout even for "users"; fine but not the PyPI path. README Option 4 `pip install -e ".[dev]"` works (verified).
- Mode coherence (Diataxis): README mixes tutorial, reference and marketing in one file (25 KB) and says "Proofs M121-M134" style claims next to measured-honesty notes; not rated, no scoring done.

## 9. Release checklist (ordered; nothing below has been done)

Legend: A = automated by existing workflows/scripts, M = maintainer-only action, AP = needs explicit approval (CLAUDE.md: tagging, merging and publishing need the maintainer's approval).

1. M. Decide CHANGELOG policy (R10-1/2): fold UNRELEASED into 0.2.0 or version 0.3.0; rewrite "Needs a real machine" to the true open list; fix format drift. Then keep VERSION, `_version.py`, pyproject and CHANGELOG top released entry equal.
2. M. Commit a LICENSE file (R10-11) and re-run `scripts/build_wheel.py`; confirm the license is inside wheel and sdist.
3. M. Fix `docs/deployment.md` (`serve`, no `devices discover`, probe path) and the stale README/docs test counts and DLL text (R10-19..24); run `python scripts/check_docs_sync.py` and `python scripts/check_delivery_docs.py` (A).
4. M. Harden workflows (R10-5..10): docker push on tags only, publish gated by `environment:` + tag==VERSION check + tests, split `id-token` to the publish job, pin actions by SHA.
5. M, hardware. Close the remaining real-machine items from RESUME.md: `pytest -m hardware` on the Arc box (the one failure above must be understood), TLC on `docs/formal/RingSlots.tla`, regression benchmark. Record results as files; no number from memory.
6. A. CI green on the release commit: Linux 3.10 + 3.13, Windows, diagrams (the Linux 3.13 path was reproduced here; 3.10, Windows and diagrams were not).
7. M. Rebuild the Windows DLL only if any `native/*.c|h` changed since 9cfb3ab (none have); otherwise record the Windows CI artifact hash next to the committed sidecar (R10-15). Bump ABI only if an export changes.
8. M. Delete stale local `dist/` and `build/` (R10-13); run `python scripts/build_wheel.py` and `python scripts/publish_pypi.py --dry-run` (A, local, safe).
9. AP + M. Create the PyPI project / pending trusted publisher for `arc-runner` -> repo `techdeveloper-org/intel-arc-models-runner`, workflow `pypi-publish.yml`, (environment `pypi`) (R10 s3: name currently 404). Optionally rehearse on TestPyPI first (`publish_pypi.py --repo testpypi` needs a token; the workflow has no TestPyPI path).
10. AP. Merge the work branch to main (not done by this review).
11. AP. Create annotated, signed tag `v0.2.0` on the merged commit (`git tag -a -s v0.2.0`), never re-push or move it (release-engineering rule). No tag exists today.
12. AP. Publish the GitHub Release for `v0.2.0` with the curated notes (separate from the changelog). This single action triggers BOTH `pypi-publish.yml` (upload, irreversible, a version number can never be reused) and `docker-build.yml` (pushes `0.2.0`, sha tag and possibly `latest`).
13. A. Watch the two workflow runs; verify `pip install arc-runner==0.2.0` in a fresh venv, `arc-runner --version`, and `docker run ... --version`; check the wheel's DLL sha against the sidecar.
14. M. If something is wrong after step 12: do NOT delete or reuse the tag or PyPI version; yank the PyPI release and ship 0.2.1. Rollback is a deployment action, not a version action.
15. M. Post-release: move to `[Unreleased]` template, add compare links, update README badge/version text from measured data.

## 10. Summary table of defects

| ID | Sev | One line |
|:--|:--|:--|
| R10-1 | HIGH | CHANGELOG `[UNRELEASED]` above `[0.2.0]` holds shipped-looking features |
| R10-5 | HIGH | docker-build pushes `latest`/sha images on every main merge touching src/ |
| R10-6 | HIGH | pypi-publish has no tag==VERSION check, no test gate, no approval environment |
| R10-11 | HIGH | No LICENSE file anywhere; wheel and sdist ship without license text |
| R10-19/20 | HIGH | deployment.md verification commands (`devices discover`, `server start`) do not exist |
| R10-2, 7, 10, 15, 16, 17, 18, 21, 22, 23, 24 | MED | stale CHANGELOG notes; workflow-wide id-token; no DLL-vs-source check; DLL not reproducible; no Linux .so in the wheel; missing sidecar accepted; unpinned Intel key; `/v1/health` route absent; PyPI 404; stale counts; stale DLL text |
| R10-3, 4, 8, 9, 12, 13, 14 | LOW | format drift, version duplication, dead `dry_run`, floating `release/v1`, unbounded setuptools, stale local dist, no dirty flag |

No defect found in: the 4-place version agreement, the DLL sidecar hash, wheel RECORD hashes, `twine check`, CI `permissions:` of ci.yml, Linux CI reproduction (878 passed), `.so` reproducibility.

## Revision history
| Version | Date | Author | What changed | Why | Commit |
|:--|:--|:--|:--|:--|:--|
| 1.0.0 | 2026-10-29 | library agent via main session | first version | independent review wave A | pending |
