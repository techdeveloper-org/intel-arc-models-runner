---
doc_id: DOC-DECISION-LOG
title: Decision log
stage: all
version: 1.1.0
status: draft
created: 2026-10-29
updated: 2026-10-29
authors: [main session]
reviewers: [user]
---

# Decision log

Append-only. A changed decision gets a new entry that names the one it supersedes; earlier rows are never edited.

| DL | Decision | Why | Alternative rejected | Stage |
|:--|:--|:--|:--|:--|
| DL-001 | Gemini plan stays the base spec | user instruction; most work done and tests pass | restart from scratch | S00 |
| DL-002 | Route to silicon-architecture, math expert for derivations | discovery tool and rules | generic systems-programming route | S00 |
| DL-003 | Library agents read all their skills with quote proof | user rule | sampling skills | S00 (see 023, 039 for what applied in practice) |
| DL-004 | Do not build non-temporal-load L3 bypass | ineffective on write-back pages; no CPU copy exists | implement as planned | S00 |
| DL-005 | Host USM is the default residency | iGPU shares system memory; device USM adds a copy | device or shared USM | S00 |
| DL-006 | Sysman returns None plus a reason, never fabricated values | machine reports no power domains / memory modules | keep 22.0 W and RAM-as-VRAM | S-A |
| DL-007 | Sealed-after-registration flag (release/acquire) | table immutable after load; zero reader cost | rwlock | S-B |
| DL-008 | Direct-mapped ring slots (layer % slots) | cyclic scan gives LRU a 0% hit rate | LRU | S-C |
| DL-009 | Remove global -mavx2 | no AVX2 code exists; SIGILL risk | runtime-dispatched AVX2 | S-B |
| DL-010 | Commit a prebuilt DLL with ABI/hash safeguards | user decision | gitignore and CI build | S00 (see 040) |
| DL-011 | Release and tag only on the maintainer's explicit OK | PyPI and GHCR publishing is irreversible | auto-release | S00 |
| DL-012 | Keep tracked uml/ and drawio/ | repo precedent | untrack | S00 |
| DL-013 | Leave root CONTRIBUTING.md | pre-existing, out of scope | move to docs/ | S00 |
| DL-014 | Keep float telemetry fields and add validity flags | avoid breaking consumers | Optional fields | S-A |
| DL-015 | Tests written by a different author than the implementer | avoid Goodhart on weak tests | implementers write tests | S00 (not achieved in cloud sessions; mutation checks used instead) |
| DL-016 | Agents write their own report files | token cost and audit trail | main session transcribes | S00 |
| DL-017 | Version 0.1.0 to 0.2.0 | new feature and behaviour fixes | patch bump | S-F |
| DL-018 | Restate or remove the unverified theorem and gain claims | math re-checked | keep | S-E |
| DL-019 | LEAN track skips optional BA/PM and quality stages | proportionality | FULL track | S00 |
| DL-020 | Regenerate all 13 UML/drawio types once | new package | selective regeneration | S-F |
| DL-021 | Rules for unrelated languages and frameworks marked not applicable | not touched | re-audit each session | S00 |
| DL-022 | LEAN is the track | token cost vs scope | FULL by default | S00 |
| DL-023 | Library agents and MCPs optional; absence logged | cloud session lacked the library at that time | pretend they ran | S00 (superseded in part by section 22 of the plan) |
| DL-024 | Work in slices S-A..S-F, honesty first | smallest harm-first units | one big wave | S00 |
| DL-025 | POSIX branch fixed with the same helper; sanitizer harness is the memory-safety proof | runnable without Windows | rely on review only | S-B |
| DL-026 | One probe script for hardware facts; claims stay unverified until its JSON is committed | hardware not reachable from the cloud | guess values | S-E |
| DL-027 | Canonical DLL build is a CI cross-compile | stale-binary risk | commit DLL separately | S00 (see 040) |
| DL-028 | Baseline for red-first is c7312e0 | Gemini work was already committed | extra snapshot commit | S00 |
| DL-029 | Benchmark scope: all domains plus native micro, roofline inputs, regression | requested coverage | LLM-only | S-E |
| DL-030 | One statistics implementation, both runners delegate | avoid drifting CI code | duplicate implementations | S-E |
| DL-031 | Results are schema-validated JSON; doc tables generated from it | root cause of hand-edited numbers | manual edits | S-E |
| DL-032 | A770 and other unowned hardware are projected or community-submitted | not in the testbed | leave as measured | S-E |
| DL-033 | Claim thresholds registered in METHOD.md before runs | prevents post-hoc tuning | decide after seeing data | S-E |
| DL-034 | CPU-only/Linux micro-benchmark subset as CI smoke, no claims | catch harness breakage | local only | S-E |
| DL-035 | POSIX native branch is best-effort; Windows x64 is the supported DLL platform | users and hardware are Windows | drop POSIX; claim full support | S-F |
| DL-036 | Speculative acceptance and speed-up are reported as 0 and 1.0 until measured | no verified source | keep constants | S-A |
| DL-037 | Shared telemetry builders in engine/telemetry.py | one place enforces measured-or-unavailable | per-engine code | S-A |
| DL-038 | Heavy-model benchmark rows carry model_kind synthetic_container until real weights run | they describe a toy model | leave unlabelled | S-A |
| DL-039 | India regulatory skills, integration-testing agent not applied; threat model scoped to GGUF parser and DLL loader; C SAST via clang-tidy, analyzer, sanitizers | no Indian data, local runner, no Semgrep in the cloud image | apply all agent defaults | S-B |
| DL-040 | Windows DLL is not rebuilt in the cloud; the maintainer rebuilds locally and commits it with its sidecar | a cross-compiled binary cannot be executed here; the ABI check refuses the old DLL | commit an untested binary | S-B |
| DL-041 | get_tensor_buffer seals the table on first use | keeps call sites working and makes the table immutable before concurrent reads | require explicit seal | S-B |
| DL-042 | GGUF parser caps dimensions at 4 and validates declared lengths | hostile input; matches GGML_MAX_DIMS | trust the file | S-B |
| DL-043 | Sidecar file name is the full library file name plus .build.json | .dll and .so share a stem | shared stem name | S-B |
| DL-044 | No pin counts in the ring buffer; arrays keep the mapping alive | colliding layers could make a thread wait for itself | blocking pins with timeout | S-C |
| DL-045 | Slot is always layer % slots; prefetch ignores its slot argument | deterministic, no timestamps | LRU | S-C |
| DL-046 | BF16 raw uint16, quantized raw uint8 in both paths | old fallback misread BF16 and inflated quantized data to float32 | convert | S-C |
| DL-047 | Row-major shape is the reversed GGUF dimension list | ne[0] is the fastest dimension | use dimensions as stored | S-C |
| DL-048 | close() defers the native unmap while views are alive | prevent use after unmap | unmap immediately | S-C |
| DL-049 | ARC_NATIVE_ABI_VERSION 3 | Level Zero entry points changed | keep 2 | S-D |
| DL-050 | u8 KV cache is opt-in | no quality gate exists | default on | S-D |
| DL-051 | ARC_ZE_LOADER_PATH and ARC_ZE_SYNC_TIMEOUT_MS are supported overrides | testability without hardware; clamp below TDR | no overrides | S-D |
| DL-052 | Prompt plus completion over the window raises | caller learns the request cannot be honoured | clamp silently | S-D |
| DL-053 | Version lives in src/arc_runner/_version.py | importing it from the package root created a circular import | duplicate strings | S-F |
| DL-054 | Diagrams are extended additively with a generated native-layer section instead of regenerating all earlier content | a repository-wide hand-built model was not rebuilt; the generated parts are tied to the code by ast and checked for drift | full regeneration of all earlier content | S-F |
| DL-055 | Stale tests that failed before this work are corrected or skipped with a reason rather than left red | CI must be able to go green; hardware-dependent behaviour is reported as skipped, never as passed | leave them failing | S-F |
| DL-056 | Non-LLM results extend schema v1 additively (workload + metrics) | existing files and tooling stay valid | schema v2 | S-G |
| DL-057 | Power is measured with the Sysman energy counter or is None | wattage constants are fabrication | assumed TDP | S-G |
| DL-058 | The fixture audit script is deleted, not patched | there was no measurement to keep | keep with a banner | S-G |
| DL-059 | u8 KV stays opt-in whatever the gate says | a default change is a human decision | auto-enable on pass | S-G |
| DL-060 | Mermaid validity is checked by rendering, in CI | structural checks cannot see syntax errors | none | S-G |

## Revision history
| Version | Date | Author | What changed | Why | Commit |
|:--|:--|:--|:--|:--|:--|
| 1.0.0 | 2026-10-29 | main session | DL-001..DL-055 consolidated from the plan and the stage records | create the log | pending |
| 1.1.0 | 2026-10-29 | main session | DL-056..DL-060 | S-G decisions | pending |
