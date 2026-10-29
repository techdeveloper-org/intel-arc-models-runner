---
doc_id: DOC-SF-STAGE
title: Slice S-F - documentation, version, CI, and checks
stage: S-F
version: 1.0.0
status: draft
created: 2026-10-29
updated: 2026-10-29
authors: [main session]
reviewers: [user]
inputs: [docs/plans/MASTER_PLAN_v4.md sections 15, 17.4, 21]
---

# Slice S-F: documentation, version, CI, and checks

Gate status: `cloud-green`. Release (tag, merge, publish) is not done and needs the maintainer's explicit approval.

## 1. Library use (honest record)
`technical-writer-agent`, `uml-from-code-engineer`, `mermaid-diagram-engineer`, `drawio-diagram-architect`, `devops-engineer`, `release-engineering-specialist` agent files and skills were not read; the work follows plan sections 15 and 17.

## 2. Changes
| Area | Change |
|:--|:--|
| Version | 0.2.0 in `VERSION`, `src/arc_runner/_version.py` (single source; importing from the package root is circular), `pyproject.toml`, CHANGELOG; the API app and the Ollama-compatible version endpoint follow it; the packaging test reads the version instead of hard-coding it |
| Packaging | native sources, libraries and sidecars included in the wheel and sdist |
| Documents | `CHANGELOG.md`, `SRS.md` (FR-001..FR-007, NFR-001..NFR-005), `CLAUDE.md`, `docs/delivery/` (RESUME, INDEX, DECISION_LOG DL-001..053, TRACEABILITY, stage records), architecture, testing, deployment and diagrams documentation |
| Diagrams | `scripts/generate_diagrams.py` adds a stamped "v0.2.0 native layer" section to all 13 `uml/*.md` and an extra page to all 13 `drawio/*.drawio`; class and package diagrams come from the source with `ast`, call-graph edges are verified against the Python AST and the C sources. Earlier diagram content is not regenerated (DL-054) |
| Checks | `scripts/check_docs_sync.py` (diagram pairs and stamps, class/package coverage, provenance notes, generated tables, relative links, version consistency, delivery documents) and `scripts/check_delivery_docs.py` |
| CI | `.github/workflows/ci.yml`: Linux (Python 3.10 and 3.13, native build, tests, ring model, benchmark files, docs sync), Windows (mingw build, tests, DLL artifact), manual red-first job against c7312e0 |
| Tests | the three tests that failed in the Linux environment before this work are fixed or skip with a stated reason: the Dockerfile health-probe contract now checks the probe the Dockerfile actually uses (`/`; no `/v1/health` route exists), the Level Zero bridge test skips without a driver and checks the device type, the ffmpeg direct-stream test skips when ffmpeg lists `h264_qsv` but no Quick Sync device works |

## 3. Results
741 non-hardware tests: 733 passed, 8 skipped, 0 failures. `check_docs_sync.py` and `check_delivery_docs.py` exit 0.

## 4. Not verified here
- Mermaid rendering: the generated blocks were not rendered with a Mermaid engine (none in the cloud image); the Draw.io files were checked as well-formed XML.
- The Windows CI job and the Linux 3.10 job have not run (no GitHub Actions run from this session); `.github/workflows/ci.yml` parses as YAML.
- No release was created.

## 5. Decisions
| DL | Decision | Why |
|:--|:--|:--|
| 054 | Diagrams are extended additively with a generated native-layer section instead of regenerating all earlier content | a repository-wide hand-built model was not rebuilt; the generated parts are tied to the code by `ast` and checked for drift |
| 055 | Stale tests that failed before this work are corrected or skipped with a reason rather than left red | CI must be able to go green; hardware-dependent behaviour is reported as skipped, never as passed |

## Revision history
| Version | Date | Author | What changed | Why | Commit |
|:--|:--|:--|:--|:--|:--|
| 1.0.0 | 2026-10-29 | main session | first version | S-F implementation | pending |
