---
doc_id: DOC-INDEX
title: Delivery document index
stage: all
version: 1.24.0
status: draft
created: 2026-10-29
updated: 2026-10-30
authors: [main session, CLI session]
reviewers: [user]
---

# Delivery document index

Every document under `docs/delivery/` is listed with its current version; `python scripts/check_delivery_docs.py` keeps this table in sync with the files.

| doc_id | Title | Version | Status | Stage | Path |
|:--|:--|:--|:--|:--|:--|
| DOC-DECISION-LOG | Decision log | 1.13.0 | draft | all | DECISION_LOG.md |
| DOC-RESUME | Resume checkpoint | 1.14.0 | draft | all | RESUME.md |
| DOC-TRACEABILITY | Finding to requirement to code to test traceability | 1.1.0 | draft | all | TRACEABILITY.md |
| DOC-SA-STAGE | Slice S-A - remove fabricated values | 1.0.0 | draft | S-A | stages/S-A-honesty/STAGE.md |
| DOC-SB-STAGE | Slice S-B - native memory safety | 1.0.0 | draft | S-B | stages/S-B-memory-safety/STAGE.md |
| DOC-SC-STAGE | Slice S-C - ring buffer, dtype, and view lifetime | 1.0.0 | draft | S-C | stages/S-C-ring-and-lifetime/STAGE.md |
| DOC-SD-STAGE | Slice S-D - engine context window and Level Zero bridge | 1.0.0 | draft | S-D | stages/S-D-engine-and-level-zero/STAGE.md |
| DOC-SE-STAGE | Slice S-E - benchmark protocol, statistics, and provenance | 1.0.0 | draft | S-E | stages/S-E-benchmarks/STAGE.md |
| DOC-SF-STAGE | Slice S-F - documentation, version, CI, and checks | 1.0.0 | draft | S-F | stages/S-F-docs-and-release-prep/STAGE.md |
| DOC-SG-STAGE | Slice S-G - remaining technical items | 1.9.0 | draft | S-G | stages/S-G-remaining-items/STAGE.md |
| DOC-R01-REVIEW | R01 - Independent review of the native C layer and ctypes bridge | 1.0.0 | draft | S-G | reviews/R01_c_native_review.md |
| DOC-R02-REVIEW | R02 - STRIDE threat model with attack trees | 1.0.0 | draft | S-G | reviews/R02_threat_model.md |
| DOC-R03-REVIEW | R03 SAST review: src/arc_runner (Python) and src/arc_runner/native (C) | 1.0.0 | draft | S-G | reviews/R03_sast.md |
| DOC-R04-REVIEW | R04 - Secrets scan and dependency audit | 1.0.0 | draft | S-G | reviews/R04_secrets_and_dependencies.md |
| DOC-R05-REVIEW | R05 - TLA+ review of RingSlots against the ring-buffer weight streamer | 1.0.0 | draft | S-G | reviews/R05_tla_review.md |
| DOC-R06-REVIEW | R06 Independent test authorship (integration-testing-engineer + testing-mathematics-expert) | 1.0.0 | draft | S-G | reviews/R06_independent_tests.md |
| DOC-R08-REVIEW | R08 Benchmark methodology review | 1.0.0 | draft | S-G | reviews/R08_benchmark_methodology.md |
| DOC-R09-REVIEW | R09 - Level Zero / Sysman / OpenVINO device handling: second opinion | 1.0.0 | draft | S-G | reviews/R09_gpu_driver_second_opinion.md |
| DOC-R10-REVIEW | R10 Release-readiness review for v0.2.0 (nothing tagged, published or pushed) | 1.0.0 | draft | S-G | reviews/R10_release_readiness.md |
| DOC-R07-REVIEW | R07 Faithfulness audit of repository claims | 1.0.0 | draft | S-G | reviews/R07_faithfulness_audit.md |
| DOC-INDEX | Delivery document index | 1.24.0 | draft | all | INDEX.md |

## Revision history
| Version | Date | Author | What changed | Why | Commit |
|:--|:--|:--|:--|:--|:--|
| 1.0.0 | 2026-10-29 | main session | first version | index of the delivery documents | 93868c6 |
| 1.1.0 | 2026-10-29 | main session | added S-G stage | remaining technical items | 8605588 |
| 1.2.0 | 2026-10-29 | main session | traceability to 1.1.0 | plan test-list audit | 0fe6981 |
| 1.3.0 | 2026-10-29 | main session | S-G stage to 1.1.0 | E5, E6, audit | 0fe6981 |
| 1.4.0 | 2026-10-29 | main session | listed review reports R01-R10 | independent review wave A | a64e2ce |
| 1.5.0 | 2026-10-29 | main session | decision log and resume to 1.2.0 | performance programme, plan section 23 | d6043f3 |
| 1.6.0 | 2026-10-29 | main session | decision log and resume to 1.3.0 | plan section 23 v8.4, review resolution | 58ff19e |
| 1.7.0 | 2026-10-29 | main session | decision log and resume to 1.4.0 | plan section 23 v8.5, round-2 review resolution | d0744e3 |
| 1.8.0 | 2026-10-29 | main session | decision log and resume to 1.5.0 | plan section 23 v9.0, round-3 review resolution by simplification | ae428e4 |
| 1.9.0 | 2026-10-29 | main session | decision log and resume to 1.6.0 | plan section 23 v9.1, round-4 review resolution (fix only) | 3aa9f72 |
| 1.10.0 | 2026-10-29 | main session | decision log and resume to 1.7.0 | plan section 23 v9.2, round-5 review resolution (fix only) | f4db269 |
| 1.11.0 | 2026-10-29 | main session | decision log and resume to 1.8.0 | plan section 23 v9.3, round-6 review resolution (simplify and fix) | 3d95ee0 |
| 1.12.0 | 2026-10-29 | main session | decision log and resume to 1.9.0 | plan section 23 v9.4, round-7 review resolution (fix only) | 8cdf045 |
| 1.13.0 | 2026-10-29 | main session | decision log and resume to 1.10.0 | plan section 23 v9.5, round-8 review resolution (fix only) | a312ee1 |
| 1.14.0 | 2026-10-30 | main session | decision log and resume to 1.11.0 | plan section 23 v9.6, round-9 review resolution (fix only) | 1a4ccfe |
| 1.15.0 | 2026-10-30 | main session | decision log and resume to 1.12.0 | plan section 23 v9.7, round-10 review resolution (fix only) | 2abe58f |
| 1.16.0 | 2026-10-30 | main session | decision log and resume to 1.13.0 | plan section 23 v9.8, round-11 final-check resolution (fix only) | 35b2dc3 |
| 1.17.0 | 2026-10-30 | main session | resume to 1.14.0 | handoff for the CLI session: code since the plan, measured results, open decisions | next revision |
| 1.18.0 | 2026-10-30 | CLI session | S-G stage to 1.3.0 | WP-02 02a/02e/02f measurement support items | pending |
| 1.19.0 | 2026-10-30 | CLI session | S-G stage to 1.4.0 | WP-02 02b bench_arms.py paired gate runner | pending |
| 1.20.0 | 2026-10-30 | CLI session | S-G stage to 1.5.0 | WP-02 02c bench_native_micro --warmup/--seed flags | pending |
| 1.21.0 | 2026-10-30 | CLI session | S-G stage to 1.6.0 | WP-02 02d run_benchmark long_prompt parameter | pending |
| 1.22.0 | 2026-10-30 | CLI session | S-G stage to 1.7.0 | WP-17 17b PromptCapExceededError NPU prompt cap enforcement | pending |
| 1.23.0 | 2026-10-30 | CLI session | S-G stage to 1.8.0 | WP-05 engine_key, unload all engine kinds, _meta_lock | pending |
| 1.24.0 | 2026-10-30 | CLI session | S-G stage to 1.9.0 | WP-07 classify_device_error, DEVICE_LOST/OOM raise ModelLoadError | pending |
