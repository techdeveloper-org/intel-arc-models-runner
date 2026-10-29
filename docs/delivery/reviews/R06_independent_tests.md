---
doc_id: DOC-R06-REVIEW
title: R06 Independent test authorship (integration-testing-engineer + testing-mathematics-expert)
stage: S-G
version: 1.0.0
status: draft
created: 2026-10-29
updated: 2026-10-29
authors: [library agent via main session]
reviewers: [user]
---

# R06 Independent test authorship (integration-testing-engineer + testing-mathematics-expert)

Scope: new tests in `tests/independent/` written from the specification only. No existing `tests/test_*.py`
was opened. Allowed exceptions used: `tests/conftest.py` (first 60 lines) and `tests/test_flash_stream_triad.py`
(imports and `build_synthetic_flash_gguf`, to confirm the GGUF layout; I then wrote my own builder in
`tests/independent/ind_helpers.py`). Specification read: SRS.md (FR-001, FR-005, NFR-002 lines via grep),
docs/architecture.md (bounds paragraph), docs/benchmarks/METHOD.md (claim gates), docs/benchmarks/schema/bench_result.v1.json,
and the public docstrings of bridge.py, weight_streamer.py, gguf_parser.py, stats.py, results.py, kv_quality.py,
harness_common.py. I also read quant/models.py, core/exceptions.py and core/constants.py for type names.
I did not read the C sources or any existing test.

## Skills (proof of reading)

Read (partially, as stated):
- `property-based-testing-core` lines 1-140. Quote: "Property-based testing (PBT) automatically generates thousands of inputs to verify universally quantified properties — predicates that must hold for all values in a domain."
- `mutation-testing-core` lines 1-120. Quote: "Survived mutants are the actionable output: each one identifies a class of faults undetected by the current test suite."
- `integration-testing-core` lines 1-120. Quote: "The key word is \"real interface\": integration tests exercise actual persistence mechanisms, actual serialization formats, and actual network protocols within a controlled environment."

Not read: `contract-testing-core`, `unit-testing-core`, `performance-testing-core`, `chaos-engineering-core`,
`ai-model-testing-core`, `e2e-testing-core`, `test-automation-architecture-core`, and the remainder of the three
skills above. Their Pact/queueing/chaos/fairness content is not applicable to this task.

Environment note: `hypothesis` is NOT installed in the venv, so all property tests are seeded `random.Random`
loops (fixed seeds, deterministic). scipy is also absent; the Student-t reference is a numerical inverse of the
regularized incomplete beta function written in the test and validated against the textbook table in the same file.

## Tests written

| File | Tests (collected) | Pass | Fail | Content |
|---|---|---|---|---|
| `test_ind_native_ranges.py` | 30 | 30 | 0 | 400 random single extents + 120 random multi-tensor layers vs a Python reference `0<=off and off+size<=file_size` (big-int); uint64 edges (2^64-1, 2^63+2^63, wrap sums); non-u64 integers; layer idx / 64-tensor / name-bytes / 4-dim limits; bytes returned equal file bytes for 150 random valid layers; seal behaviour; lock/unlock never succeed outside the file |
| `test_ind_ring_buffer.py` | 21 | 21 | 0 | exact bytes vs file for random acquire sequences (1,2,3,5 slots); occupied <= slots; direct mapping (layer L only in slot L%S); arrays valid after recycle; 6 threads x 80 random acquire/release/prefetch ops with a sampler thread; one-load-for-8-concurrent-callers; close idempotence and acquire-after-close; prefetch+close does not hang; corrupt-offset layer fails cleanly, slot recovers, 6 waiters on a failing load all terminate |
| `test_ind_gguf_fuzz.py` | 33 | 21 | 12 | own GGUF writer; every truncation before end of tensor info (must raise ArcRunnerError), truncations after (must parse); 500 byte-flip mutants; 400 u64-word and 400 u32-word edge-value replacements; hostile length fields (huge strings/arrays/counts/dims); unknown value/element/ggml types; alignment 0/str/list/neg; nested arrays depth 50/500/5000; directory path; per-parse 10 s thread timeout and 64 MB tracemalloc cap |
| `test_ind_stats.py` | 51 | 49 | 2 | t critical vs textbook and vs numeric inverse for df 1..120; `student_t_ci95` vs hand formula on 300 random samples, worked example, translation/scale laws, 3000-trial coverage 93.5-96.5%; mean/median/IQR/CV vs numpy; warm-up, roofline; bootstrap properties (determinism, bounds, shift/scale equivariance, constant collapse, 60-trial coverage); speed-up and non-inferiority gates with exactly known ratios (2.0, 1.0, 0.98...), n=9/10 boundary, CV 9%/11% boundary, veto by one derived/projected/fallback/noisy row at 3 positions, warm-up rows ignored, roofline SUSPECT, ttft direction |
| `test_ind_results.py` | 264 | 179 | 85 | every required key deleted at top/env/protocol/sample level; type/range grids for every sample field (incl. NaN, inf, bool-as-int); source enum; workload/metrics rules; failures records; schema-only rules (env value types, protocol types/minimums, noisy, run_id minLength); write_result/load_samples/build_result round trips; "never raises" grid |
| `test_ind_kv_quality.py` | 31 | 26 | 5 | `common_prefix_length` vs reference on 2000 random pairs; `compare_kv_precision` on 300 random cases vs exact rational gate; threshold boundary; factory call order and greedy config; error cases; `write_report`; `harness_common` target_device, select_tiers, collect_tiers, gpu_context_lost, default_result_path |
| **Total** | **430** | **326** | **104** | wall time about 9 s for the whole directory |

Supporting files: `ind_helpers.py` (GGUF writer), `conftest.py` (adds the directory to `sys.path`).
Skipped/not-run: none skipped; the native tests ran against `libarc_native_sys.so` (Linux, ABI 3). Nothing
hardware-bound was exercised (Level Zero, Sysman and real devices are NOT tested here and are not reported as passed).

Mutation sanity check (done in a scratch plugin, source untouched): 6 hand-written mutants were injected by
monkeypatching: slot index `% (S+1)`, `student_t_ci95` dividing by n instead of n-1, gate `>` weakened to `>=`,
`MIN_TIMED_ITERATIONS` 10 to 9, `MAX_CV_FOR_CLAIMS` 0.10 to 0.15, and a wrong `common_prefix_length`. All six are
killed by this suite. The first run of the slot mutant SURVIVED the ring tests, so I added
`test_layer_is_resident_only_in_slot_layer_mod_slots` (reads `mgr._slots`, a private field, justified by the
`_RingSlot` docstring); it now kills that mutant (3 failures).

## Failing tests: verdicts (code wrong vs test wrong)

All 104 failures are the code. For each I read the code line. Tests whose expectation I judged arbitrary were
changed to pin documented behaviour (see "Ambiguous spec" below).

### D1 MED (verified) - GGUF parser leaks `ValueError` for unknown type ids
`src/arc_runner/quant/gguf_parser.py:230` (array element type), `:253` (metadata value type), `:285` (tensor type)
call `GGUFValueType(x)` / `GGMLType(x)` on untrusted integers; an unknown id raises `ValueError`, not an
`ArcRunnerError` subclass. CLAUDE.md: "raise the specific ArcRunnerError subclass" and the quant package "treats
input as hostile".
Repro: file = `b"GGUF" + pack("<IQQ",3,0,1) + pack_string("k") + pack("<I",13)` -> `ValueError: 13 is not a valid GGUFValueType`.
Same with tensor type 4, 5 or 31 (4 and 5 are retired ggml ids, so real old files hit this). In the fuzz tests
109 of 500 random byte-flip mutants and 122 of 400 u64 mutants end in this error.
Failing tests: `test_unknown_metadata_value_type...`, `test_unknown_array_element_type...`, `test_unknown_ggml_tensor_type...`,
`test_random_byte_flips...`, `test_random_u64_word_replacement...`, `test_u32_word_replacement...`.
Impact: a caller catching `ArcRunnerError` (the API error handler maps it to HTTP 400) sees an uncaught `ValueError` -> 500.

### D2 MED (verified) - hostile `general.alignment` crashes or yields a wrong data offset
`gguf_parser.py:309-312`: the value is used as an integer without validation.
- `0` -> `ZeroDivisionError`; string or array -> `TypeError` (3 failing tests).
- Negative (`I32 -32`): parses "ok" but `data_offset` = 32, which is BEFORE the end of the tensor-info block
  (end is byte 54 in the test file), so tensor payload reads would start inside the header. Verified by running the parser.
- Float (`F32 32.0`): `data_offset` becomes the float `64.0` (type leak). BOOL True gives alignment 1 (accepted).
The first two tests above fail; negative/float are verified by a manual run only (the test accepts "ok" for them).
Suggested rule (spec ambiguity, see below): alignment must be a positive power-of-two int, else `GGUFParseError`.

### D3 LOW-MED (verified) - `RecursionError` on deeply nested arrays
`gguf_parser.py:232` recurses per nesting level; each level costs only 12 bytes of file. Depth 5000 (about 60 KB
file) raises `RecursionError`; depth 500 is fine. Test: `test_deeply_nested_arrays...[5000]`.

### D4 LOW (verified) - directory path escapes as `IsADirectoryError`
`gguf_parser.py:301-304`: `exists()` is true for a directory, `open()` raises `IsADirectoryError` (an `OSError`).
Test: `test_directory_path_raises_arc_error`.

### D5 MED-LOW (verified) - `validate_result` raises instead of returning problems
`src/arc_runner/benchmark/results.py:135-136`: `k not in document["environment"]` / `["protocol"]` raise
`TypeError` when the value is `null`, a number or a bool (`[]`, `"str"`, lists happen to work).
Repro: `validate_result({"schema_version":1,"run_id":"r","created_utc":"t","environment":None,"protocol":{},"samples":[]})`
-> `TypeError: argument of type 'NoneType' is not iterable`. The docstring promises "Human-readable problems; empty if
valid" for any "Parsed JSON value". 8 failing tests. Any tool that validates untrusted result files crashes with a traceback.

### D6 LOW-MED (verified) - `validate_result` is much weaker than the published schema
The schema file (`docs/benchmarks/schema/bench_result.v1.json`) says it "documents the same rules" as the validator,
but the validator accepts, and `write_result` therefore writes: non-string `run_id`/`created_utc`/`model_id`/`backend`
(`results.py:128-149,164-180`), empty `run_id`, non-string environment values, any `protocol` values (negative
`warmup_runs`, `timed_runs: 0`, `max_new_tokens: 0`, `labels` as a list, string `temperature`), non-boolean `noisy`,
non-boolean `on_ac_power`, `schema_version: true` (`True != 1` is False, `results.py:133`), and `NaN`/`inf` in
`ttft_ms`/`decode_tps` (`results.py:171`: `value < 0` is False for NaN; metrics do check finiteness, top-level fields do not).
About 80 of the 85 results failures are this. Verified the three committed result files in `docs/benchmarks/results/`
satisfy the stricter schema rules (labels dict, int runs, string env), so tightening would not break existing data.
Why it matters for the honesty rule: `noisy` is a claim-gate input; `noisy: "no"` or `noisy: 0` passes validation, and a
NaN decode rate would flow into `stats` unchecked.

### D7 MED-LOW (verified) - speed-up / non-inferiority gate raises `ValueError` for a zero-throughput side
`stats.py:222` (`bootstrap_ratio_ci` raises when the denominator mean is not positive), called unguarded from
`stats.py:340` and `:379`. A reference whose 10+ timed rows all have `decode_tps == 0.0` (measured, CV defined as 0 by
`coefficient_of_variation`, enough rows) passes every eligibility check and then the gate throws instead of returning
`ClaimGateResult(False, ...)`. Same for `ttft_ms` with an all-zero arc side (ratio is ref/arc). Schema allows 0.
Docstring of both gates: "Returns: ClaimGateResult: Verdict". Tests: `test_speedup_gate_zero_throughput_reference...`,
`test_speedup_gate_zero_ttft_arc...`.

### D8 LOW (verified) - KV gate threshold uses float `ceil(fraction * n)`
`src/arc_runner/benchmark/kv_quality.py:144`: `pass_count >= math.ceil(required_pass_fraction * len(comparisons))`.
`0.55 * 100 == 55.00000000000001`, so `ceil` gives 56 and a run with exactly 55 of 100 passing FAILS a gate that asks
for "at least" 0.55. Verified failing for 0.07, 0.14, 0.28, 0.55 at n=100 (27 (fraction,n) pairs among 2-decimal
fractions with n<=200 are wrong). The defaults and the common 0.5/0.75/0.8/0.9/0.95/0.99 are exact for n up to 5000, so
the shipped default is not affected. Fix: compare `pass_count / n >= fraction` or use `Fraction`.

### D9 LOW (verified) - `select_tiers` does not return a copy
`src/arc_runner/benchmark/harness_common.py:92-93` returns the same `spec` object when `tiers` is falsy, although the
docstring says "Return a copy". With tiers given it is a shallow copy (new `workloads` dict, shared entries).
Test: `test_select_tiers_returns_a_copy_as_documented` (mutating the result changes the input).

### D10 LOW (suspected from code, behaviour verified by a test that pins it) - `gpu_context_lost` can miss the marker
`harness_common.py:144-145`: the failure record stores `str(exc)[:500]` but `collect_tiers` tests the marker on the full
string. If `CL_OUT_OF_RESOURCES` appears after character 500, `collect_tiers` correctly returns `lost=True`, but a later
`gpu_context_lost(failures)` (whose docstring: "any recorded failure was a GPU out-of-resources error") returns False.
`test_collect_tiers_marker_beyond_500_chars_is_still_detected` pins the current behaviour; it passes.

### Statistical observation, not counted as a defect
`stats.py:91`: beyond df 120 the multiplier is 1.960 while the true value is 1.9798 at df 121 (about 1.0% low, step
down of 0.020 from 1.980 at df 120). Docstring states this ("the normal value beyond 120"), so it is by design, but a
Student-t 95% interval for n > 121 is slightly anti-conservative (about 0.9% narrow at n=122). Benchmarks use 10-30
runs, so no published number is affected. Test pins the documented value and bounds the gap at 1.5%.

## No defect found (verified by passing tests)
- Native extent checks: 400 random single extents + 120 random multi-tensor layers + all uint64 edge cases (including
  `offset + size` wrapping modulo 2^64 and `2^63 + 2^63`) agree with the big-int reference; integers outside [0, 2^64)
  and non-int/bool types raise `NativeBoundsError`; 64-tensor and 256-layer limits are exact; name limit is in bytes
  (127 two-byte characters rejected); zero-size extent at offset == file size is accepted and offset > file size rejected
  (consistent with the reference). Returned views equal the file bytes. `lock_range` never succeeds outside the file.
- Ring buffer: no invariant violation, deadlock or wrong byte in 6-thread random operation; a failing load releases the
  slot and all waiters terminate; one load for 8 concurrent callers.
- GGUF: every truncation before the end of the tensor-info block raises `GGUFParseError`/`InvalidGGUFError`; huge
  length fields (strings, arrays, counts, dims) are rejected quickly with peak traced memory under 64 MB; no hangs.
- `student_t_ci95`, bootstrap functions and both gates match hand-derived values; empirical coverage is as expected.

## Ambiguous specification statements (decisions I had to make)
1. Zero-size extents: SRS says an extent is `[offset, offset+size)` inside the file; it does not say whether an empty
   extent or one at offset == file_size is valid. Implementation accepts it, matching `offset+size <= file_size`; I
   tested against that reference. `get_tensor_buffer` returns None for size 0 (docstring) so it is harmless.
2. `general.alignment`: no document states valid values (positive, power of two, integer type). D2 rule is my proposal.
3. "Raise the specific ArcRunnerError subclass" (CLAUDE.md) vs `FlashStreamWeightManager._read_tensor_array`
   docstring "Raises: ValueError" for corrupt offsets: the code follows its docstring, not the project rule. I accepted
   `(ValueError, ArcRunnerError)` in the ring tests.
4. Whether `validate_result` must enforce the JSON schema or only the presence rules listed in its docstring. The schema
   claims "the same rules"; I treated the schema as normative (D6). If the intent is presence-only, D6 is a doc defect.
5. Claim gate with zero throughput: METHOD.md gives no rule; I required a verdict (not passed), not an exception (D7).
6. `student_t_critical_95` beyond df 120: design choice documented in the docstring; no tolerance is stated in METHOD.md.
7. METHOD.md says the Student-t and bootstrap intervals are compared and a row "flagged non-normal" when they disagree by
   more than 5%; this is not implemented in `stats.py` as a function (no such helper exists in stats.py),
   so I could not test it (grep for non-normal/non_normal over src/ and scripts/ returns nothing). Suspected documentation/implementation gap.
8. `select_tiers` with a separator-only string (`" , "`) returns an empty workload set (not "all", not an error); the
   spec is silent. Not asserted.
9. Required fraction semantics ("at least") is clear; rounding mode of the threshold is not (D8 assumes exact arithmetic).

## How to run
`/tmp/claude-0/-home-user/9f5d9574-39ea-55de-8475-bf2e506385a2/scratchpad/venv/bin/python -m pytest tests/independent -p no:warnings -q --show-capture=no`
(about 9 s). The 104 red tests are deliberate evidence of D1-D9; they are not marked xfail. If CI must stay green until
the fixes land, mark them `xfail(strict=True)` so that fixing the code flips them to a visible XPASS failure.
Source code was not modified.

## Revision history
| Version | Date | Author | What changed | Why | Commit |
|:--|:--|:--|:--|:--|:--|
| 1.0.0 | 2026-10-29 | library agent via main session | first version | independent review wave A | pending |
