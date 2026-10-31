---
doc_id: DOC-SC-STAGE
title: Slice S-C - ring buffer, dtype, and view lifetime
stage: S-C
version: 1.0.0
status: draft
created: 2026-10-29
updated: 2026-10-29
authors: [main session applying tla-plus-specification-engineer and c-cpp-systems-engineer roles]
reviewers: [user]
inputs: [DOC-SB-STAGE@1.0.0, docs/plans/MASTER_PLAN_v4.md sections 10.4, 17.4]
---

# Slice S-C: ring buffer, dtype, and view lifetime

Gate status: `cloud-green`. TLC itself was NOT run (see section 3).

## 1. Scope
C4 (dtype), C5 (view lifetime), C6 (slot races, shutdown), X1 (LRU on cyclic scans), X2 (resident vs mapped memory), X3 (GGUF dimension order), C7 wiring left open.

## 2. Library use (honest record)
Agent file read in this session: `tla-plus-specification-engineer` (uploaded by the user) and the library's `c-cpp-systems-engineer`. The skill `tla-plus-formal-methods-core` (60 KB) was NOT read; the spec follows the agent's rules from its agent file (canonical `Init /\ [][Next]_vars /\ Fairness`, stuttering wrapper, weak fairness on every process action, safety before liveness).

## 3. Formal model
`docs/formal/RingSlots.tla` + `RingSlots.cfg`: direct-mapped slots (EMPTY/PENDING/READY), threads that hit, wait, or become the single loader, and a two-step close that joins loaders. Invariants: SingleLoader, DirectMapped, ServedMatches, NoConcurrentLoadsPerSlot, ClosedIsClean; property: every acquire ends in done or error.
TLC was not executed: running the downloaded `tla2tools.jar` was blocked by the session's permission classifier. Command to run it locally (needs Java 11+ and the official jar from the tlaplus releases page):
```
java -cp tla2tools.jar tlc2.TLC -workers 4 -config docs/formal/RingSlots.cfg docs/formal/RingSlots.tla
```
Substitute check: `scripts/check_ring_model.py` re-implements the same transitions in Python and explores the full reachable state space. 3 threads, 4 layers, 2 slots, 2 operations each: 750,420 states, 3,606,900 transitions, 1,728 final states; safety invariants hold in every state, no deadlock, no trap (every state can still reach a clean closed final state). A mutant that lets a second thread load into a PENDING slot is detected (`NoConcurrentLoadsPerSlot`). This approximates, but is not, TLC's weak-fairness liveness check.

## 4. Design decisions
| DL | Decision | Why |
|:--|:--|:--|
| 044 | No pin counts; consumers keep their own references, and arrays keep the mapping alive (`_MappingGuard`) | with `layer % slots` collisions a thread holding layer 0 could wait forever for layer 3; the model found no need for blocking pins |
| 045 | Slot is always `layer % ring_buffer_slots`; `prefetch_layer_async(layer, slot)` ignores `slot` | a cyclic scan over more layers than slots gives LRU a 0% hit rate; direct mapping is deterministic and needs no timestamps |
| 046 | BF16 is exposed as raw uint16 bits and quantized types as raw uint8 bytes in both native and Python paths | the old fallback read BF16 as float16 and converted quantized bytes to float32 (4x memory, meaningless values) |
| 047 | Row-major shape = reversed GGUF dimensions | GGUF `ne[0]` is the fastest dimension |
| 048 | `close()` defers the native unmap while NumPy views are alive | prevents use-after-unmap; the unmap runs when the last view is collected |

## 5. Changes
`weight_streamer.py` rewritten around `_RingSlot` / `SlotState`: `_reserve_slot` (hit / wait / become loader under the lock), loading outside the lock, `_publish_load`, failed loads reset the slot, waiters time out after 30 s instead of hanging, `close()` sets a closing flag, joins prefetch threads and waits for loads in flight, `get_resident_memory_bytes()` (Windows working set, Linux RSS), `gc.collect()` only outside the lock, block-size table with `ValueError` for unknown layouts. `bridge.py`: `_MappingGuard`, read-only zero-copy views, `dtype_for_tensor_type`, `view_shape_for_dims`, `__del__`.

## 6. Results
25 new tests in `tests/test_flash_stream_ring.py` (single load under 8 threads, 8 x 150 stress with at most one in-flight load per slot, colliding layers, failed-load retry, close during prefetch, wait timeout, view lifetime, read-only views, dtype and shape parity, formal-model tie-in). Full non-hardware suite: see the commit message of this slice.

## 7. Open items
1. Run TLC locally (command above) and report the result.
2. `try_pin` (best-effort `lock_range` of the active layer) is still not called from the streamer (C7 wiring); do it with the local benchmark run so the effect can be measured.
3. On Windows an unmap deferred by a live view keeps the file open until the view dies.

## Revision history
| Version | Date | Author | What changed | Why | Commit |
|:--|:--|:--|:--|:--|:--|
| 1.0.0 | 2026-10-29 | main session | first version | S-C implementation | pending |
