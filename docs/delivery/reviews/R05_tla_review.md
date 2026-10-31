---
doc_id: DOC-R05-REVIEW
title: R05 - TLA+ review of RingSlots against the ring-buffer weight streamer
stage: S-G
version: 1.0.0
status: draft
created: 2026-10-29
updated: 2026-10-29
authors: [library agent via main session]
reviewers: [user]
---

# R05 - TLA+ review of RingSlots against the ring-buffer weight streamer

Reviewer: tla-plus-specification-engineer (library agent). Scope: `docs/formal/RingSlots.tla|cfg`,
`src/arc_runner/engine/flash_stream/weight_streamer.py` (WS), `scripts/check_ring_model.py` (CHK).
Nothing in the repo was edited. Scratch artifacts (all in
`/tmp/claude-0/-home-user/9f5d9574-39ea-55de-8475-bf2e506385a2/scratchpad/r05/`):
`mutants.py`, `liveness.py`, `repro.py`, `stress.py`, `repro_fakes.py`.

## 0. Tooling status, skills read

- TLC was NOT run (instruction). `java` exists on PATH (`/usr/bin/java`, OpenJDK 21.0.12.1); no `tla2tools.jar`
  was fetched or looked for. So the agent-definition rule "never present an unrun spec as verified" applies:
  **RingSlots.tla has never been parsed or checked by TLC or SANY**; every statement below about the spec
  is by reading, and about the model by running CHK, which is a separate Python re-implementation.
- CHK was run unmodified: `ring model OK threads=3 layers=4 slots=2 max_ops=2 {'states': 750420,
  'transitions': 3606900, 'final_states': 1728}` (12 s).
- Skill read: `tla-plus-formal-methods-core` (all 298 lines). Proof line: "Always state the canonical spec shape
  `Spec ≡ Init ∧ □[Next]_vars ∧ Fairness` explicitly when discussing what makes a formula a well-formed TLA+
  specification of a reactive system". Not read: `dafny-verification-core`, `formal-methods-case-studies-core`
  (optional). No delegation to formal-methods-mathematics-expert (no derivation needed beyond the acyclicity
  argument in 2.3, which I checked by computation instead).
- **Defect in the skill itself (library, not repo):** SKILL.md section 9 M5 (lines 213-214) defines
  `WF = []<>ENABLED => []<>taken` and `SF = <>[]ENABLED => []<>taken`. Lamport (Specifying Systems, ch. 8) has these
  the other way round: WF_v(A) = `<>[]ENABLED<A>_v => []<><A>_v` (continuously enabled), SF_v(A) = `[]<>ENABLED<A>_v
  => []<><A>_v` (infinitely often enabled). The skill's own prose then contradicts itself ("WF suffices ... SF strictly
  stronger" vs "WF is the stronger promise"). I am using Lamport's definitions below; the book should be checked
  before the skill is fixed. Same text is repeated in the agent's rule 6.

## 1. Action-by-action correspondence (question 1)

Spec shape is correct: `Spec == Init /\ [][Next]_vars /\ Fairness` (tla:114); `Begin/BeginClose` carry no
fairness (environment choices), which is the right call.

| Spec action | Code | Verdict |
|---|---|---|
| `Begin(t)` tla:45-50 | caller enters `acquire_layer_weights` ws:503; prefetch worker ws:533-536 | match. Worker's `if not closing` pre-check (ws:535) is an unlocked read; benign because `Decide` re-checks. |
| `Decide(t)`, 4 branches tla:52-71 | `_reserve_slot` ws:452-466, one `_lock` critical section | **match branch for branch**: closing->error ws:453-454 (checked before hit, as in tla:55); hit ws:456-457; PENDING->wait ws:458-459; else load ws:460-466 (slot EMPTY or READY-other-layer, same guard as tla:65). Code adds eviction bookkeeping `_pending_unpin.append` (ws:460-461): extra state, see 3. |
| `Wake(t)` tla:73-76 | `event.wait` returns True, loop retries ws:518 | **approximate.** Spec guard is state-based (`slotState # PENDING`); code is event-based: the waiter holds the event of one load instance (captured under the lock at ws:459) and wakes when that instance publishes (ws:484), even if another load already re-set the slot to PENDING. Code behaviours are a superset; the extra wake just re-runs `Decide` and is harmless (stuttering-like). |
| `LoadDone(t)` tla:78-85 | `_publish_load` ws:475-484 | match for the success path. Arrays and READY are set in one critical section (ws:482-483), so "READY => arrays published" holds by construction. |
| *(absent)* | load failure: `_publish_load(None)` ws:477-480, called from ws:511-513 | **missing action `LoadFail`**: PENDING->EMPTY, layer -1, waiters woken, loader ends in error. |
| *(absent)* | `release_layer_slot` ws:549-567 (READY->EMPTY, layer -1) | **missing action `Release(s)`**. Header comment tla:5-7 ("no pin counts") predates `_pending_unpin/_pinned`. |
| *(absent)* | wait timeout `TimeoutError` ws:518-519 | **missing `WaitTimeout(t)`**: wait->error. |
| `BeginClose` tla:93-96 | ws:606-611 | match. A second concurrent `close()` returns at ws:607-608 *without waiting for the drain*; not modelled. |
| `FinishClose` tla:98-104 | ws:613-624 | **spec is stronger than code.** Spec: guard `\A s: loader[s]=NONE`, so close only completes when no load is in flight. Code: bounded waits only (`join(5s)` ws:614, `event.wait(5s)` ws:616), then unconditional reset of every slot including PENDING (ws:619-624, which also never sets the slot event). S4 below shows the consequence. |
| `Finish(t)` | client side only | n/a |

Cfg: `NONE = none`, `Threads={t1,t2,t3}`, `Layers={0..3}`, `N=2`, `MaxOps=2`. Two TLC-level problems visible without
running it: (a) **no `CHECK_DEADLOCK FALSE` and no terminal "Done" stutter action.** The legitimate final state
(closed, every thread idle with `ops = MaxOps`) has no enabled `Next` disjunct, so TLC would report it as a deadlock
on the first run. CHK handles this with `is_final`; the spec does not. (b) no `SYMMETRY` for `Threads` although all
threads are interchangeable (up to 3! = 6x reduction; Layers are not freely symmetric because of `% N`).

## 2. Invariants and what CHK actually checks (question 2)

### 2.1 Fault injection (mutants of CHK, `mutants.py`, config 2 threads/3 layers/2 slots/2 ops)

| Mutant | Result |
|---|---|
| load allowed over a PENDING slot (double loader) | caught: `NoConcurrentLoadsPerSlot` |
| `LoadDone` keeps `loader[s]` | caught: `SingleLoader` |
| `FinishClose` without the drain guard | caught: `SingleLoader` |
| `Wake` removed (lost wakeup) | caught: deadlock |
| `Decide` ignores `closing` | caught: `ClosedIsClean` |
| **hit path serves a READY slot holding another layer** | **NOT caught** |

The last row is a real weakness: **`ServedMatches` (tla:123) is vacuous.** `served` is assigned from `want[t]`
(tla:60, tla:84; chk lines 77-78, 87-88), so `served[t] = want[t]` is a tautology. Fix in 4(b).
`DirectMapped` (tla:119) is likewise true by construction of `SlotOf`; cheap, harmless, but it cannot fail.

### 2.2 Per the four invariants asked about

- **Single loader per slot:** right invariant, correctly checked (mutants 1-3). Caveat: `loader[]` is a ghost
  variable; the code has no loader field. The spec proves `PENDING <=> loader # NONE` about its own bookkeeping.
  In the code the analogous property ("PENDING iff a live thread is between reserve and publish") is broken by
  an exception at ws:508 (S3) and by the close reset (S4); the model cannot express either because it has no
  `LoadFail` and a guarded close.
- **No deadlock:** CHK flags only states with literally zero successors that are not final. Adequate here because
  the only non-fair enabled actions are `Begin` (bounded by MaxOps) and `BeginClose` (once). Not a general
  fairness-aware check. Also see the TLC `CHECK_DEADLOCK` issue in 1.
- **READY implies arrays published:** **not in the spec and not checked.** The model has no array/payload
  variable, so a swapped order in `_publish_load` (READY before `arrays`) is unrepresentable. Honest verdict:
  this property is protected by a single critical section (ws:475-484), not by the protocol; a model would only
  prove something if publication were split into two non-atomic actions, which the code does not do (every read of
  `slot.state` is under `_lock`: ws:381, 409, 425-431, 456-460, 576, 589, 611). A unit test is the right tool.
- **Close drains:** `ClosedIsClean` is enforced by the `FinishClose` guard (tla:100) rather than discovered; it
  is only meaningful together with "no load starts after close", which mutant 5 shows is checked. It describes
  the *spec's* close, which the code does not implement (S4).

### 2.3 Liveness

`Progress == acq ~> done \/ error` (tla:136) is a PROPERTY in the cfg, but **CHK does not check it**; its
docstring says liveness is "approximated by trap-freedom" (chk:7-8). Trap-freedom (a final state is reachable from
every state) is existential and does not imply Progress under WF. Evidence (`liveness.py`): a mutant in which
`Decide` may also send a thread from READY-other-layer to `wait` (so acq->wait->acq can repeat forever, a legal
WF(Decide) behaviour because WF applies to the whole action, not a branch) **passes `check()`** (6345 states,
no assertion), while the graph is cyclic (`acyclic: False`). 

What does hold for the real model: with stuttering self-loops removed, the unmutated transition graph is
**acyclic** (computed, Kahn's algorithm; (2,3,2,2): 6109 states; (3,4,2,2): 750420 states). In an acyclic finite
graph every behaviour ends in a stutter state S; at S, WF(Decide(t)) forces `pc[t] # acq` (Decide is enabled in every
acq state: the four branches are exhaustive), WF(LoadDone) forces `pc[t] # loading`, and `pc[t]=wait` implies a PENDING
slot whose loader has LoadDone enabled (SingleLoader), a contradiction. So Progress holds for the **MaxOps-bounded**
model, by this argument plus computation, not by TLC. Two honest limits: (i) the bound does the work: with
unbounded operations another thread can win the slot every time and a waiter ping-pongs wait<->acq under any
fairness; in the code `event.wait` restarts its 30 s timeout on every wake (ws:518), so there is no overall bound
and the `TimeoutError` never fires in that ping-pong. (ii) In Lamport's terms `Wake(t)` can flicker (another
thread's reserve disables it, the next publish re-enables it), so the textbook fairness for it is SF, not WF;
WF is enough only because the bounded model is acyclic.

## 3. Pin / unpin / stage protocol (question 3)

### 3.1 Lock ordering: no deadlock risk found (verified by reading, plus stress)

Acquisition edges in WS: `_pin_lock -> _lock` at ws:379-380 (`_pin_layer`) and ws:429-431 (`_flush_unpins`).
There is **no `_lock -> _pin_lock` edge**: every `with self._lock` body (ws:199-222, 408-410, 425-426, 431, 452-466,
475-484, 540, 544, 556-564, 575, 587, 606-611, 618-624, 627-630) calls nothing that takes `_pin_lock`;
`_flush_unpins` is called only outside `_lock` (ws:508, 566, 625). `index_layers` (ws:199-222) holds `_lock` while
calling `native.register_layer` only. `DeviceLayerStager._lock` (device_stager.py:90) never nests with manager locks
(stager calls at ws:403, 412, 428, 633 are all outside manager locks). Both are non-reentrant `threading.Lock`;
I found no re-acquisition path. Stress: 100 runs x 4 threads x 60 random acquire/release/prefetch ops, ring 2,
6 layers, `sys.setswitchinterval(1e-6)`, fake native + real stager: 0 hangs. Residual (LOW): `lock_range` runs under
`_pin_lock` (ws:386), so a slow OS lock stalls every loader's `_flush_unpins` (ws:508) and its waiters (30 s timeout).

### 3.2 Leaked pin: no interleaving found

`_pin_layer` does check-then-act under `_pin_lock` (ws:379-387); an eviction enqueues under `_lock` atomically with
the state change (ws:460-461, 560-563, 621-622) and its flush needs `_pin_lock` to consume the entry (ws:429-434), so
a pin created between check and eviction is unpinned afterwards. A stale unpin of a re-pinned layer is also
impossible: reload follows eviction, and the reloader's own `_flush_unpins` (ws:508) drains the entry before it pins.
Stress result at quiescence over 100 runs: pinned spans == locked spans, pinned subset of resident, 0 leaks.
Lower-probability residuals: an exception in `_flush_unpins` part 1 (stager) skips the unlock until the next flush
(self-heals; close flushes); after `close` sets `_native_engine=None` (ws:628) a late `_flush_unpins` pops the span
without unlocking (ws:435); whether `native.close()` releases the OS lock is not verified here (UNCERTAIN, native
layer not read).

### 3.3 Leaked / wrongly freed USM buffer: two concrete interleavings (VERIFIED, `repro.py`)

Root cause (ws:422-436): `_flush_unpins` splits the work into two critical sections. Part 1 releases the stager
buffers for a **non-consuming copy** of `_pending_unpin` (ws:424-428), outside `_pin_lock`; part 2 drains the queue
and unpins (ws:429-434). The queue is thus consumed by someone other than the thread that releases the buffer.

**S1 - leak (MED).** T1 = a loader whose flush snapshot is empty (ws:426); T2 = `release_layer_slot(0)` evicting
a staged layer 0. Order: T1 snapshot (empty) -> T2 evicts and enqueues 0 (ws:560) -> T1 part 2 swaps the queue and
consumes 0 (ws:431) -> T2 flush snapshots an empty queue. `stager.release_layer(0)` is never called. Output:
`slot0 state: EMPTY | staged layers: [0] | live USM buffers: 1 -> LEAK`. The buffer survives until `stager.close()`
(live=0 after close) or a re-stage of the same layer. Effect: staged USM can exceed the ring size
(up to one extra layer per leaked eviction), which breaks the O(1) memory claim for the staged copy.

**S2 - stale free of a live buffer (MED, latent HIGH once consumed).** T2 evicts 0 and snapshots `[0]`
(ws:426) but is preempted before `release_layer(0)` (ws:428). T3 reloads layer 0, flushes, publishes, stages it
(ws:508-516). T2 resumes and frees T3's fresh buffer. Output: `before: slot0 READY layer 0 staged [0]; after:
slot0 READY layer 0 staged [] -> STALE FREE`. Nothing in `src/` reads `stager.get()` today (grep), so there is no
use-after-free yet; the first consumer that holds `StagedLayer.ptr` across a kernel launch will have one.

**Not a defect (checked):** the `_stage_layer` re-check (ws:408-412) correctly covers eviction during staging; the
natural-schedule stress run (100 runs) never hit S1/S2 because the windows are a few bytecodes wide, which is why
forced interleavings were needed.

Suggested fix (text only): drain once under `_lock` (`queued, self._pending_unpin = self._pending_unpin, []`),
then release+unpin that same list (fixes S1). S2 needs ownership, not just draining: stamp each slot with a
generation (incremented at reserve, ws:462), queue `(layer, gen)`, and have `release_layer` / unpin act only when
the staged/pinned entry carries that generation.

### 3.4 Other defects found while mapping (VERIFIED by repro)

**S3 - stuck PENDING slot (MED, low probability).** `self._flush_unpins()` at ws:508 is *outside* the `try` at
ws:509-513. Any exception from it (stager/native fault, or a KeyboardInterrupt delivered during the possibly
blocking `_pin_lock`) leaves the slot PENDING with no loader. Injected `RuntimeError` in `stager.release_layer`:
slot0 stays `PENDING layer 2`; after healing the fault both `acquire(2)` and `acquire(0)` raise `TimeoutError`
(30 s each in production), and `close()` waits the full 5 s on the dead event (measured 5.1 s). Fix: move the flush
inside the `try`, or flush before `_reserve_slot` returns "load".

**S4 - close() past its join timeout (LOW-MED).** With `PREFETCH_JOIN_TIMEOUT_SECONDS` shortened and a loader
stalled after reading, `close()` returns with slot1 EMPTY (ws:619-624); the loader then publishes: result ok,
`slot1 READY`, `get_occupied_slot_count()=1` after close, `layer_idx=-1` (publish does not restore it), arrays held
by a closed manager and never dropped (a second `close()` returns at ws:607). This falsifies the code-level
analogue of `ClosedIsClean` (tla:130-131). The stager leak half did **not** reproduce: `_stage_layer`'s
`still_resident` check sees `layer_idx=-1` and releases the buffer (by accident). Fix: `_publish_load` should discard
(drop arrays, set EMPTY) when `self._closing.is_set()`.

## 4. Proposed spec additions (question 4; text only, `.tla` not edited)

(a) Make the cfg runnable: add `CHECK_DEADLOCK FALSE` (or `Done == closed /\ \A t: pc[t]="idle" /\ UNCHANGED vars`);
add `SYMMETRY Perm` with `Perm == Permutations(Threads)`; update header tla:5-7.

(b) Make `ServedMatches` falsifiable: add `served' = [served EXCEPT ![t] = slotLayer[s]]` in the hit branch and
`slotLayer[SlotOf(want[t])]` in `LoadDone`; keep `pc[t]="done" => served[t] = want[t]`. Re-run the hit-ignores-layer
mutant; it must now be caught.

(c) Missing code actions: `LoadFail(t)` (`pc[t]="loading"`; slot -> EMPTY, `slotLayer=-1`, `loader=NONE`,
`pc[t]="error"`; WF); `Release(s)` (`slotState[s]="READY"` -> EMPTY, `-1`; no fairness; enqueues eviction, see (e));
`WaitTimeout(t)` (`pc[t]="wait"` -> `"error"`, no fairness). Replace `Progress` by `Progress` under the MaxOps
bound plus an explicit statement that unbounded starvation is allowed and only `WaitTimeout` bounds a single wait.

(d) Close, two variants. Model the code: `ForceClose == closing /\ ~closed /\ closed'=TRUE` resetting all slots
without the loader guard, and let `LoadDone` still fire after close. Then `ClosedIsClean` fails (S4); choose either
the code fix (publish discards after close) with a new invariant `closed => \A t: pc[t] # "loading" \/ ~published`, or
weaken the spec to `closed => \A s: slotState[s] = "EMPTY"` after the drain window. Add `closeCalls` to model the
second `close()` returning early.

(e) Pin/stage extension (new variables `pendingUnpin`, `pinned`, `staged` as sets of layers, `gen[s]` Nat,
`snap[t]` set, `pinLock, mgrLock \in Threads \cup {NONE}`; stager has its own atomic actions). Actions, one per
atomic region of the code: `Evict` (merged into the load branch of `Decide`, `Release`, `ForceClose`: slot change and
`pendingUnpin' = pendingUnpin \cup {layer}` atomically); `FlushSnap(t)` (`snap'[t] = pendingUnpin`, ws:426);
`StagerRelease(t,l)` for `l \in snap[t]` (staged' = staged \ {l}); `FlushSwap(t)` (pinLock then mgrLock; `pinned' =
pinned \ pendingUnpin`, `pendingUnpin' = {}`, ws:429-434); `PinCheck(t)`/`PinDo(t)` (check residency under pinLock,
then `pinned' = pinned \cup {l}`); `StageStart(t)`/`StageDone(t)` (`staged' = staged \cup {l}`) and `StageRecheck(t)`.
Encode lock order by guards (`PinCheck`/`FlushSwap` take pinLock before mgrLock; nothing takes them in the other
order), so a TLC deadlock check covers 3.1 mechanically. Invariants:
`Resident == {slotLayer[s] : s with READY}`;
`NoLeakAtQuiescence == Quiescent => staged \subseteq Resident /\ pinned \subseteq Resident` (S1 violates it);
`NoStaleFree == \A l \in Resident: (staged has l with gen = gen[SlotOf(l)]) \/ stagingDisabled \/ stageInFlight(l)`
(S2 violates it; needs the generation stamp from 3.3's fix to be expressible);
`PinImpliesResident == \A l \in pinned: l \in Resident \/ l \in pendingUnpin \/ flushInFlight`.
`Quiescent == (\A t: pc[t] \in {"idle","done","error"}) /\ pendingUnpin = {} /\ \A t: snap[t] = {}`.

(f) Conformance gap: CHK mirrors the TLA+ text 1:1, so it can only find spec bugs, never code bugs, and both were
written from the same understanding. Cheap way to close the gap: run the real `FlashStreamWeightManager` under the
gate-and-barrier technique in `repro.py` and promote S1-S4 to regression tests (red first, per CLAUDE.md) once
fixed. Also remove the dead check `all(o >= 0 for o in ops)` in `is_final` (chk:134; always true, docstring claims
"operations used up").

## 5. Severity summary

| Id | Finding | Severity | Status |
|---|---|---|---|
| S1 | staged USM buffer leaked by split snapshot/swap in `_flush_unpins` (ws:424-431) | MED | reproduced |
| S2 | stale snapshot frees a live staged buffer of a re-loaded layer (ws:426-428) | MED (latent HIGH when a consumer exists) | reproduced |
| S3 | `_flush_unpins` at ws:508 outside `try` -> slot stuck PENDING | MED (low probability) | reproduced with injected fault |
| S4 | `close()` timeout path leaves READY zombie, arrays retained (ws:613-624 vs spec FinishClose) | LOW-MED | reproduced |
| P1 | `ServedMatches` vacuous; hit-ignores-layer mutant undetected | MED (verification quality) | verified |
| P2 | cfg has no `CHECK_DEADLOCK FALSE`/Done action; TLC would flag the final state | LOW (will fail loudly) | by reading |
| P3 | CHK checks trap-freedom, not Progress; fair-livelock mutant passes | MED (verification quality) | verified; real model acyclic so Progress holds bounded |
| P4 | skill M5 swaps WF/SF definitions | MED (library) | by comparison with Lamport; confirm vs book |
| - | lock-order deadlock (`_pin_lock`/`_lock`) | no defect found | read + stress |
| - | leaked pin | no defect found | read + stress; native.close lock release UNCERTAIN |

## Revision history
| Version | Date | Author | What changed | Why | Commit |
|:--|:--|:--|:--|:--|:--|
| 1.0.0 | 2026-10-29 | library agent via main session | first version | independent review wave A | pending |
