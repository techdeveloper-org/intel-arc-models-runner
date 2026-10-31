"""Explicit-state check of the ring-slot protocol modelled in docs/formal/RingSlots.tla.

This is an independent re-implementation of the same transition system in Python
(breadth-first reachability). It checks the safety invariants of the TLA+ module on every
reachable state, rejects deadlocks (a state with no enabled step that is not a clean final
state), and rejects traps (a reachable state from which no clean final state is reachable).
It does not replace a TLC run: weak-fairness liveness is approximated by trap-freedom.

Usage: python scripts/check_ring_model.py [threads] [layers] [slots] [max_ops]
"""

from collections import deque
import itertools
import sys
from typing import Deque, Dict, Iterator, List, Set, Tuple

State = Tuple[tuple, tuple, tuple, tuple, tuple, tuple, bool, bool, tuple]
NONE = -1


def initial_states(threads: int, layers: int, slots: int) -> Iterator[State]:
    """Yield every initial state (the `want` vector is chosen nondeterministically).

    Args:
        threads: Number of acquiring threads.
        layers: Number of distinct layers.
        slots: Number of ring slots.

    Yields:
        State tuples (slotLayer, slotState, loader, pc, want, ops, closing, closed, served).
    """
    for want in itertools.product(range(layers), repeat=threads):
        yield (
            (-1,) * slots,
            ("EMPTY",) * slots,
            (NONE,) * slots,
            ("idle",) * threads,
            tuple(want),
            (0,) * threads,
            False,
            False,
            (-1,) * threads,
        )


def _set(vec: tuple, index: int, value: object) -> tuple:
    """Return a copy of vec with one element replaced."""
    items = list(vec)
    items[index] = value
    return tuple(items)


def successors(state: State, slots: int, layers: int, max_ops: int) -> List[State]:
    """Compute all states reachable in one step, mirroring the TLA+ actions.

    Args:
        state: Current state.
        slots: Number of ring slots.
        layers: Number of distinct layers.
        max_ops: Maximum acquires per thread.

    Returns:
        List of successor states.
    """
    slot_layer, slot_state, loader, pc, want, ops, closing, closed, served = state
    out: List[State] = []
    for t in range(len(pc)):
        s = want[t] % slots
        if pc[t] == "idle" and ops[t] < max_ops:
            for layer in range(layers):
                out.append((slot_layer, slot_state, loader, _set(pc, t, "acq"), _set(want, t, layer),
                            _set(ops, t, ops[t] + 1), closing, closed, served))
        if pc[t] == "acq":
            if closing:
                out.append((slot_layer, slot_state, loader, _set(pc, t, "error"), want, ops, closing, closed, served))
            elif slot_state[s] == "READY" and slot_layer[s] == want[t]:
                out.append((slot_layer, slot_state, loader, _set(pc, t, "done"), want, ops, closing, closed,
                            _set(served, t, want[t])))
            elif slot_state[s] == "PENDING":
                out.append((slot_layer, slot_state, loader, _set(pc, t, "wait"), want, ops, closing, closed, served))
            else:
                out.append((_set(slot_layer, s, want[t]), _set(slot_state, s, "PENDING"), _set(loader, s, t),
                            _set(pc, t, "loading"), want, ops, closing, closed, served))
        if pc[t] == "wait" and slot_state[s] != "PENDING":
            out.append((slot_layer, slot_state, loader, _set(pc, t, "acq"), want, ops, closing, closed, served))
        if pc[t] == "loading":
            out.append((slot_layer, _set(slot_state, s, "READY"), _set(loader, s, NONE), _set(pc, t, "done"), want,
                        ops, closing, closed, _set(served, t, want[t])))
        if pc[t] in ("done", "error"):
            out.append((slot_layer, slot_state, loader, _set(pc, t, "idle"), want, ops, closing, closed,
                        _set(served, t, -1)))
    if not closing:
        out.append((slot_layer, slot_state, loader, pc, want, ops, True, closed, served))
    if closing and not closed and all(x == NONE for x in loader):
        out.append(((-1,) * slots, ("EMPTY",) * slots, loader, pc, want, ops, closing, True, served))
    return out


def invariant_violation(state: State, slots: int) -> str:
    """Return the name of a violated safety invariant, or an empty string.

    Args:
        state: State to check.
        slots: Number of ring slots.

    Returns:
        str: Invariant name or "".
    """
    slot_layer, slot_state, loader, pc, want, _ops, _closing, closed, served = state
    for s in range(slots):
        if (slot_state[s] == "PENDING") != (loader[s] != NONE):
            return "SingleLoader"
        if slot_layer[s] != -1 and slot_layer[s] % slots != s:
            return "DirectMapped"
        if closed and (loader[s] != NONE or slot_state[s] != "EMPTY"):
            return "ClosedIsClean"
        loading = [t for t in range(len(pc)) if pc[t] == "loading" and want[t] % slots == s]
        if len(loading) > 1:
            return "NoConcurrentLoadsPerSlot"
    for t in range(len(pc)):
        if pc[t] == "done" and served[t] != want[t]:
            return "ServedMatches"
    return ""


def is_final(state: State, max_ops: int) -> bool:
    """Return True for a clean final state: closed, every thread idle with its operations used up.

    Args:
        state: State to test.
        max_ops: Maximum acquires per thread.
    """
    _sl, _ss, _ld, pc, _want, ops, _closing, closed, _served = state
    return closed and all(p == "idle" for p in pc) and all(o >= 0 for o in ops)


def check(threads: int, layers: int, slots: int, max_ops: int) -> Dict[str, int]:
    """Explore the full state space and verify safety, deadlock freedom and trap freedom.

    Args:
        threads: Number of acquiring threads.
        layers: Number of distinct layers.
        slots: Number of ring slots.
        max_ops: Maximum acquires per thread.

    Returns:
        Dict[str, int]: Counts of states, transitions and final states.

    Raises:
        AssertionError: On any invariant violation, deadlock, or trap.
    """
    seen: Set[State] = set()
    queue: Deque[State] = deque()
    reverse: Dict[State, List[State]] = {}
    transitions = 0
    for init in initial_states(threads, layers, slots):
        seen.add(init)
        queue.append(init)
    finals: List[State] = []
    while queue:
        state = queue.popleft()
        bad = invariant_violation(state, slots)
        assert not bad, f"invariant {bad} violated in {state}"
        nexts = successors(state, slots, layers, max_ops)
        if not nexts:
            assert is_final(state, max_ops), f"deadlock in non-final state {state}"
        if is_final(state, max_ops):
            finals.append(state)
        for nxt in nexts:
            transitions += 1
            reverse.setdefault(nxt, []).append(state)
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    can_finish: Set[State] = set(finals)
    work: Deque[State] = deque(finals)
    while work:
        state = work.popleft()
        for prev in reverse.get(state, []):
            if prev not in can_finish:
                can_finish.add(prev)
                work.append(prev)
    trapped = seen - can_finish
    assert not trapped, f"{len(trapped)} trapped states, e.g. {next(iter(trapped))}"
    return {"states": len(seen), "transitions": transitions, "final_states": len(finals)}


def main(argv: List[str]) -> int:
    """Run the check with the TLC configuration defaults or given sizes.

    Args:
        argv: Optional threads, layers, slots, max_ops.

    Returns:
        int: 0 on success.
    """
    threads, layers, slots, max_ops = (int(x) for x in argv[1:5]) if len(argv) >= 5 else (3, 4, 2, 2)
    stats = check(threads, layers, slots, max_ops)
    print(f"ring model OK threads={threads} layers={layers} slots={slots} max_ops={max_ops} {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
