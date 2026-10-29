---------------------------- MODULE RingSlots ----------------------------
(***************************************************************************)
(* Model of FlashStreamWeightManager's direct-mapped ring buffer:          *)
(* layer L always lives in slot L mod N; a slot is EMPTY, PENDING (one     *)
(* loader is reading it) or READY. Threads acquire layers, optionally wait *)
(* for an in-flight load, and the manager can be closed. Consumers hold    *)
(* their own references to loaded arrays, so there are no pin counts.      *)
(***************************************************************************)
EXTENDS Naturals, Integers, FiniteSets

CONSTANTS Threads, Layers, N, MaxOps, NONE

VARIABLES slotLayer, slotState, loader, pc, want, ops, closing, closed, served

vars == <<slotLayer, slotState, loader, pc, want, ops, closing, closed, served>>

Slots == 0..(N - 1)
States == {"EMPTY", "PENDING", "READY"}
Pcs == {"idle", "acq", "wait", "loading", "done", "error"}

TypeOK ==
    /\ slotLayer \in [Slots -> Layers \cup {-1}]
    /\ slotState \in [Slots -> States]
    /\ loader \in [Slots -> Threads \cup {NONE}]
    /\ pc \in [Threads -> Pcs]
    /\ want \in [Threads -> Layers]
    /\ ops \in [Threads -> 0..MaxOps]
    /\ closing \in BOOLEAN
    /\ closed \in BOOLEAN
    /\ served \in [Threads -> Layers \cup {-1}]

Init ==
    /\ slotLayer = [s \in Slots |-> -1]
    /\ slotState = [s \in Slots |-> "EMPTY"]
    /\ loader = [s \in Slots |-> NONE]
    /\ pc = [t \in Threads |-> "idle"]
    /\ want \in [Threads -> Layers]
    /\ ops = [t \in Threads |-> 0]
    /\ closing = FALSE
    /\ closed = FALSE
    /\ served = [t \in Threads |-> -1]

SlotOf(l) == l % N

Begin(t) ==
    /\ pc[t] = "idle" /\ ops[t] < MaxOps
    /\ \E l \in Layers : want' = [want EXCEPT ![t] = l]
    /\ pc' = [pc EXCEPT ![t] = "acq"]
    /\ ops' = [ops EXCEPT ![t] = @ + 1]
    /\ UNCHANGED <<slotLayer, slotState, loader, closing, closed, served>>

Decide(t) ==
    LET l == want[t] s == SlotOf(l) IN
    /\ pc[t] = "acq"
    /\ \/ /\ closing
          /\ pc' = [pc EXCEPT ![t] = "error"]
          /\ UNCHANGED <<slotLayer, slotState, loader, served>>
       \/ /\ ~closing /\ slotState[s] = "READY" /\ slotLayer[s] = l
          /\ pc' = [pc EXCEPT ![t] = "done"]
          /\ served' = [served EXCEPT ![t] = l]
          /\ UNCHANGED <<slotLayer, slotState, loader>>
       \/ /\ ~closing /\ slotState[s] = "PENDING"
          /\ pc' = [pc EXCEPT ![t] = "wait"]
          /\ UNCHANGED <<slotLayer, slotState, loader, served>>
       \/ /\ ~closing /\ slotState[s] \in {"EMPTY", "READY"} /\ ~(slotState[s] = "READY" /\ slotLayer[s] = l)
          /\ slotLayer' = [slotLayer EXCEPT ![s] = l]
          /\ slotState' = [slotState EXCEPT ![s] = "PENDING"]
          /\ loader' = [loader EXCEPT ![s] = t]
          /\ pc' = [pc EXCEPT ![t] = "loading"]
          /\ UNCHANGED served
    /\ UNCHANGED <<want, ops, closing, closed>>

Wake(t) ==
    /\ pc[t] = "wait" /\ slotState[SlotOf(want[t])] # "PENDING"
    /\ pc' = [pc EXCEPT ![t] = "acq"]
    /\ UNCHANGED <<slotLayer, slotState, loader, want, ops, closing, closed, served>>

LoadDone(t) ==
    LET s == SlotOf(want[t]) IN
    /\ pc[t] = "loading"
    /\ slotState' = [slotState EXCEPT ![s] = "READY"]
    /\ loader' = [loader EXCEPT ![s] = NONE]
    /\ pc' = [pc EXCEPT ![t] = "done"]
    /\ served' = [served EXCEPT ![t] = want[t]]
    /\ UNCHANGED <<slotLayer, want, ops, closing, closed>>

Finish(t) ==
    /\ pc[t] \in {"done", "error"}
    /\ pc' = [pc EXCEPT ![t] = "idle"]
    /\ served' = [served EXCEPT ![t] = -1]
    /\ UNCHANGED <<slotLayer, slotState, loader, want, ops, closing, closed>>

BeginClose ==
    /\ ~closing
    /\ closing' = TRUE
    /\ UNCHANGED <<slotLayer, slotState, loader, pc, want, ops, closed, served>>

FinishClose ==
    /\ closing /\ ~closed
    /\ \A s \in Slots : loader[s] = NONE
    /\ closed' = TRUE
    /\ slotLayer' = [s \in Slots |-> -1]
    /\ slotState' = [s \in Slots |-> "EMPTY"]
    /\ UNCHANGED <<loader, pc, want, ops, closing, served>>

Next ==
    \/ \E t \in Threads : Begin(t) \/ Decide(t) \/ Wake(t) \/ LoadDone(t) \/ Finish(t)
    \/ BeginClose \/ FinishClose

Fairness ==
    /\ \A t \in Threads : WF_vars(Decide(t)) /\ WF_vars(Wake(t)) /\ WF_vars(LoadDone(t)) /\ WF_vars(Finish(t))
    /\ WF_vars(FinishClose)

Spec == Init /\ [][Next]_vars /\ Fairness

\* A slot is PENDING exactly while one loader owns it.
SingleLoader == \A s \in Slots : (slotState[s] = "PENDING") <=> (loader[s] # NONE)

\* Direct mapping: a slot only ever holds layers that map to it, so one layer never occupies two slots.
DirectMapped == \A s \in Slots : slotLayer[s] # -1 => SlotOf(slotLayer[s]) = s

\* A thread is only told "done" for the layer it asked for.
ServedMatches == \A t \in Threads : pc[t] = "done" => served[t] = want[t]

\* Two threads never load into the same slot at once.
NoConcurrentLoadsPerSlot ==
    \A s \in Slots : Cardinality({t \in Threads : pc[t] = "loading" /\ SlotOf(want[t]) = s}) <= 1

\* After close completes no load is in flight and nothing is resident.
ClosedIsClean ==
    closed => (\A s \in Slots : loader[s] = NONE /\ slotState[s] = "EMPTY")

Safety == TypeOK /\ SingleLoader /\ DirectMapped /\ ServedMatches /\ NoConcurrentLoadsPerSlot /\ ClosedIsClean

\* Every started acquire ends in done or error.
Progress == \A t \in Threads : (pc[t] = "acq") ~> (pc[t] \in {"done", "error"})

=============================================================================
