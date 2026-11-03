Thank you for catching this! You are completely right.

### Resolution:
1. **Mathematical Set Difference**: Removed the raw edge count check (`connected_edges < expected_input_ports`). We now collect all unique incoming destination port IDs in a `std::unordered_set<size_t> connected_ports` and compute the strict set difference against declared input ports:
   $$\mathcal{M} = \mathcal{P}_{\text{in}} \setminus \mathcal{D}$$
   If `!missing_ports.empty()`, validation immediately throws an `ov::Exception` listing all missing XML port IDs. This is completely invariant to duplicate destination edges and non-contiguous port numbering.
2. **Regression & Positive Tests Added**:
   - Added `duplicate_destination_edges_with_noncontiguous_ports_throws_descriptive_exception` in `src/frontends/ir/tests/disconnected_port_diagnostic_test.cpp` testing declared ports `10` and `20` with duplicate incoming edges targeting port `10`. Confirmed that it immediately throws and reports missing port `[20]`.
   - Added `valid_connected_ports_deserializes_successfully` confirming zero regressions on valid graphs.

Pushed in commit a795d087.
