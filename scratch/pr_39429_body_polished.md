### Details:
 - Fix uninformative assertion failure when deserializing OpenVINO IR models with missing or disconnected input port edges.
 - Validate input connections in `XmlDeserializer::parse_function` prior to node creation, reporting exact layer friendly name, layer type, layer ID, and missing input port index.
 - Enhance `ov::Node::get_input_*` assertion messages with friendly name, node type, requested index, and total available input ports.
 - Add unit tests in `src/frontends/ir/tests/disconnected_port_diagnostic_test.cpp` (registered under `ov_ir_frontend_tests`) and out-of-bounds node input assertion tests in `src/core/tests/node_test.cpp`.

### Tickets:
 - None (Defensive hardening for IR deserializer diagnostics)

### AI Assistance:
 - AI assistance used: yes
 - If yes: Architecture consultation, test fixture synthesis, and diagnostic message formatting.
 - Human validation performed:
   1. Built `openvino` and `openvino_ir_frontend` on Windows 11 with Visual Studio 18 2026 / CMake.
   2. Validated with positive, negative, and edge-case test models.
   3. Live verification in real-world LLM inference pipeline on physical Intel Core Ultra 5 125H silicon (Arc Xe-LPG iGPU & AI Boost NPU 3720).
