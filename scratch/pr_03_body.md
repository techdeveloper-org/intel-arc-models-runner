### Details:
 - Fix uninformative exception when deserializing OpenVINO IR models with constant tensor size or shape mismatches.
 - Enhance `XmlDeserializer::set_constant_num_buffer` to report exact layer friendly name, layer type, layer ID, element type, shape string, expected minimum memory size, actual attribute size, and byte delta.
 - Include safe fallbacks (`[Unnamed Constant Node]` and `[Unknown ID]`) for malformed or hand-edited XML nodes where attributes are empty.
 - Enrich `ov::op::v0::Constant` constructor and reshape assertion messages with element type, shape dimensions, and byte delta.
 - Add comprehensive unit test suite in `src/frontends/ir/tests/const_shape_mismatch_diagnostic_test.cpp` registered under `ov_ir_frontend_tests`.

### Tickets:
 - None (Defensive hardening for IR constant deserialization diagnostics)

### AI Assistance:
 - AI assistance used: yes
 - If yes: Architecture consultation, test fixture synthesis, and diagnostic message formatting.
 - Human validation performed:
   1. Built `openvino` and `openvino_ir_frontend` on Windows 11 with Visual Studio 18 2026 / CMake.
   2. Validated with positive, negative, and edge-case test models.
   3. Live verification in real-world `intel-arc-models-runner` inference pipeline on physical Intel Core Ultra 5 125H silicon (Arc Xe-LPG iGPU & AI Boost NPU 3720), proving 0 regressions on Qwen 2.5 1.5B (2,059 constants) and DeepSeek R1 1.5B (1,973 constants).
