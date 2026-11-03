Hi @as-suvorov,

Here is a minimal, self-contained Python reproducer using only `openvino` and `openvino-genai`:

```python
import tempfile
import json
from pathlib import Path
import openvino as ov
import openvino.opset13 as ops
import openvino_genai

# Synthesize a model with valid dynamic shapes but missing required 'input_ids' input
param1 = ops.parameter([-1, -1], ov.Type.i64, name="wrong_input")
param1.get_output_tensor(0).set_names({"wrong_input"})
param2 = ops.parameter([-1, -1], ov.Type.i64, name="attention_mask")
param2.get_output_tensor(0).set_names({"attention_mask"})
param3 = ops.parameter([-1], ov.Type.i32, name="beam_idx")
param3.get_output_tensor(0).set_names({"beam_idx"})

res = ops.result(ops.add(param1, param2), name="logits")
res.get_output_tensor(0).set_names({"logits"})
model = ov.Model([res], [param1, param2, param3], "malformed_model")

with tempfile.TemporaryDirectory() as tmp_dir:
    p = Path(tmp_dir)
    ov.save_model(model, p / "openvino_model.xml")

    # Minimal tokenizer and config definitions
    inp = ops.parameter([1], ov.Type.string, name="string_input")
    ov.save_model(ov.Model([ops.result(inp, name="tokens")], [inp]), p / "openvino_tokenizer.xml")
    with open(p / "config.json", "w") as f:
        json.dump({"model_type": "custom", "vocab_size": 32000, "eos_token_id": 2}, f)

    print("1. Instantiating LLMPipeline...")
    # Pipeline instantiates and compiles to device without validating graph inputs/outputs
    pipe = openvino_genai.LLMPipeline(str(p), "CPU")
    print("2. LLMPipeline initialized without warning.")

    # Fails only when execution reaches low-level infer_request:
    pipe.generate("Hello world", max_new_tokens=5)
```

### Output:
```text
1. Instantiating LLMPipeline...
2. LLMPipeline initialized without warning.
RuntimeError: Exception from src\inference\src\cpp\infer_request.cpp:194:
Check '::getPort(port, name, {_impl->get_inputs(), _impl->get_outputs()})' failed at src\inference\src\cpp\infer_request.cpp:194:
Port for tensor name attention_mask was not found.
```

### Observation:
1. When a model with missing primary inputs (`input_ids`), missing outputs (`logits`), or disconnected parameters is provided, `LLMPipeline` constructs successfully and performs plugin compilation.
2. The error only surfaces later during inference execution (`generate()`) inside low-level runtime request handlers or KV-cache initialization loops.
3. Catching these malformed graph states early in `LLMPipeline` constructor (pre-flight validation of input/output contracts and basic connectivity) allows failing fast with clear, actionable diagnostics before expensive compilation.
