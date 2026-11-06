"""Check whether the logits row a sampler would read lies on the fp16 value grid.

The implicit-grid sampler idea (GAP-11 round 5) only helps when every logit equals the float32
value of some float16 number. This script runs the stateful target on one device, feeds each
prompt, then greedily decodes a few more tokens, and for every produced logits row counts the
entries that are not exactly representable in float16. It also records the dtype of the output
tensor. Nothing is sampled and no repository source is touched.

Usage:
    python scratch/gap11_logits_fp16_grid.py --device GPU --out docs/benchmarks/results/gap11_logits_fp16_grid.json

Exit codes: 0 measured, 2 not run (device or model missing; reason written to the JSON).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from arc_runner.core.logging import get_logger  # noqa: E402

LOGGER = get_logger("gap11.fp16_grid")
MODEL_DIR = ROOT / "models" / "qwen-2.5-1.5b"
PROMPTS = [
    "Once upon a time",
    "The capital of France is",
    "In machine learning, a neural network",
    "def fibonacci(n):",
]
DECODE_STEPS = 4


def fp16_grid_counts(row: np.ndarray) -> tuple[int, int]:
    """Count logits that are not on the float16 grid, and non-finite logits.

    Args:
        row: One row of float32 logits.

    Returns:
        A pair (off_grid, non_finite). A finite value is on the grid when converting it to
        float16 and back reproduces it exactly; values beyond the float16 range count as off grid.
    """
    finite = np.isfinite(row)
    non_finite = int(row.size - np.count_nonzero(finite))
    values = row[finite].astype(np.float32)
    with np.errstate(over="ignore"):
        round_trip = values.astype(np.float16).astype(np.float32)
    off_grid = int(np.count_nonzero(round_trip != values))
    return off_grid, non_finite


def infer_row(request, token_ids: list, start: int) -> tuple[np.ndarray, str]:
    """Run one infer and return the last logits row and the tensor dtype name.

    Args:
        request: Infer request holding KV state.
        token_ids: Tokens to feed now.
        start: Number of tokens already in the KV cache.

    Returns:
        The final logits row as a float32 copy and the element type name of the output tensor.
    """
    n = len(token_ids)
    request.infer({
        "input_ids": np.array([token_ids], dtype=np.int64),
        "attention_mask": np.ones((1, start + n), dtype=np.int64),
        "position_ids": np.arange(start, start + n, dtype=np.int64)[None, :],
        "beam_idx": np.array([0], dtype=np.int32),
    })
    tensor = request.get_tensor("logits")
    return np.array(tensor.data)[0, -1, :].astype(np.float32), str(tensor.get_element_type())


def write_result(path: str, payload: dict) -> None:
    """Write the JSON result when a path was given and always print it.

    Args:
        path: Output path, or an empty string to only print.
        payload: JSON-serialisable result.
    """
    text = json.dumps(payload, indent=2)
    print(text)
    if path:
        Path(path).write_text(text + "\n", encoding="utf-8")


def measure(device: str) -> dict:
    """Run the prompts on the device and collect per-row grid counts.

    Args:
        device: OpenVINO device name such as GPU or CPU.

    Returns:
        The result dictionary with one entry per logits row.

    Raises:
        FileNotFoundError: If the model directory is missing.
        RuntimeError: If compilation or inference fails on the device.
    """
    import openvino as ov
    import openvino_genai as ov_genai

    xml_path = MODEL_DIR / "openvino_model.xml"
    if not xml_path.is_file():
        raise FileNotFoundError(str(xml_path))
    tokenizer = ov_genai.Tokenizer(str(MODEL_DIR))
    request = ov.Core().compile_model(str(xml_path), device).create_infer_request()
    rows, dtype_names = [], set()
    for prompt_id, prompt in enumerate(PROMPTS):
        ids = [int(t) for t in tokenizer.encode(prompt).input_ids.data[0]]
        request.reset_state()
        logits, dtype_name = infer_row(request, ids, 0)
        position = len(ids)
        for step in range(DECODE_STEPS + 1):
            off_grid, non_finite = fp16_grid_counts(logits)
            dtype_names.add(dtype_name)
            rows.append({"prompt_id": prompt_id, "step": step, "vocab": int(logits.size),
                         "off_grid": off_grid, "non_finite": non_finite})
            LOGGER.info("logits row checked", extra={"device": device, "prompt_id": prompt_id,
                                                     "step": step, "off_grid": off_grid})
            if step == DECODE_STEPS:
                break
            logits, dtype_name = infer_row(request, [int(np.argmax(logits))], position)
            position += 1
    total_off = sum(r["off_grid"] for r in rows)
    return {"status": "measured", "device": device, "output_dtypes": sorted(dtype_names),
            "rows": rows, "rows_checked": len(rows), "total_off_grid": total_off,
            "all_rows_on_fp16_grid": total_off == 0}


def main() -> int:
    """Parse arguments, run the measurement and write the result.

    Returns:
        0 when measured, 2 when the run could not be performed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    try:
        result = measure(args.device)
    except (FileNotFoundError, RuntimeError, ImportError) as error:
        LOGGER.error("fp16 grid check not run", extra={"device": args.device, "error": str(error)})
        write_result(args.out, {"status": "not_run", "device": args.device, "reason": str(error)})
        return 2
    write_result(args.out, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
