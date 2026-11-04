"""Measure how much the target's next-token law depends on chunked vs single-token inference.

Speculative verification feeds K+1 tokens to the target at once, while plain decoding feeds one
token at a time. This script computes the same next-token distribution both ways on the stateful
IR directly and reports the total-variation distance after temperature 1 and top_k 20, per device
and KV precision. A non-zero distance is a numeric confound for any speculative-vs-plain
distribution test, independent of the sampling algorithm.

Usage:
    python scratch/measure_chunk_vs_single_logits.py --device GPU --kv-precision f16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import openvino as ov
import openvino_genai as ov_genai

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "qwen-2.5-1.5b"
PROMPTS = [
    "Once upon a time",
    "The capital of France is",
    "In machine learning, a neural network",
    "Explain why the sky is blue and",
    "def fibonacci(n):",
    "The quick brown fox jumps over the lazy dog and then",
]
TOP_K = 20
CHUNK = 3


def filtered_probs(logits: np.ndarray) -> np.ndarray:
    """Apply temperature 1, top-k and softmax in float64.

    Args:
        logits: One row of vocabulary logits.

    Returns:
        A probability vector with mass only on the top-k entries.
    """
    x = logits.astype(np.float64)
    keep = np.argsort(-x, kind="stable")[:TOP_K]
    out = np.zeros_like(x)
    e = np.exp(x[keep] - x[keep].max())
    out[keep] = e / e.sum()
    return out


def last_logits(request, token_ids: list, start: int) -> np.ndarray:
    """Run one infer on token_ids and return the final row of logits.

    Args:
        request: Infer request holding KV state.
        token_ids: Tokens to feed now.
        start: Number of tokens already in the KV cache.

    Returns:
        The logits row for the last fed token.
    """
    n = len(token_ids)
    request.infer({
        "input_ids": np.array([token_ids], dtype=np.int64),
        "attention_mask": np.ones((1, start + n), dtype=np.int64),
        "position_ids": np.arange(start, start + n, dtype=np.int64)[None, :],
        "beam_idx": np.array([0], dtype=np.int32),
    })
    return np.array(request.get_tensor("logits").data)[0, -1, :].copy()


def main() -> None:
    """Measure the distance per prompt and print a JSON summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--kv-precision", default="f16")
    parser.add_argument("--out", default="", help="write the JSON summary to this path")
    parser.add_argument("--prop", action="append", default=[], help="extra KEY=VALUE compile property")
    args = parser.parse_args()

    tokenizer = ov_genai.Tokenizer(str(MODEL_DIR))
    config = {"KV_CACHE_PRECISION": args.kv_precision} if args.kv_precision != "default" else {}
    for item in args.prop:
        key, value = item.split("=", 1)
        config[key] = int(value) if value.lstrip("-").isdigit() else value
    model = ov.Core().compile_model(str(MODEL_DIR / "openvino_model.xml"), args.device, config)
    request = model.create_infer_request()

    rows = []
    for prompt in PROMPTS:
        ids = [int(t) for t in tokenizer.encode(prompt).input_ids.data[0]]
        n = len(ids)
        request.reset_state()
        last_logits(request, ids[: n - CHUNK], 0)
        for i in range(n - CHUNK, n - 1):
            last_logits(request, [ids[i]], i)
        single = last_logits(request, [ids[n - 1]], n - 1)

        request.reset_state()
        last_logits(request, ids[: n - CHUNK], 0)
        chunked = last_logits(request, ids[n - CHUNK:], n - CHUNK)

        request.reset_state()
        last_logits(request, ids[: n - CHUNK], 0)
        for i in range(n - CHUNK, n - 1):
            last_logits(request, [ids[i]], i)
        single_again = last_logits(request, [ids[n - 1]], n - 1)

        p, q = filtered_probs(single), filtered_probs(chunked)
        rows.append({
            "prompt": prompt,
            "max_abs_logit_diff": float(np.max(np.abs(single - chunked))),
            "tv_distance": float(0.5 * np.abs(p - q).sum()),
            "single_vs_single_max_diff": float(np.max(np.abs(single - single_again))),
            "argmax_equal": bool(np.argmax(single) == np.argmax(chunked)),
        })
    summary = {"device": args.device, "kv_precision": args.kv_precision, "properties": args.prop,
               "rows": rows, "mean_tv": float(np.mean([r["tv_distance"] for r in rows]))}
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
