"""Deterministic numerics audit of the verifier target for speculative sampling (GAP-11).

Three checks, no sampling noise:
  1. Chunk-size sweep: the next-token law from a chunk of c tokens vs one token, for c in CHUNKS.
  2. Later-token dependence: row 0 of a chunk [t, a, b] must not change with the later tokens a, b.
     If it does, exactness of speculative sampling is broken even relative to the verifier's own law.
  3. Cross-device reference: single-token rows vs a saved reference (CPU f32, DQ off).

One model per process; the compiled model is released before exit.

Usage:
    python scratch/gap11_numerics_audit.py --device CPU --kv-precision f32 \
        --prop INFERENCE_PRECISION_HINT=f32 --prop DYNAMIC_QUANTIZATION_GROUP_SIZE=0 \
        --save-ref ref_cpu.npz --out docs/benchmarks/results/gap11_audit_cpu_f32_dq_off.json
    python scratch/gap11_numerics_audit.py --device GPU --ref ref_cpu.npz --out ...
    python scratch/gap11_numerics_audit.py --device GPU --list-props
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
    "In machine learning, a neural network is trained by minimizing a loss function using gradient descent, "
    "and the weights are updated after each batch of examples from the training set. The key idea is",
    "def fibonacci(n):",
    "Explain why the sky is blue and why sunsets are red, using the physics of Rayleigh scattering of "
    "sunlight by the molecules of the atmosphere. In short,",
]
CHUNKS = (2, 3, 8, 32)
LATER_VARIANTS = ((11, 22), (3000, 4000), (151000, 5), (1052, 264), (7, 7))
TOP_K = 20


def filtered_probs(logits: np.ndarray) -> np.ndarray:
    """Return the temperature-1, top-k filtered softmax in float64.

    Args:
        logits: One row of vocabulary logits.

    Returns:
        Probability vector with mass only on the top-k entries.
    """
    x = logits.astype(np.float64)
    keep = np.argsort(-x, kind="stable")[:TOP_K]
    out = np.zeros_like(x)
    e = np.exp(x[keep] - x[keep].max())
    out[keep] = e / e.sum()
    return out


def tv(a: np.ndarray, b: np.ndarray) -> float:
    """Total-variation distance between the filtered laws of two logit rows.

    Args:
        a: First logits row.
        b: Second logits row.

    Returns:
        TV distance after temperature 1 and top-k.
    """
    return float(0.5 * np.abs(filtered_probs(a) - filtered_probs(b)).sum())


def run(request, token_ids: list, start: int) -> np.ndarray:
    """Infer token_ids on top of start cached tokens and return all logits rows.

    Args:
        request: Infer request holding KV state.
        token_ids: Tokens to feed now.
        start: Number of tokens already in the KV cache.

    Returns:
        Array of shape [len(token_ids), vocab].
    """
    n = len(token_ids)
    request.infer({
        "input_ids": np.array([token_ids], dtype=np.int64),
        "attention_mask": np.ones((1, start + n), dtype=np.int64),
        "position_ids": np.arange(start, start + n, dtype=np.int64)[None, :],
        "beam_idx": np.array([0], dtype=np.int32),
    })
    return np.array(request.get_tensor("logits").data)[0].copy()


def chunk_rows(request, ids: list, prefix_len: int, feed: list) -> np.ndarray:
    """Reset, prefill ids[:prefix_len] in one pass, then feed a chunk and return its rows.

    Args:
        request: Infer request.
        ids: Full token list used for the prefix.
        prefix_len: Number of prefix tokens to prefill.
        feed: Tokens fed as a single chunk after the prefill.

    Returns:
        Logits rows for the fed chunk.
    """
    request.reset_state()
    run(request, ids[:prefix_len], 0)
    return run(request, feed, prefix_len)


def main() -> None:
    """Run the requested audit and write the JSON summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--kv-precision", default="f16")
    parser.add_argument("--prop", action="append", default=[])
    parser.add_argument("--out", default="")
    parser.add_argument("--save-ref", default="")
    parser.add_argument("--ref", default="")
    parser.add_argument("--list-props", action="store_true")
    args = parser.parse_args()

    core = ov.Core()
    if args.list_props:
        names = [str(p) for p in core.get_property(args.device, "SUPPORTED_PROPERTIES")]
        print(json.dumps([n for n in names if any(k in n.upper() for k in
                          ("DYNAMIC", "QUANT", "SDPA", "ATTENTION", "PRECISION", "KV", "HINT"))], indent=1))
        return

    config = {"KV_CACHE_PRECISION": args.kv_precision} if args.kv_precision != "default" else {}
    for item in args.prop:
        key, value = item.split("=", 1)
        config[key] = int(value) if value.lstrip("-").isdigit() else value
    tokenizer = ov_genai.Tokenizer(str(MODEL_DIR))
    model = core.compile_model(str(MODEL_DIR / "openvino_model.xml"), args.device, config)
    request = model.create_infer_request()
    reference = dict(np.load(args.ref)) if args.ref else {}

    report = {"device": args.device, "kv_precision": args.kv_precision, "properties": args.prop, "prompts": []}
    saved = {}
    for pi, prompt in enumerate(PROMPTS):
        ids = [int(t) for t in tokenizer.encode(prompt).input_ids.data[0]]
        n = len(ids)
        single = chunk_rows(request, ids, n - 1, ids[n - 1:])[-1]
        saved[f"single_{pi}"] = single.astype(np.float32)
        entry = {"prompt": prompt[:40], "tokens": n, "sweep": {}, "later_token": {}}

        for c in CHUNKS:
            if c > n - 1:
                continue
            row = chunk_rows(request, ids, n - c, ids[n - c:])[-1]
            entry["sweep"][str(c)] = {"tv": tv(single, row), "max_logit_diff": float(np.max(np.abs(single - row)))}

        rows0 = [chunk_rows(request, ids, n - 1, ids[n - 1:])[0]]
        for later in LATER_VARIANTS:
            rows0.append(chunk_rows(request, ids, n - 1, [ids[n - 1], *later])[0])
        entry["later_token"] = {
            "max_logit_diff_row0": float(max(np.max(np.abs(rows0[0] - r)) for r in rows0[1:])),
            "max_tv_row0": float(max(tv(rows0[0], r) for r in rows0[1:])),
        }
        if f"single_{pi}" in reference:
            entry["single_vs_reference"] = {"tv": tv(single, reference[f"single_{pi}"]),
                                            "max_logit_diff": float(np.max(np.abs(single - reference[f"single_{pi}"])))}
        report["prompts"].append(entry)

    def agg(key_fn):
        vals = [key_fn(e) for e in report["prompts"]]
        vals = [v for v in vals if v is not None]
        return float(np.max(vals)) if vals else None

    report["summary"] = {
        "max_later_token_logit_diff_row0": agg(lambda e: e["later_token"]["max_logit_diff_row0"]),
        "max_later_token_tv_row0": agg(lambda e: e["later_token"]["max_tv_row0"]),
        "max_single_vs_reference_tv": agg(lambda e: e.get("single_vs_reference", {}).get("tv")),
        "max_sweep_tv": {str(c): agg(lambda e, c=c: e["sweep"].get(str(c), {}).get("tv")) for c in CHUNKS},
    }
    print(json.dumps(report["summary"], indent=1))
    if args.save_ref:
        np.savez(args.save_ref, **saved)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    del request, model


if __name__ == "__main__":
    main()
