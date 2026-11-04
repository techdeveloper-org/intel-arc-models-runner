"""Hardware verification of GAP-11: stochastic sampling in stateful speculative decoding.

Runs a GPU target with an NPU draft (the FastDraft stateful path) and records, without
interpretation, what each case did: raised, or produced text. Run once per genai build
(baseline pip package vs locally built fix) and compare the JSON result files.

Usage:
    python scratch/verify_gap11_stochastic_speculative.py --label baseline
    python scratch/verify_gap11_stochastic_speculative.py --label fixed --target-device GPU
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import openvino_genai as ov_genai

ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = ROOT / "models" / "qwen-2.5-1.5b"
DRAFT_DIR = ROOT / "models" / "qwen-0.5b-npu-certified"
PROMPT = "Explain in two sentences why the sky is blue."
MAX_NEW_TOKENS = 40


def make_config(**overrides) -> "ov_genai.GenerationConfig":
    """Build a generation config for the speculative pipeline.

    Args:
        **overrides: GenerationConfig attributes to set.

    Returns:
        A configured GenerationConfig.
    """
    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = MAX_NEW_TOKENS
    cfg.num_assistant_tokens = 3
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def run_case(pipe, name: str, cfg) -> dict:
    """Run one generate() call and capture the outcome verbatim.

    Args:
        pipe: The LLMPipeline under test.
        name: Case identifier stored in the result.
        cfg: Generation config for this case.

    Returns:
        A dict with status, text or error, and elapsed seconds.
    """
    start = time.perf_counter()
    try:
        res = pipe.generate(PROMPT, cfg)
        text = res.texts[0] if hasattr(res, "texts") else str(res)
        return {"case": name, "status": "ok", "text": text,
                "elapsed_sec": round(time.perf_counter() - start, 3)}
    except Exception as exc:  # recorded as data, never swallowed silently
        return {"case": name, "status": "raised", "error_type": type(exc).__name__,
                "error": str(exc)[:400],
                "elapsed_sec": round(time.perf_counter() - start, 3)}


def main() -> int:
    """Run all cases and write the result file.

    Returns:
        Process exit code (0 when the file was written).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--target-device", default="GPU")
    parser.add_argument("--draft-device", default="NPU")
    args = parser.parse_args()

    draft = ov_genai.draft_model(str(DRAFT_DIR), args.draft_device)
    pipe = ov_genai.LLMPipeline(str(TARGET_DIR), args.target_device, draft_model=draft, KV_CACHE_PRECISION="f16")

    stoch = dict(do_sample=True, temperature=0.8, top_k=40, top_p=0.9)
    cases = [
        ("greedy_a", make_config()),
        ("greedy_b", make_config()),
        ("stochastic_seed1_a", make_config(rng_seed=1, **stoch)),
        ("stochastic_seed1_b", make_config(rng_seed=1, **stoch)),
        ("stochastic_seed2", make_config(rng_seed=2, **stoch)),
        ("stochastic_temp_only", make_config(rng_seed=3, do_sample=True, temperature=1.0)),
        ("stochastic_top_k1", make_config(rng_seed=3, do_sample=True, top_k=1)),
        ("unsupported_repetition_penalty",
         make_config(rng_seed=3, do_sample=True, repetition_penalty=1.2)),
    ]
    results = [run_case(pipe, name, cfg) for name, cfg in cases]

    by = {r["case"]: r for r in results}
    checks = {
        "greedy_deterministic": by["greedy_a"].get("text") == by["greedy_b"].get("text")
        and by["greedy_a"]["status"] == "ok",
        "same_seed_identical": by["stochastic_seed1_a"].get("text") == by["stochastic_seed1_b"].get("text")
        and by["stochastic_seed1_a"]["status"] == "ok",
        "different_seed_differs": by["stochastic_seed2"]["status"] == "ok"
        and by["stochastic_seed2"].get("text") != by["stochastic_seed1_a"].get("text"),
        "top_k1_equals_greedy": by["stochastic_top_k1"].get("text") == by["greedy_a"].get("text")
        and by["stochastic_top_k1"]["status"] == "ok",
        "unsupported_option_rejected": by["unsupported_repetition_penalty"]["status"] == "raised",
    }
    out = {
        "label": args.label,
        "genai_version": getattr(ov_genai, "__version__", "unknown"),
        "genai_file": str(Path(ov_genai.__file__).resolve()),
        "target_device": args.target_device,
        "draft_device": args.draft_device,
        "prompt": PROMPT,
        "results": results,
        "checks": checks,
    }
    dest = ROOT / "docs" / "benchmarks" / "results" / f"gap11_{args.label}_{args.target_device}_{args.draft_device}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"label": args.label, "checks": checks}, indent=2))
    print("wrote", dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
