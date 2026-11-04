"""Two-sample distribution test: speculative sampling vs plain target sampling on silicon.

Draws N short generations from the plain target pipeline and N from the speculative pipeline
(same prompt, same sampling config, distinct seeds) and compares the law of the first
NUM_TOKENS generated token ids with a chi-square homogeneity test. Speculative sampling is
exact, so the two laws must coincide; a failure here means a real distribution bug.

Usage:
    python scratch/verify_gap11_distribution.py --n 300 --label gpu_npu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import openvino_genai as ov_genai
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = ROOT / "models" / "qwen-2.5-1.5b"
DRAFT_DIR = ROOT / "models" / "qwen-0.5b-npu-certified"
PROMPT = "Once upon a time"
NUM_TOKENS = 3
MIN_EXPECTED = 5.0
ALPHA = 1e-3
NUM_ASSISTANT = 3


def sample_tokens(pipe, tokenizer, seed: int, speculative: bool) -> tuple:
    """Generate NUM_TOKENS tokens with a fixed sampling config and a given seed.

    Args:
        pipe: Pipeline to run.
        tokenizer: Tokenizer used to encode the prompt.
        seed: rng_seed for this draw.
        speculative: Whether to set num_assistant_tokens.

    Returns:
        The generated token ids as a tuple.
    """
    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = NUM_TOKENS
    cfg.min_new_tokens = 0
    cfg.do_sample = True
    cfg.temperature = 1.0
    cfg.top_k = 20
    cfg.top_p = 1.0
    cfg.rng_seed = seed
    if speculative:
        cfg.num_assistant_tokens = NUM_ASSISTANT
    res = pipe.generate(tokenizer.encode(PROMPT), cfg)
    return tuple(int(t) for t in res.tokens[0][:NUM_TOKENS])


def homogeneity(counts_a: Counter, counts_b: Counter) -> dict:
    """Chi-square test of homogeneity between two categorical samples.

    Categories whose pooled expected count is below MIN_EXPECTED are merged into one bin.

    Args:
        counts_a: Category counts of sample A.
        counts_b: Category counts of sample B.

    Returns:
        A dict with the statistic, degrees of freedom, p-value and bin count.
    """
    na, nb = sum(counts_a.values()), sum(counts_b.values())
    keys = set(counts_a) | set(counts_b)
    rows, rare = [], [0, 0]
    for k in keys:
        a, b = counts_a.get(k, 0), counts_b.get(k, 0)
        pooled = a + b
        if pooled * min(na, nb) / (na + nb) < MIN_EXPECTED:
            rare[0] += a
            rare[1] += b
        else:
            rows.append((a, b))
    if rare[0] + rare[1] > 0:
        rows.append(tuple(rare))
    stat = 0.0
    for a, b in rows:
        pooled = a + b
        ea, eb = pooled * na / (na + nb), pooled * nb / (na + nb)
        stat += (a - ea) ** 2 / ea + (b - eb) ** 2 / eb
    dof = len(rows) - 1
    return {"statistic": stat, "dof": dof, "bins": len(rows),
            "p_value": float(chi2.sf(stat, dof)) if dof > 0 else None}


def main() -> int:
    """Run both samplers, test homogeneity, and write the result file.

    Returns:
        0 when the speculative law is not rejected at ALPHA, else 1.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--label", default="gpu_npu")
    parser.add_argument("--target-device", default="GPU")
    parser.add_argument("--kv-precision", default="f16")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--prop", action="append", default=[], help="extra KEY=VALUE target property")
    args = parser.parse_args()
    global NUM_ASSISTANT
    NUM_ASSISTANT = args.k
    extra = {}
    for item in args.prop:
        key, value = item.split("=", 1)
        extra[key] = int(value) if value.lstrip("-").isdigit() else value
    props = {"KV_CACHE_PRECISION": args.kv_precision} if args.kv_precision != "default" else {}
    props.update(extra)

    plain = ov_genai.LLMPipeline(str(TARGET_DIR), args.target_device, **props)
    tok = plain.get_tokenizer()
    plain_draws = []
    start = time.perf_counter()
    for i in range(args.n):
        plain_draws.append(sample_tokens(plain, tok, 1000 + i, False))
    plain_sec = time.perf_counter() - start
    del plain

    draft = ov_genai.draft_model(str(DRAFT_DIR), "NPU")
    spec = ov_genai.LLMPipeline(str(TARGET_DIR), args.target_device, draft_model=draft, **props)
    stok = spec.get_tokenizer()
    spec_draws = []
    start = time.perf_counter()
    for i in range(args.n):
        spec_draws.append(sample_tokens(spec, stok, 500000 + i, True))
    spec_sec = time.perf_counter() - start

    report = {"n_per_arm": args.n, "alpha": ALPHA, "prompt": PROMPT,
              "plain_seconds": round(plain_sec, 1), "spec_seconds": round(spec_sec, 1),
              "tests": {}}
    for pos in range(NUM_TOKENS):
        report["tests"][f"token_{pos + 1}_marginal"] = homogeneity(
            Counter(d[pos] for d in plain_draws), Counter(d[pos] for d in spec_draws))
    report["tests"]["joint_first_3"] = homogeneity(Counter(plain_draws), Counter(spec_draws))
    report["plain_distinct_joint"] = len(set(plain_draws))
    report["spec_distinct_joint"] = len(set(spec_draws))
    report["rejected_at_alpha"] = {
        k: (v["p_value"] is not None and v["p_value"] < ALPHA) for k, v in report["tests"].items()}
    for pos in (1,):
        ca, cb = Counter(d[pos] for d in plain_draws), Counter(d[pos] for d in spec_draws)
        report[f"token_{pos + 1}_top_bins"] = sorted(
            ({"token": k, "plain": ca.get(k, 0), "spec": cb.get(k, 0)} for k in set(ca) | set(cb)),
            key=lambda r: -(r["plain"] + r["spec"]))[:12]
    report["draws"] = {"plain": plain_draws, "spec": spec_draws}
    report["config"] = {"target_device": args.target_device, "kv_precision": args.kv_precision, "k": args.k, "properties": args.prop}
    dest = ROOT / "docs" / "benchmarks" / "results" / f"gap11_distribution_{args.label}.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "draws"}, indent=2))
    return 1 if any(report["rejected_at_alpha"].values()) else 0


if __name__ == "__main__":
    sys.exit(main())
