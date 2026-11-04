"""End-to-end distribution check on a batch-invariant target (CPU, f32, DQ off) with an NPU draft.

One process runs one arm and appends one JSON line per draw, so a killed run keeps its data:
  --arm plain  : plain target sampling
  --arm spec   : speculative sampling (NPU draft)
  --arm law    : exact next-token laws of the target (token 1 and token 2 marginal), no sampling
  --arm report : one-sample goodness of fit and two-sample homogeneity from the files above

Usage:
    python scratch/verify_gap11_cpu_invariant.py --arm plain --n 3000
    python scratch/verify_gap11_cpu_invariant.py --arm spec  --n 3000
    python scratch/verify_gap11_cpu_invariant.py --arm law
    python scratch/verify_gap11_cpu_invariant.py --arm report
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "docs" / "benchmarks" / "results"
TARGET_DIR = ROOT / "models" / "qwen-2.5-1.5b"
DRAFT_DIR = ROOT / "models" / "qwen-0.5b-npu-certified"
PROMPT = "Once upon a time"
NUM_TOKENS = 3
TOP_K = 20
K = 3
PROPS = {"INFERENCE_PRECISION_HINT": "f32", "DYNAMIC_QUANTIZATION_GROUP_SIZE": 0, "KV_CACHE_PRECISION": "f32"}
ALPHA = 1e-3
DEVICE = "CPU"
TAG = "cpu_invariant"
MIN_EXPECTED = 5.0


def rss_mb() -> float:
    """Return the resident set size of this process in MB, or -1 if psutil is missing."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 2**20
    except ImportError:
        return -1.0


def draw_file(arm: str) -> Path:
    """Return the JSONL path that stores the draws of one arm."""
    return RES / f"gap11_{TAG}_{arm}.jsonl"


def run_arm(arm: str, n: int, seed_base: int) -> None:
    """Draw n short generations and append each to the arm's JSONL file.

    Args:
        arm: "plain" or "spec".
        n: Number of draws wanted in total; existing lines count.
        seed_base: First rng_seed; draw i uses seed_base + i.
    """
    import openvino_genai as ov_genai

    path = draw_file(arm)
    done = sum(1 for _ in path.open()) if path.exists() else 0
    kwargs = dict(PROPS)
    if arm == "spec":
        draft = ov_genai.draft_model(str(DRAFT_DIR), "NPU")
        pipe = ov_genai.LLMPipeline(str(TARGET_DIR), DEVICE, draft_model=draft, **kwargs)
    else:
        pipe = ov_genai.LLMPipeline(str(TARGET_DIR), DEVICE, **kwargs)
    tokenizer = pipe.get_tokenizer()
    encoded = tokenizer.encode(PROMPT)
    start = time.perf_counter()
    with path.open("a", encoding="utf-8") as out:
        for i in range(done, n):
            cfg = ov_genai.GenerationConfig()
            cfg.max_new_tokens = NUM_TOKENS
            cfg.do_sample = True
            cfg.temperature = 1.0
            cfg.top_k = TOP_K
            cfg.top_p = 1.0
            cfg.rng_seed = seed_base + i
            if arm == "spec":
                cfg.num_assistant_tokens = K
            res = pipe.generate(encoded, cfg)
            tokens = [int(t) for t in res.tokens[0][:NUM_TOKENS]]
            out.write(json.dumps({"i": i, "tokens": tokens}) + "\n")
            out.flush()
            if (i + 1) % 100 == 0:
                print(f"{arm} {i + 1}/{n} elapsed={time.perf_counter() - start:.0f}s rss_mb={rss_mb():.0f}", flush=True)


def filtered(logits: np.ndarray) -> np.ndarray:
    """Return the temperature-1 top-k softmax of one logits row in float64."""
    x = logits.astype(np.float64)
    keep = np.argsort(-x, kind="stable")[:TOP_K]
    out = np.zeros_like(x)
    e = np.exp(x[keep] - x[keep].max())
    out[keep] = e / e.sum()
    return out


def compute_laws() -> None:
    """Compute q(t1 | prompt) and the token-2 marginal exactly, one token at a time."""
    import openvino as ov
    import openvino_genai as ov_genai

    tokenizer = ov_genai.Tokenizer(str(TARGET_DIR))
    ids = [int(t) for t in tokenizer.encode(PROMPT).input_ids.data[0]]
    config = {k: v for k, v in PROPS.items()}
    model = ov.Core().compile_model(str(TARGET_DIR / "openvino_model.xml"), DEVICE, config)
    request = model.create_infer_request()

    def feed(tokens: list, start: int) -> np.ndarray:
        n = len(tokens)
        request.infer({
            "input_ids": np.array([tokens], dtype=np.int64),
            "attention_mask": np.ones((1, start + n), dtype=np.int64),
            "position_ids": np.arange(start, start + n, dtype=np.int64)[None, :],
            "beam_idx": np.array([0], dtype=np.int32),
        })
        return np.array(request.get_tensor("logits").data)[0, -1].copy()

    request.reset_state()
    q1 = filtered(feed(ids, 0))
    marginal2 = np.zeros_like(q1)
    for t1 in np.nonzero(q1)[0]:
        request.reset_state()
        feed(ids, 0)
        marginal2 += q1[t1] * filtered(feed([int(t1)], len(ids)))
    np.savez(RES / f"gap11_{TAG}_laws.npz", q1=q1, marginal2=marginal2)
    print("laws saved; support sizes", int((q1 > 0).sum()), int((marginal2 > 0).sum()))


def gof(counts: Counter, law: np.ndarray) -> dict:
    """One-sample chi-square test of observed counts against an exact law."""
    from scipy.stats import chi2

    total = sum(counts.values())
    impossible = sum(c for t, c in counts.items() if law[t] == 0.0)
    stat, bins, pool_e, pool_o = 0.0, 0, 0.0, 0.0
    for t in np.nonzero(law)[0]:
        e = law[t] * total
        o = counts.get(int(t), 0)
        if e < MIN_EXPECTED:
            pool_e += e
            pool_o += o
        else:
            stat += (o - e) ** 2 / e
            bins += 1
    if pool_e > 0:
        stat += (pool_o - pool_e) ** 2 / pool_e
        bins += 1
    dof = bins - 1
    return {"n": total, "statistic": stat, "dof": dof, "p_value": float(chi2.sf(stat, dof)),
            "impossible_hits": impossible}


def homogeneity(a: Counter, b: Counter) -> dict:
    """Two-sample chi-square test of homogeneity with rare bins merged."""
    from scipy.stats import chi2

    na, nb = sum(a.values()), sum(b.values())
    rows, rare = [], [0, 0]
    for k in set(a) | set(b):
        x, y = a.get(k, 0), b.get(k, 0)
        if (x + y) * min(na, nb) / (na + nb) < MIN_EXPECTED:
            rare[0] += x
            rare[1] += y
        else:
            rows.append((x, y))
    if sum(rare) > 0:
        rows.append(tuple(rare))
    stat = 0.0
    for x, y in rows:
        ex, ey = (x + y) * na / (na + nb), (x + y) * nb / (na + nb)
        stat += (x - ex) ** 2 / ex + (y - ey) ** 2 / ey
    dof = len(rows) - 1
    return {"statistic": stat, "dof": dof, "p_value": float(chi2.sf(stat, dof)) if dof > 0 else None}


def report() -> int:
    """Run the tests on the stored draws and write the summary JSON."""
    laws = np.load(RES / f"gap11_{TAG}_laws.npz")
    draws = {}
    for arm in ("plain", "spec"):
        if not draw_file(arm).exists():
            continue
        draws[arm] = [tuple(json.loads(line)["tokens"]) for line in draw_file(arm).open()]
    out = {"alpha": ALPHA, "n": {a: len(d) for a, d in draws.items()}, "props": PROPS, "k": K, "tests": {}}
    for arm, d in draws.items():
        out["tests"][f"{arm}_token1_vs_exact_law"] = gof(Counter(t[0] for t in d), laws["q1"])
        out["tests"][f"{arm}_token2_marginal_vs_exact_law"] = gof(Counter(t[1] for t in d), laws["marginal2"])
    if len(draws) < 2:
        out["rejected_at_alpha"] = {k: v["p_value"] < ALPHA for k, v in out["tests"].items()}
        (RES / f"gap11_{TAG}_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        return 1 if any(out["rejected_at_alpha"].values()) else 0
    m = min(len(draws["plain"]), len(draws["spec"]))
    for pos in range(NUM_TOKENS):
        out["tests"][f"plain_vs_spec_token{pos + 1}"] = homogeneity(
            Counter(t[pos] for t in draws["plain"][:m]), Counter(t[pos] for t in draws["spec"][:m]))
    out["tests"]["plain_vs_spec_joint"] = homogeneity(Counter(draws["plain"][:m]), Counter(draws["spec"][:m]))
    out["rejected_at_alpha"] = {k: (v["p_value"] is not None and v["p_value"] < ALPHA) for k, v in out["tests"].items()}
    (RES / f"gap11_{TAG}_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 1 if any(out["rejected_at_alpha"].values()) else 0


def main() -> int:
    """Dispatch on --arm."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=["plain", "spec", "law", "report"])
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--tag", default="cpu_invariant")
    args = parser.parse_args()
    global DEVICE, TAG, PROPS
    DEVICE, TAG = args.device, args.tag
    if DEVICE != "CPU":
        PROPS = {"KV_CACHE_PRECISION": "f16"}
    if args.arm in ("plain", "spec"):
        base = args.seed_base if args.seed_base is not None else (1000 if args.arm == "plain" else 500000)
        run_arm(args.arm, args.n, base)
    elif args.arm == "law":
        compute_laws()
    else:
        return report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
