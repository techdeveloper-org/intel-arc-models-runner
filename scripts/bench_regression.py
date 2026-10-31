"""Interleaved regression benchmark of the current checkout against a baseline commit (plan B10).

The baseline is checked out into a temporary git worktree; each iteration runs both versions in
fresh subprocesses in a seeded random order. Results are written as a schema-v1 file and judged with
evaluate_noninferiority_gate (default margin 3%).

Workloads:
  flash_acquire  layer loads per second of FlashStreamWeightManager on a synthetic container (runs anywhere;
                 recorded under the label layer_loads_per_s, not a model throughput)
  llm            time to first token and decode tokens/s of OpenVINOGenAIEngine on a real model directory
                 (needs --model-dir and a device; counts tokens with the pipeline tokenizer)

Usage: python scripts/bench_regression.py --baseline c7312e0 --workload flash_acquire [--runs 12] [--out FILE]
       python scripts/bench_regression.py --baseline c7312e0 --workload llm --model-dir DIR --device GPU
"""

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]

FLASH_WORKER = r'''
import json, sys, time, tempfile
from pathlib import Path
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
from tests.test_flash_stream_triad import build_synthetic_flash_gguf
from arc_runner.engine.flash_stream.weight_streamer import FlashStreamWeightManager
with tempfile.TemporaryDirectory() as tmp:
    model = build_synthetic_flash_gguf(Path(tmp) / "m.gguf", num_layers=6, hidden_dim=64)
    manager = FlashStreamWeightManager(model, ring_buffer_slots=3)
    try:
        start = time.perf_counter(); first = None; loads = 0
        for step in range(300):
            manager.acquire_layer_weights(step % 6); loads += 1
            if first is None: first = (time.perf_counter() - start) * 1000.0
        elapsed = time.perf_counter() - start
    finally:
        manager.close()
print(json.dumps({"ttft_ms": first, "decode_tps": loads / elapsed, "prompt_tokens": 0, "completion_tokens": loads}))
'''

LLM_WORKER = r'''
import json, sys, time
sys.path.insert(0, sys.argv[1])
from arc_runner.engine.openvino_engine import OpenVINOGenAIEngine
from arc_runner.engine.models import GenerationConfig
engine = OpenVINOGenAIEngine(); engine.load_model(sys.argv[2], device=sys.argv[3])
prompt = "Explain in two sentences why memory bandwidth limits token generation speed."
cfg = GenerationConfig(max_new_tokens=int(sys.argv[4]), temperature=0.0)
start = time.perf_counter(); first = None; pieces = []
for chunk in engine.generate_stream(prompt, cfg):
    if first is None and chunk.text: first = time.perf_counter()
    pieces.append(chunk.text)
end = time.perf_counter()
tok = engine._pipeline.get_tokenizer()
count = int(tok.encode("".join(pieces)).input_ids.shape[-1]); prompt_tokens = int(tok.encode(prompt).input_ids.shape[-1])
ttft = (first - start) * 1000.0 if first else None
tps = (count - 1) / (end - first) if first and count > 1 and end > first else None
print(json.dumps({"ttft_ms": ttft, "decode_tps": tps, "prompt_tokens": prompt_tokens, "completion_tokens": count}))
'''


def run_worker(version_root: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """Run one workload iteration for one version in a fresh subprocess.

    Args:
        version_root: Checkout root of the version under test.
        args: Parsed command line.

    Returns:
        Dict[str, Any]: The worker's JSON result line.
    """
    if args.workload == "flash_acquire":
        cmd = [sys.executable, "-c", FLASH_WORKER, str(version_root / "src"), str(version_root)]
    else:
        cmd = [sys.executable, "-c", LLM_WORKER, str(version_root / "src"), args.model_dir, args.device, str(args.max_new_tokens)]
    env = {**os.environ, "ARC_NATIVE_DISABLE": os.environ.get("ARC_NATIVE_DISABLE", "")}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    if result.returncode != 0:
        raise RuntimeError(f"worker failed for {version_root}:\n{result.stderr[-2000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def main(argv: List[str]) -> int:
    """Run the interleaved comparison and print the gate verdicts.

    Args:
        argv: Command line arguments without the program name.

    Returns:
        int: 0 if every gate passes, 1 if any gate fails or the run could not complete.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="git commit of the version to compare against")
    parser.add_argument("--workload", choices=["flash_acquire", "llm"], required=True)
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-dir")
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--margin", type=float, default=0.03)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.workload == "llm" and not args.model_dir:
        parser.error("--model-dir is required for the llm workload")

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from arc_runner.benchmark.results import build_result, capture_environment, write_result
    from arc_runner.benchmark.stats import BenchSample, evaluate_noninferiority_gate

    rng = random.Random(args.seed)
    with tempfile.TemporaryDirectory() as tmp:
        baseline_root = Path(tmp) / "baseline"
        subprocess.run(["git", "-C", str(REPO_ROOT), "worktree", "add", "--detach", str(baseline_root), args.baseline], check=True, capture_output=True)
        try:
            roots = {"new": REPO_ROOT, "old": baseline_root}
            samples: Dict[str, List[BenchSample]] = {"new": [], "old": []}
            for iteration in range(args.warmup + args.runs):
                order = ["new", "old"]
                rng.shuffle(order)
                for label in order:
                    row = run_worker(roots[label], args)
                    samples[label].append(BenchSample(
                        model_id=args.workload, backend=label, iteration=iteration, ttft_ms=row["ttft_ms"], decode_tps=row["decode_tps"],
                        prompt_tokens=row["prompt_tokens"], completion_tokens=row["completion_tokens"], source="measured",
                        is_warmup=iteration < args.warmup,
                    ))
        finally:
            subprocess.run(["git", "-C", str(REPO_ROOT), "worktree", "remove", "--force", str(baseline_root)], capture_output=True)

    protocol = {
        "warmup_runs": args.warmup, "timed_runs": args.runs, "temperature": 0.0, "max_new_tokens": args.max_new_tokens,
        "labels": {"new": f"current checkout, workload {args.workload}", "old": f"baseline {args.baseline}"},
    }
    document = build_result(f"regression-{args.workload}-{args.baseline[:8]}", samples["new"] + samples["old"], protocol, capture_environment(REPO_ROOT))
    if args.out:
        write_result(args.out, document)

    exit_code = 0
    for metric in ("decode_tps", "ttft_ms"):
        verdict = evaluate_noninferiority_gate(samples["new"], samples["old"], metric=metric, margin=args.margin, min_n=args.runs)
        print(f"{metric}: passed={verdict.passed} ratio_ci=({verdict.ratio_lo:.3f}, {verdict.ratio_hi:.3f}) n=({verdict.n_arc},{verdict.n_ref}) {verdict.reason}")
        exit_code |= 0 if verdict.passed else 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
