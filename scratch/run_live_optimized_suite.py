"""Targeted sequential physical benchmark runner for comparing Old vs New Native Governor & Telemetry."""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from huggingface_hub import snapshot_download
from arc_runner.engine.openvino_engine import OpenVINOGenAIEngine
from arc_runner.engine.models import GenerationConfig
from arc_runner.monitoring.telemetry import HardwareTelemetryCollector

TIERS = {
    "low": {
        "name": "Entity & Fact Extraction",
        "prompt": "Extract the city, country, and currency amount: 'Alice converted 450 British Pounds into Euros at a currency exchange in Berlin, Germany.' Reply with a JSON object containing keys: city, country, amount.",
        "max_tokens": 40,
    },
    "medium": {
        "name": "Algorithm & Technical Synthesis",
        "prompt": "Write a clean Python function find_longest_palindrome(s: str) -> str using expand-around-center algorithm. Include docstring and type hints.",
        "max_tokens": 160,
    },
    "hard": {
        "name": "Finite State Machine Architecture",
        "prompt": "Design a complete finite state machine class in Python for TCP connection states: CLOSED, LISTEN, SYN_SENT, SYN_RCVD, ESTABLISHED. Implement transition(event: str) -> str and raise ValueError on illegal transitions.",
        "max_tokens": 300,
    },
}

MODELS = [
    {
        "id": "qwen-2.5-1.5b",
        "repo": "OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov",
    },
    {
        "id": "deepseek-r1-1.5b",
        "repo": "OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-ov",
    },
    {
        "id": "llama-3.2-3b",
        "repo": "AIFunOver/Llama-3.2-3B-Instruct-openvino-4bit",
    },
]

def run_suite():
    results = {}
    engine = OpenVINOGenAIEngine()

    for m in MODELS:
        m_id = m["id"]
        repo = m["repo"]
        print(f"\n=======================================================")
        print(f"BENCHMARKING: {m_id} ({repo})")
        print(f"=======================================================")
        try:
            print(f"Downloading / locating cache for {repo}...")
            model_path = snapshot_download(repo_id=repo)
            print(f"Cached at: {model_path}")
            
            print("Compiling on Intel Arc Xe-LPG GPU (with Native Governor & Queue Prioritization)...")
            t_load0 = time.perf_counter()
            engine.load_model(model_path, device="GPU")
            t_load1 = time.perf_counter()
            print(f"Model compiled in {(t_load1 - t_load0):.2f}s")
            
            print("Warming up engine...")
            engine.warmup()

            results[m_id] = {}
            for tier_name, tier_cfg in TIERS.items():
                print(f"\n--- Tier: {tier_name.upper()} ({tier_cfg['name']}) ---")
                collector = HardwareTelemetryCollector(sample_interval_sec=0.03)
                config = GenerationConfig(max_new_tokens=tier_cfg["max_tokens"])
                
                collector.start()
                t0 = time.perf_counter()
                out = engine.generate(tier_cfg["prompt"], config)
                t1 = time.perf_counter()
                summary = collector.stop()
                
                telemetry = engine.get_telemetry()
                tps = telemetry.tokens_per_second
                ttft = telemetry.ttft_ms
                tpot = telemetry.tpot_ms
                tokens = telemetry.total_tokens

                results[m_id][tier_name] = {
                    "tokens_per_second": round(tps, 2),
                    "ttft_ms": round(ttft, 2),
                    "tpot_ms": round(tpot, 2),
                    "total_tokens": tokens,
                    "generation_time_s": round(t1 - t0, 2),
                    "stability_score": summary.stability_score,
                    "jitter_ms": summary.jitter_ms,
                    "mean_cpu_load": summary.mean_cpu_load,
                    "mean_ram_mb": summary.mean_ram_mb,
                    "disk_read_mb": summary.disk_read_total_mb,
                }
                print(f"  Result: {tps:.2f} tok/s | TTFT: {ttft:.2f} ms | TPOT: {tpot:.2f} ms | Stability: {summary.stability_score:.1f}%")
            
            engine.unload()
        except Exception as exc:
            print(f"Error evaluating {m_id}: {exc}")
            results[m_id] = {"error": str(exc)}

    out_file = Path("scratch/live_real_models_post_governor_benchmark.json")
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    run_suite()
