"""Physical Silicon Empirical Verification Harness for 60+ Tokens/Second.

Runs real in-engine speculative decoding on Intel Core Ultra 5 125H Arc Xe-LPG iGPU using:
1. Verifier (Target): models/qwen-2.5-1.5b (Arc Xe-LPG iGPU)
2. Draft: Qwen2.5-0.5B-Instruct-openvino-4bit (Arc Xe-LPG iGPU co-located zero-copy)
3. WindowsSystemGovernor: MMCSS Pro Audio, P-Core 0-7 pinning, 1ms timer, PowerThrottling disabled
4. Level Zero Driver: GPU_QUEUE_THROTTLE=LOW, GPU_QUEUE_PRIORITY=HIGH, SDPA Fused Attention
5. Dynamic Gamma and Low-Alpha Circuit Breaker
"""

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List

from arc_runner.core.logging import get_logger, log_benchmark_telemetry, setup_logging
from arc_runner.engine.models import GenerationConfig, SpeculativeConfig
from arc_runner.engine.speculative_engine import HybridSpeculativeEngine
from arc_runner.engine.system_governor import WindowsSystemGovernor

setup_logging(level="INFO")
logger = get_logger("verification.speculative_60tps")


def run_physical_60tps_verification() -> Dict[str, Any]:
    """Execute end-to-end physical verification on Intel Arc Xe-LPG iGPU.

    Returns:
        Dict[str, Any]: Full empirical benchmarking telemetry results.
    """
    target_path = Path("models/qwen-2.5-1.5b")
    draft_snap = Path(
        r"C:\Users\techd\.cache\huggingface\hub\models--hsuwill000--Qwen2.5-0.5B-Instruct-openvino-4bit\snapshots\50b455248a3c33f66052b2169031bf216edf4d18"
    )

    if not target_path.exists() or not draft_snap.exists():
        logger.error("Required model directories not found on physical filesystem.")
        return {}

    governor = WindowsSystemGovernor()
    results: Dict[str, Any] = {
        "platform": "Intel Core Ultra 5 125H (Meteor Lake) Arc Xe-LPG iGPU",
        "configuration": "Zero-Copy GPU In-Engine Speculative Decoding + Native Governor",
        "target_model": "Qwen-2.5-1.5B INT4",
        "draft_model": "Qwen-2.5-0.5B INT4",
        "tiers": {},
    }

    with governor.benchmark_isolation(pause_services=False) as active_gov:
        logger.info("Windows System Governor active: MMCSS Pro Audio + P-Core 0-7 pinning engaged.")

        engine = HybridSpeculativeEngine()
        spec_cfg = SpeculativeConfig(
            draft_model_path=str(draft_snap),
            draft_device="GPU",
            colocate_on_gpu=True,
            num_assistant_tokens=3,
            dynamic_gamma=True,
            min_gamma=2,
            max_gamma=5,
            circuit_breaker_threshold=0.55,
        )

        gpu_properties = {
            "PERFORMANCE_HINT": "LATENCY",
            "EXECUTION_MODE_HINT": "PERFORMANCE",
            "INFERENCE_PRECISION_HINT": "f16",
            "GPU_ENABLE_SDPA_OPTIMIZATION": True,
            "GPU_HOST_TASK_PRIORITY": "HIGH",
            "GPU_QUEUE_PRIORITY": "HIGH",
            "GPU_QUEUE_THROTTLE": "LOW",
        }

        logger.info("Compiling target and draft pipelines co-located on Intel Arc Xe-LPG iGPU...")
        engine.load_model(str(target_path), speculative_config=spec_cfg, device="GPU", **gpu_properties)

        # Warmup GPU execution units
        logger.info("Warming up GPU execution units...")
        warmup_cfg = GenerationConfig(max_new_tokens=8)
        engine.generate("Warmup silicon instruction cache.", warmup_cfg)

        prompts = [
            (
                "low",
                "Factual Geography",
                "The capital city of France is Paris. The longest river in Europe is the Volga. The tallest mountain in Japan is",
                40,
            ),
            (
                "medium",
                "TCP FSM Architecture",
                "Explain the TCP finite state machine three-way handshake sequence covering SYN, SYN-ACK, and ACK states concisely:",
                60,
            ),
            (
                "hard",
                "Cryptarithmetic Logic",
                "Solve the cryptarithmetic puzzle SEND + MORE = MONEY where each distinct letter represents a distinct base-10 digit. Provide full deduction:",
                80,
            ),
        ]

        baseline_tps = {
            "low": 45.02,
            "medium": 46.04,
            "hard": 45.41,
        }

        for tier_key, tier_name, prompt_text, max_toks in prompts:
            logger.info("Starting Benchmark: %s (%s, max_new_tokens=%d)...", tier_name, tier_key, max_toks)
            gen_cfg = GenerationConfig(max_new_tokens=max_toks, ignore_eos=True)

            t0 = time.monotonic()
            completion = engine.generate(prompt_text, gen_cfg)
            elapsed_sec = max(time.monotonic() - t0, 1e-6)

            telem = engine.get_telemetry()
            measured_tps = (
                telem.tokens_per_second
                if telem.is_tps_valid and telem.tokens_per_second > 0
                else (telem.total_tokens / elapsed_sec)
            )

            old_baseline = baseline_tps.get(tier_key, 45.5)
            delta_pct = ((measured_tps - old_baseline) / old_baseline) * 100.0
            stats = engine.get_speculative_stats()

            logger.info(
                "[%s] Tokens: %d | Elapsed: %.3fs | Speed: %.2f tok/s (Delta: %+.1f%%) | TTFT: %.1fms | Gamma: %d | Alpha: %.2f",
                tier_name,
                telem.total_tokens,
                elapsed_sec,
                measured_tps,
                delta_pct,
                telem.ttft_ms,
                stats.get("current_gamma", 3),
                stats.get("rolling_acceptance_rate", 0.0),
            )

            results["tiers"][tier_key] = {
                "name": tier_name,
                "tokens": telem.total_tokens,
                "elapsed_sec": round(elapsed_sec, 3),
                "measured_tps": round(measured_tps, 2),
                "old_baseline_tps": old_baseline,
                "delta_pct": round(delta_pct, 1),
                "ttft_ms": round(telem.ttft_ms, 2),
                "gamma": stats.get("current_gamma", 3),
                "acceptance_rate": round(stats.get("rolling_acceptance_rate", 0.0), 2),
                "circuit_breaker_tripped": stats.get("circuit_breaker_tripped", False),
                "completion_preview": completion[:120].strip() if completion else "",
            }

    out_file = Path("scratch/live_60tps_empirical_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Empirical benchmark data successfully written to %s", out_file)
    return results


if __name__ == "__main__":
    run_physical_60tps_verification()
