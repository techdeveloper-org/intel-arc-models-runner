"""Live LLM and embedding benchmark that writes schema-v1 result files.

LLM rows use the engine's own perf-metric telemetry (TTFT, decode rate, token counts; never a word
count); embedding rows time real embed_documents calls. Warmups are discarded, timed runs are recorded
one sample each, and package power/energy come from the Sysman energy counter when present (otherwise
None). Backends: arc-runner on the model's device, npu where the model supports it and an NPU exists,
and a real CPU baseline run. If the wanted device is missing the run falls back to CPU and every row is
flagged fallback_used. Nothing is simulated.

Usage: python scratch/run_live_sequential_model_benchmarks.py --model qwen-2.5-1.5b [--warmups 3] [--runs 10]
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import shutil
import sys
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from arc_runner.benchmark.harness_common import (  # noqa: E402
    HarnessConfig, add_harness_args, begin_run, config_from_args, default_result_path, directory_size_bytes, download_snapshot,
    finish_run, optional_sysman, target_device,
)
from arc_runner.benchmark.nonllm import embedding_samples, llm_samples, llm_samples_interleaved  # noqa: E402
from arc_runner.benchmark.stats import BenchSample  # noqa: E402
from arc_runner.benchmark.workloads import export_workload_result  # noqa: E402

MODEL_WORKLOAD_SPECS = {
    "qwen-2.5-1.5b": {
        "type": "llm",
        "alias": "qwen-2.5-1.5b",
        "repo_id": "OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov",
        "device": "GPU",
        "npu_supported": True,
        "size_bytes": 940000000,
        "workloads": {
            "low": {
                "name": "Factual Entity Extraction",
                "prompt": "What is the capital of France and what is its most famous tower? Reply in one short sentence.",
                "max_tokens": 30,
            },
            "medium": {
                "name": "Python Palindrome Algorithm",
                "prompt": "Write a clean Python function find_longest_palindrome(s: str) -> str using expand-around-center with O(1) auxiliary space. Include type hints and docstring.",
                "max_tokens": 200,
            },
            "hard": {
                "name": "TCP Connection State Machine",
                "prompt": "Design a finite state machine in Python for a TCP connection (CLOSED, LISTEN, SYN_SENT, SYN_RCVD, ESTABLISHED). Include handle_event(event) transitions and raise ValueError on invalid transitions.",
                "max_tokens": 300,
            },
        },
    },
    "all-minilm-l6-v2": {
        "type": "embedding",
        "alias": "all-minilm-l6-v2",
        "repo_id": "AIFunOver/all-MiniLM-L6-v2-openvino-fp16",
        "device": "NPU",
        "npu_supported": True,
        "size_bytes": 45000000,
        "workloads": {
            "low": {
                "name": "Direct Cosine Similarity & Unit Normalization",
                "text_a": "The dog chased the ball across the park.",
                "text_b": "A puppy ran after a toy on the grass.",
            },
            "medium": {
                "name": "Paraphrase vs Contradiction Discrimination",
                "anchor": "Quantum computing utilizes qubits in superposition.",
                "positive": "Qubits in superposition states power quantum computers.",
                "negative": "Classical computers use silicon transistors in binary states.",
            },
            "hard": {
                "name": "High-Density 16-Sentence Batching Matrix",
                "batch_size": 16,
            },
        },
    },
    "nomic-embed-text": {
        "type": "embedding",
        "alias": "nomic-embed-text",
        "repo_id": "OpenVINO/nomic-embed-text-v1.5-fp16-ov",
        "device": "NPU",
        "npu_supported": True,
        "size_bytes": 274000000,
        "workloads": {
            "low": {
                "name": "Short Query Unit Embedding",
                "text_a": "What is Intel Arc Level Zero architecture?",
            },
            "medium": {
                "name": "Technical RAG Semantic Search Ranking",
                "query": "how to prevent GPU out of memory in continuous batching",
                "corpus": [
                    "Little's Law and token slot preemption prevent VRAM saturation in continuous batching.",
                    "Python dictionaries maintain insertion order since version 3.7.",
                    "Mount Everest is the highest mountain above sea level.",
                    "Photosynthesis in plants converts solar radiation into glucose.",
                    "TCP uses a three-way handshake SYN, SYN-ACK, ACK to establish connection.",
                ],
            },
            "hard": {
                "name": "Long-Document 2048-Token Semantic Invariance",
                "seq_length": 2048,
            },
        },
    },
    "llama-3.2-1b": {
        "type": "llm",
        "alias": "llama-3.2-1b",
        "repo_id": "AIFunOver/Llama-3.2-1B-Instruct-openvino-4bit",
        "device": "GPU",
        "npu_supported": True,
        "size_bytes": 740000000,
        "workloads": {
            "low": {
                "name": "Instant Entity Extraction",
                "prompt": "Extract the country and currency: 'He exchanged 500 US dollars for Japanese Yen in Tokyo.'",
                "max_tokens": 25,
            },
            "medium": {
                "name": "Two-Sentence Summary with Constraint",
                "prompt": "Summarize the principle of speculative decoding in exactly two sentences. Do not use the word 'magic'.",
                "max_tokens": 100,
            },
            "hard": {
                "name": "DMA Architecture & Technical Paragraph",
                "prompt": "Write a technical paragraph explaining how Direct Memory Access (DMA) bypasses the CPU for high-speed I/O.",
                "max_tokens": 200,
            },
        },
    },
    "deepseek-r1-1.5b": {
        "type": "llm",
        "alias": "deepseek-r1-1.5b",
        "repo_id": "OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-ov",
        "device": "GPU",
        "npu_supported": True,
        "size_bytes": 1120000000,
        "workloads": {
            "low": {
                "name": "Division by Zero Rejection",
                "prompt": "What is 750 divided by 0? State the exact mathematical result.",
                "max_tokens": 40,
            },
            "medium": {
                "name": "Warehouse Multi-Step Deductive Arithmetic",
                "prompt": "Solve step-by-step: A warehouse has 45 boxes with 12 items each. They receive an extra 256 items divided evenly into 8 cartons, but find that 47 items are damaged. How many undamaged items remain?",
                "max_tokens": 200,
            },
            "hard": {
                "name": "Cryptarithmetic SEND+MORE=MONEY Deduction",
                "prompt": "Solve the cryptarithmetic puzzle SEND + MORE = MONEY step by step. What digit does each letter represent? Show the logical deduction for M, S, and O.",
                "max_tokens": 400,
            },
        },
    },
    "bge-m3": {
        "type": "embedding",
        "alias": "bge-m3",
        "repo_id": "OpenVINO/bge-m3-fp16-ov",
        "device": "NPU",
        "npu_supported": True,
        "size_bytes": 1140000000,
        "workloads": {
            "low": {
                "name": "Monolingual Similarity Unit Check",
                "text_a": "Artificial Intelligence is transforming enterprise software.",
                "text_b": "Enterprise applications are being reshaped by AI technologies.",
            },
            "medium": {
                "name": "Cross-Lingual Hindi to English Alignment",
                "text_a": "कृत्रिम बुद्धिमत्ता और मशीन लर्निंग के बुनियादी सिद्धांत क्या हैं?",
                "text_b": "What are the foundational principles of artificial intelligence and machine learning?",
            },
            "hard": {
                "name": "Multi-Lingual Indic/German/English Triplet Scoring",
                "anchor": "डेटा सुरक्षा और उपयोगकर्ता गोपनीयता",
                "positive": "Datensicherheit und Privatsphäre der Benutzer",
                "negative": "Stock market fluctuations and interest rate policies",
            },
        },
    },
    "llama-3.2-3b": {
        "type": "llm",
        "alias": "llama-3.2-3b",
        "repo_id": "OpenVINO/Llama-3.2-3B-Instruct-int4-ov",
        "device": "GPU",
        "npu_supported": False,
        "size_bytes": 1820000000,
        "workloads": {
            "low": {
                "name": "Counter-Factual Premise Rejection",
                "prompt": "In what year did Thomas Jefferson walk on the Moon during the Apollo missions?",
                "max_tokens": 40,
            },
            "medium": {
                "name": "Strict RFC 8259 eCommerce Schema",
                "prompt": "Generate a strict JSON schema for an eCommerce product with fields: id (uuid), title (string), price (float >= 0), tags (array of strings), in_stock (boolean). Output ONLY JSON.",
                "max_tokens": 150,
            },
            "hard": {
                "name": "Zero-Copy USM Memory & DPAS Verification",
                "prompt": "Explain how Unified Shared Memory (USM) zero-copy pointers enable host DDR5 and integrated Intel Xe-LPG GPU DPAS matrix engines to share tensors without PCIe serialization bottlenecks.",
                "max_tokens": 300,
            },
        },
    },
    "qwen-2.5-3b": {
        "type": "llm",
        "alias": "qwen-2.5-3b",
        "repo_id": "OpenVINO/Qwen2.5-3B-Instruct-int4-ov",
        "device": "GPU",
        "npu_supported": False,
        "size_bytes": 1910000000,
        "workloads": {
            "low": {
                "name": "Fictitious Term Detection",
                "prompt": "Define the medical procedure known as 'flimflambulatory pneumonectomy'.",
                "max_tokens": 50,
            },
            "medium": {
                "name": "Historical Anachronism Trap",
                "prompt": "Explain how Napoleon Bonaparte used radar and radio transmitters to coordinate his artillery at the Battle of Waterloo in 1815.",
                "max_tokens": 150,
            },
            "hard": {
                "name": "Concurrent Thread Pool with Sentinel Shutdown",
                "prompt": "Write a robust Python producer-consumer pattern using queue.Queue, threading.Event for graceful shutdown, and a worker pool processing 1,000 tasks. Ensure no deadlocks and all exceptions are caught.",
                "max_tokens": 350,
            },
        },
    },
    "phi-3.5-mini": {
        "type": "llm",
        "alias": "phi-3.5-mini",
        "repo_id": "OpenVINO/Phi-3.5-mini-instruct-int4-ov",
        "device": "GPU",
        "npu_supported": False,
        "size_bytes": 2210000000,
        "workloads": {
            "low": {
                "name": "Single Token Arithmetic",
                "prompt": "Compute 144 / 12 and output only the number.",
                "max_tokens": 10,
            },
            "medium": {
                "name": "Epimenides Paradox Analysis",
                "prompt": "Analyze this statement: 'This statement is false.' Is it true or false? Answer logically without repeating yourself.",
                "max_tokens": 120,
            },
            "hard": {
                "name": "Needle-In-Haystack Context Retrieval",
                "prompt": "Context: In the northern quadrant facility, security protocols dictate badge access. The secret override passcode for Vault 7 is OMEGA-992-ALPHA. All perimeter gates require dual biometric authorization. Question: What is the secret override passcode for Vault 7?",
                "max_tokens": 40,
            },
        },
    },
    "qwen-2.5-7b": {
        "type": "llm",
        "alias": "qwen-2.5-7b",
        "repo_id": "OpenVINO/Qwen2.5-7B-Instruct-int4-ov",
        "device": "GPU",
        "npu_supported": False,
        "size_bytes": 4420000000,
        "workloads": {
            "low": {
                "name": "Algebraic Resolution",
                "prompt": "Solve for x: 3x + 15 = 45. Output only the value of x.",
                "max_tokens": 20,
            },
            "medium": {
                "name": "CWE-95 Arbitrary Code Injection Guard",
                "prompt": "Write a Python script that takes user input string and runs eval(user_input) to evaluate algebraic formulas securely.",
                "max_tokens": 180,
            },
            "hard": {
                "name": "Distributed Systems & 4.096 Gbps NIC Derivation",
                "prompt": "Design a fault-tolerant event streaming engine processing 1,000,000 events/second with sub-10ms P99 latency: 1) Compare Raft vs Paxos, 2) Zero-copy kernel page cache persistence (sendfile/splice), 3) Token-bucket backpressure, 4) Derive mathematically the minimum NIC bandwidth in Gigabits/second for 512-byte events.",
                "max_tokens": 450,
            },
        },
    },
    "qwen2-vl-7b": {
        "type": "vision",
        "alias": "qwen2-vl-7b",
        "repo_id": "OpenVINO/Qwen2-VL-7B-Instruct-int4-ov",
        "device": "GPU",
        "npu_supported": False,
        "size_bytes": 4500000000,
        "workloads": {
            "low": {
                "name": "Visual Scene & Color Recognition",
                "prompt": "Describe what color and type of vehicle appears in the image.",
                "max_tokens": 30,
            },
            "medium": {
                "name": "Document Receipt Key Extraction",
                "prompt": "Extract the Invoice Number, Total Amount, and Due Date into valid JSON.",
                "max_tokens": 100,
            },
            "hard": {
                "name": "System Architecture Graph Reasoning",
                "prompt": "Examine the architecture diagram. List all microservices connected directly to Kafka, and identify the database engine attached to Payment Service.",
                "max_tokens": 200,
            },
        },
    },
    "llama-3.2-11b-vision": {
        "type": "vision",
        "alias": "llama-3.2-11b-vision",
        "repo_id": "OpenVINO/Llama-3.2-11B-Vision-Instruct-int4-ov",
        "device": "GPU",
        "npu_supported": False,
        "size_bytes": 6200000000,
        "workloads": {
            "low": {
                "name": "Visual Landscape Captioning",
                "prompt": "Describe the scene in one concise sentence.",
                "max_tokens": 30,
            },
            "medium": {
                "name": "Quarterly Revenue Bar Chart Extraction",
                "prompt": "Which quarter had the highest revenue, what was the value, and what was the total annual revenue?",
                "max_tokens": 100,
            },
            "hard": {
                "name": "Electrical Circuit Schematic Tracing",
                "prompt": "Trace the circuit from the 12V DC input: 1) What component is in series? 2) Identify capacitor C1, 3) Explain diode D1 flyback suppression.",
                "max_tokens": 250,
            },
        },
    },
}


def embedding_batch(workload: Dict[str, Any]) -> List[str]:
    """Build the batch of texts embedded per run for one tier.

    Args:
        workload: Tier entry of an embedding model spec.

    Returns:
        List[str]: The tier's texts (pair, triplet, query plus corpus), batch_size generated sentences,
        or one document of seq_length words.

    Raises:
        ValueError: If the tier has none of the known input shapes.
    """
    if "text_a" in workload:
        return [workload["text_a"], *([workload["text_b"]] if "text_b" in workload else [])]
    if "anchor" in workload:
        return [workload["anchor"], workload["positive"], workload["negative"]]
    if "corpus" in workload:
        return [workload["query"], *workload["corpus"]]
    if "batch_size" in workload:
        return [f"Benchmark sentence {i} about GPU memory management and batching." for i in range(int(workload["batch_size"]))]
    if "seq_length" in workload:
        return [" ".join(["token"] * int(workload["seq_length"]))]
    raise ValueError(f"embedding tier {workload.get('name')!r} has no recognised input shape")


def backend_plan(spec: Dict[str, Any], topology: Any, cpu_baseline: bool) -> List[tuple]:
    """List the (backend, device, fallback_used) combinations to measure.

    Args:
        spec: Model spec.
        topology: Detected device topology.
        cpu_baseline: Whether to add a CPU baseline.

    Returns:
        List[tuple]: Backend plan; embeddings prefer the NPU, LLMs the spec's device.
    """
    preferred = "NPU" if spec["type"] == "embedding" and spec.get("npu_supported") else spec["device"]
    device, fallback = target_device(topology, preferred)
    plan = [("arc-runner", device, fallback)]
    if spec["type"] == "llm" and spec.get("npu_supported") and device != "NPU" and topology.has_intel_npu:
        plan.append(("npu", "NPU", False))
    if cpu_baseline and device != "CPU":
        plan.append(("cpu-baseline", "CPU", False))
    return plan


def collect_samples(
    alias: str,
    spec: Dict[str, Any],
    engine_factory: Callable[[str, Dict[str, Any]], Any],
    plan: List[tuple],
    warmups: int,
    runs: int,
    sysman: Any,
    harness: Optional[HarnessConfig] = None,
) -> List[BenchSample]:
    """Measure every tier of one model on every planned backend.

    Args:
        alias: Model alias.
        spec: Entry of MODEL_WORKLOAD_SPECS.
        engine_factory: Function taking (device, workload) and returning a loaded engine.
        plan: Output of backend_plan.
        warmups: Discarded warmup runs per tier.
        runs: Timed runs per tier.
        sysman: Optional Sysman for package power.
        harness: Harness settings (converged warmup, interleaving, cooldown); None keeps fixed warmups with no cooldown.

    Returns:
        List[BenchSample]: Samples for all tiers and backends.
    """
    cfg = harness or HarnessConfig(max_warmups=None, cooldown_s=0.0)
    samples: List[BenchSample] = []
    if cfg.interleave and spec["type"] == "llm":
        for index, (tier, workload) in enumerate(spec["workloads"].items()):
            if index:
                cfg.cooldown()
            systems = [(backend, engine_factory(device, workload), fallback) for backend, device, fallback in plan]
            try:
                samples += llm_samples_interleaved(
                    systems, workload["prompt"], workload["max_tokens"], f"{alias}/{tier}", warmups, runs,
                    seed=cfg.seed, sysman=sysman, max_warmups=cfg.max_warmups,
                )
            finally:
                del systems
                gc.collect()
        return samples
    for plan_index, (backend, device, fallback) in enumerate(plan):
        if plan_index:
            cfg.cooldown()
        for tier, workload in spec["workloads"].items():
            engine = engine_factory(device, workload)
            model_id = f"{alias}/{tier}"
            try:
                if spec["type"] == "llm":
                    samples += llm_samples(engine, workload["prompt"], workload["max_tokens"], model_id, backend, warmups, runs, sysman=sysman, fallback_used=fallback, max_warmups=cfg.max_warmups)
                else:
                    samples += embedding_samples(engine, embedding_batch(workload), model_id, backend, warmups, runs, sysman=sysman, fallback_used=fallback, max_warmups=cfg.max_warmups)
            finally:
                del engine
                gc.collect()
    return samples


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: Arguments without the program name, or None for sys.argv.

    Returns:
        int: 0 on success.
    """
    parser = argparse.ArgumentParser(description="Live LLM/embedding benchmark writing schema-v1 results.")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_WORKLOAD_SPECS))
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--models-root", default="models")
    parser.add_argument("--no-cpu-baseline", action="store_true")
    parser.add_argument("--keep", action="store_true", help="keep the model files on disk")
    parser.add_argument("--out", type=Path)
    add_harness_args(parser)
    args = parser.parse_args(argv)

    from arc_runner.devices.detector import DeviceDetector
    from arc_runner.engine.embedding_engine import EmbeddingEngine
    from arc_runner.engine.openvino_engine import OpenVINOGenAIEngine

    spec = MODEL_WORKLOAD_SPECS[args.model]
    model_dir = Path(args.models_root).resolve() / args.model
    download_snapshot(spec["repo_id"], model_dir, ("openvino_model.xml", "openvino_tokenizer.xml"))
    plan = backend_plan(spec, DeviceDetector().detect_topology(), not args.no_cpu_baseline)
    harness = config_from_args(args)
    resident = len(plan) if harness.interleave and spec["type"] == "llm" else 1
    on_disk = directory_size_bytes(model_dir)
    report = begin_run(harness, on_disk * resident if on_disk is not None else None)

    def factory(device: str, workload: Dict[str, Any]) -> Any:
        if spec["type"] == "llm":
            engine = OpenVINOGenAIEngine()
            engine.load_model(model_dir, device=device)
            return engine
        return EmbeddingEngine(str(model_dir), device=device, normalize=True, max_length=max(512, int(workload.get("seq_length", 512))))

    try:
        samples = collect_samples(args.model, spec, factory, plan, args.warmups, args.runs, optional_sysman(), harness)
    finally:
        if not args.keep:
            shutil.rmtree(model_dir, ignore_errors=True)
    samples, protocol_extra, environment_extra = finish_run(samples, harness, report)
    workload = "llm" if spec["type"] == "llm" else "embeddings"
    out = args.out or default_result_path(REPO_ROOT, workload, args.model)
    labels = {backend: f"OpenVINO on {device}" for backend, device, _ in plan}
    protocol_labels = dict(labels)
    if spec["type"] == "llm":
        from arc_runner.benchmark.results import build_result, capture_environment, write_result

        protocol = {"warmup_runs": args.warmups, "timed_runs": args.runs, "temperature": 0.0, "max_new_tokens": max(w["max_tokens"] for w in spec["workloads"].values()), "labels": protocol_labels, **protocol_extra}
        write_result(out, build_result(f"{workload}-{args.model}", samples, protocol, {**capture_environment(REPO_ROOT), **environment_extra}))
    else:
        export_workload_result(out, f"{workload}-{args.model}", samples, protocol_labels, args.warmups, args.runs, REPO_ROOT, protocol_extra=protocol_extra, environment_extra=environment_extra)
    print(f"wrote {out} ({len(samples)} samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
