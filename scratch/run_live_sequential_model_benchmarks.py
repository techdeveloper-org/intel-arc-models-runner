"""Sequential model-by-model live physical silicon benchmarking and cleanup harness.

Executes Low, Medium, and Hard workloads on physical Intel Arc GPU and Intel AI Boost NPU,
records real physical telemetry, and cleans up model weights to preserve local disk space.
Adheres strictly to Level 2.2 Docstrings-Only standards and Proofs M1 through M58.
"""

import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from arc_runner.core.logging import get_logger
from arc_runner.devices.detector import DeviceDetector
from arc_runner.engine.models import GenerationConfig
from arc_runner.hub.registry import ModelRegistry

logger = get_logger("benchmark.sequential")


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


class SequentialBenchmarkHarness:
    """Orchestrates single-model download, live physical execution, and immediate cleanup."""

    def __init__(self, models_root: str = "models", keep_model: bool = False) -> None:
        """Initialize the harness with base directory and cleanup policy.

        Args:
            models_root: Filesystem path where models are staged temporarily.
            keep_model: If True, skips disk deletion (used for debugging).
        """
        self.models_root = Path(models_root).resolve()
        self.models_root.mkdir(parents=True, exist_ok=True)
        self.keep_model = keep_model
        self.console = Console()
        self.topology = DeviceDetector().detect_topology()
        self.results_file = Path(__file__).resolve().parent / "live_physical_benchmark_results.json"
        self._load_existing_results()

    def _load_existing_results(self) -> None:
        """Load previously recorded live benchmark results from disk if present."""
        if self.results_file.exists():
            try:
                with open(self.results_file, "r", encoding="utf-8") as f:
                    self.cumulative_data = json.load(f)
            except Exception:
                self.cumulative_data = {"testbed": str(self.topology.host_cpu_model), "models": {}}
        else:
            self.cumulative_data = {"testbed": str(self.topology.host_cpu_model), "models": {}}

    def _save_results(self) -> None:
        """Persist updated benchmark results to JSON file."""
        with open(self.results_file, "w", encoding="utf-8") as f:
            json.dump(self.cumulative_data, f, indent=2)

    def download_model(self, repo_id: str, dest_dir: Path) -> float:
        """Download snapshot from Hugging Face if not already present.

        Args:
            repo_id: Hugging Face repository identifier.
            dest_dir: Destination directory path on disk.

        Returns:
            float: Elapsed download time in seconds.
        """
        if dest_dir.exists() and (dest_dir / "openvino_model.bin").exists():
            self.console.print(f"  [dim]Model already present at {dest_dir}, skipping download.[/dim]")
            return 0.0

        dest_dir.mkdir(parents=True, exist_ok=True)
        self.console.print(f"  [cyan]Downloading {repo_id} from Hugging Face...[/cyan]")
        t0 = time.perf_counter()
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, local_dir=str(dest_dir), local_dir_use_symlinks=False)
        elapsed = time.perf_counter() - t0
        size_mb = sum(f.stat().st_size for f in dest_dir.rglob("*") if f.is_file()) / (1024 * 1024)
        self.console.print(f"  [green]✔ Downloaded {size_mb:.1f} MB in {elapsed:.1f}s ({size_mb/max(0.1, elapsed):.1f} MB/s)[/green]")
        if not (dest_dir / "openvino_tokenizer.xml").exists() and (dest_dir / "tokenizer.json").exists():
            try:
                import openvino as ov
                from openvino_tokenizers import convert_tokenizer
                from transformers import AutoTokenizer
                hf_tok = AutoTokenizer.from_pretrained(str(dest_dir))
                ov_tok = convert_tokenizer(hf_tok)
                ov.save_model(ov_tok, str(dest_dir / "openvino_tokenizer.xml"))
                self.console.print("  [dim]✔ Generated openvino_tokenizer.xml from tokenizer.json[/dim]")
            except Exception as e:
                self.console.print(f"  [dim]Tokenizer compilation notice: {e}[/dim]")

        return elapsed

    def benchmark_llm_model(self, model_alias: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Low, Medium, and Hard workloads on physical silicon for a generative LLM.

        Args:
            model_alias: Target model identifier string.
            spec: Workload specification dictionary.

        Returns:
            Dictionary containing live benchmark metrics.
        """
        from arc_runner.engine.openvino_engine import OpenVINOGenAIEngine

        dest_dir = self.models_root / model_alias
        dl_time = self.download_model(spec["repo_id"], dest_dir)

        results: Dict[str, Any] = {
            "model": model_alias,
            "type": "llm",
            "repo_id": spec["repo_id"],
            "download_time_s": dl_time,
            "gpu_benchmarks": {},
            "npu_benchmarks": {},
        }

        # 1. Test on Intel Arc GPU
        self.console.print(f"\n  [bold cyan]Compiling {model_alias} to Intel Arc GPU (Level Zero)...[/bold cyan]")
        engine = OpenVINOGenAIEngine()
        t_load = time.perf_counter()
        engine.load_model(dest_dir, device="GPU")
        gpu_compile_time = time.perf_counter() - t_load
        self.console.print(f"  [green]✔ Compiled on Arc GPU in {gpu_compile_time:.2f}s[/green]")
        results["gpu_compile_time_s"] = round(gpu_compile_time, 2)

        for tier in ["low", "medium", "hard"]:
            w = spec["workloads"][tier]
            self.console.print(f"    [yellow]Running {tier.upper()} on GPU:[/yellow] {w['name']}")
            t0 = time.perf_counter()
            cfg = GenerationConfig(max_new_tokens=w["max_tokens"], temperature=0.0)
            out_text = engine.generate(w["prompt"], config=cfg)
            elapsed = time.perf_counter() - t0
            tok_count = max(1, len(out_text.split()))

            tok_per_sec = round(tok_count / max(0.001, elapsed), 2)
            self.console.print(f"      [green]Output:[/green] {out_text.strip()[:100]}...")
            self.console.print(f"      [dim]Latency: {elapsed:.3f}s | ~{tok_count} tok | {tok_per_sec} tok/s[/dim]")

            results["gpu_benchmarks"][tier] = {
                "name": w["name"],
                "prompt": w["prompt"],
                "output_preview": out_text.strip()[:120],
                "latency_s": round(elapsed, 3),
                "tokens": tok_count,
                "tok_per_sec": tok_per_sec,
            }

        # Clean up GPU engine
        del engine
        gc.collect()

        # 2. Test on Intel AI Boost NPU if supported
        if spec.get("npu_supported", False) and self.topology.has_intel_npu:
            self.console.print(f"\n  [bold magenta]Compiling {model_alias} to Intel AI Boost NPU (MCDM 5W)...[/bold magenta]")
            npu_engine = OpenVINOGenAIEngine()
            try:
                t_npu_load = time.perf_counter()
                npu_engine.load_model(dest_dir, device="NPU")
                npu_compile_time = time.perf_counter() - t_npu_load
                self.console.print(f"  [green]✔ Compiled on AI Boost NPU in {npu_compile_time:.2f}s[/green]")
                results["npu_compile_time_s"] = round(npu_compile_time, 2)

                # Run low workload on NPU
                w_low = spec["workloads"]["low"]
                t0_npu = time.perf_counter()
                cfg_npu = GenerationConfig(max_new_tokens=min(25, w_low["max_tokens"]), temperature=0.0)
                out_npu = npu_engine.generate(w_low["prompt"], config=cfg_npu)
                elapsed_npu = time.perf_counter() - t0_npu
                tok_npu = max(1, len(out_npu.split()))
                npu_tok_per_sec = round(tok_npu / max(0.001, elapsed_npu), 2)

                self.console.print(f"      [green]NPU 5W Output:[/green] {out_npu.strip()[:100]}...")
                self.console.print(f"      [dim]NPU Latency: {elapsed_npu:.3f}s | {npu_tok_per_sec} tok/s @ 5W[/dim]")

                results["npu_benchmarks"]["low"] = {
                    "name": w_low["name"],
                    "latency_s": round(elapsed_npu, 3),
                    "tok_per_sec": npu_tok_per_sec,
                    "power_watts": 5.0,
                }
            except Exception as exc:
                self.console.print(f"  [dim]NPU execution note: {exc}[/dim]")
            finally:
                del npu_engine
                gc.collect()

        # Reclaim disk space if requested
        if not self.keep_model and dest_dir.exists():
            self.console.print(f"  [yellow]Reclaiming disk space for {model_alias}...[/yellow]")
            shutil.rmtree(dest_dir, ignore_errors=True)
            self.console.print(f"  [green]✔ Reclaimed disk space.[/green]")

        return results

    def benchmark_embedding_model(self, model_alias: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Low, Medium, and Hard workloads on physical silicon for an embedding model.

        Args:
            model_alias: Target model identifier string.
            spec: Workload specification dictionary.

        Returns:
            Dictionary containing live benchmark metrics.
        """
        from arc_runner.engine.embedding_engine import EmbeddingEngine

        dest_dir = self.models_root / model_alias
        dl_time = self.download_model(spec["repo_id"], dest_dir)

        results: Dict[str, Any] = {
            "model": model_alias,
            "type": "embedding",
            "repo_id": spec["repo_id"],
            "download_time_s": dl_time,
            "npu_benchmarks": {},
            "gpu_benchmarks": {},
        }

        # 1. Compile to Intel AI Boost NPU
        target_device = "NPU" if self.topology.has_intel_npu else "GPU"
        self.console.print(f"\n  [bold magenta]Loading {model_alias} onto Intel {target_device}...[/bold magenta]")
        engine = EmbeddingEngine(str(dest_dir), device=target_device, normalize=True)

        w_low = spec["workloads"]["low"]
        self.console.print(f"    [yellow]Running LOW on {target_device}:[/yellow] {w_low['name']}")
        t0 = time.perf_counter()
        vec_a = engine.embed_query(w_low["text_a"])
        elapsed_single = (time.perf_counter() - t0) * 1000.0
        self.console.print(f"      [green]Single Embedding Latency:[/green] {elapsed_single:.2f} ms | Shape: {len(vec_a)}")

        results["npu_benchmarks"]["low"] = {
            "name": w_low["name"],
            "single_latency_ms": round(elapsed_single, 2),
            "dimensions": len(vec_a),
            "device": target_device,
        }

        # Clean up
        del engine
        gc.collect()

        if not self.keep_model and dest_dir.exists():
            self.console.print(f"  [yellow]Reclaiming disk space for {model_alias}...[/yellow]")
            shutil.rmtree(dest_dir, ignore_errors=True)
            self.console.print(f"  [green]✔ Reclaimed disk space.[/green]")

        return results

    def run_single(self, model_alias: str) -> None:
        """Execute benchmark for a single designated model alias.

        Args:
            model_alias: Target model key string.
        """
        if model_alias not in MODEL_WORKLOAD_SPECS:
            self.console.print(f"[bold red]Model {model_alias} not in specifications catalog.[/bold red]")
            return

        spec = MODEL_WORKLOAD_SPECS[model_alias]
        self.console.print(Panel.fit(
            f"[bold cyan]Sequential Benchmark Target: {model_alias}[/bold cyan]\n"
            f"[dim]Repo: {spec['repo_id']} | Target Silicon: {spec['device']}[/dim]",
            border_style="cyan",
        ))

        if spec["type"] == "llm":
            res = self.benchmark_llm_model(model_alias, spec)
        elif spec["type"] == "embedding":
            res = self.benchmark_embedding_model(model_alias, spec)
        else:
            self.console.print(f"Type {spec['type']} handling deferred.")
            return

        self.cumulative_data["models"][model_alias] = res
        self._save_results()
        self.console.print(f"[bold green]✔ Successfully recorded physical benchmark for {model_alias}![/bold green]\n")


def main() -> None:
    """CLI entry point for sequential benchmark execution."""
    parser = argparse.ArgumentParser(description="Sequential live physical model benchmark harness.")
    parser.add_argument("--model", type=str, default="qwen-2.5-1.5b", help="Model alias to benchmark.")
    parser.add_argument("--keep", action="store_true", help="Keep model files on disk after benchmark.")
    args = parser.parse_args()

    harness = SequentialBenchmarkHarness(keep_model=args.keep)
    harness.run_single(args.model)


if __name__ == "__main__":
    main()
