"""Live audio (speech recognition and synthesis) benchmark that writes schema-v1 result files.

Every number comes from timing real AudioEngine calls: warmup runs are discarded, timed runs are
recorded one sample each, and package power/energy come from the Sysman energy counter when the machine
has one (otherwise they are None). RTF is derived from measured latency and the known clip length. The
speech-recognition input is a generated test tone (timing is valid; transcripts are not evaluated).
There is no simulated fallback: a model that cannot load raises. If no Intel Arc GPU is present the
target device is CPU and every row is flagged fallback_used. Result files go to
docs/benchmarks/results/ and are rendered into the docs by scripts/render_bench_docs.py.

Usage: python scratch/run_live_audio_benchmarks.py --model whisper-tiny [--warmups 3] [--runs 10]
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import shutil
import sys
import tempfile
import wave
from typing import Any, Callable, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from arc_runner.benchmark.harness_common import (  # noqa: E402
    HarnessConfig, add_harness_args, begin_run, config_from_args, default_result_path, directory_size_bytes, download_snapshot,
    finish_run, optional_sysman, target_device,
)
from arc_runner.benchmark.nonllm import stt_samples, tts_samples  # noqa: E402
from arc_runner.benchmark.stats import BenchSample  # noqa: E402
from arc_runner.benchmark.workloads import export_workload_result  # noqa: E402

AUDIO_MODEL_WORKLOAD_SPECS: dict[str, dict[str, Any]] = {
    "whisper-tiny": {
        "type": "stt",
        "alias": "whisper-tiny",
        "repo_id": "OpenVINO/whisper-tiny-fp16-ov",
        "device": "GPU",
        "npu_supported": True,
        "size_bytes": 75000000,
        "workloads": {
            "low": {
                "name": "Short Command & Control ASR",
                "text": "Computer, activate Level Zero hardware accelerator.",
                "duration_s": 3.0,
            },
            "medium": {
                "name": "Technical Spec Utterance with Digits & Acronyms",
                "text": "Intel Arc Xe-LPG contains 8 Xe cores with 1024 FLOPs per cycle per XMX DPAS tile.",
                "duration_s": 8.5,
            },
            "hard": {
                "name": "Fast Conversational Discourse with Ambient Noise",
                "text": "In continuous batching with paged block tables, Little's Law guarantees token slot admission while preventing GPU VRAM exhaustion during concurrent streaming requests.",
                "duration_s": 18.0,
            },
        },
    },
    "whisper-base": {
        "type": "stt",
        "alias": "whisper-base",
        "repo_id": "OpenVINO/whisper-base-fp16-ov",
        "device": "GPU",
        "npu_supported": True,
        "size_bytes": 145000000,
        "workloads": {
            "low": {
                "name": "Factual Interrogative Utterance",
                "text": "What is the clock frequency of the Intel Core Ultra processor?",
                "duration_s": 4.0,
            },
            "medium": {
                "name": "Multi-Clause Compound Sentence with Temporal Segments",
                "text": "Although traditional OpenCL driver queues introduce kernel transition overhead, Level Zero immediate command lists submit tasks in less than 2.1 microseconds.",
                "duration_s": 10.0,
            },
            "hard": {
                "name": "Dense Hardware Microarchitecture Protocol Passage",
                "text": "Direct Memory Access bypasses the central processing unit, allowing peripheral devices like network interface controllers to transfer packets straight into physical DDR5 RAM without bus starvation.",
                "duration_s": 22.0,
            },
        },
    },
    "whisper-small": {
        "type": "stt",
        "alias": "whisper-small",
        "repo_id": "OpenVINO/whisper-small-fp16-ov",
        "device": "GPU",
        "npu_supported": False,
        "size_bytes": 480000000,
        "workloads": {
            "low": {
                "name": "Proper Named Entity & Geographic Landmark ASR",
                "text": "Dr. Thomas Vance visited CERN in Geneva to inspect the Large Hadron Collider.",
                "duration_s": 5.0,
            },
            "medium": {
                "name": "Mathematical Derivation & DSP Acoustical Passage",
                "text": "The Fourier transform of a rectangular window yields a sinc function with sidelobe attenuation of negative thirteen decibels.",
                "duration_s": 12.0,
            },
            "hard": {
                "name": "30-Second Complete System Architecture Lecture",
                "text": "Heterogeneous compute architectures integrate discrete matrix accelerators and systolic neural processing units across shared virtual memory buses, enabling client-side speculative decoding where draft models execute at 5 watts while verification passes saturate GPU GDDR6 bandwidth.",
                "duration_s": 30.0,
            },
        },
    },
    "whisper-large-v3-turbo": {
        "type": "stt",
        "alias": "whisper-large-v3-turbo",
        "repo_id": "OpenVINO/whisper-large-v3-turbo-int4-ov",
        "device": "GPU",
        "npu_supported": False,
        "size_bytes": 500000000,
        "workloads": {
            "low": {
                "name": "Low-Amplitude Whispered Acoustic Utterance",
                "text": "The secret passcode for root authentication is Alpha-7-Zero-Niner.",
                "duration_s": 4.5,
            },
            "medium": {
                "name": "Accented & Dialectal Complex Technical Passage",
                "text": "Operating system kernels employ lock-free single-producer single-consumer ring buffers with atomic acquire-release semantics to communicate between audio callbacks and user interface threads.",
                "duration_s": 14.0,
            },
            "hard": {
                "name": "Multilingual Code-Switching Speech",
                "text": "Artificial intelligence (कृत्रिम बुद्धिमत्ता) is revolutionizing enterprise software architectures across global distributed clusters.",
                "duration_s": 25.0,
            },
        },
    },
    "distil-whisper-large-v3": {
        "type": "stt",
        "alias": "distil-whisper-large-v3",
        "repo_id": "OpenVINO/distil-whisper-large-v3-int4-ov",
        "device": "GPU",
        "npu_supported": False,
        "size_bytes": 450000000,
        "workloads": {
            "low": {
                "name": "High-Cadence Conversational Burst",
                "text": "Yes, acknowledged. The deployment finished with zero errors.",
                "duration_s": 3.5,
            },
            "medium": {
                "name": "Formal Legal & Regulatory Software License Clause",
                "text": "The licensee shall not decompile, reverse-engineer, or disassemble any binary components without explicit written authorization.",
                "duration_s": 11.0,
            },
            "hard": {
                "name": "Dense 30-Second Distributed Database Architecture Exposition",
                "text": "Telemetry collectors stream time-series metrics over gRPC protocol buffers, persisting events into distributed LSM-tree columnar storage engines with guaranteed write-ahead logging durability.",
                "duration_s": 30.0,
            },
        },
    },
    "kokoro-82m": {
        "type": "tts",
        "alias": "kokoro-82m",
        "repo_id": "OpenVINO/Kokoro-82M-fp16-ov",
        "device": "GPU",
        "npu_supported": False,
        "size_bytes": 160000000,
        "workloads": {
            "low": {
                "name": "Instant System Status Speech Synthesis",
                "voice": "af_heart",
                "prompt": "System online. All heterogeneous silicon devices are operational.",
                "speed": 1.0,
            },
            "medium": {
                "name": "Hardware Technical Narration with Numbers",
                "voice": "am_adam",
                "prompt": "The Intel Arc matrix architecture utilizes 8 XMX systolic engines per Xe core, executing 1024 floating point operations every single clock cycle.",
                "speed": 1.0,
            },
            "hard": {
                "name": "Long-Form Expressive Discourse with British Styling",
                "voice": "bf_alice",
                "prompt": "Welcome to the Intel Arc Models Runner studio. By combining client-side speculative decoding with Level Zero immediate command lists, local artificial intelligence achieves cloud-grade responsiveness while cutting power consumption by up to sixty two percent.",
                "speed": 1.0,
            },
        },
    },
    "kokoro-82m-int8": {
        "type": "tts",
        "alias": "kokoro-82m-int8",
        "repo_id": "OpenVINO/Kokoro-82M-int8-ov",
        "device": "GPU",
        "npu_supported": False,
        "size_bytes": 85000000,
        "workloads": {
            "low": {
                "name": "Ultra-Low-Latency Haptic Voice Confirmation",
                "voice": "af_bella",
                "prompt": "Task complete. Model deleted from disk.",
                "speed": 1.0,
            },
            "medium": {
                "name": "Accelerated Playback Speed Scaling (1.25x)",
                "voice": "bm_george",
                "prompt": "Real-time factor measures the ratio of inference wall-clock time to the generated audio duration.",
                "speed": 1.25,
            },
            "hard": {
                "name": "Extended Multi-Sentence Discourse Stress Test",
                "voice": "af_heart",
                "prompt": "Heterogeneous client silicon represents a paradigm shift in local machine learning. By orchestrating lightweight neural processing units alongside high-throughput discrete graphics processing units, local applications can run both generative language models and speech synthesis simultaneously. This architectural blueprint demonstrates that mathematical rigor and hardware-level optimization yield zero bloat and maximum efficiency.",
                "speed": 1.0,
            },
        },
    },
}


def write_test_tone(duration_s: float, dest_path: Path, sample_rate: int = 16000) -> None:
    """Write a deterministic speech-band test tone as a 16-bit mono WAV.

    Args:
        duration_s: Clip length in seconds.
        dest_path: Output file.
        sample_rate: Sampling rate in Hz.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    waveform = (0.5 * np.sin(2 * np.pi * 120.0 * t) + 0.3 * np.sin(2 * np.pi * 500.0 * t) + 0.2 * np.sin(2 * np.pi * 1500.0 * t))
    waveform *= 0.5 * (1.0 + np.sin(2 * np.pi * 3.0 * t)) * 0.7
    with wave.open(str(dest_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((waveform * 32767.0).astype(np.int16).tobytes())


def collect_samples(
    alias: str,
    spec: Dict[str, Any],
    engine_factory: Callable[[str], Any],
    device: str,
    fallback: bool,
    cpu_baseline: bool,
    warmups: int,
    runs: int,
    sysman: Any,
    workdir: Path,
    harness: Optional[HarnessConfig] = None,
) -> List[BenchSample]:
    """Measure every workload tier of one audio model on the target device and, if distinct, on CPU.

    Args:
        alias: Model alias.
        spec: Entry of AUDIO_MODEL_WORKLOAD_SPECS.
        engine_factory: Function taking a device name and returning a loaded engine.
        device: Target device.
        fallback: True if device is a fallback for the wanted one.
        cpu_baseline: Also measure on CPU when device is not CPU.
        warmups: Discarded warmup runs per tier.
        runs: Timed runs per tier.
        sysman: Optional Sysman for package power.
        workdir: Directory for generated test clips.
        harness: Harness settings (converged warmup, interleaving, cooldown); None keeps fixed warmups with no cooldown.

    Returns:
        List[BenchSample]: Samples for all tiers and backends.
    """
    plan = [("arc-runner", device, fallback)]
    if cpu_baseline and device != "CPU":
        plan.append(("cpu-baseline", "CPU", False))
    cfg = harness or HarnessConfig(max_warmups=None, cooldown_s=0.0)
    samples: List[BenchSample] = []
    for plan_index, (backend, dev, is_fallback) in enumerate(plan):
        if plan_index:
            cfg.cooldown()
        engine = engine_factory(dev)
        try:
            for tier, workload in spec["workloads"].items():
                model_id = f"{alias}/{tier}"
                if spec["type"] == "stt":
                    clip = workdir / f"{alias}_{tier}.wav"
                    write_test_tone(workload["duration_s"], clip)
                    samples += stt_samples(engine, clip, workload["duration_s"], model_id, backend, warmups, runs, sysman=sysman, fallback_used=is_fallback, max_warmups=cfg.max_warmups)
                else:
                    samples += tts_samples(engine, workload["prompt"], model_id, backend, warmups, runs, sysman=sysman, fallback_used=is_fallback, max_warmups=cfg.max_warmups)
        finally:
            engine.unload()
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
    parser = argparse.ArgumentParser(description="Live audio benchmark writing schema-v1 results.")
    parser.add_argument("--model", required=True, choices=sorted(AUDIO_MODEL_WORKLOAD_SPECS))
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--models-root", default="models")
    parser.add_argument("--no-cpu-baseline", action="store_true")
    parser.add_argument("--keep", action="store_true", help="keep the model files on disk")
    parser.add_argument("--out", type=Path)
    add_harness_args(parser)
    args = parser.parse_args(argv)

    from arc_runner.devices.detector import DeviceDetector
    from arc_runner.engine.audio_engine import AudioEngine

    spec = AUDIO_MODEL_WORKLOAD_SPECS[args.model]
    model_dir = Path(args.models_root).resolve() / args.model
    download_snapshot(spec["repo_id"], model_dir, ("openvino_model.xml", "openvino_decoder_model.xml"))
    device, fallback = target_device(DeviceDetector().detect_topology(), spec["device"])

    def factory(dev: str) -> Any:
        engine = AudioEngine()
        engine.load_model(model_dir, device=dev)
        return engine

    harness = config_from_args(args)
    report = begin_run(harness, directory_size_bytes(model_dir))
    workdir = Path(tempfile.mkdtemp(prefix="audio_bench_"))
    try:
        samples = collect_samples(args.model, spec, factory, device, fallback, not args.no_cpu_baseline, args.warmups, args.runs, optional_sysman(), workdir, harness)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        if not args.keep:
            shutil.rmtree(model_dir, ignore_errors=True)
    samples, protocol_extra, environment_extra = finish_run(samples, harness, report)
    out = args.out or default_result_path(REPO_ROOT, f"audio_{spec['type']}", args.model)
    labels = {"arc-runner": f"OpenVINO on {device}", "cpu-baseline": "OpenVINO on CPU"}
    export_workload_result(out, f"audio-{args.model}", samples, labels, args.warmups, args.runs, REPO_ROOT, protocol_extra=protocol_extra, environment_extra=environment_extra)
    print(f"wrote {out} ({len(samples)} samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
