"""Live text-to-video benchmark that writes schema-v1 result files.

Every number comes from timing real VideoEngine calls: warmup runs are discarded, timed runs are recorded
one sample each, and package power/energy come from the Sysman energy counter when the machine has one
(otherwise they are None). Stage timings are the engine's own OpenVINO perf metrics and are None when
OpenVINO reported none. Video output is not scored for temporal quality here; it measures speed.
There is no simulated fallback: a model that cannot load raises. If no Intel Arc GPU is present the target
device is CPU and every row is flagged fallback_used. The CPU baseline is a real second run on CPU, not a
multiple of the GPU time.

Usage: python scratch/run_live_video_benchmarks.py --model <alias> [--download] [--warmups 3] [--runs 10]
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import subprocess

from arc_runner.benchmark.harness_common import (  # noqa: E402
    HarnessConfig,
    add_harness_args,
    begin_run,
    collect_tiers,
    config_from_args,
    default_result_path,
    directory_size_bytes,
    finish_run,
    gpu_context_lost,
    retain_lost_engine,
    optional_sysman,
    select_tiers,
    target_device,
)
from arc_runner.benchmark.nonllm import video_samples  # noqa: E402
from arc_runner.benchmark.stats import BenchSample  # noqa: E402
from arc_runner.benchmark.workloads import export_workload_result  # noqa: E402


def download_and_convert(source_repo_id: str, dest_dir: Path, task: str, extra_args: list) -> None:
    """Download and convert a Hugging Face model to OpenVINO IR using optimum-cli.

    Args:
        source_repo_id: Hugging Face model repo identifier (e.g. 'Lightricks/LTX-Video').
        dest_dir: Local directory path to write the converted OpenVINO IR files into.
        task: optimum-cli export task name (e.g. 'text-to-video', 'image-to-video').
        extra_args: Additional CLI flags forwarded verbatim to optimum-cli (e.g. ['--weight-format', 'int8']).

    Raises:
        RuntimeError: If optimum-cli is not installed or the conversion process exits non-zero.
    """
    cmd = [
        "optimum-cli", "export", "openvino",
        "--model", source_repo_id,
        "--task", task,
        *extra_args,
        str(dest_dir),
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise RuntimeError(
            "optimum-cli not found. Install it with: pip install optimum[openvino]"
        ) from None
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"optimum-cli conversion failed for '{source_repo_id}' (exit {exc.returncode}). "
            "Check the output above for details."
        ) from exc


VIDEO_MODEL_WORKLOAD_SPECS: dict[str, dict[str, Any]] = {
    "ltx-video": {
        "alias": "ltx-video",
        "source_repo_id": "Lightricks/LTX-Video",
        "ov_export_task": "text-to-video",
        "ov_extra_args": [],
        "precision": "FP16 IR",
        "parameters": "2B",
        "architecture": "DiT Spatial-Temporal 3D Transformer + Causal 3D VAE",
        "size_bytes": 4200000000,
        "device": "GPU",
        "workloads": {
            "low": {
                "name": "Golden Retriever Bouncing Tennis Ball",
                "prompt": "A golden retriever puppy chasing a bouncing tennis ball on green grass, sunlight, 4k",
                "negative_prompt": "blurry, low quality, flickering, jittery, distorted",
                "num_frames": 8,
                "fps": 8,
                "width": 256,
                "height": 256,
                "steps": 10,
                "guidance_scale": 5.0,
                "seed": 42,
            },
            "medium": {
                "name": "Cinematic Drone Stormy Coastline",
                "prompt": "Cinematic drone flyover of a stormy coastline with crashing ocean waves against volcanic rocks, sunset glow, 8k resolution",
                "negative_prompt": "blurry, low quality, jittery, deformed, artifacts, flickering",
                "num_frames": 16,
                "fps": 16,
                "width": 512,
                "height": 512,
                "steps": 25,
                "guidance_scale": 7.5,
                "seed": 101,
            },
            "hard": {
                "name": "Sci-Fi Starship Nebula Cruise",
                "prompt": "Ultra-detailed sci-fi starship cruising through an ethereal cosmic nebula, glowing plasma thrusters, asteroids tumbling past in space, volumetric starlight, hyperrealistic",
                "negative_prompt": "blurry, low quality, flickering, geometric distortion, low framerate, ghosting, tearing",
                "num_frames": 33,
                "fps": 24,
                "width": 768,
                "height": 512,
                "steps": 40,
                "guidance_scale": 8.5,
                "seed": 202,
            },
        },
    },
    "ltx-video-int8": {
        "alias": "ltx-video-int8",
        "source_repo_id": "Lightricks/LTX-Video",
        "ov_export_task": "text-to-video",
        "ov_extra_args": ["--weight-format", "int8"],
        "precision": "INT8 NNCF",
        "parameters": "2B",
        "architecture": "DiT Spatial-Temporal 3D Transformer (INT8 DPAS)",
        "size_bytes": 2100000000,
        "device": "GPU",
        "workloads": {
            "low": {
                "name": "Golden Retriever Bouncing Tennis Ball",
                "prompt": "A golden retriever puppy chasing a bouncing tennis ball on green grass, sunlight, 4k",
                "negative_prompt": "blurry, low quality, flickering, jittery",
                "num_frames": 8,
                "fps": 8,
                "width": 256,
                "height": 256,
                "steps": 10,
                "guidance_scale": 5.0,
                "seed": 42,
            },
            "medium": {
                "name": "Cinematic Drone Stormy Coastline",
                "prompt": "Cinematic drone flyover of a stormy coastline with crashing ocean waves against volcanic rocks, sunset glow, 8k resolution",
                "negative_prompt": "blurry, low quality, jittery, deformed, artifacts, flickering",
                "num_frames": 16,
                "fps": 16,
                "width": 512,
                "height": 512,
                "steps": 25,
                "guidance_scale": 7.5,
                "seed": 101,
            },
            "hard": {
                "name": "Sci-Fi Starship Nebula Cruise",
                "prompt": "Ultra-detailed sci-fi starship cruising through an ethereal cosmic nebula, glowing plasma thrusters, asteroids tumbling past in space, volumetric starlight, hyperrealistic",
                "negative_prompt": "blurry, low quality, flickering, geometric distortion, low framerate, ghosting, tearing",
                "num_frames": 33,
                "fps": 24,
                "width": 768,
                "height": 512,
                "steps": 40,
                "guidance_scale": 8.5,
                "seed": 202,
            },
        },
    },
    "cogvideox-2b": {
        "alias": "cogvideox-2b",
        "source_repo_id": "THUDM/CogVideoX-2b",
        "ov_export_task": "text-to-video",
        "ov_extra_args": [],
        "precision": "FP16 IR",
        "parameters": "2.2B",
        "architecture": "Expert 3D DiT + Causal 3D VAE + T5 Conditioning",
        "size_bytes": 4600000000,
        "device": "GPU",
        "workloads": {
            "low": {
                "name": "Golden Retriever Bouncing Tennis Ball",
                "prompt": "A golden retriever puppy chasing a bouncing tennis ball on green grass, sunlight, 4k",
                "negative_prompt": "blurry, low quality, flickering, jittery",
                "num_frames": 8,
                "fps": 8,
                "width": 256,
                "height": 256,
                "steps": 10,
                "guidance_scale": 5.0,
                "seed": 42,
            },
            "medium": {
                "name": "Cinematic Drone Stormy Coastline",
                "prompt": "Cinematic drone flyover of a stormy coastline with crashing ocean waves against volcanic rocks, sunset glow, 8k resolution",
                "negative_prompt": "blurry, low quality, jittery, deformed, artifacts, flickering",
                "num_frames": 16,
                "fps": 16,
                "width": 512,
                "height": 512,
                "steps": 25,
                "guidance_scale": 7.5,
                "seed": 101,
            },
            "hard": {
                "name": "Sci-Fi Starship Nebula Cruise",
                "prompt": "Ultra-detailed sci-fi starship cruising through an ethereal cosmic nebula, glowing plasma thrusters, asteroids tumbling past in space, volumetric starlight, hyperrealistic",
                "negative_prompt": "blurry, low quality, flickering, geometric distortion, low framerate, ghosting, tearing",
                "num_frames": 33,
                "fps": 24,
                "width": 768,
                "height": 512,
                "steps": 40,
                "guidance_scale": 8.5,
                "seed": 202,
            },
        },
    },
    "animatediff-v1-5": {
        "alias": "animatediff-v1-5",
        "source_repo_id": "guoyww/animatediff-motion-adapter-v1-5-3",
        "ov_export_task": "text-to-video",
        "ov_extra_args": [],
        "precision": "FP16 IR",
        "parameters": "1.2B",
        "architecture": "AnimateDiff Temporal Motion Adapter + SD 1.5 UNet",
        "size_bytes": 2400000000,
        "device": "GPU",
        "workloads": {
            "low": {
                "name": "Golden Retriever Bouncing Tennis Ball",
                "prompt": "A golden retriever puppy chasing a bouncing tennis ball on green grass, sunlight, 4k",
                "negative_prompt": "blurry, low quality, flickering, jittery",
                "num_frames": 8,
                "fps": 8,
                "width": 256,
                "height": 256,
                "steps": 10,
                "guidance_scale": 5.0,
                "seed": 42,
            },
            "medium": {
                "name": "Cinematic Drone Stormy Coastline",
                "prompt": "Cinematic drone flyover of a stormy coastline with crashing ocean waves against volcanic rocks, sunset glow, 8k resolution",
                "negative_prompt": "blurry, low quality, jittery, deformed, artifacts, flickering",
                "num_frames": 16,
                "fps": 16,
                "width": 512,
                "height": 512,
                "steps": 25,
                "guidance_scale": 7.5,
                "seed": 101,
            },
            "hard": {
                "name": "Sci-Fi Starship Nebula Cruise",
                "prompt": "Ultra-detailed sci-fi starship cruising through an ethereal cosmic nebula, glowing plasma thrusters, asteroids tumbling past in space, volumetric starlight, hyperrealistic",
                "negative_prompt": "blurry, low quality, flickering, geometric distortion, low framerate, ghosting, tearing",
                "num_frames": 33,
                "fps": 24,
                "width": 768,
                "height": 512,
                "steps": 40,
                "guidance_scale": 8.5,
                "seed": 202,
            },
        },
    },
    "svd-xt": {
        "alias": "svd-xt",
        "source_repo_id": "stabilityai/stable-video-diffusion-img2vid-xt",
        "ov_export_task": "image-to-video",
        "ov_extra_args": [],
        "precision": "FP16 IR",
        "parameters": "1.5B",
        "architecture": "SVD Temporal UNet + Spatio-Temporal Cross Attention",
        "size_bytes": 3200000000,
        "device": "GPU",
        "workloads": {
            "low": {
                "name": "Golden Retriever Bouncing Tennis Ball",
                "prompt": "A golden retriever puppy chasing a bouncing tennis ball on green grass, sunlight, 4k",
                "negative_prompt": "blurry, low quality, flickering, jittery",
                "num_frames": 8,
                "fps": 8,
                "width": 256,
                "height": 256,
                "steps": 10,
                "guidance_scale": 5.0,
                "seed": 42,
            },
            "medium": {
                "name": "Cinematic Drone Stormy Coastline",
                "prompt": "Cinematic drone flyover of a stormy coastline with crashing ocean waves against volcanic rocks, sunset glow, 8k resolution",
                "negative_prompt": "blurry, low quality, jittery, deformed, artifacts, flickering",
                "num_frames": 16,
                "fps": 16,
                "width": 512,
                "height": 512,
                "steps": 25,
                "guidance_scale": 7.5,
                "seed": 101,
            },
            "hard": {
                "name": "Sci-Fi Starship Nebula Cruise",
                "prompt": "Ultra-detailed sci-fi starship cruising through an ethereal cosmic nebula, glowing plasma thrusters, asteroids tumbling past in space, volumetric starlight, hyperrealistic",
                "negative_prompt": "blurry, low quality, flickering, geometric distortion, low framerate, ghosting, tearing",
                "num_frames": 33,
                "fps": 24,
                "width": 768,
                "height": 512,
                "steps": 40,
                "guidance_scale": 8.5,
                "seed": 202,
            },
        },
    },
}


def build_request(workload: Dict[str, Any]) -> Any:
    """Build the engine request for one workload tier.

    Args:
        workload: Tier entry of the model spec.

    Returns:
        Any: The engine request object.
    """
    from arc_runner.engine.video_engine import VideoGenerationRequest

    return VideoGenerationRequest(
        prompt=workload["prompt"],
        negative_prompt=workload.get("negative_prompt") or None,
        num_frames=workload["num_frames"],
        fps=workload["fps"],
        width=workload["width"],
        height=workload["height"],
        num_inference_steps=workload["steps"],
        guidance_scale=workload["guidance_scale"],
        rng_seed=workload["seed"],
    )


def collect_samples(
    alias: str,
    spec: Dict[str, Any],
    engine_factory: Callable[[str], Any],
    request_builder: Callable[[Dict[str, Any]], Any],
    device: str,
    fallback: bool,
    cpu_baseline: bool,
    warmups: int,
    runs: int,
    sysman: Any,
    failures: Optional[List[Dict[str, str]]] = None,
    harness: Optional[HarnessConfig] = None,
) -> List[BenchSample]:
    """Measure every workload tier of one model on the target device and, if distinct, on CPU.

    Args:
        alias: Model alias.
        spec: Entry of VIDEO_MODEL_WORKLOAD_SPECS.
        engine_factory: Function taking a device name and returning a loaded engine.
        request_builder: Function turning a tier entry into an engine request.
        device: Target device.
        fallback: True if device is a fallback for the wanted one.
        cpu_baseline: Also measure on CPU when device is not CPU.
        warmups: Discarded warmup runs per tier.
        runs: Timed runs per tier.
        sysman: Optional Sysman for package power.
        failures: List that receives {model_id, backend, error} for tiers that failed; after a GPU
            CL_OUT_OF_RESOURCES the engine is not unloaded (the driver may hang) and later GPU tiers are skipped.
        harness: Harness settings (converged warmup, cooldown); None keeps fixed warmups with no cooldown. Interleaving is not used for this workload.

    Returns:
        List[BenchSample]: Samples for all tiers and backends.
    """
    plan = [("arc-runner", device, fallback)]
    if cpu_baseline and device != "CPU":
        plan.append(("cpu-baseline", "CPU", False))
    cfg = harness or HarnessConfig(max_warmups=None, cooldown_s=0.0)
    samples: List[BenchSample] = []
    failures = failures if failures is not None else []
    for plan_index, (backend, dev, is_fallback) in enumerate(plan):
        if plan_index:
            cfg.cooldown()
        engine = engine_factory(dev)

        def make(workload: Dict[str, Any], model_id: str, engine: Any = engine, backend: str = backend, is_fallback: bool = is_fallback) -> List[BenchSample]:
            return video_samples(
                engine, request_builder(workload), model_id, backend, warmups, runs, sysman=sysman, fallback_used=is_fallback, max_warmups=cfg.max_warmups,
            )

        got, lost = collect_tiers(spec["workloads"], alias, backend, make, failures)
        samples += got
        if lost:
            retain_lost_engine(engine)
            continue
        engine.unload()
        del engine
        gc.collect()
    return samples


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: Arguments without the program name, or None for sys.argv.

    Returns:
        int: 0 on success, 2 if the model is not on disk and --download was not given.
    """
    parser = argparse.ArgumentParser(description="Live text-to-video benchmark writing schema-v1 results.")
    parser.add_argument("--model", required=True, choices=sorted(VIDEO_MODEL_WORKLOAD_SPECS))
    parser.add_argument("--download", action="store_true", help="download the model from Hugging Face if it is not on disk")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--models-root", default="models")
    parser.add_argument("--no-cpu-baseline", action="store_true")
    parser.add_argument("--tiers", help="comma-separated tiers to run, for example low,medium (default: all)")
    parser.add_argument("--keep", action="store_true", help="keep the model files on disk")
    parser.add_argument("--out", type=Path)
    add_harness_args(parser)
    args = parser.parse_args(argv)

    from arc_runner.devices.detector import DeviceDetector
    from arc_runner.engine.video_engine import VideoEngine

    spec = select_tiers(VIDEO_MODEL_WORKLOAD_SPECS[args.model], args.tiers)
    model_dir = Path(args.models_root).resolve() / args.model
    if args.download:
        download_and_convert(spec["source_repo_id"], model_dir, spec["ov_export_task"], spec["ov_extra_args"])
    if not model_dir.exists():
        print(f"model directory {model_dir} not found; rerun with --download", file=sys.stderr)
        return 2
    device, fallback = target_device(DeviceDetector().detect_topology(), spec["device"])

    def factory(dev: str) -> Any:
        engine = VideoEngine()
        engine.load_model(model_dir, device=dev)
        return engine

    harness = config_from_args(args)
    report = begin_run(harness, directory_size_bytes(model_dir))
    failures: List[Dict[str, str]] = []
    try:
        samples = collect_samples(args.model, spec, factory, build_request, device, fallback, not args.no_cpu_baseline, args.warmups, args.runs, optional_sysman(), failures=failures, harness=harness)
    finally:
        if not args.keep:
            shutil.rmtree(model_dir, ignore_errors=True)
    samples, protocol_extra, environment_extra = finish_run(samples, harness, report)
    out = args.out or default_result_path(REPO_ROOT, "video", args.model)
    labels = {"arc-runner": f"OpenVINO on {device} ({spec['precision']})", "cpu-baseline": "OpenVINO on CPU"}
    export_workload_result(out, f"video-{args.model}", samples, labels, args.warmups, args.runs, REPO_ROOT, failures=failures, protocol_extra=protocol_extra, environment_extra=environment_extra)
    print(f"wrote {out} ({len(samples)} samples, {len(failures)} failed tier(s))")
    for item in failures:
        print(f"FAILED {item['model_id']} on {item['backend']}: {item['error'][:160]}", file=sys.stderr)
    if gpu_context_lost(failures):
        print("GPU context lost (CL_OUT_OF_RESOURCES): exiting without cleanup because the driver may hang; "
              "try --tiers low,medium or raise Shared GPU Memory Override in Intel Graphics Software.", file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(3)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
