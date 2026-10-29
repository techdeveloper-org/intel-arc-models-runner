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
from pathlib import Path
import shutil
import sys
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from arc_runner.benchmark.harness_common import default_result_path, download_snapshot, optional_sysman, target_device  # noqa: E402
from arc_runner.benchmark.nonllm import video_samples  # noqa: E402
from arc_runner.benchmark.stats import BenchSample  # noqa: E402
from arc_runner.benchmark.workloads import export_workload_result  # noqa: E402

VIDEO_MODEL_WORKLOAD_SPECS: dict[str, dict[str, Any]] = {
    "ltx-video": {
        "alias": "ltx-video",
        "repo_id": "Lightricks/LTX-Video-openvino",
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
        "repo_id": "Lightricks/LTX-Video-openvino-int8",
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
        "repo_id": "THUDM/CogVideoX-2b-openvino",
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
        "repo_id": "OpenVINO/AnimateDiff-v1-5-ov",
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
        "repo_id": "OpenVINO/stable-video-diffusion-img2vid-xt-ov",
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

    Returns:
        List[BenchSample]: Samples for all tiers and backends.
    """
    plan = [("arc-runner", device, fallback)]
    if cpu_baseline and device != "CPU":
        plan.append(("cpu-baseline", "CPU", False))
    samples: List[BenchSample] = []
    for backend, dev, is_fallback in plan:
        engine = engine_factory(dev)
        try:
            for tier, workload in spec["workloads"].items():
                samples += video_samples(
                    engine, request_builder(workload), f"{alias}/{tier}", backend, warmups, runs,
                    sysman=sysman, fallback_used=is_fallback,
                )
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
        int: 0 on success, 2 if the model is not on disk and --download was not given.
    """
    parser = argparse.ArgumentParser(description="Live text-to-video benchmark writing schema-v1 results.")
    parser.add_argument("--model", required=True, choices=sorted(VIDEO_MODEL_WORKLOAD_SPECS))
    parser.add_argument("--download", action="store_true", help="download the model from Hugging Face if it is not on disk")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--models-root", default="models")
    parser.add_argument("--no-cpu-baseline", action="store_true")
    parser.add_argument("--keep", action="store_true", help="keep the model files on disk")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    from arc_runner.devices.detector import DeviceDetector
    from arc_runner.engine.video_engine import VideoEngine

    spec = VIDEO_MODEL_WORKLOAD_SPECS[args.model]
    model_dir = Path(args.models_root).resolve() / args.model
    if args.download:
        download_snapshot(spec["repo_id"], model_dir, ("model_index.json", "openvino_model.xml"))
    if not model_dir.exists():
        print(f"model directory {model_dir} not found; rerun with --download", file=sys.stderr)
        return 2
    device, fallback = target_device(DeviceDetector().detect_topology(), spec["device"])

    def factory(dev: str) -> Any:
        engine = VideoEngine()
        engine.load_model(model_dir, device=dev)
        return engine

    try:
        samples = collect_samples(args.model, spec, factory, build_request, device, fallback, not args.no_cpu_baseline, args.warmups, args.runs, optional_sysman())
    finally:
        if not args.keep:
            shutil.rmtree(model_dir, ignore_errors=True)
    out = args.out or default_result_path(REPO_ROOT, "video", args.model)
    labels = {"arc-runner": f"OpenVINO on {device} ({spec['precision']})", "cpu-baseline": "OpenVINO on CPU"}
    export_workload_result(out, f"video-{args.model}", samples, labels, args.warmups, args.runs, REPO_ROOT)
    print(f"wrote {out} ({len(samples)} samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
