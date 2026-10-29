"""Live text-to-image benchmark that writes schema-v1 result files.

Every number comes from timing real ImageEngine calls: warmup runs are discarded, timed runs are recorded
one sample each, and package power/energy come from the Sysman energy counter when the machine has one
(otherwise they are None). Stage timings are the engine's own OpenVINO perf metrics and are None when
OpenVINO reported none. Image output is not scored for quality here; it measures speed.
There is no simulated fallback: a model that cannot load raises. If no Intel Arc GPU is present the target
device is CPU and every row is flagged fallback_used. The CPU baseline is a real second run on CPU, not a
multiple of the GPU time.

Usage: python scratch/run_live_image_benchmarks.py --model <alias> [--download] [--warmups 3] [--runs 10]
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
from arc_runner.benchmark.nonllm import image_samples  # noqa: E402
from arc_runner.benchmark.stats import BenchSample  # noqa: E402
from arc_runner.benchmark.workloads import export_workload_result  # noqa: E402

IMAGE_MODEL_WORKLOAD_SPECS: dict[str, dict[str, Any]] = {
    "stable-diffusion-v1-5": {
        "alias": "stable-diffusion-v1-5",
        "repo_id": "OpenVINO/stable-diffusion-v1-5-fp16-ov",
        "precision": "FP16 IR",
        "parameters": "860M",
        "architecture": "UNet2DCondition + CLIP + VAE",
        "size_bytes": 1720000000,
        "device": "GPU",
        "workloads": {
            "low": {
                "name": "Minimalist Studio Ceramic Cup",
                "prompt": "A minimalist geometric ceramic coffee cup on a smooth wooden tabletop, clean studio lighting, 8k photograph",
                "negative_prompt": "",
                "width": 512,
                "height": 512,
                "steps": 10,
                "guidance_scale": 5.0,
                "seed": 42,
            },
            "medium": {
                "name": "Futuristic Cybernetic Laboratory",
                "prompt": "A vibrant futuristic cybernetic cyberpunk laboratory with holographic displays and glowing neon circuits, hyperrealistic, octane render, 8k resolution, cinematic lighting",
                "negative_prompt": "blurry, low quality, distorted, deformed, watermark, oversaturated",
                "width": 512,
                "height": 512,
                "steps": 25,
                "guidance_scale": 7.5,
                "seed": 101,
            },
            "hard": {
                "name": "Clockwork Cathedral Dragon",
                "prompt": "An ultra-detailed cinematic concept art of an intricate clockwork dragon perched on an ancient gothic cathedral tower during a thunderstorm, volumetric lightning, photorealistic brass gears and mechanical wings, 8k octane render, dramatic shadows",
                "negative_prompt": "blurry, cartoon, low resolution, bad anatomy, missing limbs, text, watermark, bad proportions, out of focus, duplicate",
                "width": 768,
                "height": 768,
                "steps": 50,
                "guidance_scale": 9.0,
                "seed": 2024,
            },
        },
    },
    "stable-diffusion-v1-5-int8": {
        "alias": "stable-diffusion-v1-5-int8",
        "repo_id": "OpenVINO/stable-diffusion-v1-5-int8-ov",
        "precision": "INT8 NNCF",
        "parameters": "860M",
        "architecture": "UNet2DCondition INT8 + CLIP + VAE",
        "size_bytes": 860000000,
        "device": "GPU",
        "workloads": {
            "low": {
                "name": "Modern Wireless Headphone Product Shot",
                "prompt": "A sleek modern wireless headphone resting on a dark marble pedestal, product shot, soft rim lighting",
                "negative_prompt": "",
                "width": 512,
                "height": 512,
                "steps": 10,
                "guidance_scale": 5.0,
                "seed": 42,
            },
            "medium": {
                "name": "Sunlit Greenhouse Solarium",
                "prompt": "A cozy sunlit solarium greenhouse overflowing with lush tropical monstera and exotic orchids, golden hour sunlight streaming through glass panes, photorealistic 8k",
                "negative_prompt": "blurry, low quality, distorted, deformed, watermark, oversaturated",
                "width": 512,
                "height": 512,
                "steps": 25,
                "guidance_scale": 7.5,
                "seed": 101,
            },
            "hard": {
                "name": "Cybernetic Neo-Tokyo Samurai",
                "prompt": "A majestic cybernetic samurai warrior in ornate glowing armor overlooking a rain-slicked neo-Tokyo skyline at night, reflections of neon signs in puddles, volumetric rain, highly detailed concept art, 8k",
                "negative_prompt": "blurry, cartoon, low resolution, bad anatomy, missing limbs, text, watermark, bad proportions, out of focus, duplicate",
                "width": 768,
                "height": 768,
                "steps": 50,
                "guidance_scale": 9.0,
                "seed": 2024,
            },
        },
    },
    "lcm-dreamshaper-v7": {
        "alias": "lcm-dreamshaper-v7",
        "repo_id": "rupeshs/LCM-dreamshaper-v7-openvino-int8",
        "precision": "INT8 NNCF",
        "parameters": "860M",
        "architecture": "Consistency UNet + CLIP + VAE",
        "size_bytes": 860000000,
        "device": "GPU",
        "workloads": {
            "low": {
                "name": "Robotic Red Panda in Enchanted Forest",
                "prompt": "A cute robotic red panda exploring an enchanted mossy forest, warm ambient glow, digital art",
                "negative_prompt": "",
                "width": 512,
                "height": 512,
                "steps": 2,
                "guidance_scale": 1.2,
                "seed": 777,
            },
            "medium": {
                "name": "Floating Sky Island with Crystal Falls",
                "prompt": "An ethereal floating island in the sky with crystal waterfalls cascading into clouds, fantasy landscape, vivid colors, highly detailed, 8k wallpaper",
                "negative_prompt": "low quality, blurry, deformed",
                "width": 512,
                "height": 512,
                "steps": 4,
                "guidance_scale": 1.5,
                "seed": 888,
            },
            "hard": {
                "name": "Steampunk Airship Mountain Sunset",
                "prompt": "An elaborate steampunk airship navigating through towering alpine mountain peaks at sunset, intricate brass cogs, billowing steam, warm golden light, cinematic composition, 8k resolution",
                "negative_prompt": "low quality, blurry, cartoon, artifacts, disfigured",
                "width": 768,
                "height": 768,
                "steps": 8,
                "guidance_scale": 2.0,
                "seed": 999,
            },
        },
    },
    "sd-turbo": {
        "alias": "sd-turbo",
        "repo_id": "rupeshs/sd-turbo-openvino",
        "precision": "FP16 IR",
        "parameters": "860M",
        "architecture": "ADD Distilled UNet + CLIP + VAE",
        "size_bytes": 1720000000,
        "device": "GPU",
        "workloads": {
            "low": {
                "name": "Coastal Highway Vintage Sports Car",
                "prompt": "A glossy vintage sports car driving along a coastal highway, ocean sunset, photorealistic",
                "negative_prompt": "",
                "width": 512,
                "height": 512,
                "steps": 1,
                "guidance_scale": 1.0,
                "seed": 1234,
            },
            "medium": {
                "name": "Winter Mountain Cabin Twilight",
                "prompt": "A cozy mountain cabin blanketed in fresh snow, warm glowing windows, winter twilight, pine trees, high quality photography",
                "negative_prompt": "blurry, cartoon, low quality",
                "width": 512,
                "height": 512,
                "steps": 2,
                "guidance_scale": 1.0,
                "seed": 5678,
            },
            "hard": {
                "name": "Bioluminescent Coral Reef Marine Life",
                "prompt": "A magnificent underwater coral reef teeming with bioluminescent jellyfish, sea turtles, and vibrant marine life, volumetric sun rays penetrating deep turquoise water, 8k national geographic",
                "negative_prompt": "blurry, ugly, deformed, oversaturated",
                "width": 768,
                "height": 768,
                "steps": 4,
                "guidance_scale": 1.0,
                "seed": 9012,
            },
        },
    },
    "stable-diffusion-xl-base-1.0": {
        "alias": "stable-diffusion-xl-base-1.0",
        "repo_id": "OpenVINO/stable-diffusion-xl-base-1.0-fp16-ov",
        "precision": "FP16 IR",
        "parameters": "3.5B",
        "architecture": "Dual CLIP + SDXL UNet + AutoencoderKL",
        "size_bytes": 6500000000,
        "device": "GPU",
        "workloads": {
            "low": {
                "name": "Soaring Mountain Eagle Preview",
                "prompt": "A majestic eagle soaring over snow-capped mountain peaks, expansive clear blue sky, sharp focus, 8k",
                "negative_prompt": "",
                "width": 768,
                "height": 768,
                "steps": 15,
                "guidance_scale": 6.0,
                "seed": 42,
            },
            "medium": {
                "name": "Norwegian Forest Villa Architecture",
                "prompt": "An architectural photograph of a futuristic curved glass villa nestled in a Norwegian pine forest, floor-to-ceiling windows, warm interior lighting, morning mist, 8k Hasselblad",
                "negative_prompt": "blurry, low quality, distorted, watermark, cartoon",
                "width": 1024,
                "height": 1024,
                "steps": 30,
                "guidance_scale": 7.5,
                "seed": 101,
            },
            "hard": {
                "name": "Forest Sorceress Mythical Portrait",
                "prompt": "An extraordinarily detailed portrait of a mythical forest sorceress with flowing emerald hair and ornate golden filigree jewelry, glowing butterflies orbiting around her hands, magical enchanted woodland background, cinematic lighting, 8k octane render, hyperrealistic",
                "negative_prompt": "blurry, cartoon, bad anatomy, bad hands, missing fingers, extra limbs, ugly, low resolution, watermark",
                "width": 1024,
                "height": 1024,
                "steps": 50,
                "guidance_scale": 8.5,
                "seed": 2024,
            },
        },
    },
    "sdxl-turbo": {
        "alias": "sdxl-turbo",
        "repo_id": "rupeshs/sdxl-turbo-openvino-int8",
        "precision": "INT8 NNCF",
        "parameters": "3.5B",
        "architecture": "ADD Distilled SDXL UNet + Dual CLIP + VAE",
        "size_bytes": 3200000000,
        "device": "GPU",
        "workloads": {
            "low": {
                "name": "High-Speed Water Droplet Splash",
                "prompt": "A dramatic high-speed photograph of a water droplet splashing into a calm turquoise pool, creating perfect concentric ripples, macro lens, 8k",
                "negative_prompt": "",
                "width": 1024,
                "height": 1024,
                "steps": 1,
                "guidance_scale": 1.0,
                "seed": 31415,
            },
            "medium": {
                "name": "Parisian Rainy Street Cafe Corner",
                "prompt": "A vintage 1960s Parisian cafe street corner in the rain, warm amber streetlamps, cobblestone reflections, people holding umbrellas, cinematic atmospheric lighting, 8k",
                "negative_prompt": "blurry, low quality, cartoon",
                "width": 1024,
                "height": 1024,
                "steps": 2,
                "guidance_scale": 1.0,
                "seed": 27182,
            },
            "hard": {
                "name": "Orbital Space Station Starship Dock",
                "prompt": "A hyper-detailed sci-fi orbital space station docking bay with massive interstellar starships, intricate maintenance gantries, cosmic nebula visible through viewport, volumetric lights, 8k unreal engine 5",
                "negative_prompt": "blurry, low resolution, distorted, disfigured",
                "width": 1024,
                "height": 1024,
                "steps": 4,
                "guidance_scale": 1.2,
                "seed": 16180,
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
    from arc_runner.engine.image_engine import ImageGenerationRequest

    return ImageGenerationRequest(
        prompt=workload["prompt"],
        negative_prompt=workload.get("negative_prompt") or None,
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
        spec: Entry of IMAGE_MODEL_WORKLOAD_SPECS.
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
                samples += image_samples(
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
    parser = argparse.ArgumentParser(description="Live text-to-image benchmark writing schema-v1 results.")
    parser.add_argument("--model", required=True, choices=sorted(IMAGE_MODEL_WORKLOAD_SPECS))
    parser.add_argument("--download", action="store_true", help="download the model from Hugging Face if it is not on disk")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--models-root", default="models")
    parser.add_argument("--no-cpu-baseline", action="store_true")
    parser.add_argument("--keep", action="store_true", help="keep the model files on disk")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    from arc_runner.devices.detector import DeviceDetector
    from arc_runner.engine.image_engine import ImageEngine

    spec = IMAGE_MODEL_WORKLOAD_SPECS[args.model]
    model_dir = Path(args.models_root).resolve() / args.model
    if args.download:
        download_snapshot(spec["repo_id"], model_dir, ("model_index.json",))
    if not model_dir.exists():
        print(f"model directory {model_dir} not found; rerun with --download", file=sys.stderr)
        return 2
    device, fallback = target_device(DeviceDetector().detect_topology(), spec["device"])

    def factory(dev: str) -> Any:
        engine = ImageEngine()
        engine.load_model(model_dir, device=dev)
        return engine

    try:
        samples = collect_samples(args.model, spec, factory, build_request, device, fallback, not args.no_cpu_baseline, args.warmups, args.runs, optional_sysman())
    finally:
        if not args.keep:
            shutil.rmtree(model_dir, ignore_errors=True)
    out = args.out or default_result_path(REPO_ROOT, "image", args.model)
    labels = {"arc-runner": f"OpenVINO on {device} ({spec['precision']})", "cpu-baseline": "OpenVINO on CPU"}
    export_workload_result(out, f"image-{args.model}", samples, labels, args.warmups, args.runs, REPO_ROOT)
    print(f"wrote {out} ({len(samples)} samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
