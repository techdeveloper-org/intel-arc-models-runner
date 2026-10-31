"""Live vision-language (VLM) benchmark that writes schema-v1 result files.

Times real VisionEngine requests on a generated test image: warmups are discarded, timed runs are
recorded one sample each, package power/energy come from the Sysman energy counter when present
(otherwise None). VisionEngine yields only after generation finishes, so time to first token is not
observable and the chunk count is a streamer-callback count, not a token count. The CPU baseline is a
real second run. A model that cannot load raises; if no Intel Arc GPU is present the target is CPU and
every row is flagged fallback_used. This script replaces the former full-catalog audit, which printed
fixed timings and a constant PASS verdict without running any model.

Usage: python scratch/run_live_vlm_benchmarks.py --model qwen2-vl-7b [--warmups 3] [--runs 10]
"""

from __future__ import annotations

import argparse
import gc
import io
from pathlib import Path
import shutil
import sys
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from arc_runner.benchmark.harness_common import (  # noqa: E402
    HarnessConfig, add_harness_args, add_keep_args, reclaim_model_dir, begin_run, config_from_args, default_result_path, directory_size_bytes, download_snapshot,
    finish_run, optional_sysman, target_device,
)
from arc_runner.benchmark.nonllm import vlm_samples, vlm_samples_interleaved  # noqa: E402
from arc_runner.benchmark.stats import BenchSample  # noqa: E402
from arc_runner.benchmark.workloads import export_workload_result  # noqa: E402

VLM_MODEL_WORKLOAD_SPECS: Dict[str, Dict[str, Any]] = {
    "qwen2-vl-7b": {
        "repo_id": "OpenVINO/Qwen2-VL-7B-Instruct-int4-ov",
        "device": "GPU",
        "precision": "INT4",
        "workloads": {
            "low": {"name": "Colour description", "prompt": "Name the two dominant colours in this image.", "image_px": 224, "max_tokens": 32},
            "medium": {"name": "Shape inventory", "prompt": "List every distinct shape you can see and its colour.", "image_px": 448, "max_tokens": 96},
            "hard": {"name": "Detailed scene description", "prompt": "Describe this image in detail, including layout and colours.", "image_px": 896, "max_tokens": 192},
        },
    },
    "llama-3.2-11b-vision": {
        "repo_id": "OpenVINO/Llama-3.2-11B-Vision-Instruct-int4-ov",
        "device": "GPU",
        "precision": "INT4",
        "workloads": {
            "low": {"name": "Colour description", "prompt": "Name the two dominant colours in this image.", "image_px": 224, "max_tokens": 32},
            "medium": {"name": "Shape inventory", "prompt": "List every distinct shape you can see and its colour.", "image_px": 448, "max_tokens": 96},
            "hard": {"name": "Detailed scene description", "prompt": "Describe this image in detail, including layout and colours.", "image_px": 896, "max_tokens": 192},
        },
    },
}


def make_test_image(side_px: int) -> bytes:
    """Render a deterministic test image (gradient with a circle and a rectangle) as PNG bytes.

    Args:
        side_px: Image width and height in pixels.

    Returns:
        bytes: PNG data.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    ramp = np.linspace(0, 255, side_px, dtype=np.uint8)
    pixels = np.stack([np.tile(ramp, (side_px, 1)), np.tile(ramp[::-1], (side_px, 1)), np.full((side_px, side_px), 96, np.uint8)], axis=-1)
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    draw.ellipse([side_px // 8, side_px // 8, side_px // 2, side_px // 2], fill=(220, 30, 30))
    draw.rectangle([side_px // 2, side_px // 2, side_px * 7 // 8, side_px * 7 // 8], fill=(30, 30, 220))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


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
    image_factory: Callable[[int], bytes] = make_test_image,
    harness: Optional[HarnessConfig] = None,
) -> List[BenchSample]:
    """Measure every tier of one VLM on the target device and, if distinct, on CPU.

    Args:
        alias: Model alias.
        spec: Entry of VLM_MODEL_WORKLOAD_SPECS.
        engine_factory: Function taking a device name and returning a loaded VisionEngine.
        device: Target device.
        fallback: True if device is a fallback for the wanted one.
        cpu_baseline: Also measure on CPU when device is not CPU.
        warmups: Discarded warmup runs per tier.
        runs: Timed runs per tier.
        sysman: Optional Sysman for package power.
        image_factory: Function returning PNG bytes for a side length.
        harness: Harness settings (converged warmup, interleaving, cooldown); None keeps fixed warmups with no cooldown.

    Returns:
        List[BenchSample]: Samples for all tiers and backends.
    """
    plan = [("arc-runner", device, fallback)]
    if cpu_baseline and device != "CPU":
        plan.append(("cpu-baseline", "CPU", False))
    cfg = harness or HarnessConfig(max_warmups=None, cooldown_s=0.0)
    samples: List[BenchSample] = []
    if cfg.interleave:
        systems = [(backend, engine_factory(dev), is_fallback) for backend, dev, is_fallback in plan]
        try:
            for index, (tier, workload) in enumerate(spec["workloads"].items()):
                if index:
                    cfg.cooldown()
                samples += vlm_samples_interleaved(
                    systems, workload["prompt"], [image_factory(workload["image_px"])], workload["max_tokens"], f"{alias}/{tier}",
                    warmups, runs, seed=cfg.seed, sysman=sysman, max_warmups=cfg.max_warmups,
                )
        finally:
            del systems
            gc.collect()
        return samples
    for plan_index, (backend, dev, is_fallback) in enumerate(plan):
        if plan_index:
            cfg.cooldown()
        engine = engine_factory(dev)
        try:
            for tier, workload in spec["workloads"].items():
                image = image_factory(workload["image_px"])
                samples += vlm_samples(
                    engine, workload["prompt"], [image], workload["max_tokens"], f"{alias}/{tier}", backend, warmups, runs,
                    sysman=sysman, fallback_used=is_fallback, max_warmups=cfg.max_warmups,
                )
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
    parser = argparse.ArgumentParser(description="Live VLM benchmark writing schema-v1 results.")
    parser.add_argument("--model", required=True, choices=sorted(VLM_MODEL_WORKLOAD_SPECS))
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--models-root", default="models")
    parser.add_argument("--no-cpu-baseline", action="store_true")
    add_keep_args(parser)
    parser.add_argument("--out", type=Path)
    add_harness_args(parser)
    args = parser.parse_args(argv)

    from arc_runner.devices.detector import DeviceDetector
    from arc_runner.engine.vision_engine import VisionEngine

    spec = VLM_MODEL_WORKLOAD_SPECS[args.model]
    model_dir = Path(args.models_root).resolve() / args.model
    download_snapshot(spec["repo_id"], model_dir, ("openvino_language_model.xml", "openvino_model.xml"))
    device, fallback = target_device(DeviceDetector().detect_topology(), spec["device"])
    harness = config_from_args(args)
    resident = 2 if harness.interleave and not args.no_cpu_baseline and device != "CPU" else 1
    on_disk = directory_size_bytes(model_dir)
    report = begin_run(harness, on_disk * resident if on_disk is not None else None)
    try:
        samples = collect_samples(args.model, spec, lambda dev: VisionEngine(str(model_dir), device=dev), device, fallback, not args.no_cpu_baseline, args.warmups, args.runs, optional_sysman(), harness=harness)
    finally:
        reclaim_model_dir(args.delete_after, model_dir, Path(args.models_root).resolve())
    samples, protocol_extra, environment_extra = finish_run(samples, harness, report)
    out = args.out or default_result_path(REPO_ROOT, "vlm", args.model)
    labels = {"arc-runner": f"OpenVINO on {device} ({spec['precision']})", "cpu-baseline": "OpenVINO on CPU"}
    export_workload_result(out, f"vlm-{args.model}", samples, labels, args.warmups, args.runs, REPO_ROOT, protocol_extra=protocol_extra, environment_extra=environment_extra)
    print(f"wrote {out} ({len(samples)} samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
