"""The live benchmark scripts measure real engine calls, flag fallbacks, and never simulate."""

import importlib.util
from pathlib import Path
import struct
from types import ModuleType, SimpleNamespace
from typing import Any, List

import pytest

from tests.test_nonllm_runners import _EmbedEngine, _ImageEngine, _LlmEngine, _SttEngine, _TtsEngine, _VideoEngine, _VlmEngine

REPO_ROOT = Path(__file__).resolve().parents[1]


def _script(name: str) -> ModuleType:
    """Import scratch/<name>.py as a module."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scratch" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Recorder:
    """Engine factory recording requested devices and unload calls."""

    def __init__(self, engine_cls: Any) -> None:
        self.engine_cls = engine_cls
        self.devices: List[str] = []
        self.unloaded = 0

    def __call__(self, device: str, *_a: Any) -> Any:
        self.devices.append(device)
        engine = self.engine_cls()
        recorder = self
        engine.unload = lambda: setattr(recorder, "unloaded", recorder.unloaded + 1)
        return engine


def test_audio_stt_collects_target_then_cpu_baseline_and_unloads(tmp_path: Path) -> None:
    """Every tier is measured on the target device and again on CPU; engines are unloaded; no fallback flag."""
    audio = _script("run_live_audio_benchmarks")
    spec = audio.AUDIO_MODEL_WORKLOAD_SPECS["whisper-tiny"]
    factory = _Recorder(_SttEngine)
    samples = audio.collect_samples("whisper-tiny", spec, factory, "GPU", False, True, 1, 2, None, tmp_path)
    tiers = len(spec["workloads"])
    assert factory.devices == ["GPU", "CPU"] and factory.unloaded == 2
    assert len(samples) == tiers * 2 * 3
    assert {s.backend for s in samples} == {"arc-runner", "cpu-baseline"}
    assert not any(s.fallback_used for s in samples) and all(s.workload == "audio_stt" for s in samples)
    assert (tmp_path / "whisper-tiny_low.wav").exists()


def test_audio_cpu_target_has_no_second_baseline_and_flags_fallback(tmp_path: Path) -> None:
    """When the target is already CPU (no Arc), no duplicate baseline is run and rows are flagged."""
    audio = _script("run_live_audio_benchmarks")
    spec = audio.AUDIO_MODEL_WORKLOAD_SPECS["whisper-tiny"]
    factory = _Recorder(_SttEngine)
    samples = audio.collect_samples("whisper-tiny", spec, factory, "CPU", True, True, 0, 1, None, tmp_path)
    assert factory.devices == ["CPU"]
    assert all(s.fallback_used for s in samples)


def test_audio_tts_model_uses_synthesis_runner(tmp_path: Path) -> None:
    """TTS specs call synthesize, not transcribe."""
    audio = _script("run_live_audio_benchmarks")
    tts_alias = next(a for a, s in audio.AUDIO_MODEL_WORKLOAD_SPECS.items() if s["type"] == "tts")
    samples = audio.collect_samples(tts_alias, audio.AUDIO_MODEL_WORKLOAD_SPECS[tts_alias], _Recorder(_TtsEngine), "GPU", False, False, 0, 1, None, tmp_path)
    assert samples and all(s.workload == "audio_tts" for s in samples)


def test_audio_test_tone_is_a_valid_wav_of_the_requested_length(tmp_path: Path) -> None:
    """The generated clip has the stated duration at 16 kHz mono 16-bit."""
    import wave

    audio = _script("run_live_audio_benchmarks")
    audio.write_test_tone(2.5, tmp_path / "t.wav")
    with wave.open(str(tmp_path / "t.wav")) as wf:
        assert wf.getframerate() == 16000 and wf.getnchannels() == 1 and wf.getsampwidth() == 2
        assert wf.getnframes() == 40000


@pytest.mark.parametrize("script,engine_cls,workload", [("run_live_image_benchmarks", _ImageEngine, "image"), ("run_live_video_benchmarks", _VideoEngine, "video")])
def test_image_and_video_collect_real_requests_for_every_tier(script: str, engine_cls: Any, workload: str) -> None:
    """Each tier builds an engine request from the spec and is measured on target and CPU."""
    module = _script(script)
    specs = getattr(module, f"{workload.upper()}_MODEL_WORKLOAD_SPECS")
    alias = next(iter(specs))
    seen: List[Any] = []

    def request_builder(workload_spec: Any) -> Any:
        seen.append(workload_spec["name"])
        return SimpleNamespace(width=64, height=64, num_frames=8, fps=8)

    factory = _Recorder(engine_cls)
    samples = module.collect_samples(alias, specs[alias], factory, request_builder, "GPU", False, True, 0, 1, None)
    assert factory.devices == ["GPU", "CPU"]
    assert len(seen) == 2 * len(specs[alias]["workloads"])
    assert all(s.workload == workload for s in samples) and len(samples) == len(seen)


@pytest.mark.parametrize("script", ["run_live_image_benchmarks", "run_live_video_benchmarks"])
def test_image_and_video_without_model_on_disk_refuse_instead_of_simulating(script: str, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """A missing model is exit code 2 with a message; no result file and no synthetic numbers."""
    module = _script(script)
    specs = getattr(module, "IMAGE_MODEL_WORKLOAD_SPECS" if "image" in script else "VIDEO_MODEL_WORKLOAD_SPECS")
    out = tmp_path / "out.json"
    code = module.main(["--model", next(iter(specs)), "--models-root", str(tmp_path), "--out", str(out)])
    assert code == 2 and not out.exists()
    assert "--download" in capsys.readouterr().err


def test_image_and_video_request_builders_map_spec_fields() -> None:
    """Spec fields land in the engine request unchanged."""
    image = _script("run_live_image_benchmarks")
    spec = next(iter(image.IMAGE_MODEL_WORKLOAD_SPECS.values()))["workloads"]["low"]
    request = image.build_request(spec)
    assert (request.width, request.height, request.num_inference_steps, request.rng_seed) == (spec["width"], spec["height"], spec["steps"], spec["seed"])
    video = _script("run_live_video_benchmarks")
    vspec = next(iter(video.VIDEO_MODEL_WORKLOAD_SPECS.values()))["workloads"]["low"]
    vrequest = video.build_request(vspec)
    assert (vrequest.num_frames, vrequest.fps, vrequest.rng_seed) == (vspec["num_frames"], vspec["fps"], vspec["seed"])


def test_sequential_backend_plan_and_embedding_batches() -> None:
    """Embeddings prefer the NPU; LLMs add an NPU backend when supported and present; CPU baseline is real."""
    seq = _script("run_live_sequential_model_benchmarks")
    topology = SimpleNamespace(has_intel_arc=True, has_intel_npu=True)
    llm = seq.MODEL_WORKLOAD_SPECS["qwen-2.5-1.5b"]
    assert [p[:2] for p in seq.backend_plan(llm, topology, True)] == [("arc-runner", "GPU"), ("npu", "NPU"), ("cpu-baseline", "CPU")]
    emb = seq.MODEL_WORKLOAD_SPECS["all-minilm-l6-v2"]
    assert [p[:2] for p in seq.backend_plan(emb, topology, False)] == [("arc-runner", "NPU")]
    no_hw = SimpleNamespace(has_intel_arc=False, has_intel_npu=False)
    assert seq.backend_plan(llm, no_hw, True) == [("arc-runner", "CPU", True)]
    assert len(seq.embedding_batch(emb["workloads"]["low"])) == 2
    assert len(seq.embedding_batch(emb["workloads"]["medium"])) == 3
    assert len(seq.embedding_batch(emb["workloads"]["hard"])) == emb["workloads"]["hard"]["batch_size"]
    nomic = seq.MODEL_WORKLOAD_SPECS["nomic-embed-text"]["workloads"]
    assert len(seq.embedding_batch(nomic["low"])) == 1
    assert len(seq.embedding_batch(nomic["medium"])) == 1 + len(nomic["medium"]["corpus"])
    assert len(seq.embedding_batch(nomic["hard"])[0].split()) == nomic["hard"]["seq_length"]
    with pytest.raises(ValueError):
        seq.embedding_batch({"name": "unknown"})
    for alias, model in seq.MODEL_WORKLOAD_SPECS.items():
        if model["type"] == "embedding":
            for tier in model["workloads"].values():
                assert seq.embedding_batch(tier), alias


def test_sequential_collect_uses_llm_telemetry_and_embedding_runner() -> None:
    """LLM specs go through telemetry-based sampling; embedding specs through batch timing."""
    seq = _script("run_live_sequential_model_benchmarks")
    llm = seq.MODEL_WORKLOAD_SPECS["qwen-2.5-1.5b"]
    llm_out = seq.collect_samples("qwen-2.5-1.5b", llm, lambda d, w: _LlmEngine(), [("arc-runner", "GPU", False)], 0, 1, None)
    assert {s.workload for s in llm_out} == {"llm"} and llm_out[0].completion_tokens == 20
    emb = seq.MODEL_WORKLOAD_SPECS["all-minilm-l6-v2"]
    emb_out = seq.collect_samples("all-minilm-l6-v2", emb, lambda d, w: _EmbedEngine(), [("arc-runner", "NPU", False)], 0, 1, None)
    assert {s.workload for s in emb_out} == {"embeddings"}


def test_vlm_collect_builds_one_image_per_tier_and_measures_both_backends() -> None:
    """Each tier gets a test image of its stated size; target and CPU are both measured."""
    vlm = _script("run_live_vlm_benchmarks")
    spec = vlm.VLM_MODEL_WORKLOAD_SPECS["qwen2-vl-7b"]
    sizes: List[int] = []

    def image_factory(side: int) -> bytes:
        sizes.append(side)
        return b"png"

    factory = _Recorder(_VlmEngine)
    samples = vlm.collect_samples("qwen2-vl-7b", spec, factory, "GPU", False, True, 0, 1, None, image_factory)
    assert factory.devices == ["GPU", "CPU"]
    assert sizes == [224, 448, 896] * 2
    assert {s.workload for s in samples} == {"vlm"}


def test_vlm_test_image_is_a_png_of_the_requested_size() -> None:
    """make_test_image returns decodable PNG bytes with the stated side length."""
    vlm = _script("run_live_vlm_benchmarks")
    data = vlm.make_test_image(64)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (64, 64)


def test_removed_fixture_script_and_legacy_result_files_stay_removed() -> None:
    """The fixed-timing audit script and the result files holding formula numbers must not return."""
    scratch = REPO_ROOT / "scratch"
    for name in ("run_full_catalog_benchmark_audit.py", "full_catalog_benchmark_results.json", "live_physical_video_benchmark_results.json", "live_physical_image_benchmark_results.json", "live_physical_audio_benchmark_results.json", "live_physical_benchmark_results.json"):
        assert not (scratch / name).exists(), name
