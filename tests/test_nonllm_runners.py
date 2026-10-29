"""Runners that turn real engine calls into schema-v1 samples for embeddings, VLM, audio, image and video."""

from types import SimpleNamespace
from typing import Any, List

import pytest

from arc_runner.benchmark.nonllm import (
    embedding_samples,
    image_samples,
    stt_samples,
    tts_samples,
    video_samples,
    vlm_samples,
)
from arc_runner.benchmark.results import validate_result, build_result
from arc_runner.benchmark.workloads import measure_workload  # noqa: F401


def _clock() -> Any:
    ticks = iter([t for i in range(100) for t in (i * 10.0, i * 10.0 + 2.0)])
    return lambda: next(ticks)


class _SttEngine:
    def __init__(self) -> None:
        self.calls: List[Any] = []

    def transcribe(self, audio: Any, **_kw: Any) -> Any:
        self.calls.append(audio)
        return SimpleNamespace(text="hello", rtf=999.0, latency_s=999.0)


class _TtsEngine:
    def synthesize(self, text: str, **_kw: Any) -> Any:
        return SimpleNamespace(duration_s=4.0, audio_samples=[0.0] * 10)


class _ImageEngine:
    def __init__(self, with_stage: bool = True) -> None:
        self.with_stage = with_stage

    def generate(self, request: Any) -> Any:
        breakdown = SimpleNamespace(
            step_latency_ms=100.0 if self.with_stage else None,
            iteration_rate_it_s=10.0 if self.with_stage else None,
            text_encoder_ms=None, denoise_unet_ms=None, vae_decoder_ms=None, megapixels_per_sec=999.0,
        )
        return SimpleNamespace(breakdown=breakdown, width=request.width, height=request.height, total_latency_s=999.0)


class _VideoEngine:
    def generate(self, request: Any) -> Any:
        breakdown = SimpleNamespace(step_latency_ms=None, iteration_rate_it_s=None, frames_per_second=999.0, real_time_factor=999.0)
        return SimpleNamespace(breakdown=breakdown, num_frames=request.num_frames, duration_sec=request.num_frames / request.fps, total_latency_ms=999000.0)


class _EmbedEngine:
    def embed_documents(self, texts: Any) -> Any:
        return [[0.0, 1.0] for _ in texts]


class _VlmEngine:
    def generate(self, prompt: str, images: Any, config: Any = None) -> Any:
        yield SimpleNamespace(text="a", is_finished=False)
        yield SimpleNamespace(text="b", is_finished=True)


def test_stt_samples_use_our_wall_clock_not_engine_reported_rtf() -> None:
    """RTF is our measured latency over the known clip length; the engine's own figure is ignored."""
    engine = _SttEngine()
    samples = stt_samples(engine, "clip.wav", audio_duration_s=4.0, model_id="whisper/low", backend="arc", warmups=1, runs=2, clock=_clock())
    assert len(engine.calls) == 3
    assert [s.is_warmup for s in samples] == [True, False, False]
    assert samples[1].metrics["latency_s"] == pytest.approx(2.0)
    assert samples[1].metrics["rtf"] == pytest.approx(0.5)
    assert samples[1].metrics["throughput_audio_s_per_s"] == pytest.approx(2.0)
    assert samples[1].workload == "audio_stt"


def test_tts_samples_rtf_uses_synthesized_duration_of_each_run() -> None:
    """RTF = latency / the duration the run actually produced."""
    samples = tts_samples(_TtsEngine(), "hello world", model_id="kokoro/low", backend="arc", warmups=0, runs=2, clock=_clock())
    assert samples[0].metrics["audio_duration_s"] == pytest.approx(4.0)
    assert samples[0].metrics["rtf"] == pytest.approx(0.5)
    assert samples[0].workload == "audio_tts"


def test_image_samples_derive_megapixel_rate_from_measured_latency() -> None:
    """MP/s = width*height/1e6 over our wall clock; stage-derived values pass through, None stays None."""
    request = SimpleNamespace(width=1000, height=1000)
    full = image_samples(_ImageEngine(True), request, model_id="sd/low", backend="arc", warmups=0, runs=1, clock=_clock())[0]
    assert full.metrics["megapixels_per_s"] == pytest.approx(0.5)
    assert full.metrics["step_latency_ms"] == pytest.approx(100.0)
    bare = image_samples(_ImageEngine(False), request, model_id="sd/low", backend="arc", warmups=0, runs=1, clock=_clock())[0]
    assert bare.metrics["step_latency_ms"] is None and bare.metrics["iteration_rate_it_s"] is None


def test_video_samples_use_wall_clock_for_fps_and_rtf_not_engine_values() -> None:
    """frames/s and RTF come from our clock and the clip's frame count and duration."""
    request = SimpleNamespace(num_frames=16, fps=8)
    sample = video_samples(_VideoEngine(), request, model_id="ltx/low", backend="arc", warmups=0, runs=1, clock=_clock())[0]
    assert sample.metrics["frames_per_s"] == pytest.approx(8.0)
    assert sample.metrics["rtf"] == pytest.approx(1.0)
    assert sample.metrics["video_duration_s"] == pytest.approx(2.0)
    assert sample.workload == "video"


def test_embedding_samples_report_texts_per_second() -> None:
    """Throughput is texts per measured second."""
    sample = embedding_samples(_EmbedEngine(), ["a", "b", "c", "d"], model_id="minilm/low", backend="npu", warmups=0, runs=1, clock=_clock())[0]
    assert sample.metrics["texts_per_s"] == pytest.approx(2.0)
    assert sample.workload == "embeddings"


def test_vlm_samples_consume_the_stream_and_count_chunks() -> None:
    """The generator is fully consumed inside the timed region; chunk count is reported as such."""
    sample = vlm_samples(_VlmEngine(), "describe", ["img.png"], max_new_tokens=8, model_id="qwen-vl/low", backend="arc", warmups=0, runs=1, clock=_clock())[0]
    assert sample.metrics["latency_s"] == pytest.approx(2.0)
    assert sample.metrics["output_chunk_count"] == 2.0
    assert sample.workload == "vlm"


def test_engine_exception_propagates_from_every_runner() -> None:
    """A failing engine is an error, never a skipped or synthetic sample."""

    class _Broken:
        def transcribe(self, *_a: Any, **_k: Any) -> Any:
            raise RuntimeError("device lost")

    with pytest.raises(RuntimeError):
        stt_samples(_Broken(), "x.wav", 1.0, "m", "arc", warmups=0, runs=1)


def test_runner_output_forms_a_valid_schema_document() -> None:
    """All workloads' samples validate together in one document."""
    request_i = SimpleNamespace(width=64, height=64)
    samples = (
        stt_samples(_SttEngine(), "c.wav", 2.0, "w", "arc", warmups=0, runs=1, clock=_clock())
        + image_samples(_ImageEngine(), request_i, "sd", "arc", warmups=0, runs=1, clock=_clock())
        + embedding_samples(_EmbedEngine(), ["a"], "e", "npu", warmups=0, runs=1, clock=_clock())
    )
    document = build_result("r", samples, {"warmup_runs": 0, "timed_runs": 1, "labels": {"arc": "x", "npu": "y"}}, {"os": "o", "python": "p", "git_sha": "g", "hardware": "h"})
    assert validate_result(document) == []


class _LlmEngine:
    """Engine whose get_telemetry reflects the last generate() call."""

    def __init__(self, source: str = "ov_perf_metrics") -> None:
        self.source = source
        self.configs: List[Any] = []

    def generate(self, prompt: str, config: Any = None) -> str:
        self.configs.append(config)
        return "one two three"

    def get_telemetry(self) -> Any:
        valid = self.source != "unavailable"
        return SimpleNamespace(
            ttft_ms=120.0 if valid else 0.0, decode_tokens_per_second=33.0 if valid else 0.0, tokens_per_second=40.0 if valid else 0.0,
            prompt_tokens=11, total_tokens=20 if valid else 0, metrics_source=self.source, is_ttft_valid=valid, is_tps_valid=valid,
        )


def test_llm_samples_use_engine_telemetry_not_word_counts() -> None:
    """Token counts and rates come from perf metrics; the generated text is never split into words."""
    from arc_runner.benchmark.nonllm import llm_samples

    engine = _LlmEngine()
    samples = llm_samples(engine, "prompt", 64, "qwen/low", "arc", warmups=0, runs=2, clock=_clock())
    sample = samples[0]
    assert sample.workload == "llm" and sample.source == "measured"
    assert (sample.ttft_ms, sample.decode_tps, sample.prompt_tokens, sample.completion_tokens) == (120.0, 33.0, 11, 20)
    assert engine.configs[0].max_new_tokens == 64 and engine.configs[0].temperature == 0.0
    assert sample.metrics["latency_s"] == pytest.approx(2.0)


def test_llm_samples_wallclock_stream_values_are_derived_not_measured() -> None:
    """Stream-callback estimates are labelled derived so claim gates refuse them."""
    from arc_runner.benchmark.nonllm import llm_samples

    sample = llm_samples(_LlmEngine("wallclock_stream"), "p", 8, "m", "arc", warmups=0, runs=1, clock=_clock())[0]
    assert sample.source == "derived"


def test_llm_samples_unavailable_metrics_leave_values_none() -> None:
    """No perf metrics: ttft and decode rate are None, never zero or estimated."""
    from arc_runner.benchmark.nonllm import llm_samples

    sample = llm_samples(_LlmEngine("unavailable"), "p", 8, "m", "arc", warmups=0, runs=1, clock=_clock())[0]
    assert sample.ttft_ms is None and sample.decode_tps is None and sample.completion_tokens == 0


def test_llm_samples_converged_warmup_runs_extra_warmups_until_ttft_is_steady() -> None:
    """max_warmups makes the LLM runner keep warming while the TTFT of the last three runs still moves."""
    from arc_runner.benchmark.nonllm import llm_samples

    class Engine(_LlmEngine):
        ttfts = iter([900.0, 400.0, 200.0, 100.0, 100.0, 100.0] + [100.0] * 5)

        def get_telemetry(self) -> Any:
            t = super().get_telemetry()
            t.ttft_ms = next(self.ttfts)
            return t

    samples = llm_samples(Engine(), "p", 8, "m", "arc", warmups=3, runs=2, clock=_clock(), max_warmups=5)
    assert sum(s.is_warmup for s in samples) == 5 and len(samples) == 7


def test_llm_samples_interleaved_alternates_engines_per_iteration_and_labels_each_backend() -> None:
    """Both engines run once per iteration in the seeded order; each backend keeps its own telemetry."""
    from arc_runner.benchmark.nonllm import llm_samples_interleaved
    from arc_runner.benchmark.workloads import interleaved_order

    trace: List[str] = []

    class Engine(_LlmEngine):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        def generate(self, prompt: str, config: Any = None) -> str:
            trace.append(self.name)
            return super().generate(prompt, config)

    systems = [("gpu", Engine("gpu"), False), ("cpu", Engine("cpu"), False)]
    samples = llm_samples_interleaved(systems, "p", 8, "m/low", warmups=1, runs=2, seed=5, clock=_clock())
    expected: List[str] = []
    for iteration in range(3):
        expected += interleaved_order(["gpu", "cpu"], iteration, 5)
    assert trace == expected
    assert [s.backend for s in samples] == ["gpu"] * 3 + ["cpu"] * 3
    assert [s.is_warmup for s in samples[:3]] == [True, False, False]


def test_vlm_samples_interleaved_records_chunk_counts_per_backend() -> None:
    """Each backend's rows carry that backend's own chunk count."""
    from arc_runner.benchmark.nonllm import vlm_samples_interleaved

    class Short(_VlmEngine):
        pass

    class Long(_VlmEngine):
        def generate(self, prompt: str, images: Any, config: Any = None) -> Any:
            yield from super().generate(prompt, images, config)
            yield SimpleNamespace(text="c", is_finished=True)

    samples = vlm_samples_interleaved([("a", Short(), False), ("b", Long(), True)], "p", [], 8, "v/low", warmups=0, runs=2, clock=_clock())
    by = {b: [s.metrics["output_chunk_count"] for s in samples if s.backend == b] for b in ("a", "b")}
    assert by == {"a": [2.0, 2.0], "b": [3.0, 3.0]}
    assert all(s.fallback_used for s in samples if s.backend == "b") and not any(s.fallback_used for s in samples if s.backend == "a")
