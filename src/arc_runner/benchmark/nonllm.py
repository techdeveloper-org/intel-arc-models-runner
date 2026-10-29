"""Schema-v1 sample runners for embeddings, VLM, audio, image and video engines.

Each runner times real engine calls with benchmark.workloads.measure_workload. Rates and real-time
factors are derived from this process's wall clock and known workload constants; figures an engine
reports about itself are used only where the engine measured them (stage timings) and are None when
the engine did not. A failing engine call raises; there is no simulated fallback.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence

from arc_runner.benchmark.stats import BenchSample
from arc_runner.benchmark.workloads import TimedRun, measure_workload, workload_samples

Extra = Callable[[TimedRun], Dict[str, Optional[float]]]


def _run(
    workload: str,
    model_id: str,
    backend: str,
    call: Callable[[], Any],
    derive: Callable[[TimedRun, Any], Dict[str, Optional[float]]],
    warmups: int,
    runs: int,
    sysman: Any,
    device_index: int,
    clock: Optional[Callable[[], float]],
    fallback_used: bool,
) -> List[BenchSample]:
    """Time a workload and derive per-run metrics from each run's own result.

    Args:
        workload: Schema workload name.
        model_id: Model and workload-tier identifier.
        backend: Backend name.
        call: Function performing one run and returning that run's engine result.
        derive: Maps (timed run, engine result) to extra metrics.
        warmups: Discarded warmup runs.
        runs: Timed runs.
        sysman: Sysman for package power, or None.
        device_index: GPU index for power.
        clock: Optional clock override (tests).
        fallback_used: Whether the engine ran on a fallback device.

    Returns:
        List[BenchSample]: Samples, warmups first.
    """
    results: List[Any] = []

    def once() -> None:
        results.append(call())

    kwargs: Dict[str, Any] = {} if clock is None else {"clock": clock}
    timed = measure_workload(once, warmups, runs, sysman=sysman, device_index=device_index, **kwargs)
    return workload_samples(
        model_id, backend, workload, timed,
        extra=lambda run: derive(run, results[run.iteration]), fallback_used=fallback_used,
    )


def stt_samples(
    engine: Any, audio: Any, audio_duration_s: float, model_id: str, backend: str, warmups: int, runs: int,
    sysman: Any = None, device_index: int = 0, clock: Optional[Callable[[], float]] = None, fallback_used: bool = False,
) -> List[BenchSample]:
    """Measure speech recognition: rtf = latency_s / audio_duration_s.

    Args:
        engine: Loaded AudioEngine (transcribe).
        audio: Audio file path or samples passed to transcribe.
        audio_duration_s: Known clip length in seconds.
        model_id: Model and tier identifier.
        backend: Backend name.
        warmups: Warmup runs.
        runs: Timed runs.
        sysman: Optional Sysman for package power.
        device_index: GPU index for power.
        clock: Optional clock override.
        fallback_used: True if the engine runs on a fallback device.

    Returns:
        List[BenchSample]: audio_stt samples.
    """
    def derive(run: TimedRun, _result: Any) -> Dict[str, Optional[float]]:
        return {
            "audio_duration_s": audio_duration_s,
            "rtf": run.elapsed_s / audio_duration_s,
            "throughput_audio_s_per_s": audio_duration_s / run.elapsed_s if run.elapsed_s > 0 else None,
        }

    return _run("audio_stt", model_id, backend, lambda: engine.transcribe(audio), derive, warmups, runs, sysman, device_index, clock, fallback_used)


def tts_samples(
    engine: Any, text: str, model_id: str, backend: str, warmups: int, runs: int,
    sysman: Any = None, device_index: int = 0, clock: Optional[Callable[[], float]] = None, fallback_used: bool = False,
) -> List[BenchSample]:
    """Measure speech synthesis: rtf = latency_s / duration_s of the audio that run produced.

    Args:
        engine: Loaded AudioEngine (synthesize).
        text: Text to synthesize.
        model_id: Model and tier identifier.
        backend: Backend name.
        warmups: Warmup runs.
        runs: Timed runs.
        sysman: Optional Sysman for package power.
        device_index: GPU index for power.
        clock: Optional clock override.
        fallback_used: True if the engine runs on a fallback device.

    Returns:
        List[BenchSample]: audio_tts samples.
    """
    def derive(run: TimedRun, result: Any) -> Dict[str, Optional[float]]:
        duration = float(result.duration_s)
        return {"audio_duration_s": duration, "rtf": run.elapsed_s / duration if duration > 0 else None}

    return _run("audio_tts", model_id, backend, lambda: engine.synthesize(text), derive, warmups, runs, sysman, device_index, clock, fallback_used)


def image_samples(
    engine: Any, request: Any, model_id: str, backend: str, warmups: int, runs: int,
    sysman: Any = None, device_index: int = 0, clock: Optional[Callable[[], float]] = None, fallback_used: bool = False,
) -> List[BenchSample]:
    """Measure text-to-image: megapixels_per_s = width*height/1e6 over measured latency.

    Stage timings (step_latency_ms, iteration_rate_it_s) pass through from the engine and are None
    when OpenVINO reported no stage metrics.

    Args:
        engine: Loaded ImageEngine (generate).
        request: ImageGenerationRequest.
        model_id: Model and tier identifier.
        backend: Backend name.
        warmups: Warmup runs.
        runs: Timed runs.
        sysman: Optional Sysman for package power.
        device_index: GPU index for power.
        clock: Optional clock override.
        fallback_used: True if the engine runs on a fallback device.

    Returns:
        List[BenchSample]: image samples.
    """
    def derive(run: TimedRun, result: Any) -> Dict[str, Optional[float]]:
        megapixels = result.width * result.height / 1_000_000.0
        return {
            "megapixels_per_s": megapixels / run.elapsed_s if run.elapsed_s > 0 else None,
            "step_latency_ms": result.breakdown.step_latency_ms,
            "iteration_rate_it_s": result.breakdown.iteration_rate_it_s,
        }

    return _run("image", model_id, backend, lambda: engine.generate(request), derive, warmups, runs, sysman, device_index, clock, fallback_used)


def video_samples(
    engine: Any, request: Any, model_id: str, backend: str, warmups: int, runs: int,
    sysman: Any = None, device_index: int = 0, clock: Optional[Callable[[], float]] = None, fallback_used: bool = False,
) -> List[BenchSample]:
    """Measure text-to-video: frames_per_s = num_frames / latency_s and rtf = latency_s / video_duration_s.

    Args:
        engine: Loaded VideoEngine (generate).
        request: VideoGenerationRequest (num_frames, fps).
        model_id: Model and tier identifier.
        backend: Backend name.
        warmups: Warmup runs.
        runs: Timed runs.
        sysman: Optional Sysman for package power.
        device_index: GPU index for power.
        clock: Optional clock override.
        fallback_used: True if the engine runs on a fallback device.

    Returns:
        List[BenchSample]: video samples.
    """
    def derive(run: TimedRun, result: Any) -> Dict[str, Optional[float]]:
        duration = float(result.duration_sec)
        return {
            "video_duration_s": duration,
            "frames_per_s": result.num_frames / run.elapsed_s if run.elapsed_s > 0 else None,
            "rtf": run.elapsed_s / duration if duration > 0 else None,
            "step_latency_ms": result.breakdown.step_latency_ms,
        }

    return _run("video", model_id, backend, lambda: engine.generate(request), derive, warmups, runs, sysman, device_index, clock, fallback_used)


def embedding_samples(
    engine: Any, texts: Sequence[str], model_id: str, backend: str, warmups: int, runs: int,
    sysman: Any = None, device_index: int = 0, clock: Optional[Callable[[], float]] = None, fallback_used: bool = False,
) -> List[BenchSample]:
    """Measure embeddings: texts_per_s = len(texts) / latency_s for one batch call.

    Args:
        engine: Loaded EmbeddingEngine (embed_documents).
        texts: Batch of texts embedded per run.
        model_id: Model and tier identifier.
        backend: Backend name.
        warmups: Warmup runs.
        runs: Timed runs.
        sysman: Optional Sysman for package power.
        device_index: GPU index for power.
        clock: Optional clock override.
        fallback_used: True if the engine runs on a fallback device.

    Returns:
        List[BenchSample]: embeddings samples.
    """
    batch = list(texts)

    def derive(run: TimedRun, _result: Any) -> Dict[str, Optional[float]]:
        return {"batch_size": float(len(batch)), "texts_per_s": len(batch) / run.elapsed_s if run.elapsed_s > 0 else None}

    return _run("embeddings", model_id, backend, lambda: engine.embed_documents(batch), derive, warmups, runs, sysman, device_index, clock, fallback_used)


def vlm_samples(
    engine: Any, prompt: str, images: Sequence[Any], max_new_tokens: int, model_id: str, backend: str, warmups: int, runs: int,
    sysman: Any = None, device_index: int = 0, clock: Optional[Callable[[], float]] = None, fallback_used: bool = False,
) -> List[BenchSample]:
    """Measure a vision-language request end to end; the chunk count is streamer callbacks, not tokens.

    VisionEngine yields only after generation completes, so time to first token is not observable and
    is not reported.

    Args:
        engine: Loaded VisionEngine (generate).
        prompt: Text instruction.
        images: Images passed to the engine.
        max_new_tokens: Generation budget.
        model_id: Model and tier identifier.
        backend: Backend name.
        warmups: Warmup runs.
        runs: Timed runs.
        sysman: Optional Sysman for package power.
        device_index: GPU index for power.
        clock: Optional clock override.
        fallback_used: True if the engine runs on a fallback device.

    Returns:
        List[BenchSample]: vlm samples.
    """
    from arc_runner.engine.models import GenerationConfig

    config = GenerationConfig(max_new_tokens=max_new_tokens, temperature=0.0)

    def call() -> List[Any]:
        return list(engine.generate(prompt, images, config))

    def derive(_run_: TimedRun, chunks: Any) -> Dict[str, Optional[float]]:
        return {"output_chunk_count": float(len(chunks))}

    return _run("vlm", model_id, backend, call, derive, warmups, runs, sysman, device_index, clock, fallback_used)


def llm_samples(
    engine: Any, prompt: str, max_new_tokens: int, model_id: str, backend: str, warmups: int, runs: int,
    sysman: Any = None, device_index: int = 0, clock: Optional[Callable[[], float]] = None, fallback_used: bool = False,
) -> List[BenchSample]:
    """Measure an LLM from the engine's own telemetry (never from splitting the output into words).

    TTFT, decode rate and token counts come from OpenVINO perf metrics (source measured) or from
    stream callbacks (source derived); with no metrics they are None and zero.

    Args:
        engine: Loaded OpenVINOGenAIEngine (generate, get_telemetry).
        prompt: Prompt text.
        max_new_tokens: Completion budget.
        model_id: Model and tier identifier.
        backend: Backend name.
        warmups: Warmup runs.
        runs: Timed runs.
        sysman: Optional Sysman for package power.
        device_index: GPU index for power.
        clock: Optional clock override.
        fallback_used: True if the engine runs on a fallback device.

    Returns:
        List[BenchSample]: llm samples with ttft_ms, decode_tps and token counts filled when measured.
    """
    from arc_runner.engine.models import GenerationConfig

    config = GenerationConfig(max_new_tokens=max_new_tokens, temperature=0.0)
    telemetry: List[Any] = []

    def once() -> None:
        engine.generate(prompt, config)
        telemetry.append(engine.get_telemetry())

    kwargs: Dict[str, Any] = {} if clock is None else {"clock": clock}
    timed = measure_workload(once, warmups, runs, sysman=sysman, device_index=device_index, **kwargs)
    base = workload_samples(model_id, backend, "llm", timed, fallback_used=fallback_used)
    samples: List[BenchSample] = []
    for run, sample in zip(timed, base):
        t = telemetry[run.iteration]
        source = "measured" if t.metrics_source in ("ov_perf_metrics", "unavailable") else "derived"
        decode = t.decode_tokens_per_second if t.is_tps_valid and t.decode_tokens_per_second > 0 else (t.tokens_per_second if t.is_tps_valid else None)
        samples.append(BenchSample(
            model_id, backend, run.iteration, t.ttft_ms if t.is_ttft_valid else None, decode,
            int(t.prompt_tokens), int(t.total_tokens), source,  # type: ignore[arg-type]
            is_warmup=run.is_warmup, fallback_used=fallback_used, workload="llm", metrics=sample.metrics,
        ))
    return samples
