"""Shared helpers that build EngineTelemetry only from measured values.

Nothing here estimates tokens, latency splits, or speculative acceptance. When a value
cannot be measured the field stays zero and its validity flag stays False.
"""

from typing import Any, Optional

from arc_runner.core.logging import get_logger
from arc_runner.engine.models import EngineTelemetry

logger = get_logger("engine.telemetry")


def _optional_int(perf_metrics: Any, accessor: str) -> int:
    """Read an optional integer counter from perf_metrics, returning 0 if absent or failing.

    Args:
        perf_metrics: OpenVINO perf_metrics object.
        accessor: Method name such as get_num_input_tokens.

    Returns:
        int: Counter value or 0.
    """
    try:
        return int(getattr(perf_metrics, accessor)())
    except Exception:
        return 0


def _optional_mean(perf_metrics: Any, accessor: str) -> float:
    """Read the mean of an optional MeanStdPair accessor, returning 0.0 if absent or failing.

    Args:
        perf_metrics: OpenVINO perf_metrics object.
        accessor: Method name such as get_tpot.

    Returns:
        float: Mean value rounded to 2 decimals, or 0.0.
    """
    try:
        return round(float(getattr(perf_metrics, accessor)().mean), 2)
    except Exception:
        return 0.0


def _decode_rate(generated_tokens: int, generate_ms: float, ttft_ms: float) -> float:
    """Derive the decode-phase token rate from documented counters.

    OpenVINO GenAI 2026.4.1 computes get_throughput as 1000 / mean TPOT, and TPOT excludes the first
    (prefill) token, so get_throughput is itself decode-only (perf_metrics.cpp, PerfMetrics::evaluate_statistics).
    This rate is a second, coarser view over wall counters: (tokens - 1) divided by (generate duration - TTFT),
    which also includes sampling and detokenization time, so it is slightly lower.

    Args:
        generated_tokens: Number of generated tokens.
        generate_ms: Total generate duration in milliseconds.
        ttft_ms: Time to first token in milliseconds.

    Returns:
        float: Tokens per second, or 0.0 if fewer than two tokens or no decode time.
    """
    decode_ms = generate_ms - ttft_ms
    if generated_tokens < 2 or decode_ms <= 0.0:
        return 0.0
    return round((generated_tokens - 1) / (decode_ms / 1000.0), 2)


def telemetry_unavailable(model_id: str, device: str, elapsed_sec: float, **extra: Any) -> EngineTelemetry:
    """Build telemetry that reports only the measured wall-clock latency.

    Args:
        model_id: Identifier of the executed model.
        device: Silicon device used.
        elapsed_sec: Measured wall-clock duration of the call in seconds.
        **extra: Additional non-timing fields such as is_speculative.

    Returns:
        EngineTelemetry: Zeroed timing fields with validity flags False.
    """
    return EngineTelemetry(
        model_id=model_id,
        device=device,
        total_latency_ms=round(elapsed_sec * 1000.0, 3),
        metrics_source="unavailable",
        **extra,
    )


def telemetry_from_perf_metrics(
    perf_metrics: Any,
    model_id: str,
    device: str,
    elapsed_sec: float,
    **extra: Any,
) -> EngineTelemetry:
    """Build telemetry from OpenVINO perf_metrics, or report it unavailable if the counters fail.

    Args:
        perf_metrics: perf_metrics object attached to a generation result.
        model_id: Identifier of the executed model.
        device: Silicon device used.
        elapsed_sec: Measured wall-clock duration of the call in seconds.
        **extra: Additional non-timing fields such as is_speculative.

    Returns:
        EngineTelemetry: Engine-reported values, or zeros flagged invalid.
    """
    try:
        tps = float(perf_metrics.get_throughput().mean)
        ttft_ms = float(perf_metrics.get_ttft().mean)
        gen_tokens = int(perf_metrics.get_num_generated_tokens())
        gen_dur_ms = float(perf_metrics.get_generate_duration().mean)
    except Exception as exc:
        logger.warning("perf_metrics unavailable, reporting no timing values: %s", exc)
        return telemetry_unavailable(model_id, device, elapsed_sec, **extra)

    return EngineTelemetry(
        model_id=model_id,
        device=device,
        ttft_ms=round(ttft_ms, 2),
        tokens_per_second=round(tps, 2),
        total_tokens=gen_tokens,
        total_latency_ms=round(gen_dur_ms, 2),
        metrics_source="ov_perf_metrics",
        is_ttft_valid=True,
        is_tps_valid=True,
        prompt_tokens=_optional_int(perf_metrics, "get_num_input_tokens"),
        tpot_ms=_optional_mean(perf_metrics, "get_tpot"),
        decode_tokens_per_second=_decode_rate(gen_tokens, gen_dur_ms, ttft_ms),
        **extra,
    )


def telemetry_from_stream(
    model_id: str,
    device: str,
    start_time: float,
    end_time: float,
    first_token_time: Optional[float],
    chunk_count: int,
    token_count: Optional[int] = None,
    is_estimate: bool = True,
    **extra: Any,
) -> EngineTelemetry:
    """Build wall-clock telemetry for a streamed generation.

    Args:
        model_id: Identifier of the executed model.
        device: Silicon device used.
        start_time: time.monotonic() value when generation started.
        end_time: time.monotonic() value when generation ended.
        first_token_time: time.monotonic() value of the first emitted chunk, or None.
        chunk_count: Number of streamer callbacks observed (chunks, not tokenizer tokens).
        token_count: Token count of the emitted text; replaces chunk_count and
            makes the rate (tokens - 1) / time after the first chunk.
        is_estimate: False when token_count came from perf_metrics (measured); True
            when derived from re-tokenization or chunk count (approximated).
        **extra: Additional non-timing fields such as is_speculative.

    Returns:
        EngineTelemetry: Stream timings, or unavailable if no chunk arrived.
    """
    total_sec = max(end_time - start_time, 1e-6)
    if first_token_time is None:
        return telemetry_unavailable(model_id, device, total_sec, **extra)

    decode_sec = max(end_time - first_token_time, 1e-6)
    if token_count is not None:
        tokens, rate = token_count, round(max(token_count - 1, 0) / decode_sec, 2)
    else:
        tokens, rate = chunk_count, round(chunk_count / decode_sec, 2)
    return EngineTelemetry(
        model_id=model_id,
        device=device,
        ttft_ms=round((first_token_time - start_time) * 1000.0, 2),
        tokens_per_second=rate,
        total_tokens=tokens,
        total_latency_ms=round(total_sec * 1000.0, 2),
        metrics_source="wallclock_stream",
        is_estimate=is_estimate,
        is_ttft_valid=True,
        is_tps_valid=tokens > 1,
        **extra,
    )
