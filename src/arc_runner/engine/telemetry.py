"""Shared helpers that build EngineTelemetry only from measured values.

Nothing here estimates tokens, latency splits, or speculative acceptance. When a value
cannot be measured the field stays zero and its validity flag stays False.
"""

import time
from typing import Any, Optional

from arc_runner.core.logging import get_logger
from arc_runner.engine.models import EngineTelemetry

logger = get_logger("engine.telemetry")


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
        **extra,
    )


def telemetry_from_stream(
    model_id: str,
    device: str,
    start_time: float,
    end_time: float,
    first_token_time: Optional[float],
    chunk_count: int,
    **extra: Any,
) -> EngineTelemetry:
    """Build wall-clock telemetry for a streamed generation, labelled as an estimate.

    Args:
        model_id: Identifier of the executed model.
        device: Silicon device used.
        start_time: time.monotonic() value when generation started.
        end_time: time.monotonic() value when generation ended.
        first_token_time: time.monotonic() value of the first emitted chunk, or None.
        chunk_count: Number of streamer callbacks observed (chunks, not tokenizer tokens).
        **extra: Additional non-timing fields such as is_speculative.

    Returns:
        EngineTelemetry: Stream timings flagged as estimates, or unavailable if no chunk arrived.
    """
    total_sec = max(end_time - start_time, 1e-6)
    if first_token_time is None:
        return telemetry_unavailable(model_id, device, total_sec, **extra)

    decode_sec = max(end_time - first_token_time, 1e-6)
    return EngineTelemetry(
        model_id=model_id,
        device=device,
        ttft_ms=round((first_token_time - start_time) * 1000.0, 2),
        tokens_per_second=round(chunk_count / decode_sec, 2),
        total_tokens=chunk_count,
        total_latency_ms=round(total_sec * 1000.0, 2),
        metrics_source="wallclock_stream",
        is_estimate=True,
        is_ttft_valid=True,
        is_tps_valid=chunk_count > 1,
        **extra,
    )
