"""Claim gates for VLM and audio workloads, and the document-level claim check."""

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from arc_runner.benchmark.results import build_result, capture_environment, evaluate_document_claim
from arc_runner.benchmark.stats import BenchSample, evaluate_speedup_gate, evaluate_workload_gate

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = {"warmup_runs": 1, "timed_runs": 10, "labels": {"arc": "GPU", "cpu": "CPU"}}


def _rows(
    workload: str, backend: str, values: List[float], model_id: str = "m/low", chunks: Optional[float] = 12.0, key: str = "latency_s", **flags: Any,
) -> List[BenchSample]:
    """One warmup row plus one timed row per value, with the metric under `key`."""
    out: List[BenchSample] = []
    for i, value in enumerate([values[0], *values]):
        metrics: Dict[str, Optional[float]] = {key: value, "latency_s": value}
        if chunks is not None:
            metrics["output_chunk_count"] = chunks
        out.append(BenchSample(model_id, backend, i, None, None, 0, 0, "measured", is_warmup=i == 0, workload=workload, metrics=metrics, **flags))
    return out


FAST = [1.0 + 0.01 * i for i in range(10)]
SLOW = [6.0 + 0.05 * i for i in range(10)]


@pytest.mark.parametrize("workload", ["vlm", "audio_stt", "audio_tts"])
def test_workload_gate_passes_for_clearly_faster_measured_system(workload: str) -> None:
    """Fast vs slow, 10 timed rows each, CV well under 10 percent: the claim is allowed."""
    key = "rtf" if workload == "audio_tts" else "latency_s"
    result = evaluate_workload_gate(_rows(workload, "arc", FAST, key=key), _rows(workload, "cpu", SLOW, key=key), workload)
    assert result.passed and result.ratio_lo > 1.0 and result.n_arc == result.n_ref == 10


def test_workload_gate_rejects_fewer_than_ten_timed_rows() -> None:
    """Nine timed values are not enough, as for the LLM gate."""
    result = evaluate_workload_gate(_rows("audio_stt", "arc", FAST[:9]), _rows("audio_stt", "cpu", SLOW), "audio_stt")
    assert not result.passed and "need 10" in result.reason


def test_workload_gate_rejects_high_variance_in_either_system() -> None:
    """A coefficient of variation above 10 percent on the reference side blocks the claim."""
    noisy = [4.0, 9.0, 5.0, 12.0, 6.0, 3.0, 11.0, 7.0, 4.5, 10.0]
    result = evaluate_workload_gate(_rows("audio_stt", "arc", FAST), _rows("audio_stt", "cpu", noisy), "audio_stt")
    assert not result.passed and "coefficient of variation" in result.reason


@pytest.mark.parametrize("flag", [{"fallback_used": True}, {"noisy": True}])
def test_workload_gate_rejects_fallback_and_noisy_rows(flag: Dict[str, bool]) -> None:
    """Rows from a fallback device or flagged noisy never support a claim."""
    result = evaluate_workload_gate(_rows("audio_stt", "arc", FAST, **flag), _rows("audio_stt", "cpu", SLOW), "audio_stt")
    assert not result.passed


def test_workload_gate_rejects_equal_systems() -> None:
    """Identical latencies give a ratio interval spanning 1.0."""
    result = evaluate_workload_gate(_rows("audio_stt", "arc", FAST), _rows("audio_stt", "cpu", FAST), "audio_stt")
    assert not result.passed and "does not clear 1.0" in result.reason


def test_workload_gate_ratio_is_reference_over_arc_for_lower_is_better_latency() -> None:
    """A slower arc system gets a ratio below one and fails."""
    result = evaluate_workload_gate(_rows("vlm", "arc", SLOW), _rows("vlm", "cpu", FAST), "vlm")
    assert not result.passed and result.ratio_hi < 1.0


def test_workload_gate_vlm_requires_one_output_length_across_timed_rows() -> None:
    """Latency scales with output length, so differing chunk counts block the VLM claim."""
    arc = _rows("vlm", "arc", FAST, chunks=96.0)
    ref = _rows("vlm", "cpu", SLOW, chunks=118.0)
    result = evaluate_workload_gate(arc, ref, "vlm")
    assert not result.passed and "output_chunk_count" in result.reason


def test_workload_gate_vlm_rejects_rows_without_output_length() -> None:
    """Missing chunk counts mean the output length cannot be checked."""
    result = evaluate_workload_gate(_rows("vlm", "arc", FAST, chunks=None), _rows("vlm", "cpu", SLOW, chunks=None), "vlm")
    assert not result.passed and "no output_chunk_count" in result.reason


def test_workload_gate_audio_does_not_require_output_length() -> None:
    """The equal-output rule is VLM-specific."""
    assert evaluate_workload_gate(_rows("audio_stt", "arc", FAST, chunks=None), _rows("audio_stt", "cpu", SLOW, chunks=None), "audio_stt").passed


def test_workload_gate_rejects_rows_of_another_workload_or_mixed_tiers() -> None:
    """A claim covers one workload and one model/tier id."""
    assert not evaluate_workload_gate(_rows("vlm", "arc", FAST), _rows("vlm", "cpu", SLOW), "audio_stt").passed
    mixed = _rows("audio_stt", "arc", FAST[:5]) + _rows("audio_stt", "arc", FAST[5:], model_id="m/hard")
    result = evaluate_workload_gate(mixed, _rows("audio_stt", "cpu", SLOW), "audio_stt")
    assert not result.passed and "mix" in result.reason


@pytest.mark.parametrize("workload", ["embeddings", "image", "video", "llm", "unknown"])
def test_workload_gate_has_no_verdict_for_workloads_without_a_gate(workload: str) -> None:
    """Workloads without a gate are refused rather than passed."""
    result = evaluate_workload_gate(_rows(workload, "arc", FAST), _rows(workload, "cpu", SLOW), workload)
    assert not result.passed and "no claim gate" in result.reason


def test_workload_gate_rejects_a_higher_is_better_metric_name() -> None:
    """Only lower-is-better workload metrics are accepted, so the ratio direction is never wrong."""
    assert not evaluate_workload_gate(_rows("vlm", "arc", FAST), _rows("vlm", "cpu", SLOW), "vlm", metric="decode_tps").passed


def test_speedup_gate_reads_metrics_keys_so_latency_no_longer_raises() -> None:
    """Regression for the review finding: metric=latency_s used to raise AttributeError on workload samples."""
    result = evaluate_speedup_gate(_rows("vlm", "arc", FAST), _rows("vlm", "cpu", SLOW), metric="latency_s")
    assert result.passed


def test_speedup_gate_unknown_metric_reports_zero_timed_values() -> None:
    """An absent metric is a refusal with a reason, not an exception."""
    result = evaluate_speedup_gate(_rows("vlm", "arc", FAST), _rows("vlm", "cpu", SLOW), metric="no_such_metric")
    assert not result.passed and "0 timed values" in result.reason


def _document(workload: str, arc: List[float], ref: List[float], chunks: Optional[float] = 12.0) -> Dict[str, Any]:
    """Schema-v1 document with arc and cpu series of one workload."""
    samples = _rows(workload, "arc", arc, chunks=chunks) + _rows(workload, "cpu", ref, chunks=chunks)
    return build_result("t", samples, PROTOCOL, capture_environment(REPO_ROOT))


def test_document_claim_passes_for_vlm_and_audio_documents() -> None:
    """The document check routes each workload to its gate."""
    assert evaluate_document_claim(_document("vlm", FAST, SLOW), "arc", "cpu", "m/low").passed
    assert evaluate_document_claim(_document("audio_stt", FAST, SLOW, chunks=None), "arc", "cpu", "m/low").passed


def test_document_claim_refuses_workloads_without_gate_missing_rows_and_invalid_documents() -> None:
    """Embeddings have no gate; a missing backend and a broken document are refusals."""
    assert "no claim gate" in evaluate_document_claim(_document("embeddings", FAST, SLOW), "arc", "cpu", "m/low").reason
    assert "no rows" in evaluate_document_claim(_document("vlm", FAST, SLOW), "arc", "npu", "m/low").reason
    broken = copy.deepcopy(_document("vlm", FAST, SLOW))
    del broken["environment"]
    assert "not valid" in evaluate_document_claim(broken, "arc", "cpu", "m/low").reason


def test_document_claim_routes_llm_documents_to_the_llm_gate() -> None:
    """LLM rows are judged on decode_tps by the existing gate."""
    samples = [BenchSample("m/low", b, i, 10.0, tps, 1, 1, "measured", is_warmup=i == 0) for b, tps in (("arc", 50.0), ("cpu", 10.0)) for i in range(11)]
    samples = [BenchSample(s.model_id, s.backend, s.iteration, s.ttft_ms, s.decode_tps + 0.1 * s.iteration, 1, 1, "measured", is_warmup=s.is_warmup) for s in samples]
    document = build_result("t", samples, {**PROTOCOL, "temperature": 0.0, "max_new_tokens": 8}, capture_environment(REPO_ROOT))
    assert evaluate_document_claim(document, "arc", "cpu", "m/low").passed
