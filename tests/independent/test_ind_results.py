"""Independent schema-v1 mutation tests for arc_runner.benchmark.results.validate_result.

Specification: results.py docstrings and docs/benchmarks/METHOD.md ("Result files follow schema v1";
every row carries a source of measured, derived or projected; metrics that could not be measured are
null; no number is invented). validate_result must RETURN a non-empty problem list for an invalid
document and must never raise on any JSON-shaped input.
"""

import copy
import json
import math
from pathlib import Path

import pytest

from arc_runner.benchmark.results import (
    SCHEMA_VERSION, build_result, load_samples, validate_result, write_result,
)
from arc_runner.benchmark.stats import BenchSample

TOP_KEYS = ("schema_version", "run_id", "created_utc", "environment", "protocol", "samples")
ENV_KEYS = ("os", "python", "git_sha", "hardware")
PROTO_KEYS = ("warmup_runs", "timed_runs", "labels")
PROTO_LLM_KEYS = ("temperature", "max_new_tokens")
SAMPLE_KEYS = ("model_id", "backend", "iteration", "ttft_ms", "decode_tps", "prompt_tokens",
               "completion_tokens", "source", "is_warmup", "fallback_used")
WORKLOADS = ("llm", "embeddings", "vlm", "audio_stt", "audio_tts", "image", "video")


def _sample(i=0):
    return {"model_id": "m", "backend": "gpu", "iteration": i, "ttft_ms": 12.5, "decode_tps": 30.25,
            "prompt_tokens": 8, "completion_tokens": 16, "source": "measured", "is_warmup": False,
            "fallback_used": False}


def _doc():
    return {
        "schema_version": 1, "run_id": "r1", "created_utc": "2026-01-01T00:00:00+00:00",
        "environment": {"os": "x", "python": "3.13", "git_sha": "abc", "hardware": "h"},
        "protocol": {"warmup_runs": 1, "timed_runs": 2, "labels": {"gpu": "Arc iGPU int4"}, "temperature": 0.0, "max_new_tokens": 8},
        "samples": [_sample(0), _sample(1)],
    }


def _validate_never_raises(doc):
    try:
        return validate_result(doc)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"validate_result raised {type(exc).__name__}: {exc} for {json.dumps(doc, default=str)[:200]}")


def test_valid_document_has_no_problems_and_survives_json_round_trip():
    doc = _doc()
    assert validate_result(doc) == []
    assert validate_result(json.loads(json.dumps(doc))) == []


@pytest.mark.parametrize("bad", [None, [], "doc", 1, 1.5, True, [_doc()]])
def test_non_object_top_level_is_rejected(bad):
    assert _validate_never_raises(bad)


@pytest.mark.parametrize("key", TOP_KEYS)
def test_deleting_each_top_level_key_is_rejected(key):
    doc = _doc()
    del doc[key]
    assert _validate_never_raises(doc)


@pytest.mark.parametrize("key", ENV_KEYS)
def test_deleting_each_environment_key_is_rejected(key):
    doc = _doc()
    del doc["environment"][key]
    assert _validate_never_raises(doc)


@pytest.mark.parametrize("key", PROTO_KEYS + PROTO_LLM_KEYS)
def test_deleting_each_protocol_key_is_rejected(key):
    doc = _doc()
    del doc["protocol"][key]
    assert _validate_never_raises(doc)


@pytest.mark.parametrize("key", SAMPLE_KEYS)
def test_deleting_each_sample_key_is_rejected(key):
    doc = _doc()
    del doc["samples"][1][key]
    assert _validate_never_raises(doc)


@pytest.mark.parametrize("bad", [0, 2, -1, 1.5, "1", None, [], {}])
def test_wrong_schema_version_is_rejected(bad):
    doc = _doc()
    doc["schema_version"] = bad
    assert _validate_never_raises(doc)


def test_schema_version_true_is_not_the_integer_one():
    doc = _doc()
    doc["schema_version"] = True
    assert _validate_never_raises(doc), "bool True must not be accepted as schema_version 1"


@pytest.mark.parametrize("field", ["environment", "protocol"])
@pytest.mark.parametrize("bad", [None, [], [1, 2], "str", 5, 1.5, True])
def test_non_object_environment_or_protocol_is_rejected_without_raising(field, bad):
    doc = _doc()
    doc[field] = bad
    assert _validate_never_raises(doc)


@pytest.mark.parametrize("bad", [None, {}, "x", 5, True])
def test_non_list_samples_is_rejected(bad):
    doc = _doc()
    doc["samples"] = bad
    assert _validate_never_raises(doc)


@pytest.mark.parametrize("field", ["run_id", "created_utc"])
@pytest.mark.parametrize("bad", [None, 5, [], {}, True])
def test_run_id_and_created_utc_must_be_strings(field, bad):
    doc = _doc()
    doc[field] = bad
    assert _validate_never_raises(doc), f"{field}={bad!r} must be rejected (schema says string)"


@pytest.mark.parametrize("row", [None, 5, "s", [], [1]])
def test_non_object_sample_row_is_rejected(row):
    doc = _doc()
    doc["samples"][0] = row
    assert _validate_never_raises(doc)


NUMBER_BAD = [-1, -0.001, "1", [], {}, True, False, float("nan"), float("inf"), float("-inf")]
INT_BAD = [-1, 1.5, 2.0, "1", None, [], {}, True, False, float("nan")]
BOOL_BAD = [0, 1, "true", None, [], {}, 1.0]
SOURCE_BAD = ["Measured", "estimated", "", None, 1, [], "measured ", "MEASURED"]


@pytest.mark.parametrize("field", ["ttft_ms", "decode_tps"])
@pytest.mark.parametrize("bad", NUMBER_BAD)
def test_latency_and_rate_fields_reject_non_numbers_negative_and_non_finite(field, bad):
    doc = _doc()
    doc["samples"][0][field] = bad
    assert _validate_never_raises(doc), f"{field}={bad!r} must be rejected"


@pytest.mark.parametrize("field", ["ttft_ms", "decode_tps"])
@pytest.mark.parametrize("ok", [None, 0, 0.0, 3, 12.5, 1e12])
def test_latency_and_rate_fields_accept_null_zero_and_positive_numbers(field, ok):
    doc = _doc()
    doc["samples"][0][field] = ok
    assert validate_result(doc) == []


@pytest.mark.parametrize("field", ["iteration", "prompt_tokens", "completion_tokens"])
@pytest.mark.parametrize("bad", INT_BAD)
def test_count_fields_must_be_non_negative_integers(field, bad):
    doc = _doc()
    doc["samples"][0][field] = bad
    assert _validate_never_raises(doc), f"{field}={bad!r} must be rejected"


@pytest.mark.parametrize("field", ["is_warmup", "fallback_used"])
@pytest.mark.parametrize("bad", BOOL_BAD)
def test_flag_fields_must_be_booleans(field, bad):
    doc = _doc()
    doc["samples"][0][field] = bad
    assert _validate_never_raises(doc)


@pytest.mark.parametrize("bad", SOURCE_BAD)
def test_source_must_be_exactly_measured_derived_or_projected(bad):
    doc = _doc()
    doc["samples"][0]["source"] = bad
    assert _validate_never_raises(doc)
    for ok in ("measured", "derived", "projected"):
        doc["samples"][0]["source"] = ok
        assert validate_result(doc) == []


@pytest.mark.parametrize("field", ["model_id", "backend"])
@pytest.mark.parametrize("bad", [None, 5, [], {}, True])
def test_model_id_and_backend_must_be_strings(field, bad):
    doc = _doc()
    doc["samples"][0][field] = bad
    assert _validate_never_raises(doc), f"{field}={bad!r} must be rejected"


@pytest.mark.parametrize("bad", ["", "tts", "LLM", None, 3, [], "llm "])
def test_unknown_workload_is_rejected(bad):
    doc = _doc()
    doc["samples"][0]["workload"] = bad
    assert _validate_never_raises(doc)


@pytest.mark.parametrize("workload", [w for w in WORKLOADS if w != "llm"])
def test_non_llm_workload_requires_non_empty_metrics(workload):
    for metrics_value in (None, {}, [], "x", 5):
        doc = _doc()
        row = doc["samples"][0]
        row["workload"] = workload
        if metrics_value is not None:
            row["metrics"] = metrics_value
        assert _validate_never_raises(doc), f"{workload} with metrics={metrics_value!r} must be rejected"
    doc = _doc()
    doc["samples"][0].update({"workload": workload, "metrics": {"latency_s": 1.25}})
    assert validate_result(doc) == []


@pytest.mark.parametrize("key", ["Latency_s", "1x", "latency-s", "latency s", "", "_x", "lat.ency", "LATENCY"])
def test_metric_keys_must_be_lowercase_identifiers(key):
    doc = _doc()
    doc["samples"][0].update({"workload": "audio_stt", "metrics": {key: 1.0}})
    assert _validate_never_raises(doc)


@pytest.mark.parametrize("bad", [-1, "1", [], {}, True, float("nan"), float("inf")])
def test_metric_values_must_be_null_or_finite_non_negative_numbers(bad):
    doc = _doc()
    doc["samples"][0].update({"workload": "image", "metrics": {"latency_s": bad}})
    assert _validate_never_raises(doc)
    doc["samples"][0]["metrics"] = {"latency_s": None, "rtf": 0.5}
    assert validate_result(doc) == []


def test_non_llm_documents_need_not_carry_llm_protocol_fields():
    doc = _doc()
    for k in PROTO_LLM_KEYS:
        del doc["protocol"][k]
    doc["samples"] = [dict(_sample(), workload="embeddings", metrics={"latency_s": 1.0}, ttft_ms=None, decode_tps=None)]
    assert validate_result(doc) == []


@pytest.mark.parametrize("bad", [{}, "x", 5, [1], [{}], [{"model_id": "m", "backend": "b"}],
                                 [{"model_id": 1, "backend": "b", "error": "e"}],
                                 [{"model_id": "m", "backend": "b", "error": None}], [None]])
def test_failures_must_be_list_of_string_records(bad):
    doc = _doc()
    doc["failures"] = bad
    assert _validate_never_raises(doc)
    doc["failures"] = [{"model_id": "m", "backend": "b", "error": "boom"}]
    assert validate_result(doc) == []


def test_every_single_field_mutation_of_a_valid_row_is_either_valid_or_rejected_never_raises():
    # exhaustive grid: each required sample field x a pool of JSON-shaped values
    pool = [None, True, False, 0, 1, -1, 1.5, float("nan"), float("inf"), "", "x", [], {}, [1], {"a": 1}]
    for field in SAMPLE_KEYS:
        for value in pool:
            doc = _doc()
            doc["samples"][0][field] = value
            _validate_never_raises(doc)


def test_write_result_rejects_invalid_and_leaves_no_file(tmp_path):
    doc = _doc()
    doc["samples"][0]["source"] = "guessed"
    target = tmp_path / "out" / "r.json"
    with pytest.raises(ValueError):
        write_result(target, doc)
    assert not target.exists()


def test_write_result_round_trips_and_load_samples_restores_rows(tmp_path):
    doc = _doc()
    target = tmp_path / "deep" / "dir" / "r.json"
    write_result(target, doc)
    again = json.loads(target.read_text(encoding="utf-8"))
    assert again == doc
    assert validate_result(again) == []
    samples = load_samples(again)
    assert [s.iteration for s in samples] == [0, 1]
    assert samples[0].decode_tps == 30.25 and samples[0].source == "measured"


def test_build_result_output_is_valid_and_keeps_failures():
    samples = [BenchSample("m", "gpu", 0, 10.0, 20.0, 4, 8, "measured")]
    env = {"os": "o", "python": "p", "git_sha": "g", "hardware": "h"}
    proto = {"warmup_runs": 0, "timed_runs": 1, "labels": {}, "temperature": 0.0, "max_new_tokens": 4}
    fails = [{"model_id": "m", "backend": "gpu", "error": "e"}]
    doc = build_result("run", samples, proto, env, failures=fails)
    assert doc["schema_version"] == SCHEMA_VERSION == 1
    assert validate_result(doc) == []
    assert doc["failures"] == fails
    assert "failures" not in build_result("run", samples, proto, env)
    assert doc["samples"][0]["decode_tps"] == 20.0


# ---- rules that only the published JSON schema (docs/benchmarks/schema/bench_result.v1.json) states ----

@pytest.mark.parametrize("key", ENV_KEYS)
@pytest.mark.parametrize("bad", [None, 5, [], {}, True])
def test_environment_required_values_must_be_strings_per_schema(key, bad):
    doc = _doc()
    doc["environment"][key] = bad
    assert _validate_never_raises(doc), f"environment.{key}={bad!r} violates the schema"


@pytest.mark.parametrize("key,bad", [
    ("warmup_runs", -1), ("warmup_runs", 1.5), ("warmup_runs", "1"), ("warmup_runs", None), ("warmup_runs", True),
    ("timed_runs", 0), ("timed_runs", -3), ("timed_runs", 2.5), ("timed_runs", "2"), ("timed_runs", None),
    ("max_new_tokens", 0), ("max_new_tokens", -1), ("max_new_tokens", 1.5), ("max_new_tokens", "8"),
    ("temperature", "0"), ("temperature", None), ("temperature", [], ), ("temperature", True),
    ("labels", []), ("labels", "x"), ("labels", None), ("labels", 5),
])
def test_protocol_field_types_and_minimums_per_schema(key, bad):
    doc = _doc()
    doc["protocol"][key] = bad
    assert _validate_never_raises(doc), f"protocol.{key}={bad!r} violates the schema"


@pytest.mark.parametrize("bad", [None, 0, 1, "yes", [], {}])
def test_noisy_flag_must_be_boolean_per_schema(bad):
    doc = _doc()
    doc["samples"][0]["noisy"] = bad
    assert _validate_never_raises(doc)
    doc["samples"][0]["noisy"] = True
    assert validate_result(doc) == []


def test_empty_run_id_is_rejected_per_schema_minlength():
    doc = _doc()
    doc["run_id"] = ""
    assert _validate_never_raises(doc)


@pytest.mark.parametrize("key", ["openvino_genai", "ollama_version"])
def test_optional_environment_strings_may_be_null_but_not_numbers(key):
    doc = _doc()
    doc["environment"][key] = None
    assert validate_result(doc) == []
    doc["environment"][key] = 3
    assert _validate_never_raises(doc)


def test_on_ac_power_must_be_boolean_or_null_per_schema():
    doc = _doc()
    for ok in (True, False, None):
        doc["environment"]["on_ac_power"] = ok
        assert validate_result(doc) == []
    for bad in ("yes", 1, []):
        doc["environment"]["on_ac_power"] = bad
        assert _validate_never_raises(doc)
