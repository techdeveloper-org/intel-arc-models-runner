"""Comprehensive test triad for speculative decoding and OpenVINO performance optimizations.

Covers:
- 5 Positive (Happy-Path) test cases: TC-P01 to TC-P05
- 5 Negative (Error-Path) test cases: TC-N01 to TC-N05
- 5 Edge-Case (Boundary) test cases: TC-E01 to TC-E05

Compliant with Claude Global Rules 33, 35, 36, 37, 38, and 41.
"""

from pathlib import Path
from unittest import mock
import pytest

from arc_runner.core.exceptions import (
    ContextWindowExceededError,
    DraftModelCompilationError,
    ModelNotFoundError,
    TokenizerMismatchError,
)
from arc_runner.engine.models import GenerationConfig, SpeculativeConfig
from arc_runner.engine.openvino_engine import OpenVINOGenAIEngine as OpenVINOEngine
from arc_runner.engine.speculative_engine import HybridSpeculativeEngine


# ==============================================================================
# 1. POSITIVE (HAPPY-PATH) TEST CASES (Rule 35)
# ==============================================================================


def test_tc_p01_openvino_engine_optimized_properties(tmp_path: Path) -> None:
    """Verify that OpenVINOEngine builds high-performance properties for GPU execution."""
    engine = OpenVINOEngine()
    props = engine._build_ov_properties("GPU", {})
    assert props.get("PERFORMANCE_HINT") == "LATENCY"
    assert props.get("EXECUTION_MODE_HINT") == "PERFORMANCE"
    assert props.get("INFERENCE_PRECISION_HINT") == "f16"
    assert props.get("GPU_ENABLE_SDPA_OPTIMIZATION") is True


def test_tc_p02_speculative_pair_execution(tmp_path: Path) -> None:
    """Verify that speculative decoding initializes and executes with mock target and draft models."""
    target_dir = tmp_path / "target"
    draft_dir = tmp_path / "draft"
    target_dir.mkdir()
    draft_dir.mkdir()

    mock_pipeline = mock.MagicMock()
    mock_pipeline.generate.return_value = ["Speculative completion text"]

    with mock.patch("openvino_genai.draft_model", return_value=mock.MagicMock()):
        with mock.patch("openvino_genai.LLMPipeline", return_value=mock_pipeline):
            engine = HybridSpeculativeEngine()
            cfg = SpeculativeConfig(
                draft_model_path=str(draft_dir),
                draft_device="CPU",
                num_assistant_tokens=4,
            )
            engine.load_model(str(target_dir), speculative_config=cfg)
            result = engine.generate("Test prompt", GenerationConfig(max_new_tokens=20))
            assert "Speculative completion text" in result
            assert engine.is_loaded is True


def test_tc_p03_prompt_lookup_speculative_decoding(tmp_path: Path) -> None:
    """Verify that prompt lookup speculative decoding configures n-gram lookahead correctly."""
    target_dir = tmp_path / "target"
    target_dir.mkdir()

    mock_pipeline = mock.MagicMock()
    mock_pipeline.generate.return_value = ["Prompt lookup result"]

    with mock.patch("openvino_genai.LLMPipeline", return_value=mock_pipeline):
        engine = HybridSpeculativeEngine()
        cfg = SpeculativeConfig(
            prompt_lookup=True,
            num_assistant_tokens=5,
            max_ngram_size=3,
        )
        engine.load_model(str(target_dir), speculative_config=cfg)
        gen_cfg = engine._build_ov_generation_config(GenerationConfig(max_new_tokens=25))
        assert gen_cfg.num_assistant_tokens == 5
        assert gen_cfg.max_ngram_size == 3


def test_tc_p04_u8_kv_cache_property_injection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that setting the environment variable enables the u8 KV cache precision property."""
    monkeypatch.setenv("ARC_KV_CACHE_PRECISION", "u8")
    engine = OpenVINOEngine()
    props = engine._build_ov_properties("GPU", {})
    assert props.get("KV_CACHE_PRECISION") == "u8"


def test_tc_p05_telemetry_captures_speculative_metadata(tmp_path: Path) -> None:
    """Verify that telemetry correctly records speculative execution flags."""
    target_dir = tmp_path / "target"
    draft_dir = tmp_path / "draft"
    target_dir.mkdir()
    draft_dir.mkdir()

    mock_pipeline = mock.MagicMock()
    mock_pipeline.generate.return_value = ["Output text"]

    with mock.patch("openvino_genai.draft_model", return_value=mock.MagicMock()):
        with mock.patch("openvino_genai.LLMPipeline", return_value=mock_pipeline):
            engine = HybridSpeculativeEngine()
            cfg = SpeculativeConfig(draft_model_path=str(draft_dir), draft_device="CPU")
            engine.load_model(str(target_dir), speculative_config=cfg)
            engine.generate("Prompt")
            telem = engine.get_telemetry()
            assert telem.is_speculative is True
            assert telem.draft_device == "CPU"


# ==============================================================================
# 2. NEGATIVE (ERROR-PATH) TEST CASES (Rule 36)
# ==============================================================================


def test_tc_n01_tokenizer_mismatch_clean_error(tmp_path: Path) -> None:
    """Verify that tokenizer mismatch raises a typed TokenizerMismatchError."""
    target_dir = tmp_path / "target"
    draft_dir = tmp_path / "draft"
    target_dir.mkdir()
    draft_dir.mkdir()

    with mock.patch("openvino_genai.draft_model", return_value=mock.MagicMock()):
        with mock.patch(
            "openvino_genai.LLMPipeline",
            side_effect=RuntimeError("Tokenizers for draft and main models are different!"),
        ):
            engine = HybridSpeculativeEngine()
            cfg = SpeculativeConfig(draft_model_path=str(draft_dir))
            with pytest.raises(TokenizerMismatchError) as exc_info:
                engine.load_model(str(target_dir), speculative_config=cfg)
            assert "Tokenizer mismatch between target" in str(exc_info.value)
            assert exc_info.value.status_code == 400


def test_tc_n02_cpu_plugin_property_isolation() -> None:
    """Verify that GPU-specific SDPA properties are stripped when compiling on CPU."""
    engine = OpenVINOEngine()
    props = engine._build_ov_properties("CPU", {"GPU_ENABLE_SDPA_OPTIMIZATION": True})
    assert "GPU_ENABLE_SDPA_OPTIMIZATION" not in props


def test_tc_n03_invalid_model_path_handling(tmp_path: Path) -> None:
    """Verify that a non-existent directory raises ModelNotFoundError."""
    non_existent = tmp_path / "does_not_exist"
    engine = HybridSpeculativeEngine()
    with pytest.raises(ModelNotFoundError):
        engine.load_model(str(non_existent))


def test_tc_n04_context_window_exceeded_error(tmp_path: Path) -> None:
    """Verify that requesting tokens beyond context window raises ContextWindowExceededError."""
    target_dir = tmp_path / "target"
    target_dir.mkdir()

    with mock.patch("openvino_genai.LLMPipeline", return_value=mock.MagicMock()):
        engine = OpenVINOEngine()
        engine.load_model(str(target_dir))
        long_prompt = "word " * 10000
        with pytest.raises(ContextWindowExceededError):
            engine.generate(long_prompt, GenerationConfig(max_new_tokens=4096))


def test_tc_n05_draft_model_compilation_failure_wraps(tmp_path: Path) -> None:
    """Verify that complete failure of candidate draft devices raises DraftModelCompilationError."""
    draft_dir = tmp_path / "draft"
    draft_dir.mkdir()

    with mock.patch("openvino_genai.draft_model", side_effect=RuntimeError("Hardware unit unreachable")):
        engine = HybridSpeculativeEngine()
        with pytest.raises(DraftModelCompilationError) as exc_info:
            engine._create_draft_model(draft_dir, "NPU")
        assert exc_info.value.status_code in (500, 502)


# ==============================================================================
# 3. EDGE-CASE (BOUNDARY CONDITION) TEST CASES (Rule 37)
# ==============================================================================


def test_tc_e01_empty_prompt_generation(tmp_path: Path) -> None:
    """Verify that generating with an empty string prompt returns an empty string without failure."""
    target_dir = tmp_path / "target"
    target_dir.mkdir()

    with mock.patch("openvino_genai.LLMPipeline", return_value=mock.MagicMock()):
        engine = HybridSpeculativeEngine()
        engine.load_model(str(target_dir))
        result = engine.generate("")
        assert result == ""
        assert engine.get_telemetry().prompt_tokens == 0


def test_tc_e02_single_token_generation_boundary(tmp_path: Path) -> None:
    """Verify single-token generation boundary (max_new_tokens=1) handles latency math cleanly."""
    target_dir = tmp_path / "target"
    target_dir.mkdir()

    mock_pipeline = mock.MagicMock()
    mock_pipeline.generate.return_value = ["A"]

    with mock.patch("openvino_genai.LLMPipeline", return_value=mock_pipeline):
        engine = HybridSpeculativeEngine()
        engine.load_model(str(target_dir))
        result = engine.generate("Prompt", GenerationConfig(max_new_tokens=1))
        assert result == "A"
        telem = engine.get_telemetry()
        assert telem.prompt_tokens >= 0


def test_tc_e03_draft_missing_tokenizer_ir_auto_sync(tmp_path: Path) -> None:
    """Verify that tokenizer IR files are automatically synchronized to draft directories."""
    target_dir = tmp_path / "target"
    draft_dir = tmp_path / "draft"
    target_dir.mkdir()
    draft_dir.mkdir()

    (target_dir / "openvino_tokenizer.xml").write_text("<xml/>", encoding="utf-8")
    (target_dir / "openvino_tokenizer.bin").write_bytes(b"\x00")

    engine = HybridSpeculativeEngine()
    engine._sync_draft_tokenizer(target_dir, draft_dir)

    assert (draft_dir / "openvino_tokenizer.xml").exists()
    assert (draft_dir / "openvino_tokenizer.bin").exists()


def test_tc_e04_gamma_boundary_adaptation() -> None:
    """Verify that dynamic gamma updates adhere strictly to configured min_gamma and max_gamma bounds."""
    engine = HybridSpeculativeEngine()
    engine._spec_config = SpeculativeConfig(
        draft_model_path="dummy",
        num_assistant_tokens=4,
        min_gamma=2,
        max_gamma=6,
        dynamic_gamma=True,
    )
    engine._current_gamma = 6

    engine._adapt_gamma(accepted_count=4, draft_count=4)
    assert engine._current_gamma == 6

    engine._current_gamma = 2
    engine._adapt_gamma(accepted_count=0, draft_count=4)
    assert engine._current_gamma == 2


def test_tc_e05_zero_draft_acceptance_fallback(tmp_path: Path) -> None:
    """Verify that 0% draft acceptance does not cause crash or divide-by-zero."""
    engine = HybridSpeculativeEngine()
    engine._spec_config = SpeculativeConfig(
        draft_model_path="dummy",
        num_assistant_tokens=4,
        min_gamma=1,
        max_gamma=8,
        dynamic_gamma=True,
    )
    engine._current_gamma = 4
    engine._adapt_gamma(accepted_count=0, draft_count=4)
    assert engine._current_gamma == 3


class TestDraftRewriteFallback:
    """Issue #69: the NPU draft rewrite failure must be visible, typed and bounded."""

    @staticmethod
    def _incompatible_adapter():
        report = mock.Mock(is_compatible=False)
        return mock.patch(
            "arc_runner.engine.npu_adapter.NpuModelAdapter.inspect_model_compatibility",
            return_value=report,
        )

    @staticmethod
    def _compatible_adapter():
        return mock.patch(
            "arc_runner.engine.npu_adapter.NpuModelAdapter.inspect_model_compatibility",
            return_value=mock.Mock(is_compatible=True),
        )

    def test_create_draft_model_rewriter_error_records_reason_and_uses_gpu(self, tmp_path, caplog):
        from arc_runner.quant.npu_graph_rewriter import NpuGraphRewriterError

        draft_dir = tmp_path / "draft"
        draft_dir.mkdir()
        error = NpuGraphRewriterError("bad range", {"layer": "w1"})
        with self._incompatible_adapter(), mock.patch(
            "arc_runner.quant.npu_graph_rewriter.NpuGraphRewriter.rewrite", side_effect=error
        ), mock.patch("openvino_genai.draft_model", return_value="handle") as draft:
            engine = HybridSpeculativeEngine()
            with caplog.at_level("ERROR"):
                handle = engine._create_draft_model(draft_dir, "NPU")
        assert handle == "handle"
        assert draft.call_args.args[1] == "GPU"
        assert engine.draft_device == "GPU"
        assert engine.draft_fallback_reason == "NPU_GRAPH_REWRITE_ERROR: bad range"
        assert any(r.levelname == "ERROR" for r in caplog.records)

    def test_create_draft_model_unexpected_rewriter_exception_is_not_swallowed(self, tmp_path):
        draft_dir = tmp_path / "draft"
        draft_dir.mkdir()
        with self._incompatible_adapter(), mock.patch(
            "arc_runner.quant.npu_graph_rewriter.NpuGraphRewriter.rewrite", side_effect=ZeroDivisionError("bug")
        ), mock.patch("openvino_genai.draft_model", return_value="handle"):
            engine = HybridSpeculativeEngine()
            with pytest.raises(ZeroDivisionError):
                engine._create_draft_model(draft_dir, "NPU")


class TestDraftDeviceChain:
    """NPU falls back to GPU, GPU falls back to CPU, CPU is the last tier."""

    @pytest.mark.parametrize(
        "preferred,expected",
        [("NPU", ["NPU", "GPU", "CPU"]), ("GPU", ["GPU", "CPU"]), ("GPU.0", ["GPU.0", "CPU"]),
         ("CPU", ["CPU"]), ("AUTO", ["AUTO", "CPU"])],
    )
    def test_draft_device_chain_returns_ordered_tiers(self, preferred, expected):
        assert HybridSpeculativeEngine._draft_device_chain(preferred) == expected

    def test_create_draft_model_npu_compile_failure_falls_back_to_gpu(self, tmp_path):
        draft_dir = tmp_path / "draft"
        draft_dir.mkdir()
        calls = []

        def fake(path, dev, **kw):
            calls.append(dev)
            if dev == "NPU":
                raise RuntimeError("npu compile failed")
            return "handle"

        with TestDraftRewriteFallback._compatible_adapter(), mock.patch("openvino_genai.draft_model", side_effect=fake):
            engine = HybridSpeculativeEngine()
            assert engine._create_draft_model(draft_dir, "NPU") == "handle"
        assert calls == ["NPU", "GPU"]
        assert engine.draft_device == "GPU"
        assert "NPU" in engine.draft_fallback_reason

    def test_create_draft_model_npu_and_gpu_fail_falls_back_to_cpu(self, tmp_path):
        draft_dir = tmp_path / "draft"
        draft_dir.mkdir()
        calls = []

        def fake(path, dev, **kw):
            calls.append(dev)
            if dev != "CPU":
                raise RuntimeError(f"{dev} failed")
            return "handle"

        with TestDraftRewriteFallback._compatible_adapter(), mock.patch("openvino_genai.draft_model", side_effect=fake):
            engine = HybridSpeculativeEngine()
            assert engine._create_draft_model(draft_dir, "NPU") == "handle"
        assert calls == ["NPU", "GPU", "CPU"]
        assert engine.draft_device == "CPU"

    def test_create_draft_model_gpu_receives_original_path_not_npu_certified_copy(self, tmp_path):
        draft_dir = tmp_path / "draft"
        draft_dir.mkdir()
        seen = {}

        def fake(path, dev, **kw):
            seen[dev] = path
            if dev == "NPU":
                raise RuntimeError("npu compile failed")
            return "handle"

        with TestDraftRewriteFallback._incompatible_adapter(), mock.patch(
            "arc_runner.quant.npu_graph_rewriter.NpuGraphRewriter.rewrite"
        ), mock.patch("openvino_genai.draft_model", side_effect=fake):
            HybridSpeculativeEngine()._create_draft_model(draft_dir, "NPU")
        assert seen["NPU"].endswith("draft-npu-certified")
        assert seen["GPU"] == str(draft_dir)
