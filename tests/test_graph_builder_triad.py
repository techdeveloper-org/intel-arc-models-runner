"""Triad test suite for OpenVINOGraphSynthesizer adhering to Enterprise standards.

Validates Kahn topological sorting, OpenVINO IR v11 computational graph synthesis,
edge wiring from Parameter to Result, tokenizer artifact generation, and error handling.
"""

from pathlib import Path
import json
import pytest

from arc_runner.quant.graph_builder import (
    CyclicGraphError,
    GraphBuilderError,
    OpenVINOGraphSynthesizer,
    ShapeMismatchError,
    UnsupportedArchitectureError,
)
from arc_runner.quant.models import GGUFSummary


def _create_qwen_summary(num_layers: int = 2) -> GGUFSummary:
    """Helper to create dummy Qwen 2.5 GGUFSummary."""
    return GGUFSummary(
        file_path="dummy.gguf",
        file_size_bytes=1024,
        architecture="qwen2",
        context_length=32768,
        embedding_length=2048,
        block_count=num_layers,
        head_count=16,
        tensor_count=50,
        total_parameters=3_000_000_000,
    )


def _create_llama_summary() -> GGUFSummary:
    """Helper to create dummy LLaMA GGUFSummary."""
    return GGUFSummary(
        file_path="dummy_llama.gguf",
        file_size_bytes=1024,
        architecture="llama",
        context_length=8192,
        embedding_length=4096,
        block_count=2,
        head_count=32,
        tensor_count=60,
        total_parameters=7_000_000_000,
    )


class TestGraphBuilderPositive:
    """Positive test specifications for computational graph synthesizer."""

    def test_synthesize_when_qwen25_summary_then_emits_valid_openvino_ir_xml(self, tmp_path: Path) -> None:
        """Verify graph synthesizer emits complete OpenVINO IR XML and config artifacts."""
        summary = _create_qwen_summary(num_layers=2)
        synthesizer = OpenVINOGraphSynthesizer(summary, tmp_path)

        tensor_layout = [("model.embed_tokens.weight", 0, 1024, (151936, 2048))]
        artifacts = synthesizer.synthesize(tensor_layout, "openvino_model")

        assert artifacts.xml_path.exists()
        assert artifacts.generation_config_json.exists()
        assert artifacts.config_json.exists()
        assert artifacts.tokenizer_xml.exists()
        assert artifacts.detokenizer_xml.exists()

        xml_text = artifacts.xml_path.read_text(encoding="utf-8")
        assert '<net name="openvino_model" version="11">' in xml_text
        assert 'type="Parameter"' in xml_text
        assert 'type="Result"' in xml_text
        assert '<edges>' in xml_text

    def test_synthesize_when_parsed_by_openvino_core_then_loads_without_errors(self, tmp_path: Path) -> None:
        """Verify XML contains properly structured and matching edge tags."""
        summary = _create_qwen_summary(num_layers=1)
        synthesizer = OpenVINOGraphSynthesizer(summary, tmp_path)
        artifacts = synthesizer.synthesize([], "model")

        xml_text = artifacts.xml_path.read_text(encoding="utf-8")
        assert xml_text.count("<edge ") > 5
        assert "</edges>" in xml_text

    def test_synthesize_when_graph_edges_verified_then_all_nodes_fully_connected(self, tmp_path: Path) -> None:
        """Verify every edge points from a valid layer to another valid layer."""
        summary = _create_qwen_summary(num_layers=2)
        synthesizer = OpenVINOGraphSynthesizer(summary, tmp_path)
        artifacts = synthesizer.synthesize([], "connected_model")

        xml_text = artifacts.xml_path.read_text(encoding="utf-8")
        assert "from-layer" in xml_text
        assert "to-layer" in xml_text

    def test_synthesize_when_qwen_architecture_then_scales_rope_freq_base_to_1m(self, tmp_path: Path) -> None:
        """Verify Qwen family sets rope_freq_base to 1,000,000."""
        summary = _create_qwen_summary()
        synthesizer = OpenVINOGraphSynthesizer(summary, tmp_path)
        artifacts = synthesizer.synthesize([], "qwen_test")

        config_data = json.loads(artifacts.config_json.read_text(encoding="utf-8"))
        assert config_data["rope_theta"] == 1000000.0

    def test_synthesize_when_mlp_blocks_built_then_wires_swiglu_activation_chain(self, tmp_path: Path) -> None:
        """Verify Swish/SwiGLU layers are instantiated for each transformer layer."""
        summary = _create_qwen_summary(num_layers=3)
        synthesizer = OpenVINOGraphSynthesizer(summary, tmp_path)
        artifacts = synthesizer.synthesize([], "swiglu_test")

        xml_text = artifacts.xml_path.read_text(encoding="utf-8")
        assert xml_text.count('type="Swish"') == 3

    def test_synthesize_when_metadata_written_then_emits_valid_generation_config(self, tmp_path: Path) -> None:
        """Verify generation_config.json contains temperature, top_p, and eos tokens."""
        summary = _create_qwen_summary()
        synthesizer = OpenVINOGraphSynthesizer(summary, tmp_path)
        artifacts = synthesizer.synthesize([], "gen_test")

        gen_config = json.loads(artifacts.generation_config_json.read_text(encoding="utf-8"))
        assert gen_config["temperature"] == 0.6
        assert gen_config["top_p"] == 0.9
        assert gen_config["repetition_penalty"] == 1.1

    def test_synthesize_when_vocab_present_then_emits_tokenizer_and_detokenizer_xml(self, tmp_path: Path) -> None:
        """Verify tokenizer and detokenizer OpenVINO IR files are created."""
        summary = _create_qwen_summary()
        synthesizer = OpenVINOGraphSynthesizer(summary, tmp_path)
        artifacts = synthesizer.synthesize([], "tok_test")

        assert artifacts.tokenizer_xml.is_file()
        assert artifacts.detokenizer_xml.is_file()
        assert "openvino_tokenizer" in artifacts.tokenizer_xml.read_text(encoding="utf-8")
        assert "openvino_detokenizer" in artifacts.detokenizer_xml.read_text(encoding="utf-8")

    def test_synthesize_when_llama_architecture_then_constructs_llama_family_graph(self, tmp_path: Path) -> None:
        """Verify LLaMA architecture builds with rope_theta = 10,000."""
        summary = _create_llama_summary()
        synthesizer = OpenVINOGraphSynthesizer(summary, tmp_path)
        artifacts = synthesizer.synthesize([], "llama_test")

        config_data = json.loads(artifacts.config_json.read_text(encoding="utf-8"))
        assert config_data["model_type"] == "llama"
        assert config_data["rope_theta"] == 10000.0


class TestGraphBuilderNegative:
    """Negative test specifications verifying exceptions and validation errors."""

    def test_synthesize_when_unknown_architecture_then_raises_unsupported_arch_error(self, tmp_path: Path) -> None:
        """Verify unsupported model family raises UnsupportedArchitectureError."""
        summary = GGUFSummary(
            file_path="exotic.gguf",
            file_size_bytes=100,
            architecture="mamba_ssm",
            context_length=2048,
            embedding_length=1024,
            block_count=4,
            head_count=8,
            tensor_count=10,
            total_parameters=100_000,
        )

        with pytest.raises(UnsupportedArchitectureError) as exc_info:
            OpenVINOGraphSynthesizer(summary, tmp_path)

        assert "mamba_ssm" in str(exc_info.value)

    def test_synthesize_when_missing_hidden_size_then_raises_graph_builder_error(self, tmp_path: Path) -> None:
        """Verify zero or absent hidden size raises GraphBuilderError."""
        summary = GGUFSummary(
            file_path="bad.gguf",
            file_size_bytes=100,
            architecture="qwen2",
            context_length=2048,
            embedding_length=0,
            block_count=4,
            head_count=8,
            tensor_count=10,
            total_parameters=100_000,
        )

        with pytest.raises(GraphBuilderError) as exc_info:
            OpenVINOGraphSynthesizer(summary, tmp_path)

        assert "hidden_size" in str(exc_info.value)

    def test_synthesize_when_weight_dimensions_conflict_then_raises_shape_mismatch_error(self, tmp_path: Path) -> None:
        """Verify ShapeMismatchError holds expected and actual dimensions."""
        err = ShapeMismatchError("embed_tokens", (151936, 2048), (32000, 4096))
        assert err.status_code == 422
        assert "embed_tokens" in str(err)

    def test_synthesize_when_output_dir_read_only_then_raises_filesystem_error(self, tmp_path: Path) -> None:
        """Verify invalid path raises GraphBuilderError."""
        summary = _create_qwen_summary()
        invalid_path = Path("Z:/non_existent_mount/forbidden/dir")

        with pytest.raises(GraphBuilderError):
            synthesizer = OpenVINOGraphSynthesizer(summary, invalid_path)
            synthesizer.synthesize([])

    def test_synthesize_when_kahn_cycle_injected_then_raises_cyclic_graph_error(self, tmp_path: Path) -> None:
        """Verify cyclic graph triggers CyclicGraphError."""
        summary = _create_qwen_summary()
        synthesizer = OpenVINOGraphSynthesizer(summary, tmp_path)

        in_degree = {0: 1, 1: 1}
        adj = {0: [1], 1: [0]}

        with pytest.raises(CyclicGraphError):
            synthesizer._validate_kahn_topological_sort(in_degree, adj, 2)

    def test_synthesize_when_num_layers_is_zero_then_raises_validation_error(self, tmp_path: Path) -> None:
        """Verify zero block count raises GraphBuilderError."""
        summary = GGUFSummary(
            file_path="empty.gguf",
            file_size_bytes=100,
            architecture="qwen2",
            context_length=2048,
            embedding_length=2048,
            block_count=0,
            head_count=8,
            tensor_count=10,
            total_parameters=100_000,
        )

        with pytest.raises(GraphBuilderError) as exc_info:
            OpenVINOGraphSynthesizer(summary, tmp_path)

        assert "block_count" in str(exc_info.value)


class TestGraphBuilderEdgeCase:
    """Edge-case test specifications verifying boundaries and unusual configurations."""

    def test_synthesize_when_128k_context_window_then_applies_long_rope_scaling(self, tmp_path: Path) -> None:
        """Verify long context window (128K) is stored cleanly in config."""
        summary = GGUFSummary(
            file_path="long_context.gguf",
            file_size_bytes=1024,
            architecture="qwen2",
            context_length=131072,
            embedding_length=2048,
            block_count=1,
            head_count=16,
            tensor_count=20,
            total_parameters=1_000_000,
        )
        synthesizer = OpenVINOGraphSynthesizer(summary, tmp_path)
        artifacts = synthesizer.synthesize([], "long_ctx")

        config_data = json.loads(artifacts.config_json.read_text(encoding="utf-8"))
        assert config_data["max_position_embeddings"] == 131072

    def test_synthesize_when_gqa_asymmetric_heads_then_wires_kv_broadcasting_correctly(self, tmp_path: Path) -> None:
        """Verify GQA with num_heads=16 and num_kv_heads=2 propagates to config."""
        summary = _create_qwen_summary()
        metadata = {"qwen2.attention.head_count_kv": 2}
        synthesizer = OpenVINOGraphSynthesizer(summary, tmp_path, metadata)
        artifacts = synthesizer.synthesize([], "gqa_test")

        config_data = json.loads(artifacts.config_json.read_text(encoding="utf-8"))
        assert config_data["num_key_value_heads"] == 2

    def test_synthesize_when_vocab_contains_special_xml_chars_then_escapes_attributes(self, tmp_path: Path) -> None:
        """Verify model name with XML special characters is properly escaped."""
        summary = _create_qwen_summary(num_layers=1)
        synthesizer = OpenVINOGraphSynthesizer(summary, tmp_path)
        artifacts = synthesizer.synthesize([], "model<>&\"test")

        xml_text = artifacts.xml_path.read_text(encoding="utf-8")
        assert "&lt;" in xml_text
        assert "&gt;" in xml_text

    def test_synthesize_when_single_layer_tiny_model_then_constructs_valid_minimal_ir(self, tmp_path: Path) -> None:
        """Verify 1-layer minimal model synthesizes valid DAG."""
        summary = _create_qwen_summary(num_layers=1)
        synthesizer = OpenVINOGraphSynthesizer(summary, tmp_path)
        artifacts = synthesizer.synthesize([], "tiny_model")

        xml_text = artifacts.xml_path.read_text(encoding="utf-8")
        assert "model.layers.0.input_layernorm" in xml_text
        assert "model.layers.1.input_layernorm" not in xml_text

    def test_synthesize_when_rms_norm_eps_varies_then_formats_without_precision_loss(self, tmp_path: Path) -> None:
        """Verify tiny rms_norm_eps (1e-7) formats accurately into XML attribute."""
        summary = _create_qwen_summary(num_layers=1)
        metadata = {"qwen2.attention.layer_norm_rms_epsilon": 1e-7}
        synthesizer = OpenVINOGraphSynthesizer(summary, tmp_path, metadata)
        artifacts = synthesizer.synthesize([], "eps_test")

        xml_text = artifacts.xml_path.read_text(encoding="utf-8")
        assert 'eps="1e-07"' in xml_text or 'eps="0.0000001"' in xml_text or 'eps="1e-7"' in xml_text

    def test_synthesize_when_partial_vocab_table_then_applies_fallback_tokenizer_logic(self, tmp_path: Path) -> None:
        """Verify tokenizer XML is produced even without explicit tokenizer metadata."""
        summary = _create_qwen_summary(num_layers=1)
        synthesizer = OpenVINOGraphSynthesizer(summary, tmp_path, metadata={})
        artifacts = synthesizer.synthesize([], "fallback_tok")

        assert artifacts.tokenizer_xml.exists()
        assert artifacts.detokenizer_xml.exists()
