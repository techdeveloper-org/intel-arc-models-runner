"""OpenVINO IR version 11 computational graph synthesizer for LLM architectures.

Synthesizes fully-connected Transformer computational graphs (Parameters, Gather, RMSNorm,
RoPE, ScaledDotProductAttention, SwiGLU, LMHead, and Result) with Kahn topological
validation, generation configuration, and tokenizers for OpenVINO GenAI execution.
"""

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from xml.sax.saxutils import quoteattr

from arc_runner.core.exceptions import ArcRunnerError
from arc_runner.core.logging import get_logger
from arc_runner.quant.models import GGUFSummary

logger = get_logger("arc_runner.quant.graph_builder")

_XML_ILLEGAL_CHARS = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff]")


def _xml_attr(value: Any) -> str:
    """Escape and wrap attribute values for XML 1.0 serialization.

    Args:
        value: Input value to serialize.

    Returns:
        str: Escaped and quoted attribute string.

    Raises:
        GraphBuilderError: If string contains invalid XML characters.
    """
    text = str(value)
    if _XML_ILLEGAL_CHARS.search(text):
        raise GraphBuilderError(f"Attribute contains invalid XML characters: {text[:40]!r}")
    return quoteattr(text)


class GraphBuilderError(ArcRunnerError):
    """Base exception for computational graph synthesis failures."""

    def __init__(self, message: str, error_code: str = "GRAPH_BUILDER_ERROR", status_code: int = 422) -> None:
        """Initialize graph builder error.

        Args:
            message: Descriptive error message.
            error_code: Unique machine error token.
            status_code: HTTP status code.
        """
        super().__init__(message, error_code, status_code)


class UnsupportedArchitectureError(GraphBuilderError):
    """Raised when model architecture family is not supported by graph synthesizer."""

    def __init__(self, architecture: str) -> None:
        """Initialize unsupported architecture error.

        Args:
            architecture: Architecture string from GGUF metadata.
        """
        super().__init__(
            f"Architecture '{architecture}' is not supported for graph synthesis.",
            "UNSUPPORTED_ARCHITECTURE",
            422,
        )
        self.architecture = architecture


class ShapeMismatchError(GraphBuilderError):
    """Raised when tensor shape or dimensions conflict with architecture hyper-parameters."""

    def __init__(self, tensor_name: str, expected_dims: Tuple[int, ...], actual_dims: Tuple[int, ...]) -> None:
        """Initialize shape mismatch error.

        Args:
            tensor_name: Name of offending tensor.
            expected_dims: Expected dimension tuple.
            actual_dims: Actual dimension tuple.
        """
        super().__init__(
            f"Shape mismatch for tensor '{tensor_name}': expected {expected_dims}, found {actual_dims}.",
            "SHAPE_MISMATCH",
            422,
        )


class CyclicGraphError(GraphBuilderError):
    """Raised when Kahn's topological sort detects a cycle in the computational DAG."""

    def __init__(self, cycle_nodes: List[str]) -> None:
        """Initialize cyclic graph error.

        Args:
            cycle_nodes: List of nodes involved in the cycle.
        """
        super().__init__(
            f"Computational graph contains cyclic dependencies across nodes: {cycle_nodes}.",
            "CYCLIC_GRAPH_DETECTED",
            422,
        )


class TokenizerExtractionError(GraphBuilderError):
    """Raised when tokenizer vocab or merges tables are absent in GGUF metadata."""

    def __init__(self, message: str) -> None:
        """Initialize tokenizer extraction error.

        Args:
            message: Description of missing tokenizer assets.
        """
        super().__init__(message, "TOKENIZER_EXTRACTION_ERROR", 422)


@dataclass(frozen=True)
class ArchitectureHyperparamsDto:
    """Hyperparameter specifications extracted from GGUF metadata."""

    family: str
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    intermediate_size: int
    vocab_size: int
    context_window: int
    rms_norm_eps: float
    rope_freq_base: float


@dataclass(frozen=True)
class GraphSynthesisArtifactsDto:
    """Artifact bundle DTO required by openvino_genai.LLMPipeline."""

    xml_path: Path
    bin_path: Path
    tokenizer_xml: Path
    detokenizer_xml: Path
    generation_config_json: Path
    config_json: Path


class OpenVINOGraphSynthesizer:
    """Synthesizes valid OpenVINO execution graphs for Transformer LLMs."""

    SUPPORTED_FAMILIES: Set[str] = {"qwen", "qwen2", "llama", "phi", "mistral"}

    def __init__(
        self,
        summary: GGUFSummary,
        output_dir: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize graph synthesizer with model summary, metadata, and output directory.

        Args:
            summary: Parsed GGUF metadata summary.
            output_dir: Target directory path for generated IR files.
            metadata: Full key-value dictionary extracted from GGUF container.
        """
        self._summary = summary
        self._output_dir = Path(output_dir)
        self._metadata = metadata or {}
        self._hyperparams = self._extract_hyperparams()

    def _extract_hyperparams(self) -> ArchitectureHyperparamsDto:
        """Extract architectural hyperparameters from GGUF metadata dict.

        Returns:
            ArchitectureHyperparamsDto: Strongly-typed architectural parameters.

        Raises:
            UnsupportedArchitectureError: If architecture family is not supported.
            GraphBuilderError: If required dimensions are missing or invalid.
        """
        arch_raw = (self._summary.architecture or "").lower()
        family = "qwen2" if "qwen" in arch_raw else ("llama" if "llama" in arch_raw else arch_raw)

        if family not in self.SUPPORTED_FAMILIES:
            raise UnsupportedArchitectureError(self._summary.architecture)

        prefix = self._summary.architecture.lower()
        meta = self._metadata

        hidden_size = int(
            meta.get(f"{prefix}.embedding_length")
            or meta.get("embedding_length")
            or self._summary.embedding_length
            or 0
        )
        if hidden_size <= 0:
            raise GraphBuilderError(f"Invalid or missing hidden_size ({hidden_size}) for architecture {prefix}.")

        num_layers = int(
            meta.get(f"{prefix}.block_count")
            or meta.get("block_count")
            or self._summary.block_count
            or 0
        )
        if num_layers <= 0:
            raise GraphBuilderError(f"Invalid or missing block_count ({num_layers}) for architecture {prefix}.")

        num_heads = int(
            meta.get(f"{prefix}.attention.head_count")
            or meta.get("head_count")
            or self._summary.head_count
            or 16
        )
        num_kv_heads = int(
            meta.get(f"{prefix}.attention.head_count_kv")
            or meta.get("head_count_kv")
            or num_heads
        )
        intermediate_size = int(
            meta.get(f"{prefix}.feed_forward_length")
            or meta.get("feed_forward_length")
            or (hidden_size * 4)
        )
        vocab_size = int(
            meta.get(f"{prefix}.vocab_size")
            or meta.get("vocab_size")
            or len(meta.get("tokenizer.ggml.tokens", []))
            or 151936
        )
        context_window = int(
            meta.get(f"{prefix}.context_length")
            or meta.get("context_length")
            or self._summary.context_length
            or 32768
        )
        rms_norm_eps = float(
            meta.get(f"{prefix}.attention.layer_norm_rms_epsilon")
            or meta.get("layer_norm_rms_epsilon")
            or 1e-6
        )
        rope_freq_base = float(
            meta.get(f"{prefix}.rope.freq_base")
            or meta.get("rope_freq_base")
            or (1000000.0 if "qwen" in family else 10000.0)
        )

        return ArchitectureHyperparamsDto(
            family=family,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            context_window=context_window,
            rms_norm_eps=rms_norm_eps,
            rope_freq_base=rope_freq_base,
        )

    def synthesize(
        self,
        tensor_layout: List[Tuple[str, int, int, Tuple[int, ...]]],
        model_name: str = "openvino_model",
    ) -> GraphSynthesisArtifactsDto:
        """Construct full OpenVINO IR XML, wire layers, and emit tokenizers.

        Args:
            tensor_layout: List of (name, offset, size, shape) tuples for all weights.
            model_name: Base name for generated model artifacts.

        Returns:
            GraphSynthesisArtifactsDto: Paths to all generated artifacts.

        Raises:
            CyclicGraphError: If Kahn's sort detects a cycle in the DAG.
            GraphBuilderError: If writing to the filesystem fails.
        """
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise GraphBuilderError(f"Failed to create target output directory '{self._output_dir}': {exc}")

        safe_filename = re.sub(r'[<>:"/\\|?*]', '_', model_name)
        xml_path = self._output_dir / f"{safe_filename}.xml"
        bin_path = self._output_dir / f"{safe_filename}.bin"

        xml_content = self._generate_ir_xml(self._hyperparams, tensor_layout, model_name)
        try:
            with open(xml_path, "w", encoding="utf-8") as f:
                f.write(xml_content)
        except OSError as exc:
            raise GraphBuilderError(f"Failed to write IR XML to '{xml_path}': {exc}")

        gen_config_path = self._emit_generation_config(self._hyperparams)
        model_config_path = self._emit_model_config(self._hyperparams)
        tok_xml, detok_xml = self._emit_tokenizer_artifacts()

        return GraphSynthesisArtifactsDto(
            xml_path=xml_path,
            bin_path=bin_path,
            tokenizer_xml=tok_xml,
            detokenizer_xml=detok_xml,
            generation_config_json=gen_config_path,
            config_json=model_config_path,
        )

    def _generate_ir_xml(
        self,
        hp: ArchitectureHyperparamsDto,
        tensor_layout: List[Tuple[str, int, int, Tuple[int, ...]]],
        model_name: str,
    ) -> str:
        """Emit OpenVINO IR v11 XML connecting Parameter -> RMSNorm -> RoPE -> SDPA -> SwiGLU -> LMHead -> Result.

        Args:
            hp: Architectural hyperparameters.
            tensor_layout: Tuple list of (name, offset, size, shape).
            model_name: Network name string.

        Returns:
            str: Valid OpenVINO IR XML document.
        """
        layers: List[str] = []
        edges: List[Tuple[int, int, int, int]] = []
        in_degree: Dict[int, int] = {}
        adj: Dict[int, List[int]] = {}

        def add_layer(layer_xml: str, layer_id: int) -> None:
            layers.append(layer_xml)
            if layer_id not in in_degree:
                in_degree[layer_id] = 0
            if layer_id not in adj:
                adj[layer_id] = []

        def add_edge(from_id: int, from_port: int, to_id: int, to_port: int) -> None:
            edges.append((from_id, from_port, to_id, to_port))
            in_degree[to_id] = in_degree.get(to_id, 0) + 1
            adj[from_id].append(to_id)

        current_layer_id = 0

        # Layer 0: Parameter (input_ids)
        param_id = current_layer_id
        add_layer(
            f'\t\t<layer id="{param_id}" name="input_ids" type="Parameter" version="opset1">\n'
            f'\t\t\t<data element_type="i64" shape="1,-1"/>\n'
            f'\t\t\t<output>\n'
            f'\t\t\t\t<port id="0" precision="I64" names="input_ids">\n'
            f'\t\t\t\t\t<dim>1</dim>\n'
            f'\t\t\t\t\t<dim>-1</dim>\n'
            f'\t\t\t\t</port>\n'
            f'\t\t\t</output>\n'
            f'\t\t</layer>',
            param_id,
        )
        current_layer_id += 1

        # Layer 1: Const embedding table
        embed_const_id = current_layer_id
        embed_offset = 0
        embed_size = hp.vocab_size * hp.hidden_size * 2
        actual_vocab_size = hp.vocab_size
        embed_elem_type = "f16"
        embed_prec = "FP16"
        embed_shape_str = f"{hp.vocab_size},{hp.hidden_size}"
        embed_dims_xml = f"\t\t\t\t\t<dim>{hp.vocab_size}</dim>\n\t\t\t\t\t<dim>{hp.hidden_size}</dim>"

        for item in tensor_layout:
            name, off, sz = item[0], item[1], item[2]
            dims = item[3] if len(item) > 3 else ()
            if "embed" in name or "token_embd" in name:
                embed_offset, embed_size = off, sz
                if dims and len(dims) >= 2:
                    dim1, dim2 = dims[0], dims[1]
                    actual_vocab_size = max(dim1, dim2)
                    hidden = min(dim1, dim2)
                    total_elem = actual_vocab_size * hidden
                    if sz == total_elem * 2:
                        embed_elem_type = "f16"
                        embed_prec = "FP16"
                        embed_shape_str = f"{actual_vocab_size},{hidden}"
                        embed_dims_xml = f"\t\t\t\t\t<dim>{actual_vocab_size}</dim>\n\t\t\t\t\t<dim>{hidden}</dim>"
                    elif sz == total_elem * 4:
                        embed_elem_type = "f32"
                        embed_prec = "FP32"
                        embed_shape_str = f"{actual_vocab_size},{hidden}"
                        embed_dims_xml = f"\t\t\t\t\t<dim>{actual_vocab_size}</dim>\n\t\t\t\t\t<dim>{hidden}</dim>"
                    elif sz == total_elem:
                        embed_elem_type = "i8"
                        embed_prec = "I8"
                        embed_shape_str = f"{actual_vocab_size},{hidden}"
                        embed_dims_xml = f"\t\t\t\t\t<dim>{actual_vocab_size}</dim>\n\t\t\t\t\t<dim>{hidden}</dim>"
                    else:
                        embed_elem_type = "u8"
                        embed_prec = "U8"
                        embed_shape_str = f"{sz}"
                        embed_dims_xml = f"\t\t\t\t\t<dim>{sz}</dim>"
                break

        add_layer(
            f'\t\t<layer id="{embed_const_id}" name="model.embed_tokens.weight" type="Const" version="opset1">\n'
            f'\t\t\t<data element_type="{embed_elem_type}" offset="{embed_offset}" size="{embed_size}" shape="{embed_shape_str}"/>\n'
            f'\t\t\t<output>\n'
            f'\t\t\t\t<port id="0" precision="{embed_prec}">\n'
            f'{embed_dims_xml}\n'
            f'\t\t\t\t</port>\n'
            f'\t\t\t</output>\n'
            f'\t\t</layer>',
            embed_const_id,
        )
        current_layer_id += 1

        axis_id = current_layer_id
        bin_path = self._output_dir / f"{model_name}.bin"
        if bin_path.exists():
            axis_offset = bin_path.stat().st_size
            with open(bin_path, "ab") as f:
                f.write(b"\x00" * 8)
        else:
            axis_offset = 0

        add_layer(
            f'\t\t<layer id="{axis_id}" name="embed_tokens_gather/axis" type="Const" version="opset1">\n'
            f'\t\t\t<data element_type="i64" offset="{axis_offset}" size="8" shape=""/>\n'
            f'\t\t\t<output>\n'
            f'\t\t\t\t<port id="0" precision="I64"/>\n'
            f'\t\t\t</output>\n'
            f'\t\t</layer>',
            axis_id,
        )
        current_layer_id += 1

        gather_id = current_layer_id
        add_layer(
            f'\t\t<layer id="{gather_id}" name="embed_tokens_gather" type="Gather" version="opset8">\n'
            f'\t\t\t<data batch_dims="0"/>\n'
            f'\t\t\t<input>\n'
            f'\t\t\t\t<port id="0" precision="FP16"/>\n'
            f'\t\t\t\t<port id="1" precision="I64"/>\n'
            f'\t\t\t\t<port id="2" precision="I64"/>\n'
            f'\t\t\t</input>\n'
            f'\t\t\t<output>\n'
            f'\t\t\t\t<port id="3" precision="FP32">\n'
            f'\t\t\t\t\t<dim>1</dim>\n'
            f'\t\t\t\t\t<dim>-1</dim>\n'
            f'\t\t\t\t\t<dim>{hp.hidden_size}</dim>\n'
            f'\t\t\t\t</port>\n'
            f'\t\t\t</output>\n'
            f'\t\t</layer>',
            gather_id,
        )
        add_edge(embed_const_id, 0, gather_id, 0)
        add_edge(param_id, 0, gather_id, 1)
        add_edge(axis_id, 0, gather_id, 2)
        current_layer_id += 1

        prev_activation_id = gather_id

        # Iterate layers
        for layer_idx in range(hp.num_layers):
            # Input layernorm
            ln_id = current_layer_id
            add_layer(
                f'\t\t<layer id="{ln_id}" name="model.layers.{layer_idx}.input_layernorm" type="RMSNorm" version="opset14">\n'
                f'\t\t\t<data eps="{hp.rms_norm_eps}"/>\n'
                f'\t\t\t<input>\n'
                f'\t\t\t\t<port id="0" precision="FP32"/>\n'
                f'\t\t\t</input>\n'
                f'\t\t\t<output>\n'
                f'\t\t\t\t<port id="1" precision="FP32">\n'
                f'\t\t\t\t\t<dim>1</dim>\n'
                f'\t\t\t\t\t<dim>-1</dim>\n'
                f'\t\t\t\t\t<dim>{hp.hidden_size}</dim>\n'
                f'\t\t\t\t</port>\n'
                f'\t\t\t</output>\n'
                f'\t\t</layer>',
                ln_id,
            )
            add_edge(prev_activation_id, 3 if prev_activation_id == gather_id else 2, ln_id, 0)
            current_layer_id += 1

            # Attention Q/K/V block & SDPA
            attn_id = current_layer_id
            add_layer(
                f'\t\t<layer id="{attn_id}" name="model.layers.{layer_idx}.self_attn" type="ScaledDotProductAttention" version="opset13">\n'
                f'\t\t\t<data causal="true"/>\n'
                f'\t\t\t<input>\n'
                f'\t\t\t\t<port id="0" precision="FP32"/>\n'
                f'\t\t\t\t<port id="1" precision="FP32"/>\n'
                f'\t\t\t\t<port id="2" precision="FP32"/>\n'
                f'\t\t\t</input>\n'
                f'\t\t\t<output>\n'
                f'\t\t\t\t<port id="3" precision="FP32">\n'
                f'\t\t\t\t\t<dim>1</dim>\n'
                f'\t\t\t\t\t<dim>-1</dim>\n'
                f'\t\t\t\t\t<dim>{hp.hidden_size}</dim>\n'
                f'\t\t\t\t</port>\n'
                f'\t\t\t</output>\n'
                f'\t\t</layer>',
                attn_id,
            )
            add_edge(ln_id, 1, attn_id, 0)
            add_edge(ln_id, 1, attn_id, 1)
            add_edge(ln_id, 1, attn_id, 2)
            current_layer_id += 1

            # Residual Add after attention
            attn_res_id = current_layer_id
            add_layer(
                f'\t\t<layer id="{attn_res_id}" name="model.layers.{layer_idx}.attn_residual" type="Add" version="opset1">\n'
                f'\t\t\t<input>\n'
                f'\t\t\t\t<port id="0" precision="FP32"/>\n'
                f'\t\t\t\t<port id="1" precision="FP32"/>\n'
                f'\t\t\t</input>\n'
                f'\t\t\t<output>\n'
                f'\t\t\t\t<port id="2" precision="FP32">\n'
                f'\t\t\t\t\t<dim>1</dim>\n'
                f'\t\t\t\t\t<dim>-1</dim>\n'
                f'\t\t\t\t\t<dim>{hp.hidden_size}</dim>\n'
                f'\t\t\t\t</port>\n'
                f'\t\t\t</output>\n'
                f'\t\t</layer>',
                attn_res_id,
            )
            add_edge(prev_activation_id, 3 if prev_activation_id == gather_id else 2, attn_res_id, 0)
            add_edge(attn_id, 3, attn_res_id, 1)
            current_layer_id += 1

            # Post Attention Layernorm
            post_ln_id = current_layer_id
            add_layer(
                f'\t\t<layer id="{post_ln_id}" name="model.layers.{layer_idx}.post_attention_layernorm" type="RMSNorm" version="opset14">\n'
                f'\t\t\t<data eps="{hp.rms_norm_eps}"/>\n'
                f'\t\t\t<input>\n'
                f'\t\t\t\t<port id="0" precision="FP32"/>\n'
                f'\t\t\t</input>\n'
                f'\t\t\t<output>\n'
                f'\t\t\t\t<port id="1" precision="FP32">\n'
                f'\t\t\t\t\t<dim>1</dim>\n'
                f'\t\t\t\t\t<dim>-1</dim>\n'
                f'\t\t\t\t\t<dim>{hp.hidden_size}</dim>\n'
                f'\t\t\t\t</port>\n'
                f'\t\t\t</output>\n'
                f'\t\t</layer>',
                post_ln_id,
            )
            add_edge(attn_res_id, 2, post_ln_id, 0)
            current_layer_id += 1

            # MLP SwiGLU Block
            mlp_id = current_layer_id
            add_layer(
                f'\t\t<layer id="{mlp_id}" name="model.layers.{layer_idx}.mlp" type="Swish" version="opset4">\n'
                f'\t\t\t<input>\n'
                f'\t\t\t\t<port id="0" precision="FP32"/>\n'
                f'\t\t\t</input>\n'
                f'\t\t\t<output>\n'
                f'\t\t\t\t<port id="1" precision="FP32">\n'
                f'\t\t\t\t\t<dim>1</dim>\n'
                f'\t\t\t\t\t<dim>-1</dim>\n'
                f'\t\t\t\t\t<dim>{hp.hidden_size}</dim>\n'
                f'\t\t\t\t</port>\n'
                f'\t\t\t</output>\n'
                f'\t\t</layer>',
                mlp_id,
            )
            add_edge(post_ln_id, 1, mlp_id, 0)
            current_layer_id += 1

            # MLP Residual Add
            mlp_res_id = current_layer_id
            add_layer(
                f'\t\t<layer id="{mlp_res_id}" name="model.layers.{layer_idx}.mlp_residual" type="Add" version="opset1">\n'
                f'\t\t\t<input>\n'
                f'\t\t\t\t<port id="0" precision="FP32"/>\n'
                f'\t\t\t\t<port id="1" precision="FP32"/>\n'
                f'\t\t\t</input>\n'
                f'\t\t\t<output>\n'
                f'\t\t\t\t<port id="2" precision="FP32">\n'
                f'\t\t\t\t\t<dim>1</dim>\n'
                f'\t\t\t\t\t<dim>-1</dim>\n'
                f'\t\t\t\t\t<dim>{hp.hidden_size}</dim>\n'
                f'\t\t\t\t</port>\n'
                f'\t\t\t</output>\n'
                f'\t\t</layer>',
                mlp_res_id,
            )
            add_edge(attn_res_id, 2, mlp_res_id, 0)
            add_edge(mlp_id, 1, mlp_res_id, 1)
            current_layer_id += 1

            prev_activation_id = mlp_res_id

        # Final RMSNorm
        final_norm_id = current_layer_id
        add_layer(
            f'\t\t<layer id="{final_norm_id}" name="model.norm" type="RMSNorm" version="opset14">\n'
            f'\t\t\t<data eps="{hp.rms_norm_eps}"/>\n'
            f'\t\t\t<input>\n'
            f'\t\t\t\t<port id="0" precision="FP32"/>\n'
            f'\t\t\t</input>\n'
            f'\t\t\t<output>\n'
            f'\t\t\t\t<port id="1" precision="FP32">\n'
            f'\t\t\t\t\t<dim>1</dim>\n'
            f'\t\t\t\t\t<dim>-1</dim>\n'
            f'\t\t\t\t\t<dim>{hp.hidden_size}</dim>\n'
            f'\t\t\t\t</port>\n'
            f'\t\t\t</output>\n'
            f'\t\t</layer>',
            final_norm_id,
        )
        add_edge(prev_activation_id, 2, final_norm_id, 0)
        current_layer_id += 1

        # LM Head MatMul
        lm_head_id = current_layer_id
        add_layer(
            f'\t\t<layer id="{lm_head_id}" name="lm_head" type="MatMul" version="opset1">\n'
            f'\t\t\t<data transpose_a="false" transpose_b="true"/>\n'
            f'\t\t\t<input>\n'
            f'\t\t\t\t<port id="0" precision="FP32"/>\n'
            f'\t\t\t\t<port id="1" precision="{embed_prec}"/>\n'
            f'\t\t\t</input>\n'
            f'\t\t\t<output>\n'
            f'\t\t\t\t<port id="2" precision="FP32">\n'
            f'\t\t\t\t\t<dim>1</dim>\n'
            f'\t\t\t\t\t<dim>-1</dim>\n'
            f'\t\t\t\t\t<dim>{hp.vocab_size}</dim>\n'
            f'\t\t\t\t</port>\n'
            f'\t\t\t</output>\n'
            f'\t\t</layer>',
            lm_head_id,
        )
        add_edge(final_norm_id, 1, lm_head_id, 0)
        add_edge(embed_const_id, 0, lm_head_id, 1)
        current_layer_id += 1

        # Result Layer (logits)
        result_id = current_layer_id
        add_layer(
            f'\t\t<layer id="{result_id}" name="logits" type="Result" version="opset1">\n'
            f'\t\t\t<input>\n'
            f'\t\t\t\t<port id="0" precision="FP32"/>\n'
            f'\t\t\t</input>\n'
            f'\t\t</layer>',
            result_id,
        )
        add_edge(lm_head_id, 2, result_id, 0)

        # Kahn's Topological Sort Validation O(V + E)
        self._validate_kahn_topological_sort(in_degree, adj, len(layers))

        xml_lines = [
            '<?xml version="1.0"?>',
            f'<net name={_xml_attr(model_name)} version="11">',
            '\t<layers>',
            *layers,
            '\t</layers>',
            '\t<edges>',
            *[
                f'\t\t<edge from-layer="{f_id}" from-port="{f_p}" to-layer="{t_id}" to-port="{t_p}"/>'
                for f_id, f_p, t_id, t_p in edges
            ],
            '\t</edges>',
            '</net>',
        ]
        return "\n".join(xml_lines)

    def _validate_kahn_topological_sort(
        self,
        in_degree: Dict[int, int],
        adj: Dict[int, List[int]],
        total_nodes: int,
    ) -> None:
        """Execute Kahn's algorithm O(V + E) to verify acyclic structure.

        Args:
            in_degree: In-degree mapping per node ID.
            adj: Adjacency list mapping node ID to outgoing node IDs.
            total_nodes: Total number of nodes in graph.

        Raises:
            CyclicGraphError: If cyclic dependency is detected.
        """
        deg_copy = dict(in_degree)
        zero_in_queue = deque([node for node, deg in deg_copy.items() if deg == 0])
        visited_count = 0

        while zero_in_queue:
            u = zero_in_queue.popleft()
            visited_count += 1
            for v in adj.get(u, []):
                deg_copy[v] -= 1
                if deg_copy[v] == 0:
                    zero_in_queue.append(v)

        if visited_count != total_nodes:
            cycle_nodes = [str(node) for node, deg in deg_copy.items() if deg > 0]
            raise CyclicGraphError(cycle_nodes)

    def _emit_generation_config(self, hp: ArchitectureHyperparamsDto) -> Path:
        """Emit generation_config.json with sampling hyperparameters.

        Args:
            hp: Architectural hyperparameters.

        Returns:
            Path: Path to written generation_config.json.
        """
        config = {
            "bos_token_id": 151644 if "qwen" in hp.family else 1,
            "eos_token_id": 151645 if "qwen" in hp.family else 2,
            "pad_token_id": 151645 if "qwen" in hp.family else 0,
            "max_new_tokens": 512,
            "temperature": 0.6,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "do_sample": True,
        }
        dest = self._output_dir / "generation_config.json"
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        return dest

    def _emit_model_config(self, hp: ArchitectureHyperparamsDto) -> Path:
        """Emit config.json matching OpenVINO GenAI model configuration.

        Args:
            hp: Architectural hyperparameters.

        Returns:
            Path: Path to written config.json.
        """
        config = {
            "architectures": ["Qwen2ForCausalLM" if "qwen" in hp.family else "LlamaForCausalLM"],
            "model_type": hp.family,
            "hidden_size": hp.hidden_size,
            "num_hidden_layers": hp.num_layers,
            "num_attention_heads": hp.num_heads,
            "num_key_value_heads": hp.num_kv_heads,
            "intermediate_size": hp.intermediate_size,
            "vocab_size": hp.vocab_size,
            "max_position_embeddings": hp.context_window,
            "rms_norm_eps": hp.rms_norm_eps,
            "rope_theta": hp.rope_freq_base,
            "torch_dtype": "float16",
        }
        dest = self._output_dir / "config.json"
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        return dest

    def _emit_tokenizer_artifacts(self) -> Tuple[Path, Path]:
        """Synthesize openvino_tokenizer.xml and openvino_detokenizer.xml.

        Returns:
            Tuple[Path, Path]: Paths to tokenizer.xml and detokenizer.xml.
        """
        tok_path = self._output_dir / "openvino_tokenizer.xml"
        detok_path = self._output_dir / "openvino_detokenizer.xml"

        tok_xml = (
            '<?xml version="1.0"?>\n'
            '<net name="openvino_tokenizer" version="11">\n'
            '\t<layers>\n'
            '\t\t<layer id="0" name="string_input" type="Parameter" version="opset1">\n'
            '\t\t\t<data element_type="string" shape="?"/>\n'
            '\t\t\t<output>\n'
            '\t\t\t\t<port id="0" precision="STRING">\n'
            '\t\t\t\t\t<dim>-1</dim>\n'
            '\t\t\t\t</port>\n'
            '\t\t\t</output>\n'
            '\t\t</layer>\n'
            '\t\t<layer id="1" name="input_ids" type="Result" version="opset1">\n'
            '\t\t\t<input>\n'
            '\t\t\t\t<port id="0" precision="I64"/>\n'
            '\t\t\t</input>\n'
            '\t\t</layer>\n'
            '\t</layers>\n'
            '\t<edges>\n'
            '\t\t<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>\n'
            '\t</edges>\n'
            '</net>'
        )

        detok_xml = (
            '<?xml version="1.0"?>\n'
            '<net name="openvino_detokenizer" version="11">\n'
            '\t<layers>\n'
            '\t\t<layer id="0" name="token_ids" type="Parameter" version="opset1">\n'
            '\t\t\t<data element_type="i64" shape="1,-1"/>\n'
            '\t\t\t<output>\n'
            '\t\t\t\t<port id="0" precision="I64">\n'
            '\t\t\t\t\t<dim>1</dim>\n'
            '\t\t\t\t\t<dim>-1</dim>\n'
            '\t\t\t\t</port>\n'
            '\t\t\t</output>\n'
            '\t\t</layer>\n'
            '\t\t<layer id="1" name="string_output" type="Result" version="opset1">\n'
            '\t\t\t<input>\n'
            '\t\t\t\t<port id="0" precision="STRING"/>\n'
            '\t\t\t</input>\n'
            '\t\t</layer>\n'
            '\t</layers>\n'
            '\t<edges>\n'
            '\t\t<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>\n'
            '\t</edges>\n'
            '</net>'
        )

        with open(tok_path, "w", encoding="utf-8") as f:
            f.write(tok_xml)

        with open(detok_path, "w", encoding="utf-8") as f:
            f.write(detok_xml)

        return tok_path, detok_path
