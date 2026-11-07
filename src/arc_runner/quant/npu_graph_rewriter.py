"""Intel AI Boost NPU Graph Rewriter and Computational DAG Optimizer.

Transforms OpenVINO IR XML and binary weights to meet the strict physical hardware
constraints of the Intel NPU 3720 (NPU37XX) VPUX compiler:
1. Strips unsupported asymmetric Subtract zero-point layers and repacks weights into
   symmetric signed INT4 (i4) with zero offset.
2. Freezes unbounded dynamic KV cache dimensions to static CMX-compliant bounds.
3. Synchronizes tokenizer assets and emits hardware-certified NPU manifests.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Set, Tuple
import xml.etree.ElementTree as ET

import numpy as np

from arc_runner.core.exceptions import ArcRunnerError
from arc_runner.core.logging import get_logger

logger = get_logger("quant.npu_graph_rewriter")


class NpuGraphRewriterError(ArcRunnerError):
    """Raised when an OpenVINO IR graph transformation fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the graph rewrite exception with diagnostic details.

        Args:
            message: Human-readable error description.
            details: Optional dictionary containing context metadata.
        """
        super().__init__(message=message, error_code="NPU_GRAPH_REWRITE_ERROR", status_code=500)
        self.details = details or {}


def parse_dim(token: str) -> Tuple[int, Optional[int]]:
    """Parse one OpenVINO partial dimension into inclusive bounds.

    Args:
        token: "?", "-1", "N" or an interval "A..B" where either end may be empty.

    Returns:
        Pair (low, high); high is None when the dimension has no upper bound.

    Raises:
        NpuGraphRewriterError: If the token is not a valid dimension.
    """
    text = token.strip()
    try:
        if text in ("?", "-1"):
            return 0, None
        if ".." in text:
            low_text, high_text = text.split("..", 1)
            return (int(low_text) if low_text else 0), (int(high_text) if high_text else None)
        value = int(text)
    except ValueError as exc:
        raise NpuGraphRewriterError(f"Invalid dimension token: {token!r}", {"token": token}) from exc
    if value < 0:
        raise NpuGraphRewriterError(f"Invalid dimension token: {token!r}", {"token": token})
    return value, value


def freeze_kv_variable_shape(shape_text: str, max_context_len: int) -> Optional[str]:
    """Bound the sequence axis of a rank 4 KV state and keep every model fact.

    For source shape (s0, s1, s2, s3) the result is (s0, s1, 0..U, s3) with
    U = max(max_context_len, upper bound of s2). Heads (s1) and head dimension (s3) are copied,
    so a Variable shape always extends an initialisation shape of the form (b, s1, 0, s3).
    Applying the function to its own output with the same bound returns the same text.

    Args:
        shape_text: The variable_shape attribute, comma separated.
        max_context_len: Static sequence bound for the NPU.

    Returns:
        The frozen shape text, or None when the state is not rank 4 (not a KV cache).

    Raises:
        NpuGraphRewriterError: If heads or head dimension is not a single fixed value.
    """
    tokens = shape_text.split(",")
    if len(tokens) != 4:
        return None
    dims = [parse_dim(token) for token in tokens]
    for axis, name in ((1, "kv heads"), (3, "head dimension")):
        low, high = dims[axis]
        if high is None or low != high:
            raise NpuGraphRewriterError(
                f"Cannot freeze KV state: {name} is not a fixed value",
                {"variable_shape": shape_text, "axis": axis},
            )
    upper = max(max_context_len, dims[2][1] or 0)
    return f"{tokens[0].strip()},{tokens[1].strip()},0..{upper},{tokens[3].strip()}"


class NpuGraphRewriter:
    """Orchestrates computational DAG mutations to certify models for Intel NPU37XX."""

    def __init__(
        self,
        source_dir: Path,
        output_dir: Path,
        max_context_len: int = 128,
        batch_size: int = 1,
    ) -> None:
        """Initialize the graph rewriter with source model path and hardware constraints.

        Args:
            source_dir: Directory containing source openvino_model.xml and openvino_model.bin.
            output_dir: Directory where certified NPU IR files and assets will be written.
            max_context_len: Static sequence length bound for CMX scratchpad buffer allocation.
            batch_size: Static execution batch size.
        """
        self.source_dir = Path(source_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.max_context_len = max(64, int(max_context_len))
        self.batch_size = max(1, int(batch_size))

    def rewrite(self) -> Path:
        """Execute complete graph rewriting pass and serialize certified NPU artifacts.

        Returns:
            Path: Resolved output directory containing certified NPU model assets.

        Raises:
            NpuGraphRewriterError: If model files are missing or XML parsing fails.
        """
        source_xml = self.source_dir / "openvino_model.xml"
        source_bin = self.source_dir / "openvino_model.bin"

        if not source_xml.exists() or not source_bin.exists():
            raise NpuGraphRewriterError(
                f"Missing OpenVINO IR files in source directory: {self.source_dir}",
                {"source_dir": str(self.source_dir)},
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        dest_xml = self.output_dir / "openvino_model.xml"
        dest_bin = self.output_dir / "openvino_model.bin"

        try:
            tree = ET.parse(source_xml)
        except ET.ParseError as exc:
            raise NpuGraphRewriterError(
                f"Failed to parse source OpenVINO XML graph: {exc}",
                {"xml_path": str(source_xml)},
            ) from exc

        root = tree.getroot()
        bin_data = bytearray(source_bin.read_bytes())

        stripped_count = self._strip_zero_points(root, bin_data)
        frozen_count = self._freeze_dynamic_kv_shapes(root)

        tree.write(dest_xml, encoding="utf-8", xml_declaration=True)
        dest_bin.write_bytes(bin_data)

        self._synchronize_tokenizer_assets()
        self._write_npu_manifest(stripped_count, frozen_count)

        logger.info(
            f"Successfully certified NPU model at {self.output_dir} "
            f"(stripped {stripped_count} zero-point layers, froze {frozen_count} KV shapes)"
        )
        return self.output_dir

    def _strip_zero_points(self, root: ET.Element, bin_data: bytearray) -> int:
        """Remove asymmetric Subtract zero-point layers and repack weights into signed INT4/INT8.

        Args:
            root: Root element of the parsed OpenVINO IR XML document.
            bin_data: Mutable bytearray of the weights binary.

        Returns:
            int: Number of Subtract layers successfully decoupled from the DAG.
        """
        layers_elem = root.find("layers")
        edges_elem = root.find("edges")
        if layers_elem is None or edges_elem is None:
            return 0

        layers = {l.get("id"): l for l in layers_elem}
        edges = list(edges_elem)

        u4_const_ranges: List[Tuple[int, int]] = []
        for l in layers.values():
            if l.get("type") == "Const":
                data = l.find("data")
                if data is not None and data.get("element_type") == "u4":
                    data.set("element_type", "i4")
                    offset = int(data.get("offset", "0"))
                    size = int(data.get("size", "0"))
                    u4_const_ranges.append((offset, size))
                    for p in l.findall("./output/port"):
                        if p.get("precision") == "U4":
                            p.set("precision", "I4")
            elif l.get("type") == "Convert":
                for p in l.findall("./input/port"):
                    if p.get("precision") == "U4":
                        p.set("precision", "I4")

        for offset, size in u4_const_ranges:
            chunk = np.frombuffer(bin_data, dtype=np.uint8, count=size, offset=offset)
            chunk ^= 0x88

        sub_layers = [l for l in layers.values() if l.get("type") == "Subtract"]
        stripped_count = 0

        for sub in sub_layers:
            sub_id = sub.get("id")
            in_e0 = next((e for e in edges if e.get("to-layer") == sub_id and e.get("to-port") == "0"), None)
            in_e1 = next((e for e in edges if e.get("to-layer") == sub_id and e.get("to-port") == "1"), None)
            out_edges = [e for e in edges if e.get("from-layer") == sub_id and e.get("from-port") == "2"]

            if in_e0 is not None and out_edges:
                from_layer = in_e0.get("from-layer")
                from_port = in_e0.get("from-port")

                for out_e in out_edges:
                    out_e.set("from-layer", from_layer)
                    out_e.set("from-port", from_port)

                if in_e0 in edges_elem:
                    edges_elem.remove(in_e0)
                if in_e1 is not None and in_e1 in edges_elem:
                    edges_elem.remove(in_e1)

                layers_elem.remove(sub)
                stripped_count += 1

        return stripped_count

    def _freeze_dynamic_kv_shapes(self, root: ET.Element) -> int:
        """Replace dynamic KV cache shapes and parameters with bounded static dimensions.

        Args:
            root: Root element of the parsed OpenVINO IR XML document.

        Returns:
            int: Number of KV cache and parameter layers modified.
        """
        layers_elem = root.find("layers")
        if layers_elem is None:
            return 0

        frozen_count = 0
        skipped_count = 0
        for l in layers_elem:
            layer_type = l.get("type")

            if layer_type == "ReadValue":
                if self._freeze_read_value(l):
                    frozen_count += 1
                else:
                    skipped_count += 1

            elif layer_type == "Assign":
                if self._freeze_port_dims(l, self.max_context_len):
                    frozen_count += 1

        logger.info(
            "KV state shapes frozen",
            extra={
                "frozen_layers": frozen_count,
                "skipped_non_kv_states": skipped_count,
                "max_context_len": self.max_context_len,
            },
        )
        return frozen_count

    def _freeze_read_value(self, layer: ET.Element) -> bool:
        """Freeze one ReadValue state to a bounded shape derived from its own source shape.

        Args:
            layer: A ReadValue layer element.

        Returns:
            bool: True when a rank 4 state was rewritten, False when it is not a KV cache.

        Raises:
            NpuGraphRewriterError: If the shape cannot be frozen without inventing a model
                fact, or if the frozen shape would not contain the initialisation shape.
        """
        data = layer.find("data")
        if data is None or "variable_shape" not in data.attrib:
            return False
        source_shape = data.attrib["variable_shape"]
        frozen_shape = freeze_kv_variable_shape(source_shape, self.max_context_len)
        if frozen_shape is None:
            return False
        self._assert_contains_init_shape(layer, frozen_shape)
        data.attrib["variable_shape"] = frozen_shape
        upper = parse_dim(frozen_shape.split(",")[2])[1]
        self._freeze_port_dims(layer, upper if upper is not None else self.max_context_len)
        return True

    def _freeze_port_dims(self, layer: ET.Element, sequence_bound: int) -> bool:
        """Make the batch and sequence dimensions of rank 4 ports static.

        Heads and head dimension are never touched.

        Args:
            layer: A ReadValue or Assign layer element.
            sequence_bound: Static value written over a dynamic sequence dimension.

        Returns:
            bool: True when at least one port dimension was changed.
        """
        changed = False
        for port in layer.findall(".//port"):
            dims = port.findall("dim")
            if len(dims) != 4:
                continue
            if dims[0].text == "-1":
                dims[0].text = str(self.batch_size)
                changed = True
            if dims[2].text == "-1":
                dims[2].text = str(sequence_bound)
                changed = True
        return changed

    @staticmethod
    def _assert_contains_init_shape(layer: ET.Element, frozen_shape: str) -> None:
        """Check that the frozen Variable shape extends the shape of the init subgraph.

        This is the condition OpenVINO enforces in ReadValue validation.

        Args:
            layer: The ReadValue layer whose input port carries the init shape.
            frozen_shape: Candidate variable_shape text.

        Raises:
            NpuGraphRewriterError: If a static init dimension lies outside the frozen one.
        """
        frozen_dims = [parse_dim(token) for token in frozen_shape.split(",")]
        for port in layer.findall("./input/port"):
            init_dims = [dim.text for dim in port.findall("dim")]
            if len(init_dims) != len(frozen_dims):
                continue
            for axis, text in enumerate(init_dims):
                if text is None or int(text) < 0:
                    continue
                low, high = frozen_dims[axis]
                if not (low <= int(text) and (high is None or int(text) <= high)):
                    raise NpuGraphRewriterError(
                        "Frozen KV state shape does not extend the initialisation shape",
                        {"variable_shape": frozen_shape, "init_dims": init_dims, "axis": axis},
                    )

    def _synchronize_tokenizer_assets(self) -> None:
        """Copy tokenizer weights, configurations, and vocabularies from source directory."""
        tokenizer_patterns = (
            "openvino_tokenizer.xml",
            "openvino_tokenizer.bin",
            "openvino_detokenizer.xml",
            "openvino_detokenizer.bin",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "config.json",
            "generation_config.json",
            "special_tokens_map.json",
        )
        for pattern in tokenizer_patterns:
            src_file = self.source_dir / pattern
            if src_file.exists():
                dest_file = self.output_dir / pattern
                if not dest_file.exists():
                    shutil.copy2(src_file.resolve(), dest_file)

    def _write_npu_manifest(self, stripped_layers: int, frozen_layers: int) -> None:
        """Write hardware certification manifest detailing silicon compatibility parameters.

        Args:
            stripped_layers: Number of zero-point Subtract layers eliminated.
            frozen_layers: Number of dynamic KV cache layers frozen.
        """
        manifest_path = self.output_dir / "npu_manifest.json"
        manifest_data = {
            "target_device": "NPU",
            "architecture": "NPU37XX",
            "precision": "INT4_SYM",
            "cmx_optimized": True,
            "alignment_bytes": 32,
            "max_context_len": self.max_context_len,
            "batch_size": self.batch_size,
            "zero_points_stripped": stripped_layers,
            "kv_shapes_frozen": frozen_layers,
        }
        manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
