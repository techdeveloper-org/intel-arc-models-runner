"""Intel AI Boost NPU Graph Rewriter and Computational DAG Optimizer.

Transforms OpenVINO IR XML and binary weights to meet the strict physical hardware
constraints of the Intel NPU 3720 (NPU37XX) VPUX compiler:
1. Strips unsupported asymmetric Subtract zero-point layers and repacks weights into
   symmetric signed INT4 (i4) with zero offset.
2. Freezes unbounded dynamic KV cache dimensions to static CMX-compliant bounds.
3. Synchronizes tokenizer assets and emits hardware-certified NPU manifests.
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from arc_runner.core.exceptions import ArcRunnerError
from arc_runner.core.logging import get_logger
from arc_runner.quant.zero_point_math import (
    ZeroPointMathError,
    decode_unsigned,
    encode_signed,
    is_symmetric_zero_point,
    requantize_symmetric,
    unsigned_to_signed_bytes,
)

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


CONVERSION_MODES = ("lossy", "exact", "off")
DEFAULT_CONVERSION_MODE = "lossy"
CONVERSION_MODE_ENV = "ARC_NPU_CONVERSION_MODE"
MANIFEST_SCHEMA_VERSION = 2
SIGNED_ELEMENT_TYPE = {"u4": "i4", "u8": "i8"}
UNSIGNED_ELEMENT_BITS = {"u4": 4, "u8": 8}


@dataclass
class _ConstInfo:
    """Validated description of one Const layer and its byte range in the weights binary."""

    layer: ET.Element
    element_type: str
    shape: Tuple[int, ...]
    offset: int
    size: int

    @property
    def end(self) -> int:
        """Return the exclusive end offset of the byte range."""
        return self.offset + self.size

    @property
    def count(self) -> int:
        """Return the number of elements."""
        total = 1
        for dim in self.shape:
            total *= dim
        return total


@dataclass
class _SubtractPlan:
    """Everything needed to rewrite one zero-point Subtract."""

    sub: ET.Element
    sub_ports: List[str]
    out_port: str
    weight: _ConstInfo
    weight_converts: List[ET.Element]
    zero_point: _ConstInfo
    scale: Optional[_ConstInfo]
    consumers: List[ET.Element]


@dataclass
class ConversionReport:
    """Measured outcome of the weight conversion pass."""

    exact_layers: int = 0
    requantized_layers: int = 0
    skipped_subtracts: int = 0
    unconverted_unsigned_consts: int = 0
    max_abs_error: float = 0.0
    max_rel_error: float = 0.0
    layers: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def converted_layers(self) -> int:
        """Return the number of removed Subtract layers."""
        return self.exact_layers + self.requantized_layers


class _GraphIndex:
    """One-pass index over the layers and edges of an IR document."""

    def __init__(self, root: ET.Element) -> None:
        """Index layers by id, edges by consumer port and by producer layer.

        Args:
            root: Root element of the parsed OpenVINO IR XML document.
        """
        self.layers_elem = root.find("layers")
        self.edges_elem = root.find("edges")
        self.layers: Dict[str, ET.Element] = {}
        self.incoming: Dict[Tuple[str, str], ET.Element] = {}
        self.outgoing: Dict[str, List[ET.Element]] = {}
        if self.layers_elem is None or self.edges_elem is None:
            return
        self.layers = {layer.get("id", ""): layer for layer in self.layers_elem}
        for edge in self.edges_elem:
            self.incoming[(edge.get("to-layer", ""), edge.get("to-port", ""))] = edge
            self.outgoing.setdefault(edge.get("from-layer", ""), []).append(edge)

    def input_ports(self, layer: ET.Element) -> List[str]:
        """Return the declared input port ids of a layer in document order."""
        return [port.get("id", "") for port in layer.findall("./input/port")]

    def trace_const(
        self, layer_id: str, port_id: str
    ) -> Optional[Tuple[ET.Element, List[ET.Element]]]:
        """Follow an input port upstream through Convert layers to a Const.

        Args:
            layer_id: Consumer layer id.
            port_id: Consumer input port id.

        Returns:
            The Const layer and the Convert layers on the way (nearest first), or None when
            the producer chain does not end in a Const.
        """
        converts: List[ET.Element] = []
        edge = self.incoming.get((layer_id, port_id))
        for _ in range(len(self.layers) + 1):
            if edge is None:
                return None
            source = self.layers.get(edge.get("from-layer", ""))
            if source is None:
                return None
            if source.get("type") == "Const":
                return source, converts
            if source.get("type") != "Convert":
                return None
            converts.append(source)
            ports = self.input_ports(source)
            edge = self.incoming.get((source.get("id", ""), ports[0])) if ports else None
        return None


def conversion_mode_from_env(explicit: Optional[str]) -> str:
    """Resolve the conversion mode from an explicit value or the environment.

    Args:
        explicit: Mode given by the caller, or None.

    Returns:
        One of CONVERSION_MODES.

    Raises:
        NpuGraphRewriterError: If the value is not a known mode.
    """
    mode = explicit if explicit is not None else os.environ.get(CONVERSION_MODE_ENV, DEFAULT_CONVERSION_MODE)
    if mode not in CONVERSION_MODES:
        raise NpuGraphRewriterError(
            f"Unknown conversion mode: {mode}", {"allowed": list(CONVERSION_MODES)}
        )
    return mode


def _int_attribute(layer: ET.Element, data: ET.Element, key: str) -> int:
    """Read a non-negative integer attribute or raise a structured error."""
    text = data.get(key, "")
    try:
        value = int(text)
    except ValueError as exc:
        raise NpuGraphRewriterError(
            f"Const attribute {key} is not an integer",
            {"layer": layer.get("name"), "attribute": key},
        ) from exc
    if value < 0:
        raise NpuGraphRewriterError(
            f"Const attribute {key} is negative", {"layer": layer.get("name"), "attribute": key}
        )
    return value


def read_const_info(layer: ET.Element, bin_length: int) -> _ConstInfo:
    """Validate a Const layer against the weights binary.

    Args:
        layer: A Const layer element.
        bin_length: Size in bytes of the weights binary.

    Returns:
        The validated description.

    Raises:
        NpuGraphRewriterError: On missing, malformed or out-of-range attributes, or when
            the stored size does not match shape and element type.
    """
    data = layer.find("data")
    if data is None:
        raise NpuGraphRewriterError("Const layer has no data element", {"layer": layer.get("name")})
    element_type = data.get("element_type", "")
    offset = _int_attribute(layer, data, "offset")
    size = _int_attribute(layer, data, "size")
    shape_text = data.get("shape", "").strip()
    try:
        shape = tuple(int(token) for token in shape_text.split(",")) if shape_text else ()
    except ValueError as exc:
        raise NpuGraphRewriterError(
            "Const shape is not a list of integers", {"layer": layer.get("name")}
        ) from exc
    info = _ConstInfo(layer, element_type, shape, offset, size)
    if info.end > bin_length:
        raise NpuGraphRewriterError(
            "Const byte range lies outside the weights binary",
            {"layer": layer.get("name"), "offset": offset, "size": size, "bin_length": bin_length},
        )
    expected = {"u4": info.count // 2, "u8": info.count, "f16": info.count * 2}.get(element_type)
    if expected is not None and (size != expected or (element_type == "u4" and info.count % 2)):
        raise NpuGraphRewriterError(
            "Const size does not match its shape and element type",
            {"layer": layer.get("name"), "size": size, "expected": expected},
        )
    return info


class NpuGraphRewriter:
    """Orchestrates computational DAG mutations to certify models for Intel NPU37XX."""

    def __init__(
        self,
        source_dir: Path,
        output_dir: Path,
        max_context_len: int = 128,
        batch_size: int = 1,
        conversion_mode: Optional[str] = None,
    ) -> None:
        """Initialize the graph rewriter with source model path and hardware constraints.

        Args:
            source_dir: Directory containing source openvino_model.xml and openvino_model.bin.
            output_dir: Directory where certified NPU IR files and assets will be written.
            max_context_len: Static sequence length bound for CMX scratchpad buffer allocation.
            batch_size: Static execution batch size.
            conversion_mode: lossy (exact where possible, requantise otherwise), exact (raise
                on an asymmetric zero-point) or off (leave weights alone). When None the
                environment variable ARC_NPU_CONVERSION_MODE is used, default lossy.

        Raises:
            NpuGraphRewriterError: If the conversion mode is unknown.
        """
        self.source_dir = Path(source_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.max_context_len = max(64, int(max_context_len))
        self.batch_size = max(1, int(batch_size))
        self.conversion_mode = conversion_mode_from_env(conversion_mode)

    def rewrite(self) -> Path:
        """Execute complete graph rewriting pass and serialize certified NPU artifacts.

        Nothing is written to the output directory unless the whole pass succeeds.

        Returns:
            Path: Resolved output directory containing certified NPU model assets.

        Raises:
            NpuGraphRewriterError: If model files are missing, the XML is malformed, a Const
                range is invalid, or a weight conversion cannot be done as requested.
        """
        source_xml = self.source_dir / "openvino_model.xml"
        source_bin = self.source_dir / "openvino_model.bin"

        if not source_xml.exists() or not source_bin.exists():
            raise NpuGraphRewriterError(
                f"Missing OpenVINO IR files in source directory: {self.source_dir}",
                {"source_dir": str(self.source_dir)},
            )

        try:
            tree = ET.parse(source_xml)
        except ET.ParseError as exc:
            raise NpuGraphRewriterError(
                f"Failed to parse source OpenVINO XML graph: {exc}",
                {"xml_path": str(source_xml)},
            ) from exc

        root = tree.getroot()
        bin_data = bytearray(source_bin.read_bytes())

        report = self._convert_weight_groups(root, bin_data)
        frozen_count = self._freeze_dynamic_kv_shapes(root)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        dest_xml = self.output_dir / "openvino_model.xml"
        dest_bin = self.output_dir / "openvino_model.bin"
        tree.write(dest_xml, encoding="utf-8", xml_declaration=True)
        dest_bin.write_bytes(bin_data)

        self._synchronize_tokenizer_assets()
        self._write_npu_manifest(report, frozen_count)

        logger.info(
            "NPU model certified",
            extra={
                "conversion_mode": self.conversion_mode,
                "exact_layers": report.exact_layers,
                "requantized_layers": report.requantized_layers,
                "kv_shapes_frozen": frozen_count,
                "max_abs_error": report.max_abs_error,
            },
        )
        return self.output_dir

    def _convert_weight_groups(self, root: ET.Element, bin_data: bytearray) -> ConversionReport:
        """Remove zero-point Subtract layers and store the weights as signed values.

        Only a Subtract whose inputs trace back to two Const layers of the same unsigned
        type (u4 or u8) is touched. A symmetric zero-point is converted exactly; any other
        zero-point is requantised (lossy mode) or rejected (exact mode).

        Args:
            root: Root element of the parsed OpenVINO IR XML document.
            bin_data: Mutable bytearray of the weights binary.

        Returns:
            The measured conversion report.

        Raises:
            NpuGraphRewriterError: On invalid ranges, shared constants that would be
                modified twice, overlapping ranges, or a conversion the mode forbids.
        """
        report = ConversionReport()
        index = _GraphIndex(root)
        if index.layers_elem is None or index.edges_elem is None:
            return report
        removed_layers: Set[str] = set()
        removed_edges: Set[int] = set()
        out_count = {layer_id: len(edges) for layer_id, edges in index.outgoing.items()}
        claims: List[Tuple[int, int, str]] = []

        if self.conversion_mode != "off":
            for sub in [l for l in index.layers_elem if l.get("type") == "Subtract"]:
                plan = self._plan_subtract(index, sub, len(bin_data), out_count)
                if plan is None:
                    report.skipped_subtracts += 1
                    continue
                self._convert_one(plan, bin_data, report, claims)
                self._detach_subtract(index, plan, removed_layers, removed_edges, out_count)

        self._assert_disjoint(claims)
        index.layers_elem[:] = [l for l in index.layers_elem if l.get("id") not in removed_layers]
        index.edges_elem[:] = [e for e in index.edges_elem if id(e) not in removed_edges]
        report.unconverted_unsigned_consts = sum(
            1
            for l in index.layers_elem
            if l.get("type") == "Const"
            and l.find("data") is not None
            and l.find("data").get("element_type") in UNSIGNED_ELEMENT_BITS
        )
        return report

    def _plan_subtract(
        self,
        index: _GraphIndex,
        sub: ET.Element,
        bin_length: int,
        out_count: Dict[str, int],
    ) -> Optional[_SubtractPlan]:
        """Decide whether a Subtract is a weight zero-point and collect its operands.

        Args:
            index: Graph index of the document.
            sub: A Subtract layer.
            bin_length: Size of the weights binary in bytes.
            out_count: Outgoing edge count per layer id.

        Returns:
            A plan, or None when the Subtract is not an unsigned weight minus zero-point.

        Raises:
            NpuGraphRewriterError: If a constant that must be rewritten has other consumers.
        """
        sub_id = sub.get("id", "")
        ports = index.input_ports(sub)
        out_ports = sub.findall("./output/port")
        if len(ports) != 2 or len(out_ports) != 1:
            return None
        weight_trace = index.trace_const(sub_id, ports[0])
        zero_trace = index.trace_const(sub_id, ports[1])
        if weight_trace is None or zero_trace is None:
            return None
        types = [
            (trace[0].find("data").get("element_type") if trace[0].find("data") is not None else "")
            for trace in (weight_trace, zero_trace)
        ]
        if types[0] != types[1] or types[0] not in UNSIGNED_ELEMENT_BITS:
            return None
        weight = read_const_info(weight_trace[0], bin_length)
        zero_point = read_const_info(zero_trace[0], bin_length)
        self._assert_exclusive(weight_trace[0], weight_trace[1], out_count)
        out_port = out_ports[0].get("id", "")
        consumers = [e for e in index.outgoing.get(sub_id, []) if e.get("from-port") == out_port]
        scale = self._find_scale(index, consumers, bin_length, out_count)
        return _SubtractPlan(sub, ports, out_port, weight, weight_trace[1], zero_point, scale, consumers)

    def _find_scale(
        self,
        index: _GraphIndex,
        consumers: List[ET.Element],
        bin_length: int,
        out_count: Dict[str, int],
    ) -> Optional[_ConstInfo]:
        """Locate the float16 scale Const of the Multiply that consumes a Subtract.

        Args:
            index: Graph index of the document.
            consumers: Edges leaving the Subtract output port.
            bin_length: Size of the weights binary in bytes.
            out_count: Outgoing edge count per layer id.

        Returns:
            The scale description, or None when there is no Multiply with a f16 Const scale.

        Raises:
            NpuGraphRewriterError: If the scale Const has other consumers.
        """
        if len(consumers) != 1:
            return None
        mul = index.layers.get(consumers[0].get("to-layer", ""))
        if mul is None or mul.get("type") != "Multiply":
            return None
        ports = index.input_ports(mul)
        if len(ports) != 2 or consumers[0].get("to-port") != ports[0]:
            return None
        trace = index.trace_const(mul.get("id", ""), ports[1])
        if trace is None:
            return None
        data = trace[0].find("data")
        if data is None or data.get("element_type") != "f16":
            return None
        scale = read_const_info(trace[0], bin_length)
        self._assert_exclusive(trace[0], trace[1], out_count)
        return scale

    @staticmethod
    def _assert_exclusive(const: ET.Element, converts: List[ET.Element], out_count: Dict[str, int]) -> None:
        """Require that a Const and its Convert chain feed exactly one consumer each.

        Args:
            const: The Const layer that will be modified.
            converts: Convert layers between the Const and its consumer.
            out_count: Outgoing edge count per layer id.

        Raises:
            NpuGraphRewriterError: If any of them is shared, because the modification would
                change another part of the graph.
        """
        for layer in [const] + converts:
            if out_count.get(layer.get("id", ""), 0) != 1:
                raise NpuGraphRewriterError(
                    "A constant that must be rewritten is shared by several consumers",
                    {"layer": layer.get("name")},
                )

    def _convert_one(
        self,
        plan: _SubtractPlan,
        bin_data: bytearray,
        report: ConversionReport,
        claims: List[Tuple[int, int, str]],
    ) -> None:
        """Rewrite the weight (and scale) bytes of one Subtract in place.

        Args:
            plan: The collected operands.
            bin_data: Mutable weights binary.
            report: Report that receives the measured numbers.
            claims: Byte ranges written so far, used for the overlap check.

        Raises:
            NpuGraphRewriterError: If the mode forbids the conversion or it is not possible.
        """
        weight, zero = plan.weight, plan.zero_point
        element_type = weight.element_type
        name = weight.layer.get("name")
        claims.append((weight.offset, weight.end, f"{name}:weights"))
        zero_raw = np.frombuffer(bytes(bin_data[zero.offset:zero.end]), dtype=np.uint8)
        entry: Dict[str, Any] = {"name": name, "element_type": element_type}
        try:
            if is_symmetric_zero_point(zero_raw, element_type):
                new_bytes = unsigned_to_signed_bytes(bytes(bin_data[weight.offset:weight.end]), element_type)
                bin_data[weight.offset:weight.end] = new_bytes
                report.exact_layers += 1
                entry.update({"mode": "exact", "max_abs_error": 0.0, "max_rel_error": 0.0})
            else:
                self._requantize(plan, bin_data, report, claims, entry)
        except ZeroPointMathError as exc:
            raise NpuGraphRewriterError(
                f"Weight conversion failed: {exc}", {"layer": name, "element_type": element_type}
            ) from exc
        report.layers.append(entry)
        self._mark_signed(plan)

    def _requantize(
        self,
        plan: _SubtractPlan,
        bin_data: bytearray,
        report: ConversionReport,
        claims: List[Tuple[int, int, str]],
        entry: Dict[str, Any],
    ) -> None:
        """Requantise one asymmetric weight group set and write weights and scales back.

        Args:
            plan: The collected operands.
            bin_data: Mutable weights binary.
            report: Report that receives the measured numbers.
            claims: Byte ranges written so far.
            entry: Per-layer report entry to fill.

        Raises:
            NpuGraphRewriterError: If exact mode is active or there is no scale to rewrite.
            ZeroPointMathError: If the numerical preconditions fail.
        """
        weight, zero, scale = plan.weight, plan.zero_point, plan.scale
        name = weight.layer.get("name")
        if self.conversion_mode == "exact":
            raise NpuGraphRewriterError(
                "Asymmetric zero-point cannot be removed exactly",
                {"layer": name, "hint": f"set {CONVERSION_MODE_ENV}=lossy to requantise"},
            )
        if scale is None:
            raise NpuGraphRewriterError(
                "Asymmetric zero-point without a float16 scale Const cannot be requantised",
                {"layer": name},
            )
        bits = UNSIGNED_ELEMENT_BITS[weight.element_type]
        q = decode_unsigned(bytes(bin_data[weight.offset:weight.end]), weight.element_type, weight.count)
        z = decode_unsigned(bytes(bin_data[zero.offset:zero.end]), zero.element_type, zero.count)
        s = np.frombuffer(bytes(bin_data[scale.offset:scale.end]), dtype=np.float16).copy()
        result = requantize_symmetric(q.reshape(weight.shape), z.reshape(zero.shape), s.reshape(scale.shape), bits)
        bin_data[weight.offset:weight.end] = encode_signed(result.q, bits)
        bin_data[scale.offset:scale.end] = result.scale.tobytes()
        claims.append((scale.offset, scale.end, f"{name}:scale"))
        report.requantized_layers += 1
        report.max_abs_error = max(report.max_abs_error, result.max_abs_error)
        report.max_rel_error = max(report.max_rel_error, result.max_rel_error)
        entry.update(
            {"mode": "requantized", "max_abs_error": result.max_abs_error, "max_rel_error": result.max_rel_error}
        )

    @staticmethod
    def _mark_signed(plan: _SubtractPlan) -> None:
        """Relabel the weight Const and its Convert inputs from unsigned to signed storage.

        Args:
            plan: The collected operands.
        """
        signed = SIGNED_ELEMENT_TYPE[plan.weight.element_type]
        plan.weight.layer.find("data").set("element_type", signed)
        for port in plan.weight.layer.findall("./output/port"):
            port.set("precision", signed.upper())
        for convert in plan.weight_converts:
            for port in convert.findall("./input/port"):
                port.set("precision", signed.upper())

    @staticmethod
    def _detach_subtract(
        index: _GraphIndex,
        plan: _SubtractPlan,
        removed_layers: Set[str],
        removed_edges: Set[int],
        out_count: Dict[str, int],
    ) -> None:
        """Bypass the Subtract and drop the zero-point chain once nothing else uses it.

        Args:
            index: Graph index of the document.
            plan: The collected operands.
            removed_layers: Layer ids scheduled for removal (updated).
            removed_edges: Identities of edges scheduled for removal (updated).
            out_count: Outgoing edge count per layer id (updated).
        """
        sub_id = plan.sub.get("id", "")
        weight_edge = index.incoming[(sub_id, plan.sub_ports[0])]
        zero_edge = index.incoming[(sub_id, plan.sub_ports[1])]
        for consumer in plan.consumers:
            consumer.set("from-layer", weight_edge.get("from-layer", ""))
            consumer.set("from-port", weight_edge.get("from-port", ""))
        out_count[weight_edge.get("from-layer", "")] = out_count.get(weight_edge.get("from-layer", ""), 0)
        removed_edges.add(id(weight_edge))
        removed_edges.add(id(zero_edge))
        removed_layers.add(sub_id)
        edge = zero_edge
        for _ in range(len(index.layers) + 1):
            source_id = edge.get("from-layer", "")
            out_count[source_id] = out_count.get(source_id, 1) - 1
            if out_count[source_id] > 0:
                break
            removed_layers.add(source_id)
            source = index.layers[source_id]
            ports = index.input_ports(source)
            if source.get("type") != "Convert" or not ports:
                break
            edge = index.incoming[(source_id, ports[0])]
            removed_edges.add(id(edge))

    @staticmethod
    def _assert_disjoint(claims: List[Tuple[int, int, str]]) -> None:
        """Require that no two rewritten byte ranges overlap.

        Args:
            claims: Rewritten (start, end, label) ranges.

        Raises:
            NpuGraphRewriterError: If two ranges overlap, which would apply a rewrite twice.
        """
        ordered = sorted(claims)
        for previous, current in itertools.pairwise(ordered):
            if current[0] < previous[1]:
                raise NpuGraphRewriterError(
                    "Rewritten weight ranges overlap",
                    {"first": previous[2], "second": current[2]},
                )


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

    def _write_npu_manifest(self, report: ConversionReport, frozen_layers: int) -> None:
        """Write the certification manifest with only measured values.

        Args:
            report: Measured weight conversion outcome.
            frozen_layers: Number of dynamic KV cache layers frozen.
        """
        manifest_path = self.output_dir / "npu_manifest.json"
        requantized = report.requantized_layers > 0
        manifest_data = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "target_device": "NPU",
            "architecture": "NPU37XX",
            "conversion_mode": self.conversion_mode,
            "max_context_len": self.max_context_len,
            "batch_size": self.batch_size,
            "zero_points_stripped": report.converted_layers,
            "kv_shapes_frozen": frozen_layers,
            "certified_exact": self.conversion_mode != "off" and not requantized,
            "quality": "unmeasured" if requantized else "exact",
            "weight_conversion": {
                "exact_layers": report.exact_layers,
                "requantized_layers": report.requantized_layers,
                "skipped_subtracts": report.skipped_subtracts,
                "unconverted_unsigned_consts": report.unconverted_unsigned_consts,
                "max_abs_error": report.max_abs_error,
                "max_rel_error": report.max_rel_error,
                "layers": report.layers,
            },
        }
        manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
