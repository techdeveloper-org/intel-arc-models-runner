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
import mmap
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


def freeze_state_variable_shape(shape_text: str, growth_axis: int, max_context_len: int) -> str:
    """Bound the growth axis of a stateful variable and keep every other model fact.

    The growth axis becomes 0..U with U = max(max_context_len, its own upper bound).
    Dimension 0 (batch) is copied. Every other dimension must be a single fixed value, because
    freezing it would otherwise invent a model fact.

    Args:
        shape_text: The variable_shape attribute, comma separated.
        growth_axis: Index of the axis along which the state grows.
        max_context_len: Static sequence bound for the NPU.

    Returns:
        The frozen shape text.

    Raises:
        NpuGraphRewriterError: If the axis is out of range or a non-batch, non-growth
            dimension is not a fixed value.
    """
    tokens = [token.strip() for token in shape_text.split(",")]
    if not 0 <= growth_axis < len(tokens):
        raise NpuGraphRewriterError(
            "Growth axis lies outside the variable rank",
            {"variable_shape": shape_text, "axis": growth_axis},
        )
    dims = [parse_dim(token) for token in tokens]
    for axis, (low, high) in enumerate(dims):
        if axis in (0, growth_axis):
            continue
        if high is None or low != high:
            raise NpuGraphRewriterError(
                "Cannot freeze state: a non-growth dimension is not a fixed value",
                {"variable_shape": shape_text, "axis": axis},
            )
    upper = max(max_context_len, dims[growth_axis][1] or 0)
    tokens[growth_axis] = f"0..{upper}"
    return ",".join(tokens)


def freeze_kv_variable_shape(shape_text: str, max_context_len: int) -> Optional[str]:
    """Bound the sequence axis (axis 2) of a rank 4 KV state.

    Kept for the common [batch, heads, sequence, head_dim] layout; the rewriter itself uses
    freeze_state_variable_shape with an axis derived from the graph.

    Args:
        shape_text: The variable_shape attribute, comma separated.
        max_context_len: Static sequence bound for the NPU.

    Returns:
        The frozen shape text, or None when the state is not rank 4.

    Raises:
        NpuGraphRewriterError: If heads or head dimension is not a single fixed value.
    """
    if len(shape_text.split(",")) != 4:
        return None
    return freeze_state_variable_shape(shape_text, 2, max_context_len)


def find_growth_axis(
    variable_dims: List[Tuple[int, Optional[int]]],
    init_dims: List[Optional[int]],
    concat_axis: Optional[int],
) -> Optional[int]:
    """Pick the axis along which a state grows, from the graph rather than from its rank.

    A candidate axis (never axis 0) has a static initial size of 0. When the ReadValue feeds a Concat, that Concat axis must be the candidate.

    Args:
        variable_dims: Parsed (low, high) bounds of every variable dimension.
        init_dims: Static size of every initialisation dimension, None when dynamic.
        concat_axis: Normalised Concat axis downstream of the ReadValue, or None.

    Returns:
        The growth axis, or None when the state does not grow (not a cache).

    Raises:
        NpuGraphRewriterError: If the candidates are ambiguous or disagree with the Concat axis.
    """
    candidates = [
        axis
        for axis in range(len(variable_dims))
        if axis > 0 and init_dims[axis] == 0
    ]
    if concat_axis is not None and candidates:
        if concat_axis in candidates:
            return concat_axis
        raise NpuGraphRewriterError(
            "Concat axis does not match the growing dimension of the state",
            {"concat_axis": concat_axis, "candidates": candidates},
        )
    if len(candidates) > 1:
        raise NpuGraphRewriterError(
            "Several dimensions of the state could be the growth axis",
            {"candidates": candidates},
        )
    return candidates[0] if candidates else None


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


@dataclass
class BoundednessReport:
    """Result of reading a rewritten model back and checking every state dimension.

    Attributes:
        variables_checked: Number of state variables inspected.
        unbounded_variables: Variable ids with a non-batch dimension that has no finite maximum.
        unbounded_parameters: Names of model inputs with a dimension without a finite maximum;
            informational, because inputs are bounded at run time by reshaping.
    """

    variables_checked: int
    unbounded_variables: List[str]
    unbounded_parameters: List[str]


def _has_unbounded_dimension(partial_shape: Any, skip_batch: bool = False) -> bool:
    """Tell whether any dimension of an OpenVINO PartialShape lacks a finite maximum.

    Args:
        partial_shape: The shape to inspect.
        skip_batch: Ignore axis 0, which the rewriter copies unchanged by design.

    Returns:
        True when the rank is dynamic or a checked dimension has no finite maximum.
    """
    if partial_shape.rank.is_dynamic:
        return True
    dims = list(partial_shape)[1:] if skip_batch else list(partial_shape)
    return any((not dim.is_static) and dim.get_max_length() < 0 for dim in dims)


def verify_bounded(model_xml: Path) -> BoundednessReport:
    """Read a rewritten IR back with OpenVINO and check that every state is bounded.

    This is a structural check on the parsed model. It does not compile for the NPU and is
    therefore not a guarantee that the NPU compiler accepts the model.

    Args:
        model_xml: Path to the rewritten openvino_model.xml.

    Returns:
        The boundedness report.

    Raises:
        NpuGraphRewriterError: If OpenVINO is unavailable or the model cannot be read.
    """
    try:
        import openvino as ov
    except ImportError as exc:
        raise NpuGraphRewriterError("OpenVINO is required to verify boundedness", {}) from exc
    try:
        model = ov.Core().read_model(str(model_xml))
    except RuntimeError as exc:
        raise NpuGraphRewriterError(
            f"OpenVINO could not read the rewritten model: {exc}", {"xml_path": str(model_xml)}
        ) from exc
    variables = list(model.get_variables())
    unbounded_variables = [
        variable.get_info().variable_id
        for variable in variables
        if _has_unbounded_dimension(variable.get_info().data_shape, skip_batch=True)
    ]
    unbounded_parameters = [
        port.get_any_name() for port in model.inputs if _has_unbounded_dimension(port.get_partial_shape())
    ]
    del model
    return BoundednessReport(len(variables), unbounded_variables, unbounded_parameters)


class NpuGraphRewriter:
    """Orchestrates computational DAG mutations to certify models for Intel NPU37XX."""

    def __init__(
        self,
        source_dir: Path,
        output_dir: Path,
        max_context_len: int = 128,
        batch_size: int = 1,
        conversion_mode: Optional[str] = None,
        verify: bool = False,
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
            verify: Read the rewritten model back with OpenVINO and fail when any state
                variable is still unbounded.

        Raises:
            NpuGraphRewriterError: If the conversion mode is unknown.
        """
        self.source_dir = Path(source_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.max_context_len = max(64, int(max_context_len))
        self.batch_size = max(1, int(batch_size))
        self.conversion_mode = conversion_mode_from_env(conversion_mode)
        self.verify = bool(verify)

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

        signature = self._source_signature(source_xml, source_bin)
        if self._is_up_to_date(signature):
            logger.info("NPU model already certified", extra={"output_dir": self.output_dir.name})
            return self.output_dir

        needed = 2 * source_bin.stat().st_size
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.output_dir.parent).free
        if free < needed:
            raise NpuGraphRewriterError(
                "Not enough free disk space for the rewritten weights",
                {"needed_bytes": needed, "free_bytes": free},
            )

        stage = self.output_dir.parent / f".{self.output_dir.name}.tmp-{os.getpid()}"
        shutil.rmtree(stage, ignore_errors=True)
        stage.mkdir(parents=True)
        try:
            root = tree.getroot()
            stage_bin = stage / "openvino_model.bin"
            shutil.copyfile(source_bin, stage_bin)
            if stage_bin.stat().st_size == 0:
                report = self._convert_weight_groups(root, bytearray())
            else:
                with open(stage_bin, "r+b") as handle, mmap.mmap(handle.fileno(), 0) as bin_data:
                    report = self._convert_weight_groups(root, bin_data)
            frozen_count = self._freeze_dynamic_kv_shapes(root)
            tree.write(stage / "openvino_model.xml", encoding="utf-8", xml_declaration=True)
            if self.verify:
                self._verify_stage(stage / "openvino_model.xml")
            self._synchronize_tokenizer_assets(stage)
            self._write_npu_manifest(report, frozen_count, stage, signature)
            shutil.rmtree(self.output_dir, ignore_errors=True)
            os.replace(stage, self.output_dir)
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise

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

    def _convert_weight_groups(self, root: ET.Element, bin_data: Any) -> ConversionReport:
        """Remove zero-point Subtract layers and store the weights as signed values.

        Only a Subtract whose inputs trace back to two Const layers of the same unsigned
        type (u4 or u8) is touched. A symmetric zero-point is converted exactly; any other
        zero-point is requantised (lossy mode) or rejected (exact mode).

        Args:
            root: Root element of the parsed OpenVINO IR XML document.
            bin_data: Mutable memory map (or bytearray) of the weights binary.

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
        bin_data: Any,
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
        bin_data: Any,
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
        """Bound every growing state variable (and its Assign) to a static sequence length.

        The growth axis is derived from the graph per ReadValue, and the bound chosen for a
        variable is reused for its Assign so both ends agree.

        Args:
            root: Root element of the parsed OpenVINO IR XML document.

        Returns:
            int: Number of state layers modified.

        Raises:
            NpuGraphRewriterError: If a growth axis is ambiguous or a model fact would have to
                be invented to freeze a state.
        """
        index = _GraphIndex(root)
        if index.layers_elem is None:
            return 0

        bounds: Dict[str, Tuple[int, int]] = {}
        frozen_count = 0
        skipped_count = 0
        for layer in index.layers_elem:
            if layer.get("type") != "ReadValue":
                continue
            result = self._freeze_read_value(layer, index)
            if result is None:
                skipped_count += 1
                continue
            data = layer.find("data")
            bounds[data.get("variable_id", "") if data is not None else ""] = result
            frozen_count += 1

        for layer in index.layers_elem:
            if layer.get("type") != "Assign":
                continue
            data = layer.find("data")
            known = bounds.get(data.get("variable_id", "") if data is not None else "")
            if known is not None and self._freeze_port_dims(layer, known[0], known[1]):
                frozen_count += 1

        logger.info(
            "State shapes frozen",
            extra={
                "frozen_layers": frozen_count,
                "skipped_non_growing_states": skipped_count,
                "max_context_len": self.max_context_len,
            },
        )
        return frozen_count

    @staticmethod
    def _concat_axis_downstream(index: _GraphIndex, layer: ET.Element, rank: int) -> Optional[int]:
        """Find the axis of the Concat that appends to a state, looking through Gather/Convert.

        Args:
            index: Graph index of the document.
            layer: A ReadValue layer.
            rank: Rank of the state, used to normalise a negative axis.

        Returns:
            The non-negative Concat axis, or None when no Concat follows the state.

        Raises:
            NpuGraphRewriterError: If the axis attribute is not an integer or out of range.
        """
        frontier = [layer.get("id", "")]
        for _ in range(4):
            next_frontier: List[str] = []
            for layer_id in frontier:
                for edge in index.outgoing.get(layer_id, []):
                    target = index.layers.get(edge.get("to-layer", ""))
                    if target is None:
                        continue
                    if target.get("type") == "Concat":
                        return NpuGraphRewriter._normalise_axis(target, rank)
                    if target.get("type") in ("Gather", "Convert"):
                        next_frontier.append(target.get("id", ""))
            frontier = next_frontier
        return None

    @staticmethod
    def _normalise_axis(concat: ET.Element, rank: int) -> int:
        """Return the Concat axis as a non-negative index.

        Args:
            concat: A Concat layer.
            rank: Rank of the concatenated tensors.

        Returns:
            The normalised axis.

        Raises:
            NpuGraphRewriterError: If the axis is missing, not an integer or out of range.
        """
        data = concat.find("data")
        text = data.get("axis", "") if data is not None else ""
        try:
            axis = int(text)
        except ValueError as exc:
            raise NpuGraphRewriterError(
                "Concat axis is not an integer", {"layer": concat.get("name"), "axis": text}
            ) from exc
        axis = axis + rank if axis < 0 else axis
        if not 0 <= axis < rank:
            raise NpuGraphRewriterError(
                "Concat axis lies outside the state rank", {"layer": concat.get("name"), "rank": rank}
            )
        return axis

    def _freeze_read_value(self, layer: ET.Element, index: _GraphIndex) -> Optional[Tuple[int, int]]:
        """Freeze one ReadValue state to a bounded shape derived from the graph.

        Args:
            layer: A ReadValue layer element.
            index: Graph index of the document.

        Returns:
            (growth axis, sequence bound) when the state was rewritten, None when it does not grow.

        Raises:
            NpuGraphRewriterError: If the shape cannot be frozen without inventing a model
                fact, or if the frozen shape would not contain the initialisation shape.
        """
        data = layer.find("data")
        if data is None or "variable_shape" not in data.attrib:
            return None
        source_shape = data.attrib["variable_shape"]
        tokens = source_shape.split(",")
        variable_dims = [parse_dim(token) for token in tokens]
        init_dims = self._init_dims(layer, len(tokens))
        concat_axis = self._concat_axis_downstream(index, layer, len(tokens))
        axis = find_growth_axis(variable_dims, init_dims, concat_axis)
        if axis is None:
            return None
        frozen_shape = freeze_state_variable_shape(source_shape, axis, self.max_context_len)
        self._assert_contains_init_shape(layer, frozen_shape)
        data.attrib["variable_shape"] = frozen_shape
        upper = parse_dim(frozen_shape.split(",")[axis])[1]
        bound = upper if upper is not None else self.max_context_len
        self._freeze_port_dims(layer, axis, bound)
        return axis, bound

    @staticmethod
    def _init_dims(layer: ET.Element, rank: int) -> List[Optional[int]]:
        """Read the static sizes of the initialisation port, None where dynamic or unknown.

        Args:
            layer: A ReadValue layer element.
            rank: Rank of the variable shape.

        Returns:
            One entry per axis.

        Raises:
            NpuGraphRewriterError: If a dimension is not an integer.
        """
        dims: List[Optional[int]] = [None] * rank
        for port in layer.findall("./input/port"):
            texts = [dim.text for dim in port.findall("dim")]
            if len(texts) != rank:
                continue
            for axis, text in enumerate(texts):
                try:
                    value = int(text) if text is not None else -1
                except ValueError as exc:
                    raise NpuGraphRewriterError(
                        "Initialisation dimension is not an integer",
                        {"layer": layer.get("name"), "axis": axis},
                    ) from exc
                dims[axis] = value if value >= 0 else None
        return dims

    def _freeze_port_dims(self, layer: ET.Element, growth_axis: int, sequence_bound: int) -> bool:
        """Make the batch and growth dimensions of the state ports static.

        Every other dimension is never touched.

        Args:
            layer: A ReadValue or Assign layer element.
            growth_axis: Axis that receives sequence_bound when dynamic.
            sequence_bound: Static value written over a dynamic growth dimension.

        Returns:
            bool: True when at least one port dimension was changed.
        """
        changed = False
        for port in layer.findall(".//port"):
            dims = port.findall("dim")
            if len(dims) <= growth_axis:
                continue
            if dims[0].text == "-1":
                dims[0].text = str(self.batch_size)
                changed = True
            if dims[growth_axis].text == "-1":
                dims[growth_axis].text = str(sequence_bound)
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
                try:
                    value = int(text) if text is not None else -1
                except ValueError as exc:
                    raise NpuGraphRewriterError(
                        "Initialisation dimension is not an integer",
                        {"layer": layer.get("name"), "axis": axis},
                    ) from exc
                if value < 0:
                    continue
                low, high = frozen_dims[axis]
                if not (low <= value and (high is None or value <= high)):
                    raise NpuGraphRewriterError(
                        "Frozen KV state shape does not extend the initialisation shape",
                        {"variable_shape": frozen_shape, "init_dims": init_dims, "axis": axis},
                    )

    @staticmethod
    def _verify_stage(stage_xml: Path) -> None:
        """Fail the rewrite when the staged model still has an unbounded state variable.

        Args:
            stage_xml: Staged rewritten XML path.

        Raises:
            NpuGraphRewriterError: If a state variable is unbounded.
        """
        report = verify_bounded(stage_xml)
        if report.unbounded_variables:
            raise NpuGraphRewriterError(
                "Rewritten model still has unbounded state variables",
                {"variables": report.unbounded_variables[:10], "count": len(report.unbounded_variables)},
            )

    def _source_signature(self, source_xml: Path, source_bin: Path) -> Dict[str, Any]:
        """Describe the inputs and settings that determine the output.

        Args:
            source_xml: Source IR XML path.
            source_bin: Source IR weights path.

        Returns:
            A JSON-serialisable dictionary compared against the stored manifest.
        """
        return {
            "xml_size": source_xml.stat().st_size,
            "xml_mtime_ns": source_xml.stat().st_mtime_ns,
            "bin_size": source_bin.stat().st_size,
            "bin_mtime_ns": source_bin.stat().st_mtime_ns,
            "conversion_mode": self.conversion_mode,
            "max_context_len": self.max_context_len,
            "batch_size": self.batch_size,
        }

    def _is_up_to_date(self, signature: Dict[str, Any]) -> bool:
        """Tell whether the output directory was built from the same inputs and settings.

        Args:
            signature: Signature of the current inputs.

        Returns:
            True when a complete, current-schema output exists.
        """
        manifest_path = self.output_dir / "npu_manifest.json"
        if not manifest_path.exists():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        files_present = all((self.output_dir / n).exists() for n in ("openvino_model.xml", "openvino_model.bin"))
        return (
            files_present
            and manifest.get("schema_version") == MANIFEST_SCHEMA_VERSION
            and manifest.get("source_signature") == signature
        )

    def _synchronize_tokenizer_assets(self, target_dir: Path) -> None:
        """Copy tokenizer weights, configurations, and vocabularies from source directory.

        Args:
            target_dir: Directory receiving the copies; existing files are overwritten so
                stale assets never survive a re-run.
        """
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
                shutil.copy2(src_file.resolve(), target_dir / pattern)

    def _write_npu_manifest(
        self, report: ConversionReport, frozen_layers: int, target_dir: Path, signature: Dict[str, Any]
    ) -> None:
        """Write the certification manifest with only measured values.

        Args:
            report: Measured weight conversion outcome.
            frozen_layers: Number of dynamic KV cache layers frozen.
            target_dir: Directory that receives the manifest.
            signature: Input signature stored for the up-to-date check.
        """
        manifest_path = target_dir / "npu_manifest.json"
        requantized = report.requantized_layers > 0
        manifest_data = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "target_device": "NPU",
            "architecture": "NPU37XX",
            "conversion_mode": self.conversion_mode,
            "source_signature": signature,
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
