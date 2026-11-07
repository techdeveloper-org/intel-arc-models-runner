"""Tests for zero-point removal in NpuGraphRewriter on synthetic weight-compression graphs.

The graph mirrors the structure measured on Qwen3-0.6B:
Const(u4|u8) -> Convert -> Subtract <- Convert <- Const(zero point), Subtract -> Multiply <- Const(f16 scale).
Oracles are written independently of the implementation (float64 dequantisation).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from arc_runner.quant.npu_graph_rewriter import NpuGraphRewriter, NpuGraphRewriterError


def _dims(shape: Tuple[int, ...]) -> str:
    return "".join(f"<dim>{d}</dim>" for d in shape)


def _port(port_id: str, precision: str, shape: Tuple[int, ...]) -> str:
    return f'<port id="{port_id}" precision="{precision}">{_dims(shape)}</port>'


def _const(layer_id: str, name: str, etype: str, shape: Tuple[int, ...], offset: int, size: int) -> str:
    shape_text = ", ".join(str(d) for d in shape)
    precision = etype.upper() if etype != "f16" else "FP16"
    return (
        f'<layer id="{layer_id}" name="{name}" type="Const" version="opset1">'
        f'<data element_type="{etype}" shape="{shape_text}" offset="{offset}" size="{size}"/>'
        f"<output>{_port('0', precision, shape)}</output></layer>"
    )


def _convert(layer_id: str, name: str, in_precision: str, shape: Tuple[int, ...]) -> str:
    return (
        f'<layer id="{layer_id}" name="{name}" type="Convert" version="opset1">'
        f'<data destination_type="f16"/>'
        f"<input>{_port('0', in_precision, shape)}</input>"
        f"<output>{_port('1', 'FP16', shape)}</output></layer>"
    )


def _binary(layer_id: str, name: str, kind: str, in_ports: Tuple[str, str], out_port: str,
            shape: Tuple[int, ...]) -> str:
    return (
        f'<layer id="{layer_id}" name="{name}" type="{kind}" version="opset1">'
        f"<input>{_port(in_ports[0], 'FP16', shape)}{_port(in_ports[1], 'FP16', shape)}</input>"
        f"<output>{_port(out_port, 'FP16', shape)}</output></layer>"
    )


def _edge(src: str, src_port: str, dst: str, dst_port: str) -> str:
    return f'<edge from-layer="{src}" from-port="{src_port}" to-layer="{dst}" to-port="{dst_port}"/>'


def _pack(values: np.ndarray, etype: str) -> bytes:
    flat = [int(v) for v in values.ravel()]
    if etype == "u8":
        return bytes(flat)
    return bytes(flat[i] | (flat[i + 1] << 4) for i in range(0, len(flat), 2))


def _unpack_signed(raw: bytes, bits: int) -> np.ndarray:
    if bits == 8:
        return np.frombuffer(raw, dtype=np.int8).astype(np.int64)
    out = []
    for byte in raw:
        for nib in (byte & 0x0F, byte >> 4):
            out.append(nib if nib < 8 else nib - 16)
    return np.array(out, dtype=np.int64)


@dataclass
class Graph:
    source: Path
    w_shape: Tuple[int, ...]
    etype: str
    q: np.ndarray
    zp: np.ndarray
    scale: np.ndarray
    w_offset: int
    z_offset: int
    s_offset: int


def make_graph(
    tmp_path: Path,
    q: np.ndarray,
    zp: np.ndarray,
    scale: np.ndarray,
    etype: str = "u4",
    with_multiply: bool = True,
    sub_ports: Tuple[str, str, str] = ("0", "1", "2"),
    mul_ports: Tuple[str, str, str] = ("0", "1", "2"),
    w_offset: Optional[int] = None,
    extra_layers: str = "",
    extra_edges: str = "",
) -> Graph:
    """Write openvino_model.xml/.bin for one compressed weight with its Subtract."""
    wb, zb, sb = _pack(q, etype), _pack(zp, etype), scale.astype(np.float16).tobytes()
    w_off = 0 if w_offset is None else w_offset
    z_off = len(wb) + 16
    s_off = z_off + len(zb) + 16
    blob = bytearray(s_off + len(sb))
    blob[w_off:w_off + len(wb)] = wb
    blob[z_off:z_off + len(zb)] = zb
    blob[s_off:s_off + len(sb)] = sb
    prec = etype.upper()
    layers = [
        _const("10", "w", etype, q.shape, w_off, len(wb)),
        _convert("11", "cw", prec, q.shape),
        _const("12", "z", etype, zp.shape, z_off, len(zb)),
        _convert("13", "cz", prec, zp.shape),
        _binary("14", "sub", "Subtract", sub_ports[:2], sub_ports[2], q.shape),
    ]
    edges = [
        _edge("10", "0", "11", "0"),
        _edge("12", "0", "13", "0"),
        _edge("11", "1", "14", sub_ports[0]),
        _edge("13", "1", "14", sub_ports[1]),
    ]
    if with_multiply:
        layers.append(_const("15", "s", "f16", scale.shape, s_off, len(sb)))
        layers.append(_binary("16", "mul", "Multiply", mul_ports[:2], mul_ports[2], q.shape))
        edges.append(_edge("14", sub_ports[2], "16", mul_ports[0]))
        edges.append(_edge("15", "0", "16", mul_ports[1]))
    xml = (
        '<?xml version="1.0"?><net name="t" version="11"><layers>'
        + "".join(layers) + extra_layers + "</layers><edges>" + "".join(edges) + extra_edges
        + "</edges></net>"
    )
    source = tmp_path / "src"
    source.mkdir()
    (source / "openvino_model.xml").write_text(xml, encoding="utf-8")
    (source / "openvino_model.bin").write_bytes(bytes(blob))
    return Graph(source, q.shape, etype, q, zp, scale, w_off, z_off, s_off)


def run(graph: Graph, tmp_path: Path, **kwargs) -> Tuple[Path, ET.Element]:
    out = tmp_path / "out"
    NpuGraphRewriter(graph.source, out, **kwargs).rewrite()
    return out, ET.parse(out / "openvino_model.xml").getroot()


def layer(root: ET.Element, layer_id: str) -> Optional[ET.Element]:
    return next((l for l in root.find("layers") if l.get("id") == layer_id), None)


def manifest(out: Path) -> dict:
    return json.loads((out / "npu_manifest.json").read_text(encoding="utf-8"))


def rng_group(seed: int, bits: int, shape=(4, 2, 32)):
    rng = np.random.default_rng(seed)
    top = (1 << bits) - 1
    q = rng.integers(0, top + 1, size=shape)
    zp = rng.integers(0, top + 1, size=shape[:-1] + (1,))
    scale = rng.uniform(0.01, 0.2, size=shape[:-1] + (1,)).astype(np.float16)
    return q, zp, scale


class TestExactPath:
    """Symmetric zero-points: the Subtract is removed and no value changes."""

    @pytest.mark.parametrize("etype,flip,signed", [("u4", 0x88, "i4"), ("u8", 0x80, "i8")])
    def test_rewrite_symmetric_zero_point_flips_sign_bit_and_removes_subtract(
        self, tmp_path, etype, flip, signed
    ):
        bits = 4 if etype == "u4" else 8
        q, _, scale = rng_group(1, bits)
        zp = np.full(scale.shape, 8 if etype == "u4" else 128)
        graph = make_graph(tmp_path, q, zp, scale, etype)
        before = (graph.source / "openvino_model.bin").read_bytes()
        out, root = run(graph, tmp_path)
        after = (out / "openvino_model.bin").read_bytes()
        size = len(_pack(q, etype))
        assert after[:size] == bytes(b ^ flip for b in before[:size])
        assert layer(root, "14") is None
        assert layer(root, "10").find("data").get("element_type") == signed
        assert layer(root, "11").find("./input/port").get("precision") == signed.upper()
        assert layer(root, "10").find("./output/port").get("precision") == signed.upper()
        s_off = graph.s_offset
        assert after[s_off:s_off + scale.nbytes] == before[s_off:s_off + scale.nbytes]

    def test_rewrite_symmetric_zero_point_reports_exact_certified_conversion(self, tmp_path):
        q, _, scale = rng_group(2, 4)
        graph = make_graph(tmp_path, q, np.full(scale.shape, 8), scale)
        out, _ = run(graph, tmp_path)
        report = manifest(out)["weight_conversion"]
        assert report["exact_layers"] == 1
        assert report["requantized_layers"] == 0
        assert report["max_abs_error"] == 0.0
        assert manifest(out)["certified_exact"] is True

    def test_rewrite_symmetric_zero_point_preserves_dequantised_weights(self, tmp_path):
        q, _, scale = rng_group(3, 4)
        zp = np.full(scale.shape, 8)
        graph = make_graph(tmp_path, q, zp, scale)
        out, _ = run(graph, tmp_path)
        raw = (out / "openvino_model.bin").read_bytes()[: len(_pack(q, "u4"))]
        new_q = _unpack_signed(raw, 4).reshape(q.shape)
        before = scale.astype(np.float64) * (q - zp)
        after = scale.astype(np.float64) * new_q
        assert np.array_equal(before, after)

    def test_rewrite_removes_zero_point_chain_and_rewires_consumer(self, tmp_path):
        q, _, scale = rng_group(4, 4)
        graph = make_graph(tmp_path, q, np.full(scale.shape, 8), scale)
        _, root = run(graph, tmp_path)
        assert layer(root, "12") is None and layer(root, "13") is None
        edge = next(e for e in root.find("edges") if e.get("to-layer") == "16" and e.get("to-port") == "0")
        assert (edge.get("from-layer"), edge.get("from-port")) == ("11", "1")
        assert all(e.get("from-layer") != "12" and e.get("to-layer") != "13" for e in root.find("edges"))

    def test_rewrite_uses_declared_port_ids_instead_of_hardcoded_ones(self, tmp_path):
        q, _, scale = rng_group(5, 4)
        graph = make_graph(
            tmp_path, q, np.full(scale.shape, 8), scale,
            sub_ports=("5", "6", "7"), mul_ports=("3", "4", "9"),
        )
        _, root = run(graph, tmp_path)
        edge = next(e for e in root.find("edges") if e.get("to-layer") == "16" and e.get("to-port") == "3")
        assert (edge.get("from-layer"), edge.get("from-port")) == ("11", "1")
        assert layer(root, "14") is None

    def test_rewrite_is_idempotent_on_converted_output(self, tmp_path):
        q, zp, scale = rng_group(6, 4)
        graph = make_graph(tmp_path, q, zp, scale)
        out, _ = run(graph, tmp_path)
        first = (out / "openvino_model.bin").read_bytes()
        again = tmp_path / "again"
        again_src = Graph(out, graph.w_shape, "i4", q, zp, scale, 0, 0, 0)
        NpuGraphRewriter(again_src.source, again).rewrite()
        assert (again / "openvino_model.bin").read_bytes() == first
        assert manifest(again)["weight_conversion"]["requantized_layers"] == 0


class TestLossyPath:
    """Asymmetric zero-points: requantised with the proven per-element bound."""

    @pytest.mark.parametrize("etype,bits,shape", [("u4", 4, (4, 2, 32)), ("u8", 8, (6, 64))])
    def test_rewrite_asymmetric_zero_point_requantizes_within_bound(self, tmp_path, etype, bits, shape):
        q, zp, scale = rng_group(7, bits, shape)
        graph = make_graph(tmp_path, q, zp, scale, etype)
        out, root = run(graph, tmp_path)
        raw = (out / "openvino_model.bin").read_bytes()
        w_bytes = len(_pack(q, etype))
        new_q = _unpack_signed(raw[:w_bytes], bits).reshape(q.shape)
        new_scale = np.frombuffer(raw[graph.s_offset:graph.s_offset + scale.nbytes], dtype=np.float16)
        new_scale = new_scale.astype(np.float64).reshape(scale.shape)
        weights = scale.astype(np.float64) * (q - zp)
        recon = new_scale * new_q
        qp, qn = (7, 8) if bits == 4 else (127, 128)
        a_plus = np.maximum(weights.max(axis=-1, keepdims=True), 0)
        a_minus = np.maximum(-weights.min(axis=-1, keepdims=True), 0)
        bound = np.maximum.reduce([new_scale / 2, a_plus - qp * new_scale, a_minus - qn * new_scale])
        assert np.all(np.abs(weights - recon) <= bound + 1e-9)
        assert layer(root, "14") is None
        assert layer(root, "10").find("data").get("element_type") == f"i{bits}"

    def test_rewrite_asymmetric_zero_point_reports_lossy_conversion(self, tmp_path):
        q, zp, scale = rng_group(8, 4)
        graph = make_graph(tmp_path, q, zp, scale)
        out, _ = run(graph, tmp_path)
        data = manifest(out)
        report = data["weight_conversion"]
        assert report["requantized_layers"] == 1 and report["exact_layers"] == 0
        assert 0.0 < report["max_abs_error"]
        assert 0.0 < report["max_rel_error"] <= 1.0 / 15.0 + 2.0 ** -11 + 1e-9
        assert data["certified_exact"] is False
        assert data["quality"] == "unmeasured"
        assert data["conversion_mode"] == "lossy"

    def test_rewrite_exact_mode_with_asymmetric_zero_point_raises(self, tmp_path):
        q, zp, scale = rng_group(9, 4)
        graph = make_graph(tmp_path, q, zp, scale)
        with pytest.raises(NpuGraphRewriterError):
            NpuGraphRewriter(graph.source, tmp_path / "out", conversion_mode="exact").rewrite()
        assert not (tmp_path / "out" / "openvino_model.bin").exists()

    def test_rewrite_asymmetric_zero_point_without_scale_raises(self, tmp_path):
        q, zp, scale = rng_group(10, 4)
        graph = make_graph(tmp_path, q, zp, scale, with_multiply=False)
        with pytest.raises(NpuGraphRewriterError):
            NpuGraphRewriter(graph.source, tmp_path / "out").rewrite()

    def test_rewrite_mode_off_leaves_subtract_and_weights_untouched(self, tmp_path):
        q, zp, scale = rng_group(11, 4)
        graph = make_graph(tmp_path, q, zp, scale)
        out, root = run(graph, tmp_path, conversion_mode="off")
        assert layer(root, "14") is not None
        assert (out / "openvino_model.bin").read_bytes() == (graph.source / "openvino_model.bin").read_bytes()
        assert manifest(out)["certified_exact"] is False

    def test_rewrite_unknown_conversion_mode_raises(self, tmp_path):
        with pytest.raises(NpuGraphRewriterError):
            NpuGraphRewriter(tmp_path, tmp_path / "out", conversion_mode="fast")


class TestStructureGuards:
    """Only real zero-point Subtracts are rewritten and nothing is changed silently."""

    def test_rewrite_subtract_between_non_constants_is_left_untouched(self, tmp_path):
        extra_layers = (
            '<layer id="20" name="p0" type="Parameter" version="opset1"><data element_type="f16" shape="2"/>'
            f"<output>{_port('0', 'FP16', (2,))}</output></layer>"
            '<layer id="21" name="p1" type="Parameter" version="opset1"><data element_type="f16" shape="2"/>'
            f"<output>{_port('0', 'FP16', (2,))}</output></layer>"
            + _binary("22", "plain_sub", "Subtract", ("0", "1"), "2", (2,))
        )
        extra_edges = _edge("20", "0", "22", "0") + _edge("21", "0", "22", "1")
        q, _, scale = rng_group(12, 4)
        graph = make_graph(tmp_path, q, np.full(scale.shape, 8), scale,
                           extra_layers=extra_layers, extra_edges=extra_edges)
        out, root = run(graph, tmp_path)
        assert layer(root, "22") is not None
        assert layer(root, "14") is None
        assert manifest(out)["zero_points_stripped"] == 1

    def test_rewrite_unsigned_const_without_subtract_is_not_flipped(self, tmp_path):
        q, _, scale = rng_group(13, 4)
        graph = make_graph(tmp_path, q, np.full(scale.shape, 8), scale)
        extra = _const("30", "lone", "u4", (2, 2), 4096, 2)
        xml_path = graph.source / "openvino_model.xml"
        xml_path.write_text(xml_path.read_text(encoding="utf-8").replace("</layers>", extra + "</layers>"),
                            encoding="utf-8")
        blob = bytearray((graph.source / "openvino_model.bin").read_bytes())
        blob.extend(b"\x00" * (4096 + 2 - len(blob)))
        blob[4096:4098] = b"\x21\x43"
        (graph.source / "openvino_model.bin").write_bytes(bytes(blob))
        out, root = run(graph, tmp_path)
        assert layer(root, "30").find("data").get("element_type") == "u4"
        assert (out / "openvino_model.bin").read_bytes()[4096:4098] == b"\x21\x43"
        assert manifest(out)["weight_conversion"]["unconverted_unsigned_consts"] == 1

    def test_rewrite_shared_scale_const_raises(self, tmp_path):
        q, zp, scale = rng_group(14, 4)
        extra_layers = _binary("40", "mul2", "Multiply", ("0", "1"), "2", q.shape)
        extra_edges = _edge("11", "1", "40", "0") + _edge("15", "0", "40", "1")
        graph = make_graph(tmp_path, q, zp, scale, extra_layers=extra_layers, extra_edges=extra_edges)
        with pytest.raises(NpuGraphRewriterError):
            NpuGraphRewriter(graph.source, tmp_path / "out").rewrite()

    def test_rewrite_offset_past_end_of_file_raises_structured_error(self, tmp_path):
        q, _, scale = rng_group(15, 4)
        graph = make_graph(tmp_path, q, np.full(scale.shape, 8), scale)
        xml_path = graph.source / "openvino_model.xml"
        text = xml_path.read_text(encoding="utf-8").replace('offset="0"', 'offset="99999999"', 1)
        xml_path.write_text(text, encoding="utf-8")
        with pytest.raises(NpuGraphRewriterError):
            NpuGraphRewriter(graph.source, tmp_path / "out").rewrite()

    def test_rewrite_non_numeric_offset_raises_structured_error(self, tmp_path):
        q, _, scale = rng_group(16, 4)
        graph = make_graph(tmp_path, q, np.full(scale.shape, 8), scale)
        xml_path = graph.source / "openvino_model.xml"
        text = xml_path.read_text(encoding="utf-8").replace('offset="0"', 'offset="abc"', 1)
        xml_path.write_text(text, encoding="utf-8")
        with pytest.raises(NpuGraphRewriterError):
            NpuGraphRewriter(graph.source, tmp_path / "out").rewrite()

    def test_rewrite_size_not_matching_shape_raises_structured_error(self, tmp_path):
        q, _, scale = rng_group(17, 4)
        graph = make_graph(tmp_path, q, np.full(scale.shape, 8), scale)
        xml_path = graph.source / "openvino_model.xml"
        text = xml_path.read_text(encoding="utf-8")
        text = text.replace(f'size="{len(_pack(q, "u4"))}"', 'size="3"', 1)
        xml_path.write_text(text, encoding="utf-8")
        with pytest.raises(NpuGraphRewriterError):
            NpuGraphRewriter(graph.source, tmp_path / "out").rewrite()

    def test_rewrite_overlapping_weight_and_scale_ranges_raise(self, tmp_path):
        q, zp, scale = rng_group(18, 4)
        graph = make_graph(tmp_path, q, zp, scale)
        xml_path = graph.source / "openvino_model.xml"
        text = xml_path.read_text(encoding="utf-8")
        text = text.replace(f'offset="{graph.s_offset}"', f'offset="{graph.w_offset + 4}"', 1)
        xml_path.write_text(text, encoding="utf-8")
        with pytest.raises(NpuGraphRewriterError):
            NpuGraphRewriter(graph.source, tmp_path / "out").rewrite()

    def test_rewrite_failure_never_writes_partial_output(self, tmp_path):
        q, zp, scale = rng_group(19, 4)
        graph = make_graph(tmp_path, q, zp, scale)
        with pytest.raises(NpuGraphRewriterError):
            NpuGraphRewriter(graph.source, tmp_path / "out", conversion_mode="exact").rewrite()
        assert not (tmp_path / "out" / "openvino_model.xml").exists()

    def test_rewrite_two_subtracts_sharing_one_zero_point_keeps_it_until_last_use(self, tmp_path):
        q, _, scale = rng_group(20, 4)
        zp = np.full(scale.shape, 8)
        q_b = q.copy()
        layers = (
            _const("50", "w2", "u4", q.shape, 8192, len(_pack(q, "u4")))
            + _convert("51", "cw2", "U4", q.shape)
            + _binary("52", "sub2", "Subtract", ("0", "1"), "2", q.shape)
        )
        edges = _edge("50", "0", "51", "0") + _edge("51", "1", "52", "0") + _edge("13", "1", "52", "1")
        graph = make_graph(tmp_path, q, zp, scale, extra_layers=layers, extra_edges=edges)
        blob = bytearray((graph.source / "openvino_model.bin").read_bytes())
        blob.extend(b"\x00" * (8192 + len(_pack(q_b, "u4")) - len(blob)))
        blob[8192:8192 + len(_pack(q_b, "u4"))] = _pack(q_b, "u4")
        (graph.source / "openvino_model.bin").write_bytes(bytes(blob))
        out, root = run(graph, tmp_path)
        assert layer(root, "14") is None and layer(root, "52") is None
        assert layer(root, "12") is None and layer(root, "13") is None
        assert manifest(out)["weight_conversion"]["exact_layers"] == 2
