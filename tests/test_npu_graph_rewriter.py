"""Unit and regression test suite for NpuGraphRewriter.

Validates OpenVINO IR computational DAG transformations, zero-point Subtract
layer decoupling, static KV cache shape freezing, and NPU manifest generation.
"""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from arc_runner.quant.npu_graph_rewriter import (
    NpuGraphRewriter,
    NpuGraphRewriterError,
    freeze_kv_variable_shape,
    verify_bounded,
    parse_dim,
)


class TestNpuGraphRewriter:
    """Test suite validating graph rewriting rules for Intel NPU37XX."""

    def test_rewriter_missing_files_raises_error(self, tmp_path: Path) -> None:
        """Verify that missing model files raise a structured NpuGraphRewriterError."""
        source_dir = tmp_path / "empty_model"
        source_dir.mkdir(parents=True)
        out_dir = tmp_path / "output_model"

        rewriter = NpuGraphRewriter(source_dir=source_dir, output_dir=out_dir)
        with pytest.raises(NpuGraphRewriterError) as exc_info:
            rewriter.rewrite()
        assert "Missing OpenVINO IR files" in str(exc_info.value)

    def test_rewriter_corrupted_xml_raises_error(self, tmp_path: Path) -> None:
        """Verify that malformed XML syntax raises NpuGraphRewriterError."""
        source_dir = tmp_path / "corrupted_model"
        source_dir.mkdir(parents=True)
        (source_dir / "openvino_model.xml").write_text("<net><unclosed>", encoding="utf-8")
        (source_dir / "openvino_model.bin").write_bytes(b"\x00" * 32)
        out_dir = tmp_path / "output_model"

        rewriter = NpuGraphRewriter(source_dir=source_dir, output_dir=out_dir)
        with pytest.raises(NpuGraphRewriterError) as exc_info:
            rewriter.rewrite()
        assert "Failed to parse source OpenVINO XML graph" in str(exc_info.value)

    def test_rewriter_strips_subtract_and_freezes_shapes(self, tmp_path: Path) -> None:
        """Verify DAG transformation decoupling Subtract layers and freezing KV shapes."""
        source_dir = tmp_path / "mock_model"
        source_dir.mkdir(parents=True)
        out_dir = tmp_path / "npu_certified"

        mock_xml = """<?xml version="1.0"?>
<net name="test_net" version="11">
    <layers>
        <layer id="0" name="weights_const" type="Const" version="opset1">
            <data element_type="u4" offset="0" shape="1, 8" size="4"/>
            <output>
                <port id="0" precision="U4">
                    <dim>1</dim><dim>8</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="convert_weights" type="Convert" version="opset1">
            <data destination_type="f16"/>
            <input>
                <port id="0" precision="U4">
                    <dim>1</dim><dim>8</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP16">
                    <dim>1</dim><dim>8</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="zp_const" type="Const" version="opset1">
            <data element_type="u4" offset="4" shape="1, 8" size="4"/>
            <output>
                <port id="0" precision="U4">
                    <dim>1</dim><dim>8</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="convert_zp" type="Convert" version="opset1">
            <data destination_type="f16"/>
            <input>
                <port id="0" precision="U4">
                    <dim>1</dim><dim>8</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP16">
                    <dim>1</dim><dim>8</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="subtract_layer" type="Subtract" version="opset1">
            <input>
                <port id="0" precision="FP16"><dim>1</dim><dim>8</dim></port>
                <port id="1" precision="FP16"><dim>1</dim><dim>8</dim></port>
            </input>
            <output>
                <port id="2" precision="FP16"><dim>1</dim><dim>8</dim></port>
            </output>
        </layer>
        <layer id="5" name="multiply_layer" type="Multiply" version="opset1">
            <input>
                <port id="0" precision="FP16"><dim>1</dim><dim>8</dim></port>
                <port id="1" precision="FP16"><dim>1</dim><dim>8</dim></port>
            </input>
            <output>
                <port id="2" precision="FP16"><dim>1</dim><dim>8</dim></port>
            </output>
        </layer>
        <layer id="6" name="kv_read_value" type="ReadValue" version="opset6">
            <data variable_id="past_kv" variable_type="f32" variable_shape="?,2,?,64"/>
            <input>
                <port id="0" precision="FP32"><dim>-1</dim><dim>2</dim><dim>0</dim><dim>64</dim></port>
            </input>
            <output>
                <port id="1" precision="FP32"><dim>-1</dim><dim>2</dim><dim>-1</dim><dim>64</dim></port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="2" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="4" to-port="0"/>
        <edge from-layer="3" from-port="1" to-layer="4" to-port="1"/>
        <edge from-layer="4" from-port="2" to-layer="5" to-port="0"/>
    </edges>
</net>
"""
        (source_dir / "openvino_model.xml").write_text(mock_xml, encoding="utf-8")
        (source_dir / "openvino_model.bin").write_bytes(b"\x00" * 4 + b"\x88" * 4 + b"\x00" * 24)
        (source_dir / "config.json").write_text("{}", encoding="utf-8")

        rewriter = NpuGraphRewriter(source_dir=source_dir, output_dir=out_dir, max_context_len=128)
        result_dir = rewriter.rewrite()

        assert result_dir == out_dir.resolve()
        assert (out_dir / "openvino_model.xml").exists()
        assert (out_dir / "openvino_model.bin").exists()
        assert (out_dir / "config.json").exists()
        assert (out_dir / "npu_manifest.json").exists()

        tree = ET.parse(out_dir / "openvino_model.xml")
        root = tree.getroot()
        layers_elem = root.find("layers")
        assert layers_elem is not None
        layer_types = [l.get("type") for l in layers_elem]

        assert "Subtract" not in layer_types

        edges_elem = root.find("edges")
        assert edges_elem is not None
        to_mult = [e for e in edges_elem if e.get("to-layer") == "5" and e.get("to-port") == "0"]
        assert len(to_mult) == 1
        assert to_mult[0].get("from-layer") == "1"
        assert to_mult[0].get("from-port") == "1"

        rv_layer = next(l for l in layers_elem if l.get("type") == "ReadValue")
        rv_data = rv_layer.find("data")
        assert rv_data is not None
        assert "0..128" in rv_data.get("variable_shape", "")

        manifest = json.loads((out_dir / "npu_manifest.json").read_text(encoding="utf-8"))
        assert manifest["target_device"] == "NPU"
        assert manifest["zero_points_stripped"] == 1
        assert manifest["kv_shapes_frozen"] == 1


UNBOUNDED = float("inf")


def _parse_dim(token: str) -> tuple:
    """Parse one OpenVINO partial dimension token into an inclusive (low, high) pair.

    Independent oracle for the tests: it shares no code with the implementation.

    Args:
        token: One of "?", "N" or "A..B".

    Returns:
        Inclusive bounds, with an unbounded high end for a fully dynamic dimension.
    """
    token = token.strip()
    if token == "?":
        return (0, UNBOUNDED)
    if ".." in token:
        low, high = token.split("..")
        return (int(low) if low else 0, int(high) if high else UNBOUNDED)
    return (int(token), int(token))


def _contains(outer: str, inner: str) -> bool:
    """Return True when every dimension interval of inner lies inside outer.

    Args:
        outer: Comma separated partial shape that must relax inner.
        inner: Comma separated partial shape of the initialisation subgraph.

    Returns:
        True when the shapes have equal rank and outer extends inner on every dimension.
    """
    outer_dims = [_parse_dim(t) for t in outer.split(",")]
    inner_dims = [_parse_dim(t) for t in inner.split(",")]
    if len(outer_dims) != len(inner_dims):
        return False
    return all(o[0] <= i[0] and i[1] <= o[1] for o, i in zip(outer_dims, inner_dims))


def _kv_model_xml(variable_shape: str, heads: int = 8, head_dim: int = 128, rank: int = 4) -> str:
    """Build a minimal IR with one stateful KV ReadValue and its Assign.

    Args:
        variable_shape: The variable_shape attribute of the ReadValue.
        heads: KV head count used in the port dimensions.
        head_dim: Head dimension used in the port dimensions.
        rank: Rank of the state; 4 builds [batch, heads, seq, head_dim].

    Returns:
        IR XML text.
    """
    if rank == 4:
        init = f"<dim>-1</dim><dim>{heads}</dim><dim>0</dim><dim>{head_dim}</dim>"
        out = f"<dim>-1</dim><dim>{heads}</dim><dim>-1</dim><dim>{head_dim}</dim>"
    else:
        init = "<dim>-1</dim><dim>0</dim><dim>16</dim>"
        out = "<dim>-1</dim><dim>-1</dim><dim>16</dim>"
    return f"""<?xml version="1.0"?>
<net name="kv_net" version="11">
    <layers>
        <layer id="0" name="kv_read" type="ReadValue" version="opset6">
            <data variable_id="past_kv" variable_type="f32" variable_shape="{variable_shape}"/>
            <input><port id="0" precision="FP32">{init}</port></input>
            <output><port id="1" precision="FP32">{out}</port></output>
        </layer>
        <layer id="1" name="kv_assign" type="Assign" version="opset6">
            <data variable_id="past_kv"/>
            <input><port id="0" precision="FP32">{out}</port></input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="1" to-layer="1" to-port="0"/>
    </edges>
</net>
"""


def _rewrite_kv(tmp_path: Path, xml_text: str, max_context_len: int = 128) -> ET.Element:
    """Run the rewriter on a one-layer KV model and return the parsed output root.

    Args:
        tmp_path: Pytest temporary directory.
        xml_text: Source IR XML.
        max_context_len: Static sequence bound given to the rewriter.

    Returns:
        Root element of the rewritten IR.
    """
    source_dir = tmp_path / "src_model"
    out_dir = tmp_path / "out_model"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "openvino_model.xml").write_text(xml_text, encoding="utf-8")
    (source_dir / "openvino_model.bin").write_bytes(b"\x00" * 16)
    (source_dir / "config.json").write_text("{}", encoding="utf-8")
    NpuGraphRewriter(source_dir=source_dir, output_dir=out_dir, max_context_len=max_context_len).rewrite()
    return ET.parse(out_dir / "openvino_model.xml").getroot()


def _read_value_shape(root: ET.Element) -> str:
    """Return the variable_shape of the first ReadValue layer.

    Args:
        root: Root element of an IR document.

    Returns:
        The variable_shape attribute text.
    """
    layer = next(l for l in root.find("layers") if l.get("type") == "ReadValue")
    return layer.find("data").get("variable_shape")


def _static_state_xml() -> str:
    """Build a model whose rank 3 state has a fixed shape and therefore never grows."""
    xml = _kv_model_xml("?,4,16", rank=3)
    xml = xml.replace("<dim>0</dim>", "<dim>4</dim>").replace("<dim>-1</dim><dim>-1</dim>", "<dim>-1</dim><dim>4</dim>")
    return xml


class TestKvShapeFreezing:
    """Freezing must derive KV geometry from the model, never from a fixed constant."""

    @pytest.mark.parametrize(
        "heads,head_dim",
        [(2, 64), (8, 128), (4, 128), (8, 64), (1, 256), (32, 128)],
    )
    def test_rewrite_kv_geometry_preserves_heads_and_head_dim(self, tmp_path, heads, head_dim):
        """Heads and head_dim from the source state must survive unchanged."""
        xml = _kv_model_xml(f"?,{heads},?,{head_dim}", heads, head_dim)
        shape = _read_value_shape(_rewrite_kv(tmp_path, xml))
        assert shape == f"?,{heads},0..128,{head_dim}"

    def test_rewrite_qwen3_geometry_output_relaxes_initialisation_shape(self, tmp_path):
        """OpenVINO requires the Variable shape to extend the initialisation shape."""
        xml = _kv_model_xml("?,8,?,128", 8, 128)
        shape = _read_value_shape(_rewrite_kv(tmp_path, xml))
        assert _contains(shape, "?,8,0,128")

    def test_rewrite_max_context_len_sets_sequence_upper_bound(self, tmp_path):
        """The sequence bound must come from max_context_len."""
        xml = _kv_model_xml("?,8,?,128", 8, 128)
        shape = _read_value_shape(_rewrite_kv(tmp_path, xml, max_context_len=256))
        assert shape == "?,8,0..256,128"

    def test_rewrite_static_sequence_dimension_is_relaxed_to_contain_zero(self, tmp_path):
        """A static non-zero sequence dim would not contain the empty init state, so it is relaxed."""
        xml = _kv_model_xml("?,8,32,128", 8, 128)
        shape = _read_value_shape(_rewrite_kv(tmp_path, xml))
        assert shape == "?,8,0..128,128"
        assert _contains(shape, "?,8,0,128")

    def test_rewrite_static_sequence_dimension_above_limit_keeps_its_upper_bound(self, tmp_path):
        """A static sequence larger than max_context_len must stay reachable."""
        xml = _kv_model_xml("?,8,512,128", 8, 128)
        assert _read_value_shape(_rewrite_kv(tmp_path, xml, max_context_len=128)) == "?,8,0..512,128"

    def test_rewrite_non_growing_state_is_not_counted_as_frozen(self, tmp_path):
        """A state with no zero-sized initial axis does not grow and is not counted."""
        source_dir = tmp_path / "src_model"
        out_dir = tmp_path / "out_model"
        source_dir.mkdir(parents=True)
        (source_dir / "openvino_model.xml").write_text(_static_state_xml(), encoding="utf-8")
        (source_dir / "openvino_model.bin").write_bytes(b"\x00" * 16)
        NpuGraphRewriter(source_dir=source_dir, output_dir=out_dir).rewrite()
        manifest = json.loads((out_dir / "npu_manifest.json").read_text(encoding="utf-8"))
        assert manifest["kv_shapes_frozen"] == 0

    def test_rewrite_rank_three_growing_state_is_frozen_on_its_growth_axis(self, tmp_path):
        """A rank 3 latent cache [batch, seq, dim] grows on axis 1 and must be bounded there."""
        root = _rewrite_kv(tmp_path, _kv_model_xml("?,?,16", rank=3))
        assert _read_value_shape(root) == "?,0..128,16"

    def test_rewrite_is_idempotent_on_certified_output(self, tmp_path):
        """Rewriting an already certified model must not change the state shape."""
        first = _rewrite_kv(tmp_path / "a", _kv_model_xml("?,8,?,128", 8, 128))
        again = _kv_model_xml(_read_value_shape(first), 8, 128)
        second = _rewrite_kv(tmp_path / "b", again)
        assert _read_value_shape(second) == _read_value_shape(first)

    def test_rewrite_dynamic_head_count_raises_structured_error(self, tmp_path):
        """A dynamic head count cannot be frozen without inventing a value."""
        xml = _kv_model_xml("?,?,?,128", 8, 128)
        with pytest.raises(NpuGraphRewriterError):
            _rewrite_kv(tmp_path, xml)

    def test_rewrite_dynamic_head_dim_raises_structured_error(self, tmp_path):
        """A dynamic head dimension cannot be frozen without inventing a value."""
        xml = _kv_model_xml("?,8,?,?", 8, 128)
        with pytest.raises(NpuGraphRewriterError):
            _rewrite_kv(tmp_path, xml)

    def test_rewrite_variable_shape_disagreeing_with_init_ports_raises(self, tmp_path):
        """A variable_shape whose heads differ from the init port dims is an inconsistent model."""
        xml = _kv_model_xml("?,2,?,64", heads=8, head_dim=128)
        with pytest.raises(NpuGraphRewriterError):
            _rewrite_kv(tmp_path, xml)

    def test_rewrite_non_growing_state_is_left_untouched(self, tmp_path):
        """A fixed-shape state keeps its original variable shape."""
        xml = _static_state_xml()
        assert _read_value_shape(_rewrite_kv(tmp_path, xml)) == "?,4,16"

    def test_rewrite_port_dims_keep_heads_and_head_dim(self, tmp_path):
        """Only batch and sequence port dimensions may be made static."""
        root = _rewrite_kv(tmp_path, _kv_model_xml("?,8,?,128", 8, 128))
        layer = next(l for l in root.find("layers") if l.get("type") == "ReadValue")
        dims = [d.text for d in layer.find("output/port").findall("dim")]
        assert dims[1] == "8" and dims[3] == "128"


class TestKvShapeEdgeCases:
    """Direct tests of the dimension parser and of the remaining freezing rules."""

    @pytest.mark.parametrize(
        "token,expected",
        [("?", (0, None)), ("-1", (0, None)), ("7", (7, 7)), ("0..128", (0, 128)),
         ("2..", (2, None)), ("..9", (0, 9)), (" 5 ", (5, 5))],
    )
    def test_parse_dim_valid_token_returns_inclusive_bounds(self, token, expected):
        """Every supported spelling of a partial dimension maps to its exact bounds."""
        assert parse_dim(token) == expected

    @pytest.mark.parametrize("token", ["", "abc", "-3", "1..x", "1.5"])
    def test_parse_dim_invalid_token_raises_structured_error(self, token):
        """Malformed or negative tokens must raise, never be guessed."""
        with pytest.raises(NpuGraphRewriterError):
            parse_dim(token)

    def test_freeze_variable_shape_head_range_raises(self):
        """A range of head counts is not one model fact, so it cannot be frozen."""
        with pytest.raises(NpuGraphRewriterError):
            freeze_kv_variable_shape("?,2..8,?,128", 128)

    def test_freeze_variable_shape_head_dim_range_raises(self):
        """A range of head dimensions is not one model fact, so it cannot be frozen."""
        with pytest.raises(NpuGraphRewriterError):
            freeze_kv_variable_shape("?,8,?,64..128", 128)

    def test_freeze_variable_shape_rank_other_than_four_returns_none(self):
        """Only rank 4 states are KV caches."""
        assert freeze_kv_variable_shape("?,?,16", 128) is None
        assert freeze_kv_variable_shape("?,8,?,128,1", 128) is None

    def test_freeze_variable_shape_open_interval_uses_context_bound(self):
        """An interval with no upper end is bounded by max_context_len."""
        assert freeze_kv_variable_shape("?,8,3..,128", 96) == "?,8,0..96,128"

    def test_rewrite_init_heads_below_variable_heads_raises(self, tmp_path):
        """Init heads smaller than the variable heads cannot be relaxed by the variable."""
        xml = _kv_model_xml("?,8,?,128", heads=4, head_dim=128)
        with pytest.raises(NpuGraphRewriterError):
            _rewrite_kv(tmp_path, xml)

    def test_rewrite_assign_port_dims_become_static(self, tmp_path):
        """The Assign that writes the state back is frozen consistently with the ReadValue."""
        root = _rewrite_kv(tmp_path, _kv_model_xml("?,8,?,128", 8, 128))
        assign = next(l for l in root.find("layers") if l.get("type") == "Assign")
        dims = [d.text for d in assign.find("input/port").findall("dim")]
        assert dims == ["1", "8", "128", "128"]

    def test_rewrite_assign_of_growing_rank_three_state_uses_the_read_value_bound(self, tmp_path):
        """The Assign growth dimension gets the bound chosen for its ReadValue."""
        root = _rewrite_kv(tmp_path, _kv_model_xml("?,512,16", rank=3))
        assign = next(l for l in root.find("layers") if l.get("type") == "Assign")
        dims = [d.text for d in assign.find("input/port").findall("dim")]
        assert dims == ["1", "512", "16"]

    def test_rewrite_assign_of_non_growing_state_is_left_dynamic(self, tmp_path):
        """An Assign whose ReadValue does not grow keeps its dimensions."""
        root = _rewrite_kv(tmp_path, _static_state_xml())
        assign = next(l for l in root.find("layers") if l.get("type") == "Assign")
        dims = [d.text for d in assign.find("input/port").findall("dim")]
        assert dims == ["-1", "4", "16"]

class TestKvShapeFreezingProperties:
    """Property based check of invariants I1 to I5 against an independent oracle."""

    @settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        heads=st.integers(min_value=1, max_value=64),
        head_dim=st.sampled_from([32, 64, 96, 128, 192, 256]),
        context=st.integers(min_value=64, max_value=4096),
        seq_dynamic=st.booleans(),
        seq_static=st.integers(min_value=0, max_value=512),
    )
    def test_rewrite_output_contains_initialisation_shape_for_any_geometry(
        self, tmp_path_factory, heads, head_dim, context, seq_dynamic, seq_static
    ):
        """Output contains the init shape, keeps geometry, and is idempotent."""
        tmp_path = tmp_path_factory.mktemp("prop")
        seq = "?" if seq_dynamic else str(seq_static)
        shape_in = f"?,{heads},{seq},{head_dim}"
        root = _rewrite_kv(tmp_path / "a", _kv_model_xml(shape_in, heads, head_dim), context)
        shape_out = _read_value_shape(root)
        assert _contains(shape_out, f"?,{heads},0,{head_dim}")
        assert _parse_dim(shape_out.split(",")[2])[1] >= max(context, 0 if seq_dynamic else seq_static)
        out_dims = shape_out.split(",")
        assert out_dims[1] == str(heads) and out_dims[3] == str(head_dim)
        again = _read_value_shape(
            _rewrite_kv(tmp_path / "b", _kv_model_xml(shape_out, heads, head_dim), context)
        )
        assert again == shape_out


def _concat_model_xml(variable_shape: str, init_dims: str, out_dims: str, axis: int, rank_marker: str = "") -> str:
    """Build ReadValue -> Gather -> Concat(axis) -> Assign with explicit dimensions."""
    return f"""<?xml version="1.0"?>
<net name="cc" version="11">
    <layers>
        <layer id="0" name="rv" type="ReadValue" version="opset6">
            <data variable_id="v" variable_type="f32" variable_shape="{variable_shape}"/>
            <input><port id="0" precision="FP32">{init_dims}</port></input>
            <output><port id="1" precision="FP32">{out_dims}</port></output>
        </layer>
        <layer id="1" name="g" type="Gather" version="opset8">
            <data batch_dims="0"/>
            <output><port id="2" precision="FP32">{out_dims}</port></output>
        </layer>
        <layer id="2" name="cat" type="Concat" version="opset1">
            <data axis="{axis}"/>
            <output><port id="3" precision="FP32">{out_dims}</port></output>
        </layer>
        <layer id="3" name="as" type="Assign" version="opset6">
            <data variable_id="v"/>
            <input><port id="0" precision="FP32">{out_dims}</port></input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="1" to-layer="1" to-port="0"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
        <edge from-layer="2" from-port="3" to-layer="3" to-port="0"/>
    </edges>
</net>
"""


class TestGrowthAxisFromGraph:
    """The growth axis comes from the init shape and the Concat, never from rank or position."""

    INIT = "<dim>-1</dim><dim>0</dim><dim>16</dim><dim>0</dim>"
    OUT = "<dim>-1</dim><dim>-1</dim><dim>16</dim><dim>-1</dim>"

    def test_rewrite_negative_concat_axis_selects_the_matching_candidate(self, tmp_path):
        init = "<dim>-1</dim><dim>8</dim><dim>0</dim><dim>128</dim>"
        out = "<dim>-1</dim><dim>8</dim><dim>-1</dim><dim>128</dim>"
        root = _rewrite_kv(tmp_path, _concat_model_xml("?,8,?,128", init, out, -2))
        assert _read_value_shape(root) == "?,8,0..128,128"

    def test_rewrite_sequence_first_layout_freezes_axis_one(self, tmp_path):
        init = "<dim>-1</dim><dim>0</dim><dim>8</dim><dim>128</dim>"
        out = "<dim>-1</dim><dim>-1</dim><dim>8</dim><dim>128</dim>"
        root = _rewrite_kv(tmp_path, _concat_model_xml("?,?,8,128", init, out, 1))
        assert _read_value_shape(root) == "?,0..128,8,128"

    def test_rewrite_concat_axis_not_matching_growth_candidate_raises(self, tmp_path):
        init = "<dim>-1</dim><dim>8</dim><dim>0</dim><dim>128</dim>"
        out = "<dim>-1</dim><dim>8</dim><dim>-1</dim><dim>128</dim>"
        with pytest.raises(NpuGraphRewriterError):
            _rewrite_kv(tmp_path, _concat_model_xml("?,8,?,128", init, out, 1))

    def test_rewrite_two_zero_sized_axes_without_concat_raises(self, tmp_path):
        xml = _kv_model_xml("?,?,?,128", 8, 128).replace(
            "<dim>8</dim><dim>0</dim>", "<dim>0</dim><dim>0</dim>"
        )
        with pytest.raises(NpuGraphRewriterError):
            _rewrite_kv(tmp_path, xml)

    def test_rewrite_non_numeric_init_dimension_raises_structured_error(self, tmp_path):
        xml = _kv_model_xml("?,8,?,128", 8, 128).replace("<dim>0</dim>", "<dim>zero</dim>", 1)
        with pytest.raises(NpuGraphRewriterError):
            _rewrite_kv(tmp_path, xml)


def _save_stateful_model(path: Path, sequence: str) -> None:
    """Serialise a one-state OpenVINO model whose variable sequence dimension is given."""
    import openvino as ov
    from openvino import opset13 as ops

    x = ops.parameter(ov.PartialShape([1, 2, 0, 4]), ov.Type.f32, name="x")
    dims = {"unbounded": ov.Dimension(), "bounded": ov.Dimension(0, 16)}
    var_shape = ov.PartialShape([ov.Dimension(), ov.Dimension(2), dims[sequence], ov.Dimension(4)])
    info = ov.op.util.VariableInfo()
    info.data_shape = var_shape
    info.data_type = ov.Type.f32
    info.variable_id = "past"
    variable = ov.op.util.Variable(info)
    read = ops.read_value(x, variable)
    assign = ops.assign(read, variable)
    model = ov.Model([ops.result(read)], [assign], [x])
    ov.save_model(model, str(path / "openvino_model.xml"))


class TestVerifyBounded:
    """verify_bounded reads the rewritten model back with OpenVINO."""

    def test_verify_bounded_unbounded_sequence_variable_is_reported(self, tmp_path):
        pytest.importorskip("openvino")
        _save_stateful_model(tmp_path, "unbounded")
        report = verify_bounded(tmp_path / "openvino_model.xml")
        assert report.variables_checked == 1
        assert report.unbounded_variables == ["past"]

    def test_verify_bounded_bounded_sequence_variable_passes_with_dynamic_batch(self, tmp_path):
        pytest.importorskip("openvino")
        _save_stateful_model(tmp_path, "bounded")
        report = verify_bounded(tmp_path / "openvino_model.xml")
        assert report.unbounded_variables == []

    def test_verify_bounded_unreadable_model_raises_structured_error(self, tmp_path):
        pytest.importorskip("openvino")
        (tmp_path / "openvino_model.xml").write_text("<net>", encoding="utf-8")
        with pytest.raises(NpuGraphRewriterError):
            verify_bounded(tmp_path / "openvino_model.xml")
