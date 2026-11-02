"""Unit and regression test suite for NpuGraphRewriter.

Validates OpenVINO IR computational DAG transformations, zero-point Subtract
layer decoupling, static KV cache shape freezing, and NPU manifest generation.
"""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from arc_runner.quant.npu_graph_rewriter import NpuGraphRewriter, NpuGraphRewriterError


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
        (source_dir / "openvino_model.bin").write_bytes(b"\x00" * 32)
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
        assert manifest["precision"] == "INT4_SYM"
        assert manifest["zero_points_stripped"] == 1
        assert manifest["kv_shapes_frozen"] == 1
