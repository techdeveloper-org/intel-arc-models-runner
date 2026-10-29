"""TM-09 tests: file-controlled names must never become live XML in the generated IR."""

from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from arc_runner.quant.transpiler import GGUFTranspiler, TranspilationError

HOSTILE_TENSOR = 'w"/><layer id="99" name="EVIL" type="Parameter"/><layer name="'
HOSTILE_MODEL = 'x"><injected_net/><net name="'


def _transpiler(tmp_path: Path) -> GGUFTranspiler:
    """Build a transpiler without touching a real model file."""
    return object.__new__(GGUFTranspiler)


def _tensor(name: str) -> SimpleNamespace:
    """Return a minimal tensor descriptor."""
    return SimpleNamespace(name=name, dimensions=(2, 3))


def test_generate_ir_xml_hostile_names_stay_attribute_values(tmp_path: Path) -> None:
    """Quotes and angle brackets in names round-trip as text; no extra elements appear."""
    text = _transpiler(tmp_path)._generate_ir_xml(HOSTILE_MODEL, [_tensor(HOSTILE_TENSOR), _tensor("ok")], "m.bin", 10)
    root = ET.fromstring(text)
    assert root.get("name") == HOSTILE_MODEL
    layers = root.findall("./layers/layer")
    assert [layer.get("name") for layer in layers] == [HOSTILE_TENSOR, "ok"]
    assert root.find(".//injected_net") is None
    assert layers[0].find("./output/port").get("names") == HOSTILE_TENSOR


def test_generate_ir_xml_ampersand_and_lt_in_bin_name_are_escaped(tmp_path: Path) -> None:
    """The companion file name is escaped like every other attribute."""
    text = _transpiler(tmp_path)._generate_ir_xml("n", [_tensor("a")], 'a&b<"c>.bin', 1)
    assert ET.fromstring(text).find("./meta_data/bin_file").get("name") == 'a&b<"c>.bin'


@pytest.mark.parametrize("bad", ["a\x00b", "a\x01b", "a\x1fb"])
def test_generate_ir_xml_xml_illegal_control_characters_raise_transpilation_error(tmp_path: Path, bad: str) -> None:
    """Characters XML 1.0 forbids are rejected, not stripped, so the name is never silently altered."""
    with pytest.raises(TranspilationError):
        _transpiler(tmp_path)._generate_ir_xml("n", [_tensor(bad)], "m.bin", 1)


def test_generate_ir_xml_plain_names_are_unchanged(tmp_path: Path) -> None:
    """Ordinary names produce the same attribute text as before."""
    text = _transpiler(tmp_path)._generate_ir_xml("llama", [_tensor("blk.0.attn_q.weight")], "openvino_model.bin", 4)
    assert 'name="blk.0.attn_q.weight"' in text and '<net name="llama" version="11">' in text
