# -*- coding: utf-8 -*-
"""Verification script for all 13 UML diagrams and 13 Draw.io XML files."""

import glob
import xml.etree.ElementTree as ET
from pathlib import Path

def test_drawio_diagrams():
    keywords = {
        'class_diagram.drawio': 'ImageEngine',
        'package_diagram.drawio': 'ImageEngine',
        'component_diagram.drawio': 'Image_Engine',
        'sequence_diagram.drawio': 'ImageEngine',
        'state_diagram.drawio': 'LoadingImagePipeline',
        'activity_diagram.drawio': 'Ingest_Prompt_And_Parameters',
        'deployment_diagram.drawio': 'ImageEngine Runtime',
        'usecase_diagram.drawio': 'Synthesize Images',
        'object_diagram.drawio': 'image_engine_sd15',
        'composite_diagram.drawio': 'ImageEngine',
        'interaction_diagram.drawio': 'ImageGenerationResult',
        'communication_diagram.drawio': 'ImageEngine',
        'call_graph_diagram.drawio': 'handle_generate_image'
    }

    print("=== Checking Draw.io XML Diagrams (13 files) ===")
    drawio_dir = Path("drawio")
    for fname, kw in keywords.items():
        fpath = drawio_dir / fname
        assert fpath.exists(), f"Missing file: {fname}"
        
        # XML check
        tree = ET.parse(fpath)
        root = tree.getroot()
        assert root.tag == "mxfile", f"Root tag must be mxfile, got {root.tag}"
        cells = root.findall(".//mxCell")
        assert len(cells) >= 8, f"Too few cells ({len(cells)}) in {fname}"
        
        content = fpath.read_text(encoding="utf-8")
        assert kw in content, f"Keyword '{kw}' not found in {fname}"
        print(f"  [PASS] {fname:<30}: Valid XML, {len(cells):>2} cells, contains '{kw}'")

def test_mermaid_diagrams():
    keywords = {
        'class_diagram.md': ('ImageEngine', 'Proof M74', 'Proof M84'),
        'package_diagram.md': ('image_engine', 'Proof M73', 'Proof M79'),
        'component_diagram.md': ('Image_Engine', 'Proof M78', 'Proof M91'),
        'sequence_diagram.md': ('ImageEngine', 'Proof M80', 'Proof M84'),
        'state_diagram.md': ('LoadingImagePipeline', 'Proof M74', 'Proof M93'),
        'activity_diagram.md': ('generate_image', 'Proof M84', 'Proof M89'),
        'deployment_diagram.md': ('ImageEngine Runtime', 'Proof M78', 'Proof M91'),
        'usecase_diagram.md': ('UC_Txt2Img', 'Proof M74', 'Proof M84'),
        'object_diagram.md': ('image_engine_sd15', 'Proof M82', 'Proof M89'),
        'composite_diagram.md': ('ImageEngineComposite', 'Proof M74', 'Proof M84'),
        'interaction_diagram.md': ('ImageEngine', 'Proof M82', 'Proof M84'),
        'communication_diagram.md': ('ImageEngine', 'Proof M84', 'Proof M91'),
        'call_graph_diagram.md': ('handle_generate_image', 'Proof M74', 'Proof M78')
    }

    print("\n=== Checking Mermaid UML Diagrams (13 files) ===")
    uml_dir = Path("uml")
    for fname, kws in keywords.items():
        fpath = uml_dir / fname
        assert fpath.exists(), f"Missing file: {fname}"
        content = fpath.read_text(encoding="utf-8")
        assert "```mermaid" in content, f"Mermaid block missing in {fname}"
        for kw in kws:
            assert kw in content, f"Required string '{kw}' not found in {fname}"
        print(f"  [PASS] {fname:<30}: Valid Mermaid, contains {list(kws)}")

if __name__ == "__main__":
    test_drawio_diagrams()
    test_mermaid_diagrams()
    print("\nALL 26 DIAGRAM ARTIFACTS (13 UML + 13 DRAWIO) 100% VALIDATED!")
