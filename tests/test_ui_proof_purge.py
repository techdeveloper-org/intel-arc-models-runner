"""Automated test suite verifying complete eradication of fabricated proof claims.

Ensures zero remnants of static marketing claims, hardcoded speedups, or fake ROI
exist in any served HTML templates, JS scripts, or styles.
"""

from pathlib import Path
import re
import pytest

TEMPLATES_DIR = Path(__file__).parent.parent / "src" / "arc_runner" / "server" / "templates"
STATIC_DIR = Path(__file__).parent.parent / "src" / "arc_runner" / "server" / "static"


def test_no_hardcoded_speedup_in_templates() -> None:
    """Verify '1.86x Active' is completely absent from all HTML templates."""
    for html_file in TEMPLATES_DIR.glob("*.html"):
        content = html_file.read_text(encoding="utf-8")
        assert "1.86x Active" not in content, f"Hardcoded 1.86x found in {html_file}"
        assert "SPEEDUP ~1.86x" not in content, f"Hardcoded speedup found in {html_file}"


def test_no_proof_m_tokens_in_html_ui() -> None:
    """Verify internal Proof M assertions are not rendered in user-facing templates."""
    for html_file in TEMPLATES_DIR.glob("*.html"):
        content = html_file.read_text(encoding="utf-8")
        matches = re.findall(r"Proof M\d+", content)
        assert len(matches) == 0, f"Lingering test proofs found in {html_file}: {matches}"


def test_no_fake_roi_calculator_in_js_or_html() -> None:
    """Verify fake pricing ROI formula (updateRoiMetrics) is purged from JS and HTML."""
    index_html = (TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")
    assert "Cloud API Replacement ROI" not in index_html
    assert "roi-monthly-savings" not in index_html

    for js_file in STATIC_DIR.glob("**/*.js"):
        content = js_file.read_text(encoding="utf-8")
        assert "updateRoiMetrics" not in content, f"updateRoiMetrics found in {js_file}"


def test_dynamic_telemetry_containers_present_in_index() -> None:
    """Verify dynamic hardware and silicon chips exist in index.html for real-time telemetry."""
    index_html = (TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")
    assert 'id="header-device-chip"' in index_html
    assert 'id="header-server-status"' in index_html
    assert 'id="header-model-chip"' in index_html
    assert 'id="gpu-usm-badge"' in index_html
    assert 'id="gpu-vram-bar"' in index_html
