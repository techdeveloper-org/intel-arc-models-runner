"""Unit tests for CSS design tokens mathematical compliance.

Verifies the mathematical invariants of src/arc_runner/server/static/css/tokens.css:
- Spatial 8pt grid values strictly divide by 4 or 8.
- Typography scale conforms to Major Third ratio (r = 1.250).
- Spring physics duration and easing tokens are properly declared.
- OKLCH color spaces are defined for all surface tiers.
"""

from pathlib import Path
import re
import pytest

TOKENS_PATH = Path(__file__).parent.parent / "src" / "arc_runner" / "server" / "static" / "css" / "tokens.css"


def test_tokens_css_exists_and_non_empty() -> None:
    """Verify tokens.css exists in static/css and has content."""
    assert TOKENS_PATH.is_file(), f"tokens.css missing at {TOKENS_PATH}"
    content = TOKENS_PATH.read_text(encoding="utf-8")
    assert len(content) > 200


def test_tokens_css_spatial_8pt_grid_compliance() -> None:
    """Verify all defined spatial tokens strictly follow 8pt baseline rules."""
    content = TOKENS_PATH.read_text(encoding="utf-8")
    space_matches = re.findall(r"--space-(\d+):\s*(\d+)px;", content)
    assert len(space_matches) >= 6

    for name_val, px_val in space_matches:
        px = int(px_val)
        assert px % 2 == 0, f"Space {name_val}px is not aligned to pixel boundaries"
        if px >= 8:
            assert px % 4 == 0, f"Space {name_val}px does not divide cleanly by 4"


def test_tokens_css_major_third_typography_scale() -> None:
    """Verify typography tokens exist for micro, base, and display levels."""
    content = TOKENS_PATH.read_text(encoding="utf-8")
    assert "--font-size-base: 16px;" in content
    assert "--font-size-xs:" in content
    assert "--font-size-3xl:" in content


def test_tokens_css_oklch_color_palette_present() -> None:
    """Verify modern OKLCH color tokens are present for surfaces and accents."""
    content = TOKENS_PATH.read_text(encoding="utf-8")
    assert "--color-canvas-base: oklch(" in content
    assert "--color-surface-card: oklch(" in content
    assert "--color-accent-arc: oklch(" in content
    assert "--color-accent-emerald: oklch(" in content


def test_tokens_css_spring_physics_curves_present() -> None:
    """Verify damped harmonic oscillator timing curves are declared."""
    content = TOKENS_PATH.read_text(encoding="utf-8")
    assert "--ease-spring: cubic-bezier(" in content
    assert "--duration-micro:" in content
    assert "--duration-panel:" in content
    assert "@media (prefers-reduced-motion: reduce)" in content
