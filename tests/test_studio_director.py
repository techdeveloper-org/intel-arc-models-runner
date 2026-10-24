"""Triad unit tests for StudioDirector LLM Agent and JSON AST Auto-Repair.

Validates positive prompt decomposition, negative boundary rejections, and edge-case
malformed JSON parsing repair (trailing commas, unclosed brackets, markdown backticks).
Strictly adheres to Level 2.2 Docstrings-Only standards and Proof M4.
"""

import pytest

from arc_runner.orchestrator.director import StudioDirector
from arc_runner.orchestrator.models import VisualModality


class TestStudioDirectorPositive:
    """Happy-path validation tests for StudioDirector agent."""

    def test_director_decompose_prompt_heuristic(self) -> None:
        """Verify decompose_prompt produces valid ScenePlan sequence from brief."""
        director = StudioDirector()
        scenes = director.decompose_prompt(
            prompt="Explain Quantum Entanglement and Bell States",
            target_duration_sec=20.0,
            visual_modality=VisualModality.ANIMATED_VIDEO,
        )

        assert len(scenes) >= 2
        total_duration = sum(s.duration_seconds for s in scenes)
        assert abs(total_duration - 20.0) <= 2.0
        assert all(s.modality == VisualModality.ANIMATED_VIDEO for s in scenes)
        assert all(len(s.narration_text) > 10 for s in scenes)
        assert all(len(s.visual_prompt) > 10 for s in scenes)

    def test_director_scene_indexing_sequential(self) -> None:
        """Verify generated scenes carry sequential 1-based scene_id indices."""
        director = StudioDirector()
        scenes = director.decompose_prompt("Photosynthesis in plants", target_duration_sec=15.0)

        for idx, scene in enumerate(scenes, start=1):
            assert scene.scene_id == idx


class TestStudioDirectorNegative:
    """Negative error-boundary validation tests for StudioDirector."""

    def test_director_empty_prompt_raises_value_error(self) -> None:
        """Verify decompose_prompt raises ValueError on empty or whitespace brief."""
        director = StudioDirector()
        with pytest.raises(ValueError) as exc_info:
            director.decompose_prompt("   ")
        assert "cannot be empty" in str(exc_info.value)

    def test_director_invalid_duration_bounds(self) -> None:
        """Verify decompose_prompt rejects duration values outside [2.0, 600.0]."""
        director = StudioDirector()
        with pytest.raises(ValueError):
            director.decompose_prompt("Valid prompt", target_duration_sec=1.0)

        with pytest.raises(ValueError):
            director.decompose_prompt("Valid prompt", target_duration_sec=700.0)


class TestStudioDirectorASTAutoRepairEdgeCases:
    """Edge-case tests verifying resilient JSON AST auto-repair."""

    def test_repair_markdown_wrapped_json(self) -> None:
        """Verify markdown code block wrapped JSON is sanitized and parsed."""
        raw = (
            "Here is your script:\n"
            "```json\n"
            "[\n"
            '  {"scene_id": 1, "title": "Intro", "narration_text": "Hello world.", "visual_prompt": "World.", "duration_seconds": 5.0}\n'
            "]\n"
            "```\n"
            "Enjoy the video!"
        )
        director = StudioDirector()
        repaired = director._clean_and_repair_json(raw)
        assert repaired.startswith("[")
        assert repaired.endswith("]")
        assert "Hello world." in repaired

    def test_repair_trailing_comma_json(self) -> None:
        """Verify trailing commas before closing braces/brackets are removed."""
        raw = (
            "<<<STUDIO_PLAN_START>>>\n"
            "[\n"
            '  {"scene_id": 1, "title": "S1", "narration_text": "T1", "visual_prompt": "V1", "duration_seconds": 5.0,},\n'
            "]\n"
            "<<<STUDIO_PLAN_END>>>"
        )
        director = StudioDirector()
        repaired = director._clean_and_repair_json(raw)
        assert ",}" not in repaired
        assert ",]" not in repaired

    def test_repair_unclosed_bracket_json(self) -> None:
        """Verify truncated JSON missing closing bracket is auto-balanced."""
        raw = (
            "<<<STUDIO_PLAN_START>>>\n"
            "[\n"
            '  {"scene_id": 1, "title": "S1", "narration_text": "T1", "visual_prompt": "V1", "duration_seconds": 5.0}'
        )
        director = StudioDirector()
        repaired = director._clean_and_repair_json(raw)
        assert repaired.endswith("]")

    def test_fallback_heuristic_on_total_garbage_output(self) -> None:
        """Verify fallback segmentation activates when LLM output is non-JSON text."""
        director = StudioDirector()
        scenes = director._fallback_heuristic_segmentation("Relativity theory", 24.0, VisualModality.IMAGE_SLIDE)
        assert len(scenes) >= 2
        assert all(isinstance(s.scene_id, int) for s in scenes)
        assert sum(s.duration_seconds for s in scenes) > 0.0
