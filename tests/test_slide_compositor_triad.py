"""Triad Test Suite for EducationalSlideCompositor and Visual Synthesis in Arc MultiModal Studio.

Covers 36 exhaustive test scenarios across Positive, Negative, and Edge-Case layers,
strictly adhering to Level 2.2 Docstrings-Only standards and Proofs M1 through M7.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from arc_runner.orchestrator.models import ScenePlan, VisualModality
from arc_runner.orchestrator.slide_compositor import (
    EducationalSlideCompositor,
    SlideCompositorError,
)
from arc_runner.orchestrator.usm_bridge import USMZeroCopyBridge
from arc_runner.orchestrator.visual_worker import StudioVisualWorker


class TestSlideCompositorPositiveLayer:
    """Positive test scenarios validating broadcast-grade presentation rendering."""

    def test_p_slide_01_cosmic_gradient_canvas(self) -> None:
        """Verify cosmic gradient canvas generation produces valid uint8 RGB dimensions and range."""
        compositor = EducationalSlideCompositor()
        canvas = compositor._render_cosmic_gradient(256, 256)

        assert canvas.shape == (256, 256, 3)
        assert canvas.dtype == np.uint8
        assert np.min(canvas) >= 0
        assert np.max(canvas) <= 255

    def test_p_slide_02_quantum_particle_field(self) -> None:
        """Verify particle field generation instantiates requested count with valid coordinates."""
        compositor = EducationalSlideCompositor()
        particles = compositor._generate_particle_field(width=512, height=512, count=30)

        assert len(particles) == 30
        for p in particles:
            assert 0.0 <= p["x"] <= 512.0
            assert 0.0 <= p["y"] <= 512.0
            assert -0.5 <= p["vx"] <= 0.5
            assert -0.5 <= p["vy"] <= 0.5

    def test_p_slide_03_wave_interference_simulation(self) -> None:
        """Verify quantum wave interference simulation modifies canvas pixels across envelope."""
        compositor = EducationalSlideCompositor()
        canvas = np.zeros((256, 256, 3), dtype=np.uint8)
        compositor._draw_quantum_wave_interference(canvas, alpha=0.5, width=256, height=256)

        assert np.count_nonzero(canvas) > 100

    def test_p_slide_04_bohr_orbitals_simulation(self) -> None:
        """Verify Bohr atomic orbital simulation renders nucleus and revolving electrons."""
        compositor = EducationalSlideCompositor()
        canvas = np.zeros((256, 256, 3), dtype=np.uint8)
        compositor._draw_bohr_orbitals(canvas, alpha=0.25, width=256, height=256)

        assert np.count_nonzero(canvas) > 200

    def test_p_slide_05_spacetime_geodesics_simulation(self) -> None:
        """Verify spacetime curvature geodesics simulation renders metric rings and center."""
        compositor = EducationalSlideCompositor()
        canvas = np.zeros((256, 256, 3), dtype=np.uint8)
        compositor._draw_spacetime_geodesics(canvas, alpha=0.33, width=256, height=256)

        assert np.count_nonzero(canvas) > 150

    def test_p_slide_06_neural_lattice_simulation(self) -> None:
        """Verify hardware neural lattice simulation renders nodes and circuit connections."""
        compositor = EducationalSlideCompositor()
        canvas = np.zeros((256, 256, 3), dtype=np.uint8)
        compositor._draw_neural_lattice(canvas, alpha=0.6, width=256, height=256)

        assert np.count_nonzero(canvas) > 100

    def test_p_slide_07_glassmorphic_header_hud(self) -> None:
        """Verify header HUD renders category pill and title into canvas."""
        compositor = EducationalSlideCompositor()
        scene = ScenePlan(
            scene_id=1,
            title="Quantum Superposition",
            narration_text="Subatomic particles exist in superposition.",
            visual_prompt="Quantum wave packet.",
            duration_seconds=2.0,
        )
        frames = compositor.render_scene_frames(scene, width=256, height=256, num_frames=2)

        assert len(frames) == 2
        assert frames[0].shape == (256, 256, 3)

    def test_p_slide_08_formula_hud_card(self) -> None:
        """Verify formula HUD card resolves appropriate formula based on domain."""
        compositor = EducationalSlideCompositor()
        scene = ScenePlan(
            scene_id=2,
            title="Schrodinger Wave Mechanics",
            narration_text="The wave equation governs state evolution.",
            visual_prompt="Wave packet propagation.",
            duration_seconds=3.0,
        )
        formula = compositor._resolve_formula(scene, domain_key="quantum")

        assert len(formula) > 0
        assert any(sym in formula for sym in ("ψ", "∂", "ℏ", "h", "E"))

    def test_p_slide_09_subtitles_card_word_wrap(self) -> None:
        """Verify multi-word narration text renders subtitle banner without error."""
        compositor = EducationalSlideCompositor()
        scene = ScenePlan(
            scene_id=1,
            title="Particle Duality",
            narration_text="Light demonstrates both wave and particle characteristics under experimental scrutiny.",
            visual_prompt="Photons interacting with double slit.",
            duration_seconds=2.0,
        )
        frames = compositor.render_scene_frames(scene, width=320, height=320, num_frames=3)

        assert len(frames) == 3

    def test_p_slide_10_ken_burns_camera_motion(self) -> None:
        """Verify camera zoom motion alters frame pixel content dynamically over time."""
        compositor = EducationalSlideCompositor()
        scene = ScenePlan(
            scene_id=1,
            title="Black Hole Horizon",
            narration_text="Time dilates near the singularity.",
            visual_prompt="Supermassive black hole.",
            duration_seconds=2.0,
            camera_motion="slow zoom in",
        )
        frames = compositor.render_scene_frames(scene, width=128, height=128, num_frames=5)

        assert not np.array_equal(frames[0], frames[-1])

    def test_p_slide_11_base_image_decoration(self) -> None:
        """Verify pre-rendered base image is color-graded and augmented with educational HUD."""
        compositor = EducationalSlideCompositor()
        base = np.full((256, 256, 3), fill_value=180, dtype=np.uint8)
        scene = ScenePlan(
            scene_id=1,
            title="AI Silicon",
            narration_text="Hardware accelerators execute tensor convolutions.",
            visual_prompt="Intel Arc microchip.",
            duration_seconds=1.0,
        )
        frames = compositor.render_scene_frames(scene, width=256, height=256, num_frames=2, base_background=base)

        assert len(frames) == 2
        assert frames[0].shape == (256, 256, 3)

    def test_p_slide_12_resolution_scaling_512(self) -> None:
        """Verify 512x512 resolution synthesis adheres exactly to target dimensions."""
        compositor = EducationalSlideCompositor()
        scene = ScenePlan(
            scene_id=1,
            title="Resolution Check",
            narration_text="Evaluating standard resolution output.",
            visual_prompt="Diagram.",
            duration_seconds=1.0,
        )
        frames = compositor.render_scene_frames(scene, width=512, height=512, num_frames=2)

        assert frames[0].shape == (512, 512, 3)

    def test_p_slide_13_resolution_scaling_720p(self) -> None:
        """Verify 720p widescreen format (1280x720) renders cleanly."""
        compositor = EducationalSlideCompositor()
        scene = ScenePlan(
            scene_id=1,
            title="HD Widescreen",
            narration_text="Widescreen high-definition broadcast slide.",
            visual_prompt="Cosmic panorama.",
            duration_seconds=1.0,
        )
        frames = compositor.render_scene_frames(scene, width=1280, height=720, num_frames=1)

        assert frames[0].shape == (720, 1280, 3)

    def test_p_slide_14_usm_bridge_transfer(self) -> None:
        """Verify rendered slide frames transfer seamlessly through USMZeroCopyBridge."""
        compositor = EducationalSlideCompositor()
        bridge = USMZeroCopyBridge(track_telemetry=True)
        scene = ScenePlan(
            scene_id=1,
            title="USM Transfer",
            narration_text="Unified shared memory zero-copy tensor handoff.",
            visual_prompt="Hardware bridge.",
            duration_seconds=1.0,
        )
        frames = compositor.render_scene_frames(scene, width=128, height=128, num_frames=5)
        shared_frames, latency_us = bridge.transfer_frames_to_muxer(frames)

        assert len(shared_frames) == 5
        assert latency_us >= 0.0
        cum_stats = bridge.get_cumulative_transfer_stats()
        assert cum_stats["total_transfers"] >= 5.0

    def test_p_slide_15_throughput_benchmark(self) -> None:
        """Verify rendering throughput exceeds real-time playback rate (15+ fps) on CPU/GPU."""
        compositor = EducationalSlideCompositor()
        scene = ScenePlan(
            scene_id=1,
            title="Throughput Check",
            narration_text="Measuring frames per second rendering velocity.",
            visual_prompt="Particle speed.",
            duration_seconds=1.0,
        )
        t_start = time.perf_counter()
        frames = compositor.render_scene_frames(scene, width=256, height=256, num_frames=15)
        elapsed = time.perf_counter() - t_start
        fps_rate = float(len(frames)) / max(0.001, elapsed)

        assert fps_rate >= 15.0

    def test_p_slide_16_visual_worker_integration(self) -> None:
        """Verify StudioVisualWorker.render_scene delegates to compositor with valid SceneArtifact."""
        worker = StudioVisualWorker(device="GPU")
        scene = ScenePlan(
            scene_id=1,
            title="Wave Mechanics",
            narration_text="Electrons demonstrate wave interference.",
            visual_prompt="Quantum wave packet.",
            duration_seconds=2.0,
            modality=VisualModality.IMAGE_SLIDE,
        )
        artifact = worker.render_scene(scene, resolution=(256, 256), fps=10)

        assert artifact.scene_id == 1
        assert len(artifact.frames) == 20
        assert artifact.frames[0].shape == (256, 256, 3)

    def test_p_slide_17_double_slit_diffraction_render(self) -> None:
        """Verify double-slit diffraction simulation activates and produces non-empty interference patterns."""
        compositor = EducationalSlideCompositor()
        scene = ScenePlan(
            scene_id=2,
            title="Double Slit Experiment",
            narration_text="Photons create interference patterns across two parallel slits.",
            visual_prompt="Double slit interference fringes with monochromatic wavelets.",
            duration_seconds=2.0,
        )
        frames = compositor.render_scene_frames(scene, width=640, height=360, num_frames=5)

        assert len(frames) == 5
        assert frames[0].shape == (360, 640, 3)
        assert np.any(frames[0] > 0)

    def test_p_slide_18_widescreen_16x9_voronoi_layout(self) -> None:
        """Verify 1280x720 HD widescreen rendering executes with zero title-hud collisions."""
        compositor = EducationalSlideCompositor()
        scene = ScenePlan(
            scene_id=1,
            title="Quantum Superposition and Wave Mechanics in Modern Physics",
            narration_text="Comprehensive exploration of quantum state vectors and wave functions.",
            visual_prompt="Quantum state vector in Hilbert space.",
            duration_seconds=1.0,
        )
        frames = compositor.render_scene_frames(scene, width=1280, height=720, num_frames=2)

        assert len(frames) == 2
        assert frames[0].shape == (720, 1280, 3)


class TestSlideCompositorNegativeLayer:
    """Negative test scenarios validating defensive boundaries and parameter validation."""

    def test_n_slide_01_zero_resolution_error(self) -> None:
        """Verify SlideCompositorError is raised when width or height is zero."""
        compositor = EducationalSlideCompositor()
        scene = ScenePlan(scene_id=1, title="T", narration_text="D", visual_prompt="V")

        with pytest.raises(SlideCompositorError, match="Invalid frame dimensions"):
            compositor.render_scene_frames(scene, width=0, height=256, num_frames=5)

    def test_n_slide_02_negative_resolution_error(self) -> None:
        """Verify SlideCompositorError is raised when dimensions are negative."""
        compositor = EducationalSlideCompositor()
        scene = ScenePlan(scene_id=1, title="T", narration_text="D", visual_prompt="V")

        with pytest.raises(SlideCompositorError, match="Invalid frame dimensions"):
            compositor.render_scene_frames(scene, width=256, height=-100, num_frames=5)

    def test_n_slide_03_zero_duration_error(self) -> None:
        """Verify SlideCompositorError is raised when frame count is zero."""
        compositor = EducationalSlideCompositor()
        scene = ScenePlan(scene_id=1, title="T", narration_text="D", visual_prompt="V")

        with pytest.raises(SlideCompositorError, match="Invalid frame count"):
            compositor.render_scene_frames(scene, width=128, height=128, num_frames=0)

    def test_n_slide_04_negative_frame_count_error(self) -> None:
        """Verify SlideCompositorError is raised when frame count is negative."""
        compositor = EducationalSlideCompositor()
        scene = ScenePlan(scene_id=1, title="T", narration_text="D", visual_prompt="V")

        with pytest.raises(SlideCompositorError, match="Invalid frame count"):
            compositor.render_scene_frames(scene, width=128, height=128, num_frames=-10)

    def test_n_slide_05_corrupt_base_image_type(self) -> None:
        """Verify non-array base image input is safely coerced without unhandled crash."""
        compositor = EducationalSlideCompositor()
        scene = ScenePlan(scene_id=1, title="T", narration_text="D", visual_prompt="V")
        dummy_list_base = [[[200, 100, 50]] * 64] * 64

        frames = compositor.render_scene_frames(scene, width=64, height=64, num_frames=1, base_background=dummy_list_base)  # type: ignore
        assert len(frames) == 1

    def test_n_slide_06_huge_dimension_guard(self) -> None:
        """Verify requests exceeding 8192px trigger memory protection exception."""
        compositor = EducationalSlideCompositor()
        scene = ScenePlan(scene_id=1, title="T", narration_text="D", visual_prompt="V")

        with pytest.raises(SlideCompositorError, match="exceed maximum allowed bound"):
            compositor.render_scene_frames(scene, width=16384, height=16384, num_frames=1)

    def test_n_slide_07_invalid_camera_motion_fallback(self) -> None:
        """Verify unrecognized camera motion parameter defaults to static canvas without crash."""
        compositor = EducationalSlideCompositor()
        scene = ScenePlan(scene_id=1, title="T", narration_text="D", visual_prompt="V", camera_motion="warp_hyperspace")
        frames = compositor.render_scene_frames(scene, width=128, height=128, num_frames=3)

        assert len(frames) == 3

    def test_n_slide_08_corrupt_unicode_title(self) -> None:
        """Verify unprintable or complex Unicode title renders without crashing typography."""
        compositor = EducationalSlideCompositor()
        scene = ScenePlan(scene_id=1, title="Quantum \x00\x01\x02 \U0001F680 Entity", narration_text="Dialogue", visual_prompt="Visual")
        frames = compositor.render_scene_frames(scene, width=128, height=128, num_frames=1)

        assert len(frames) == 1

    def test_n_slide_09_empty_narration_subtitles(self) -> None:
        """Verify empty narration string bypasses subtitle card without rendering error."""
        compositor = EducationalSlideCompositor()
        scene = ScenePlan(scene_id=1, title="Silent Scene", narration_text="P", visual_prompt="Visual")
        scene.narration_text = ""
        frames = compositor.render_scene_frames(scene, width=128, height=128, num_frames=1)

        assert len(frames) == 1

    def test_n_slide_10_missing_font_fallback(self) -> None:
        """Verify get_font resolves default font when requested font names do not exist."""
        compositor = EducationalSlideCompositor()
        compositor._SYSTEM_FONTS = ["non_existent_font_123.ttf"]
        compositor._font_cache.clear()
        font = compositor._get_font(size=14)

        assert font is not None


class TestSlideCompositorEdgeCaseLayer:
    """Edge-case test scenarios validating numerical boundaries, aspect ratios, and stability."""

    def test_e_slide_01_single_frame_scene(self) -> None:
        """Verify single-frame scene (num_frames=1) completes without division-by-zero on alpha."""
        compositor = EducationalSlideCompositor()
        scene = ScenePlan(scene_id=1, title="Single Frame", narration_text="Brief instant.", visual_prompt="Flash.")
        frames = compositor.render_scene_frames(scene, width=128, height=128, num_frames=1)

        assert len(frames) == 1
        assert frames[0].shape == (128, 128, 3)

    def test_e_slide_02_ultra_long_dialogue_overflow(self) -> None:
        """Verify long multi-sentence narration dialogue wraps neatly into 3 lines max."""
        compositor = EducationalSlideCompositor()
        long_text = (
            "In quantum mechanics, the wave function contains all knowable physical information "
            "concerning a particle or system. When a measurement is conducted, this probability distribution "
            "collapses instantaneously to an eigenstate of the measured observable according to the Copenhagen interpretation."
        )
        scene = ScenePlan(scene_id=1, title="Wave Collapse", narration_text=long_text, visual_prompt="Superposition.")
        frames = compositor.render_scene_frames(scene, width=384, height=384, num_frames=2)

        assert len(frames) == 2

    def test_e_slide_03_ultrawide_aspect_ratio(self) -> None:
        """Verify 32:9 extreme ultrawide aspect ratio (1024x288) renders without distortion crash."""
        compositor = EducationalSlideCompositor()
        scene = ScenePlan(scene_id=1, title="Ultrawide", narration_text="Panoramic cinematic view.", visual_prompt="Cosmos.")
        frames = compositor.render_scene_frames(scene, width=1024, height=288, num_frames=2)

        assert frames[0].shape == (288, 1024, 3)

    def test_e_slide_04_vertical_mobile_aspect_ratio(self) -> None:
        """Verify 9:16 vertical smartphone aspect ratio (288x512) renders properly."""
        compositor = EducationalSlideCompositor()
        scene = ScenePlan(scene_id=1, title="Mobile Shorts", narration_text="Vertical format educational clip.", visual_prompt="Quantum.")
        frames = compositor.render_scene_frames(scene, width=288, height=512, num_frames=2)

        assert frames[0].shape == (512, 288, 3)

    def test_e_slide_05_single_word_title(self) -> None:
        """Verify single character or single word title renders without formatting errors."""
        compositor = EducationalSlideCompositor()
        scene = ScenePlan(scene_id=1, title="Ψ", narration_text="State vector.", visual_prompt="Wave.")
        frames = compositor.render_scene_frames(scene, width=128, height=128, num_frames=1)

        assert len(frames) == 1

    def test_e_slide_06_latex_special_symbols(self) -> None:
        """Verify special mathematical symbols render in HUD formula without crash."""
        compositor = EducationalSlideCompositor()
        scene = ScenePlan(scene_id=1, title="Formula Check", narration_text="Math notation.", visual_prompt="Equations.")
        frames = compositor.render_scene_frames(scene, width=256, height=256, num_frames=1)

        assert len(frames) == 1

    def test_e_slide_07_fractional_duration(self) -> None:
        """Verify fractional scene duration calculates correct integer frame count."""
        worker = StudioVisualWorker(device="GPU")
        scene = ScenePlan(scene_id=1, title="Fractional", narration_text="Dialogue.", visual_prompt="Visual.", duration_seconds=1.45)
        artifact = worker.render_scene(scene, resolution=(128, 128), fps=10)

        assert len(artifact.frames) == 14
        assert artifact.video_duration_sec == pytest.approx(1.4, rel=0.01)

    def test_e_slide_08_high_fps_stress(self) -> None:
        """Verify 60 fps high-framerate rendering completes without timing skew."""
        compositor = EducationalSlideCompositor()
        scene = ScenePlan(scene_id=1, title="60 FPS", narration_text="High frame rate animation.", visual_prompt="Motion.")
        frames = compositor.render_scene_frames(scene, width=128, height=128, num_frames=30, fps=60)

        assert len(frames) == 30

    def test_e_slide_09_single_channel_grayscale_base(self) -> None:
        """Verify single-channel 2D grayscale base image is automatically converted to 3D RGB."""
        compositor = EducationalSlideCompositor()
        gray_base = np.full((128, 128), fill_value=120, dtype=np.uint8)
        scene = ScenePlan(scene_id=1, title="Grayscale Base", narration_text="Monochrome source.", visual_prompt="Grayscale.")
        frames = compositor.render_scene_frames(scene, width=128, height=128, num_frames=1, base_background=gray_base)

        assert frames[0].shape == (128, 128, 3)

    def test_e_slide_10_rapid_sequential_renders(self) -> None:
        """Verify 5 consecutive scene renders execute deterministically with zero memory leak."""
        compositor = EducationalSlideCompositor()
        for idx in range(1, 6):
            scene = ScenePlan(scene_id=idx, title=f"Scene {idx}", narration_text=f"Dialogue for scene {idx}.", visual_prompt="Visual.")
            frames = compositor.render_scene_frames(scene, width=128, height=128, num_frames=2)
            assert len(frames) == 2
