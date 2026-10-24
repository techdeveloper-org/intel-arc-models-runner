"""Broadcast-Grade Educational Presentation Slide & Simulation Engine for Arc MultiModal Studio.

Renders broadcast-quality presentation slides with dynamic quantum wave packets,
Bohr-Sommerfeld atomic orbital kinematics, glassmorphic HUD cards, authentic LaTeX formulas,
and synchronized narration dialogue subtitles.
Strictly adheres to Level 2.2 Docstrings-Only standards and Proofs M1 through M7.
"""

from __future__ import annotations

import math
from typing import ClassVar

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from arc_runner.core.exceptions import ArcRunnerError
from arc_runner.core.logging import get_logger
from arc_runner.orchestrator.models import ScenePlan

logger = get_logger("orchestrator.slide_compositor")


class SlideCompositorError(ArcRunnerError):
    """Raised when slide generation or compositing experiences an unrecoverable failure."""


class EducationalSlideCompositor:
    """High-fidelity broadcast graphics compositor rendering animated educational slides."""

    _SYSTEM_FONTS: ClassVar[list[str]] = [
        "segoeui.ttf",
        "segoeuib.ttf",
        "arial.ttf",
        "arialbd.ttf",
        "calibri.ttf",
        "DejaVuSans.ttf",
    ]

    _FORMULA_CATALOG: ClassVar[dict[str, list[str]]] = {
        "quantum": [
            "|ψ(t)> = α|0> + β|1>",
            "i·h_bar (∂ψ/∂t) = H·ψ",
            "Δx · Δp ≥ h_bar / 2",
            "E = h·ν = h_bar·ω",
            "I(y) = I₀ · cos²(β) · sinc²(α)",
        ],
        "astrophysics": [
            "E² = (pc)² + (m₀c²)²",
            "R_μν - 1/2 R g_μν = 8πG/c⁴ T_μν",
            "r_s = 2GM / c²",
            "v = H₀ · d",
        ],
        "hardware": [
            "FLOPS = Cores × Freq × DPAS_width",
            "Bandwidth = Bus_width × Freq_eff / 8",
            "P_dynamic = α · C · V² · f",
            "T_exec = IC × CPI × τ",
        ],
        "general": [
            "∇ · E = ρ / ε₀",
            "∇ × B = μ₀J + μ₀ε₀ ∂E/∂t",
            "S = k_B · ln(Ω)",
            "dE = T dS - P dV",
        ],
    }

    def __init__(self) -> None:
        """Initialize EducationalSlideCompositor and resolve optimal typography assets."""
        self._font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}

    def render_scene_frames(
        self,
        scene: ScenePlan,
        width: int,
        height: int,
        num_frames: int,
        fps: int = 16,
        base_background: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """Render complete sequence of broadcast-grade presentation frames for a scene.

        Args:
            scene: Validated ScenePlan containing title, narration, visual cues, and duration.
            width: Output raster frame width in pixels.
            height: Output raster frame height in pixels.
            num_frames: Total number of frames to synthesize.
            fps: Playback framerate in frames per second.
            base_background: Optional pre-rendered AI diffusion base image of shape (H, W, 3).

        Returns:
            List of RGB uint8 numpy arrays of shape (height, width, 3).

        Raises:
            SlideCompositorError: When resolution or frame count parameters are invalid.
        """
        if width <= 0 or height <= 0:
            raise SlideCompositorError(f"Invalid frame dimensions ({width}x{height}); dimensions must be positive.")
        if num_frames <= 0:
            raise SlideCompositorError(f"Invalid frame count ({num_frames}); count must be at least 1.")
        if width > 8192 or height > 8192:
            raise SlideCompositorError(f"Requested dimensions ({width}x{height}) exceed maximum allowed bound of 8192.")

        domain_key = self._detect_domain_key(scene)
        formula = self._resolve_formula(scene, domain_key)
        particles = self._generate_particle_field(width, height, count=min(60, max(15, width // 16)))

        frames: list[np.ndarray] = []

        for i in range(num_frames):
            alpha = float(i) / max(1.0, float(num_frames - 1))

            if base_background is not None:
                canvas = self._prepare_decorated_base(base_background, width, height)
            else:
                canvas = self._render_cosmic_gradient(width, height)
                self._draw_particles(canvas, particles, alpha, width, height)
                self._draw_physics_simulation(canvas, domain_key, alpha, width, height, scene=scene)

            canvas_pil = Image.fromarray(canvas)
            draw = ImageDraw.Draw(canvas_pil, "RGBA")

            self._draw_header_hud(draw, scene, domain_key, width, height)
            self._draw_formula_card(draw, formula, width, height)
            self._draw_subtitles_card(draw, scene.narration_text, width, height)

            frame_np = np.array(canvas_pil.convert("RGB"), dtype=np.uint8)

            if scene.camera_motion and num_frames > 1:
                frame_np = self._apply_camera_motion(frame_np, alpha, scene.camera_motion)

            frames.append(frame_np)

        return frames

    def _detect_domain_key(self, scene: ScenePlan) -> str:
        """Identify scientific domain key from scene title and prompt text.

        Args:
            scene: Target ScenePlan item.

        Returns:
            Domain identifier string ('quantum', 'astrophysics', 'hardware', 'general').
        """
        corpus = f"{scene.title} {scene.visual_prompt} {scene.narration_text}".lower()
        if any(k in corpus for k in ("quantum", "wave", "particle", "schrodinger", "bohr", "electron", "slit", "spin")):
            return "quantum"
        if any(k in corpus for k in ("black hole", "relativity", "gravity", "spacetime", "star", "galaxy", "cosmic")):
            return "astrophysics"
        if any(k in corpus for k in ("intel", "arc", "gpu", "npu", "cpu", "silicon", "hardware", "neural", "tensor")):
            return "hardware"
        return "general"

    def _resolve_formula(self, scene: ScenePlan, domain_key: str) -> str:
        """Select authentic scientific formula corresponding to scene semantics.

        Args:
            scene: Target ScenePlan item.
            domain_key: Resolved domain identifier string.

        Returns:
            Display-ready mathematical equation string.
        """
        catalog = self._FORMULA_CATALOG.get(domain_key, self._FORMULA_CATALOG["general"])
        idx = (scene.scene_id - 1) % len(catalog)
        return catalog[idx]

    def _get_font(self, size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """Resolve cached TrueType font or gracefully fall back to default bitmap font.

        Args:
            size: Desired font size in points.
            bold: Whether bold typography variant is preferred.

        Returns:
            Resolved ImageFont instance.
        """
        key = ("bold" if bold else "regular", size)
        if key in self._font_cache:
            return self._font_cache[key]

        selected_font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None
        target_names = [f for f in self._SYSTEM_FONTS if ("bd" in f or "b." in f)] if bold else self._SYSTEM_FONTS

        for name in target_names:
            try:
                selected_font = ImageFont.truetype(name, size=size)
                break
            except (OSError, ImportError):
                continue

        if selected_font is None:
            try:
                selected_font = ImageFont.truetype("arial.ttf", size=size)
            except (OSError, ImportError):
                selected_font = ImageFont.load_default()

        self._font_cache[key] = selected_font
        return selected_font

    def _render_cosmic_gradient(self, width: int, height: int) -> np.ndarray:
        """Synthesize deep-space cosmic dark gradient canvas adhering to Proof M6.

        Args:
            width: Raster width in pixels.
            height: Raster height in pixels.

        Returns:
            RGB uint8 numpy array canvas of shape (height, width, 3).
        """
        top_color = np.array([10, 15, 29], dtype=np.float32)
        bottom_color = np.array([2, 4, 10], dtype=np.float32)

        v_factors = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
        canvas = top_color * (1.0 - v_factors) + bottom_color * v_factors

        center_x, center_y = width / 2.0, height * 0.42
        max_dist = math.sqrt(center_x**2 + center_y**2)

        y_coords, x_coords = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
        glow = np.clip(1.0 - (dist_from_center / max_dist), 0.0, 1.0) ** 2.2

        glow_tint = np.array([0, 40, 70], dtype=np.float32)
        canvas = np.clip(canvas + glow[:, :, None] * glow_tint, 0.0, 255.0)

        return canvas.astype(np.uint8)

    def _prepare_decorated_base(self, base_image: np.ndarray, width: int, height: int) -> np.ndarray:
        """Resize and color-grade pre-rendered AI diffusion base image for HUD compositing.

        Args:
            base_image: Raw input image array.
            width: Target canvas width.
            height: Target canvas height.

        Returns:
            Color-graded RGB uint8 numpy array.
        """
        if not isinstance(base_image, np.ndarray):
            base_image = np.array(base_image)

        if base_image.shape[0] != height or base_image.shape[1] != width:
            resized = cv2.resize(base_image, (width, height), interpolation=cv2.INTER_AREA)
        else:
            resized = base_image.copy()

        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        elif resized.shape[2] == 4:
            resized = cv2.cvtColor(resized, cv2.COLOR_RGBA2RGB)

        graded = (resized.astype(np.float32) * 0.65).astype(np.uint8)
        overlay = self._render_cosmic_gradient(width, height)
        blended = cv2.addWeighted(graded, 0.70, overlay, 0.30, 0.0)
        return blended

    def _generate_particle_field(self, width: int, height: int, count: int = 40) -> list[dict[str, float]]:
        """Instantiate initial stochastic quantum vacuum particles for Brownian drift.

        Args:
            width: Canvas width.
            height: Canvas height.
            count: Number of particle elements.

        Returns:
            List of particle state dictionaries containing coordinates, velocities, and phase.
        """
        rng = np.random.default_rng(seed=1337)
        particles: list[dict[str, float]] = []

        for _ in range(count):
            particles.append({
                "x": float(rng.uniform(0, width)),
                "y": float(rng.uniform(0, height)),
                "vx": float(rng.uniform(-0.4, 0.4)),
                "vy": float(rng.uniform(-0.3, 0.3)),
                "phase": float(rng.uniform(0, math.tau)),
                "radius": float(rng.uniform(1.0, 2.5)),
            })

        return particles

    def _draw_particles(
        self,
        canvas: np.ndarray,
        particles: list[dict[str, float]],
        alpha: float,
        width: int,
        height: int,
    ) -> None:
        """Render drifting quantum vacuum particles with sinusoidal brightness oscillation.

        Args:
            canvas: Target RGB numpy array canvas.
            particles: Particle state list.
            alpha: Normalized timeline progress [0.0, 1.0].
            width: Frame width.
            height: Frame height.
        """
        for p in particles:
            curr_x = int((p["x"] + p["vx"] * alpha * 40.0) % width)
            curr_y = int((p["y"] + p["vy"] * alpha * 40.0) % height)

            intensity = 0.5 + 0.5 * math.sin(alpha * math.tau * 2.0 + p["phase"])
            brightness = int(140 + 115 * intensity)
            color = (int(brightness * 0.7), int(brightness * 0.9), brightness)
            r = max(1, round(p["radius"]))

            cv2.circle(canvas, (curr_x, curr_y), r, color, -1, lineType=cv2.LINE_AA)

    def _draw_physics_simulation(
        self,
        canvas: np.ndarray,
        domain_key: str,
        alpha: float,
        width: int,
        height: int,
        scene: ScenePlan | None = None,
    ) -> None:
        """Dispatch domain-specific dynamic physics simulation renderer.

        Args:
            canvas: Target RGB numpy array canvas.
            domain_key: Domain identifier string.
            alpha: Normalized progress parameter [0.0, 1.0].
            width: Canvas width.
            height: Canvas height.
            scene: Optional target ScenePlan to inspect fine-grained physics cues.
        """
        scene_cue = f"{scene.title} {scene.visual_prompt}".lower() if scene else ""

        if domain_key == "quantum":
            if any(k in scene_cue for k in ("slit", "duality", "interference")):
                self._draw_double_slit_experiment(canvas, alpha, width, height)
            elif any(k in scene_cue for k in ("atom", "bohr", "electron", "realm")):
                self._draw_bohr_orbitals(canvas, alpha, width, height)
            elif any(k in scene_cue for k in ("superposition", "schrodinger", "packet")):
                self._draw_quantum_wave_interference(canvas, alpha, width, height)
            elif width >= 256:
                self._draw_quantum_wave_interference(canvas, alpha, width, height)
                self._draw_bohr_orbitals(canvas, alpha, width, height)
            else:
                self._draw_bohr_orbitals(canvas, alpha, width, height)
        elif domain_key == "astrophysics":
            self._draw_spacetime_geodesics(canvas, alpha, width, height)
        elif domain_key == "hardware":
            self._draw_neural_lattice(canvas, alpha, width, height)
        else:
            self._draw_quantum_wave_interference(canvas, alpha, width, height)

    def _draw_double_slit_experiment(
        self,
        canvas: np.ndarray,
        alpha: float,
        width: int,
        height: int,
    ) -> None:
        """Render Young's double-slit experiment showing wave-particle duality (Proof M10).

        Args:
            canvas: Target RGB canvas.
            alpha: Timeline progress.
            width: Width in pixels.
            height: Height in pixels.
        """
        center_y = int(height * 0.48)
        barrier_x = int(width * 0.36)
        slit_gap = max(16, int(height * 0.16))
        slit_w = max(4, height // 50)

        s1_y = center_y - slit_gap // 2
        s2_y = center_y + slit_gap // 2

        wave_spacing = max(16, width // 28)
        wave_speed = 1.5 * alpha * wave_spacing
        for i in range(12):
            wx = int((i * wave_spacing + wave_speed) % max(1, barrier_x))
            cv2.line(canvas, (wx, int(height * 0.18)), (wx, int(height * 0.78)), (0, 160, 255), 1, lineType=cv2.LINE_AA)

        barrier_color = (60, 90, 140)
        cv2.line(canvas, (barrier_x, int(height * 0.15)), (barrier_x, s1_y - slit_w), barrier_color, 3, cv2.LINE_AA)
        cv2.line(canvas, (barrier_x, s1_y + slit_w), (barrier_x, s2_y - slit_w), barrier_color, 3, cv2.LINE_AA)
        cv2.line(canvas, (barrier_x, s2_y + slit_w), (barrier_x, int(height * 0.82)), barrier_color, 3, cv2.LINE_AA)

        cv2.circle(canvas, (barrier_x, s1_y), 3, (0, 240, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (barrier_x, s2_y), 3, (0, 240, 255), -1, cv2.LINE_AA)

        max_r = int(width * 0.52)
        for r_base in range(12, max_r, max(12, width // 32)):
            r = int((r_base + alpha * max(16, width // 28)) % max_r)
            if r > 4:
                cv2.ellipse(canvas, (barrier_x, s1_y), (r, r), 0, -85, 85, (0, 210, 255), 1, cv2.LINE_AA)
                cv2.ellipse(canvas, (barrier_x, s2_y), (r, r), 0, -85, 85, (160, 70, 255), 1, cv2.LINE_AA)

        screen_x = int(width * 0.88)
        cv2.line(canvas, (screen_x, int(height * 0.16)), (screen_x, int(height * 0.80)), (100, 120, 150), 2, cv2.LINE_AA)

        for y in range(int(height * 0.18), int(height * 0.78), 2):
            dy = float(y - center_y) / (height * 0.22)
            beta = math.pi * 3.5 * dy - alpha * math.tau
            sinc_arg = max(0.001, math.pi * 0.8 * dy)
            intensity = (math.cos(beta) ** 2) * ((math.sin(sinc_arg) / sinc_arg) ** 2)
            fringe_w = int(24 * intensity)
            if fringe_w > 0:
                bright = int(120 + 135 * intensity)
                color = (int(bright * 0.1), int(bright * 0.8), bright)
                cv2.line(canvas, (screen_x + 3, y), (screen_x + 3 + fringe_w, y), color, 1, lineType=cv2.LINE_AA)

    def _draw_quantum_wave_interference(self, canvas: np.ndarray, alpha: float, width: int, height: int) -> None:
        """Render time-dependent quantum probability wave packet and interference fringes (Proof M1).

        Args:
            canvas: Target RGB canvas.
            alpha: Timeline progress.
            width: Width in pixels.
            height: Height in pixels.
        """
        center_y = int(height * 0.50)
        amplitude = height * 0.08
        k = 2.0 * math.pi / max(32.0, float(width) * 0.25)
        omega = 2.0 * math.pi * 1.5

        pts_primary: list[tuple[int, int]] = []
        pts_secondary: list[tuple[int, int]] = []

        for x in range(0, width, max(1, width // 200)):
            x_norm = (float(x) - width * 0.5) / (width * 0.35)
            envelope = math.exp(-0.5 * (x_norm**2))
            phase = k * float(x) - omega * alpha

            y1 = int(center_y + amplitude * envelope * math.sin(phase))
            y2 = int(center_y + (amplitude * 0.7) * envelope * math.cos(phase * 1.2 + 0.5))

            pts_primary.append((x, y1))
            pts_secondary.append((x, y2))

        if len(pts_primary) > 1:
            arr_primary = np.array(pts_primary, dtype=np.int32).reshape((-1, 1, 2))
            arr_secondary = np.array(pts_secondary, dtype=np.int32).reshape((-1, 1, 2))

            cv2.polylines(canvas, [arr_primary], False, (0, 160, 255), max(1, width // 256), lineType=cv2.LINE_AA)
            cv2.polylines(canvas, [arr_secondary], False, (180, 70, 255), max(1, width // 384), lineType=cv2.LINE_AA)

    def _draw_bohr_orbitals(self, canvas: np.ndarray, alpha: float, width: int, height: int) -> None:
        """Render Bohr-Sommerfeld atomic orbital kinematics with decaying electron trails (Proof M2).

        Args:
            canvas: Target RGB canvas.
            alpha: Timeline progress.
            width: Width in pixels.
            height: Height in pixels.
        """
        center_x = int(width * 0.50)
        center_y = int(height * 0.44)

        core_radius = max(4, int(width * 0.025))
        for r_glow, c_glow in [
            (core_radius * 3, (20, 40, 80)),
            (core_radius * 2, (50, 90, 180)),
            (core_radius, (255, 220, 100)),
        ]:
            cv2.circle(canvas, (center_x, center_y), r_glow, c_glow, -1, lineType=cv2.LINE_AA)

        orbital_params = [
            {"a": width * 0.22, "b": height * 0.09, "angle": -25.0, "speed": 2.0, "phase": 0.0},
            {"a": width * 0.26, "b": height * 0.10, "angle": 35.0, "speed": 1.7, "phase": 2.1},
            {"a": width * 0.30, "b": height * 0.12, "angle": 85.0, "speed": 1.4, "phase": 4.2},
        ]

        for p in orbital_params:
            axes = (int(p["a"]), int(p["b"]))
            angle = float(p["angle"])
            cv2.ellipse(
                canvas, (center_x, center_y), axes, angle, 0, 360, (60, 110, 170), max(1, width // 512), cv2.LINE_AA
            )

            theta = p["phase"] + p["speed"] * alpha * math.tau
            rad_angle = math.radians(angle)

            cos_a, sin_a = math.cos(rad_angle), math.sin(rad_angle)
            raw_x = p["a"] * math.cos(theta)
            raw_y = p["b"] * math.sin(theta)

            e_x = int(center_x + (raw_x * cos_a - raw_y * sin_a))
            e_y = int(center_y + (raw_x * sin_a + raw_y * cos_a))

            electron_r = max(2, int(width * 0.012))
            cv2.circle(canvas, (e_x, e_y), electron_r + 2, (0, 200, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (e_x, e_y), electron_r, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    def _draw_spacetime_geodesics(self, canvas: np.ndarray, alpha: float, width: int, height: int) -> None:
        """Render gravitational curvature metric grid lines and celestial center (Proof M1).

        Args:
            canvas: Target RGB canvas.
            alpha: Timeline progress.
            width: Canvas width.
            height: Canvas height.
        """
        center_x = int(width * 0.50)
        center_y = int(height * 0.44)

        for ring_r in range(int(width * 0.08), int(width * 0.40), max(8, width // 12)):
            wobble = int(4.0 * math.sin(alpha * math.tau + ring_r))
            cv2.circle(canvas, (center_x, center_y), ring_r + wobble, (40, 80, 140), 1, lineType=cv2.LINE_AA)

        for deg in range(0, 360, 30):
            rad = math.radians(deg + alpha * 30.0)
            x_end = int(center_x + width * 0.42 * math.cos(rad))
            y_end = int(center_y + height * 0.38 * math.sin(rad))
            cv2.line(canvas, (center_x, center_y), (x_end, y_end), (25, 55, 95), 1, lineType=cv2.LINE_AA)

        cv2.circle(canvas, (center_x, center_y), max(6, int(width * 0.04)), (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (center_x, center_y), max(7, int(width * 0.045)), (255, 170, 40), 2, lineType=cv2.LINE_AA)

    def _draw_neural_lattice(self, canvas: np.ndarray, alpha: float, width: int, height: int) -> None:
        """Render microchip circuit traces and pulsing neural synaptic nodes.

        Args:
            canvas: Target RGB canvas.
            alpha: Timeline progress.
            width: Canvas width.
            height: Canvas height.
        """
        rows, cols = 4, 6
        step_x = width // (cols + 1)
        step_y = int(height * 0.55) // (rows + 1)
        y_offset = int(height * 0.18)

        nodes: list[tuple[int, int]] = []
        for r in range(rows):
            for c in range(cols):
                nodes.append(((c + 1) * step_x, y_offset + (r + 1) * step_y))

        for idx, (nx, ny) in enumerate(nodes):
            if idx + 1 < len(nodes) and (idx + 1) % cols != 0:
                cv2.line(canvas, (nx, ny), nodes[idx + 1], (30, 70, 110), 1, lineType=cv2.LINE_AA)
            if idx + cols < len(nodes):
                cv2.line(canvas, (nx, ny), nodes[idx + cols], (30, 70, 110), 1, lineType=cv2.LINE_AA)

        active_idx = int(alpha * len(nodes)) % len(nodes)
        for idx, (nx, ny) in enumerate(nodes):
            color = (0, 230, 255) if idx == active_idx else (80, 140, 200)
            cv2.circle(canvas, (nx, ny), max(2, width // 160), color, -1, lineType=cv2.LINE_AA)

    def _draw_header_hud(self, draw: ImageDraw.ImageDraw, scene: ScenePlan, domain_key: str, width: int, height: int) -> None:
        """Render glassmorphic category chip badge, scene title, and accent gradient line.

        Args:
            draw: Active PIL ImageDraw instance configured for RGBA.
            scene: Target ScenePlan item.
            domain_key: Resolved domain identifier string.
            width: Raster canvas width.
            height: Raster canvas height.
        """
        margin_x = max(12, int(width * 0.04))
        badge_y = max(10, int(height * 0.03))

        badge_font = self._get_font(size=max(9, int(height * 0.024)), bold=True)
        title_font = self._get_font(size=max(12, int(height * 0.042)), bold=True)

        badge_text = f" {domain_key.upper()} MECHANICS • SCENE {scene.scene_id:02d} "
        badge_bbox = badge_font.getbbox(badge_text)
        badge_w = badge_bbox[2] - badge_bbox[0] + 16
        badge_h = badge_bbox[3] - badge_bbox[1] + 8

        badge_rect = [margin_x, badge_y, margin_x + badge_w, badge_y + badge_h]
        draw.rounded_rectangle(badge_rect, radius=max(3, int(height * 0.01)), fill=(15, 30, 55, 190), outline=(0, 210, 255, 230), width=1)
        draw.text((margin_x + 8, badge_y + 3), badge_text, font=badge_font, fill=(0, 240, 255, 255))

        title_y = badge_y + badge_h + max(4, int(height * 0.012))
        clean_title = scene.title.upper().strip()

        card_w_est = max(int(width * 0.38), 180)
        card_x_est = width - card_w_est - margin_x
        max_title_w = max(100, card_x_est - margin_x - 16)

        words = clean_title.split()
        lines: list[str] = []
        curr_line: list[str] = []

        for word in words:
            candidate = " ".join([*curr_line, word])
            bbox = title_font.getbbox(candidate)
            if bbox[2] - bbox[0] <= max_title_w:
                curr_line.append(word)
            else:
                if curr_line:
                    lines.append(" ".join(curr_line))
                curr_line = [word]

        if curr_line:
            lines.append(" ".join(curr_line))

        lines = lines[:2]
        line_step = max(14, int(height * 0.042))

        for idx, line in enumerate(lines):
            draw.text((margin_x, title_y + (idx * line_step)), line, font=title_font, fill=(255, 255, 255, 255))

        accent_y = title_y + (len(lines) * line_step) + max(4, int(height * 0.008))
        accent_w = min(max_title_w, max(60, int(width * 0.35)))
        draw.line([(margin_x, accent_y), (margin_x + accent_w, accent_y)], fill=(0, 220, 255, 220), width=max(1, height // 300))

    def _draw_formula_card(self, draw: ImageDraw.ImageDraw, formula: str, width: int, height: int) -> None:
        """Render translucent glassmorphic HUD formula card (Proof M4, M5).

        Args:
            draw: Active PIL ImageDraw instance.
            formula: Mathematical equation string.
            width: Canvas width.
            height: Canvas height.
        """
        font = self._get_font(size=max(10, int(height * 0.028)), bold=True)
        label_font = self._get_font(size=max(8, int(height * 0.018)), bold=False)

        f_bbox = font.getbbox(formula)
        card_w = max(int(width * 0.40), f_bbox[2] - f_bbox[0] + 28)
        card_h = max(int(height * 0.08), f_bbox[3] - f_bbox[1] + 24)

        card_x = width - card_w - max(12, int(width * 0.04))
        card_y = max(10, int(height * 0.03))

        draw.rounded_rectangle(
            [card_x, card_y, card_x + card_w, card_y + card_h],
            radius=max(4, int(height * 0.012)),
            fill=(8, 18, 36, 175),
            outline=(70, 130, 210, 160),
            width=1,
        )

        draw.text((card_x + 12, card_y + 4), "PRINCIPLE / FORMULA", font=label_font, fill=(140, 180, 230, 210))
        draw.text((card_x + 12, card_y + int(card_h * 0.40)), formula, font=font, fill=(255, 225, 120, 255))

    def _draw_subtitles_card(self, draw: ImageDraw.ImageDraw, narration_text: str, width: int, height: int) -> None:
        """Render word-wrapped synchronized dialogue subtitle card at canvas bottom.

        Args:
            draw: Active PIL ImageDraw instance.
            narration_text: Speech transcript string.
            width: Raster canvas width.
            height: Raster canvas height.
        """
        if not narration_text or not narration_text.strip():
            return

        font = self._get_font(size=max(10, int(height * 0.028)), bold=False)
        margin_x = max(16, int(width * 0.05))
        card_w = width - (2 * margin_x)
        max_text_w = card_w - 32

        words = narration_text.strip().split()
        lines: list[str] = []
        current_line: list[str] = []

        for word in words:
            candidate = " ".join([*current_line, word])
            bbox = font.getbbox(candidate)
            if bbox[2] - bbox[0] <= max_text_w:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]

        if current_line:
            lines.append(" ".join(current_line))

        lines = lines[:3]
        line_h = max(14, int(height * 0.035))
        card_h = (len(lines) * line_h) + max(14, int(height * 0.025))
        card_y = height - card_h - max(12, int(height * 0.03))

        draw.rounded_rectangle(
            [margin_x, card_y, margin_x + card_w, card_y + card_h],
            radius=max(4, int(height * 0.012)),
            fill=(6, 12, 24, 210),
            outline=(0, 190, 255, 140),
            width=1,
        )

        for idx, line in enumerate(lines):
            line_bbox = font.getbbox(line)
            w_line = line_bbox[2] - line_bbox[0]
            x_line = margin_x + (card_w - w_line) // 2
            y_line = card_y + 8 + (idx * line_h)
            draw.text((x_line, y_line), line, font=font, fill=(240, 248, 255, 255))

    def _apply_camera_motion(self, frame: np.ndarray, alpha: float, motion: str) -> np.ndarray:
        """Apply affine transformation zoom or pan motion interpolation across frames (Proof M3).

        Args:
            frame: Input RGB frame array.
            alpha: Progress factor [0.0, 1.0].
            motion: Camera motion descriptor string.

        Returns:
            Transformed RGB frame array.
        """
        h, w = frame.shape[:2]
        motion_lower = motion.lower()

        if "zoom" in motion_lower or "push" in motion_lower:
            scale = 1.0 + 0.05 * alpha
            center = (w / 2.0, h / 2.0)
            matrix = cv2.getRotationMatrix2D(center, 0.0, scale)
            return cv2.warpAffine(frame, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        elif "pan" in motion_lower or "drift" in motion_lower:
            shift_x = int(14.0 * alpha)
            return np.roll(frame, shift_x, axis=1)
        else:
            return frame
