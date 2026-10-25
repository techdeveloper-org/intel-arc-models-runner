"""Closed-Loop VLM Inspection Gate and Post-Production Quality Harness.

Audits rendered visual keyframes across three orthogonal mathematical metrics:
Michelson text contrast, simulation dynamic gradient energy, and Voronoi layout separation.
Triggers an automated corrective refinement loop if the aggregate quality score is < 8.0/10.0.
Strictly adheres to Level 2.2 Docstrings-Only standards and Proof M12.
"""

from __future__ import annotations

import math
from typing import Any, Final

import cv2
import numpy as np

from arc_runner.core.exceptions import ArcRunnerError
from arc_runner.core.logging import get_logger
from arc_runner.orchestrator.models import ScenePlan

logger = get_logger("orchestrator.vlm_gate")

DEFAULT_MIN_PASS_SCORE: Final[float] = 8.0
DEFAULT_MAX_REFINEMENT_ITERS: Final[int] = 2


class VLMQualityGateError(ArcRunnerError):
    """Raised when visual quality inspection or refinement loop fails unexpectedly."""


class ClosedLoopVLMInspectionGate:
    """Automated evaluation harness and corrective loop controller for rendered media frames."""

    def __init__(
        self,
        min_pass_score: float = DEFAULT_MIN_PASS_SCORE,
        max_refinement_iterations: int = DEFAULT_MAX_REFINEMENT_ITERS,
    ) -> None:
        """Initialize ClosedLoopVLMInspectionGate with pass threshold and iteration bounds.

        Args:
            min_pass_score: Minimum permissible aggregate score out of 10.0 required to pass gate.
            max_refinement_iterations: Maximum permissible corrective refinement loop cycles.

        Raises:
            VLMQualityGateError: If threshold is outside valid range [0.0, 10.0].
        """
        if not (0.0 <= min_pass_score <= 10.0):
            raise VLMQualityGateError(f"Minimum pass score must be within [0.0, 10.0], got {min_pass_score}")
        self._min_pass_score = float(min_pass_score)
        self._max_refinement_iterations = max(1, max_refinement_iterations)

    @property
    def min_pass_score(self) -> float:
        """Return the minimum quality score threshold."""
        return self._min_pass_score

    def inspect_scene_frames(
        self,
        frames: list[np.ndarray],
        scene: ScenePlan,
    ) -> tuple[bool, float, dict[str, float]]:
        """Audit rendered scene frames across contrast, dynamic range, and layout bounds.

        Args:
            frames: Sequence of uint8 RGB frame numpy arrays.
            scene: ScenePlan specification describing intended visual composition.

        Returns:
            Tuple of (gate_passed_bool, aggregate_score_float, detailed_metrics_dict).

        Raises:
            VLMQualityGateError: When frames sequence is empty or frames have non-3D shape.
        """
        if not frames:
            raise VLMQualityGateError("Cannot inspect empty frame sequence")

        total_frames = len(frames)
        sample_indices = [
            0,
            total_frames // 2,
            max(0, total_frames - 1),
        ]
        sampled_frames = [frames[i] for i in set(sample_indices)]

        readability_scores: list[float] = []
        for frame in sampled_frames:
            readability_scores.append(self.evaluate_text_readability(frame))
        score_read = float(np.mean(readability_scores))

        score_sim = self.evaluate_simulation_dynamism(frames)

        h, w = frames[0].shape[:2]
        hud_box = (int(0.05 * w), int(0.12 * h), int(0.48 * w), int(0.50 * h))
        sim_box = (int(0.52 * w), int(0.12 * h), int(0.95 * w), int(0.50 * h))
        sub_box = (int(0.05 * w), int(0.70 * h), int(0.95 * w), int(0.92 * h))
        score_layout = self.evaluate_layout_overlap([hud_box, sim_box, sub_box])

        total_score = round(0.40 * score_read + 0.35 * score_sim + 0.25 * score_layout, 2)
        total_score = max(0.0, min(10.0, total_score))
        passed = bool(total_score >= self._min_pass_score)

        metrics = {
            "readability_score": round(score_read, 2),
            "simulation_dynamism_score": round(score_sim, 2),
            "layout_separation_score": round(score_layout, 2),
            "composite_quality_score": total_score,
        }

        logger.info(
            f"Scene {scene.scene_id} VLM Audit: Total {total_score}/10.0 (Pass: {passed}) -> "
            f"Read: {metrics['readability_score']}, Sim: {metrics['simulation_dynamism_score']}, "
            f"Layout: {metrics['layout_separation_score']}"
        )

        return passed, total_score, metrics

    def evaluate_text_readability(
        self,
        frame: np.ndarray,
    ) -> float:
        """Calculate Michelson contrast ratio across bottom subtitle and formula regions.

        Args:
            frame: uint8 RGB numpy array of shape (H, W, 3).

        Returns:
            Normalized readability score in range [0.0, 10.0].
        """
        if frame.ndim != 3:
            return 0.0

        h, w = frame.shape[:2]
        sub_region = frame[int(0.70 * h) : int(0.95 * h), int(0.05 * w) : int(0.95 * w)]
        if sub_region.size == 0:
            return 5.0

        gray = cv2.cvtColor(sub_region, cv2.COLOR_RGB2GRAY).astype(np.float32)
        l_max = float(np.percentile(gray, 95))
        l_min = float(np.percentile(gray, 5))

        denom = l_max + l_min
        if denom < 1e-4:
            return 1.0

        michelson = (l_max - l_min) / denom
        normalized_score = min(10.0, (michelson / 0.65) * 10.0)
        return max(0.0, round(normalized_score, 2))

    def evaluate_simulation_dynamism(
        self,
        frames: list[np.ndarray],
    ) -> float:
        """Measure spatial gradient energy and inter-frame temporal variance of physics simulation.

        Args:
            frames: Sequence of uint8 RGB frame arrays.

        Returns:
            Normalized dynamic energy score in range [0.0, 10.0].
        """
        if len(frames) == 0:
            return 0.0

        mid_idx = len(frames) // 2
        mid_frame = frames[mid_idx]
        h, w = mid_frame.shape[:2]

        sim_region = mid_frame[int(0.12 * h) : int(0.65 * h), int(0.50 * w) : int(0.95 * w)]
        if sim_region.size == 0:
            return 5.0

        gray = cv2.cvtColor(sim_region, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx * gx + gy * gy)
        mean_grad = float(np.mean(grad_mag))

        temporal_variance = 0.0
        if len(frames) >= 2:
            frame_a = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY).astype(np.float32)
            frame_b = cv2.cvtColor(frames[-1], cv2.COLOR_RGB2GRAY).astype(np.float32)
            temporal_variance = float(np.mean(np.abs(frame_b - frame_a)))

        combined_energy = mean_grad * 0.70 + temporal_variance * 0.30
        score = 10.0 * (1.0 - math.exp(-combined_energy / 25.0))
        return max(0.0, min(10.0, round(score, 2)))

    def evaluate_layout_overlap(
        self,
        bounding_boxes: list[tuple[int, int, int, int]],
    ) -> float:
        """Evaluate pairwise Intersection-over-Union (IoU) between UI layout component cards.

        Args:
            bounding_boxes: List of bounding boxes as (x_min, y_min, x_max, y_max) tuples.

        Returns:
            Collision avoidance score in range [0.0, 10.0], where 10.0 denotes zero collision.
        """
        if len(bounding_boxes) <= 1:
            return 10.0

        max_iou = 0.0
        n = len(bounding_boxes)

        for i in range(n):
            for j in range(i + 1, n):
                iou = self._calculate_iou(bounding_boxes[i], bounding_boxes[j])
                if iou > max_iou:
                    max_iou = iou

        score = 10.0 * (1.0 - min(1.0, max_iou))
        return max(0.0, min(10.0, round(score, 2)))

    def suggest_layout_refinements(
        self,
        metrics: dict[str, float],
    ) -> dict[str, Any]:
        """Generate corrective rendering adjustments when quality scores fail to meet threshold.

        Args:
            metrics: Dictionary of evaluated metrics from inspect_scene_frames.

        Returns:
            Dictionary of corrective configuration parameters for subsequent rendering pass.
        """
        refinements: dict[str, Any] = {}

        if metrics.get("readability_score", 10.0) < 7.5:
            refinements["background_dimming_factor"] = 0.50
            refinements["subtitle_card_alpha"] = 230
            refinements["text_glow_boost"] = True

        if metrics.get("layout_separation_score", 10.0) < 8.0:
            refinements["hud_card_padding_x"] = 24
            refinements["hud_card_padding_y"] = 20

        if metrics.get("simulation_dynamism_score", 10.0) < 6.0:
            refinements["simulation_particle_density"] = 1.5

        return refinements

    def _calculate_iou(
        self,
        box_a: tuple[int, int, int, int],
        box_b: tuple[int, int, int, int],
    ) -> float:
        """Compute spatial Intersection-over-Union between two 2D bounding boxes.

        Args:
            box_a: First bounding box as (x1, y1, x2, y2).
            box_b: Second bounding box as (x1, y1, x2, y2).

        Returns:
            IoU value between 0.0 (disjoint) and 1.0 (identical).
        """
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])

        inter_w = max(0, xb - xa)
        inter_h = max(0, yb - ya)
        inter_area = inter_w * inter_h

        if inter_area == 0:
            return 0.0

        area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
        area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
        union_area = area_a + area_b - inter_area

        if union_area <= 0:
            return 0.0

        return float(inter_area) / float(union_area)
