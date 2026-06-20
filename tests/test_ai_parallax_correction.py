"""Unit tests for :mod:`ai_parallax_correction`.

These tests pin the *current* behavior of the AI parallax-correction math used to
place the targeting crosshair. The module is pure, deterministic geometry
(distance estimation, parallax correction, target selection and smoothing), which
makes it a high-value, side-effect-free target for regression tests.

No production behavior is changed by this file. Reference values were produced by
running the shipped implementation and are asserted with ``pytest.approx`` where
floating point is involved.
"""

import math

import pytest

from ai_parallax_correction import AIParallaxCorrector


# Default geometry constants used by AIParallaxCorrector() with no overrides.
DEFAULT_CAMERA_HEIGHT_MM = 50
DEFAULT_SERVO_OFFSET_MM = 30
DEFAULT_FOV_H = 60
DEFAULT_FOV_V = 45

FW, FH = 1920, 1080  # A common test frame size (1080p).


@pytest.fixture
def corrector():
    """A fresh corrector with default geometry for each test."""
    return AIParallaxCorrector()


# ---------------------------------------------------------------------------
# Construction / configuration
# ---------------------------------------------------------------------------
class TestConstruction:
    def test_defaults(self, corrector):
        assert corrector.camera_height_mm == DEFAULT_CAMERA_HEIGHT_MM
        assert corrector.servo_offset_mm == DEFAULT_SERVO_OFFSET_MM
        assert corrector.camera_fov_h == DEFAULT_FOV_H
        assert corrector.camera_fov_v == DEFAULT_FOV_V
        assert corrector.drone_only_mode is True
        assert corrector.min_confidence_drone == 0.3
        assert corrector.smoothing_factor == 0.7
        # No history until the first crosshair is computed.
        assert corrector.last_crosshair_x is None
        assert corrector.last_crosshair_y is None

    def test_custom_geometry(self):
        c = AIParallaxCorrector(camera_height_mm=10, servo_offset_mm=5,
                                camera_fov_h=90, camera_fov_v=70)
        assert c.camera_height_mm == 10
        assert c.servo_offset_mm == 5
        assert c.camera_fov_h == 90
        assert c.camera_fov_v == 70

    def test_known_object_sizes_present(self, corrector):
        # The lookup table drives distance estimation; pin a couple of entries.
        assert corrector.object_sizes["drone"] == 300
        assert corrector.object_sizes["sports ball"] == 220
        assert "clock" in corrector.object_sizes


# ---------------------------------------------------------------------------
# estimate_distance
# ---------------------------------------------------------------------------
class TestEstimateDistance:
    def test_known_class_reference_value(self, corrector):
        # clock (200mm), 100px box, high confidence (weight 1.0, no fallback blend).
        d = corrector.estimate_distance(100, 100, "clock", 0.85, FW, FH)
        assert d == pytest.approx(3666.0208181128146, rel=1e-9)

    def test_unknown_class_uses_200mm_default(self, corrector):
        # Unknown classes fall back to a 200mm assumed size; clock is also 200mm,
        # so the two must agree exactly.
        unknown = corrector.estimate_distance(100, 100, "banana", 0.85, FW, FH)
        clock = corrector.estimate_distance(100, 100, "clock", 0.85, FW, FH)
        assert unknown == pytest.approx(clock, rel=1e-9)

    def test_larger_known_object_is_estimated_farther(self, corrector):
        # A physically larger object (drone 300mm) at the same pixel size must be
        # estimated as farther away than a smaller one (clock 200mm).
        drone = corrector.estimate_distance(100, 100, "drone", 0.85, FW, FH)
        clock = corrector.estimate_distance(100, 100, "clock", 0.85, FW, FH)
        assert drone > clock

    @pytest.mark.parametrize(
        "confidence, expected",
        [
            (0.85, 3666.0208181128146),  # > 0.8  -> high weight 1.0
            (0.80, 3132.816654490252),   # == 0.8 -> medium weight 0.8 (not high)
            (0.40, 2599.612490867689),   # == 0.4 -> low weight 0.6 (not medium)
        ],
    )
    def test_confidence_weighting_tiers(self, corrector, confidence, expected):
        d = corrector.estimate_distance(100, 100, "clock", confidence, FW, FH)
        assert d == pytest.approx(expected, rel=1e-9)

    def test_lower_confidence_pulls_toward_1m_fallback(self, corrector):
        # Lower confidence blends the estimate toward the 1000mm fallback. Here the
        # raw estimate (~3666) is above 1000, so reducing confidence lowers it.
        high = corrector.estimate_distance(100, 100, "clock", 0.85, FW, FH)
        med = corrector.estimate_distance(100, 100, "clock", 0.80, FW, FH)
        low = corrector.estimate_distance(100, 100, "clock", 0.40, FW, FH)
        assert high > med > low > 1000

    def test_bigger_box_is_closer(self, corrector):
        near = corrector.estimate_distance(400, 400, "clock", 0.85, FW, FH)
        far = corrector.estimate_distance(100, 100, "clock", 0.85, FW, FH)
        assert near < far

    def test_clamped_to_minimum_100mm(self, corrector):
        # A box filling the frame implies an extremely close object; the result is
        # clamped to the 100mm floor.
        d = corrector.estimate_distance(5000, 5000, "clock", 0.85, FW, FH)
        assert d == 100

    def test_clamped_to_maximum_10000mm(self, corrector):
        # A 1px box implies a very distant object; clamped to the 10000mm ceiling.
        d = corrector.estimate_distance(1, 1, "clock", 0.85, FW, FH)
        assert d == 10000

    def test_uses_larger_bbox_dimension(self, corrector):
        # The estimator keys off max(width, height); swapping which dimension is
        # larger but keeping the max constant must not change the result.
        a = corrector.estimate_distance(120, 80, "clock", 0.85, FW, FH)
        b = corrector.estimate_distance(80, 120, "clock", 0.85, FW, FH)
        assert a == pytest.approx(b, rel=1e-9)

    def test_matches_independent_formula(self, corrector):
        # Cross-check against an independent re-derivation of the documented math.
        bbox, frame, size = 150, FW, 200  # clock
        ang_deg = (bbox / max(FW, FH)) * max(DEFAULT_FOV_H, DEFAULT_FOV_V)
        ang_rad = math.radians(ang_deg)
        expected = size / (2 * math.tan(ang_rad / 2))  # high confidence weight 1.0
        got = corrector.estimate_distance(bbox, 10, "clock", 0.9, FW, FH)
        assert got == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# calculate_parallax_correction
# ---------------------------------------------------------------------------
class TestParallaxCorrection:
    def test_returns_integer_tuple(self, corrector):
        out = corrector.calculate_parallax_correction(1000, 960, 540, FW, FH, 1.0)
        assert isinstance(out, tuple) and len(out) == 2
        assert all(isinstance(v, int) for v in out)

    def test_center_target_reference_value(self, corrector):
        # Even a centered target is offset by the camera/servo geometry.
        assert corrector.calculate_parallax_correction(1000, 960, 540, FW, FH, 1.0) == (905, 471)

    def test_output_clamped_within_frame(self, corrector):
        x, y = corrector.calculate_parallax_correction(5000, 5000, 5000, FW, FH, 1.0)
        assert 0 <= x <= FW - 1
        assert 0 <= y <= FH - 1

    def test_negative_target_clamped_to_zero(self, corrector):
        x, y = corrector.calculate_parallax_correction(5000, -500, -500, FW, FH, 1.0)
        assert x >= 0 and y >= 0

    def test_zoom_changes_result(self, corrector):
        # A different zoom factor narrows the effective FOV and changes the mapping.
        z1 = corrector.calculate_parallax_correction(2000, 1200, 540, FW, FH, 1.0)
        z2 = corrector.calculate_parallax_correction(2000, 1200, 540, FW, FH, 2.0)
        assert z1 != z2

    def test_deterministic(self, corrector):
        a = corrector.calculate_parallax_correction(1500, 1000, 600, FW, FH, 1.0)
        b = corrector.calculate_parallax_correction(1500, 1000, 600, FW, FH, 1.0)
        assert a == b


# ---------------------------------------------------------------------------
# get_adaptive_crosshair
# ---------------------------------------------------------------------------
class TestAdaptiveCrosshair:
    def test_no_detections_returns_center(self, corrector):
        x, y, info = corrector.get_adaptive_crosshair([], FW, FH, 1.0)
        assert (x, y) == (FW // 2, FH // 2)
        assert info["status"] == "no_targets"
        assert info["zoom_factor"] == 1.0

    def test_no_detections_records_history(self, corrector):
        corrector.get_adaptive_crosshair([], FW, FH, 1.0)
        assert corrector.last_crosshair_x == FW // 2
        assert corrector.last_crosshair_y == FH // 2

    def test_non_drone_class_is_filtered_out(self, corrector):
        # 'person' is not in the strict drone class set -> treated as no targets.
        det = [{"bbox": [0, 0, 10, 10], "class": "person", "confidence": 0.99}]
        x, y, info = corrector.get_adaptive_crosshair(det, FW, FH, 1.0)
        assert (x, y) == (FW // 2, FH // 2)
        assert info["status"] == "no_targets"

    def test_low_confidence_drone_is_filtered(self, corrector):
        # Below min_confidence_drone (0.3) the detection is dropped.
        det = [{"bbox": [0, 0, 10, 10], "class": "kite", "confidence": 0.2}]
        _, _, info = corrector.get_adaptive_crosshair(det, FW, FH, 1.0)
        assert info["status"] == "no_targets"

    def test_valid_detection_is_ai_corrected(self, corrector):
        det = [{"bbox": [900, 500, 1020, 620], "class": "clock", "confidence": 0.9}]
        x, y, info = corrector.get_adaptive_crosshair(det, FW, FH, 1.0)
        assert info["status"] == "ai_corrected"
        assert info["target_class"] == "clock"
        assert (x, y) == (941, 537)
        assert info["estimated_distance_m"] == pytest.approx(
            info["estimated_distance_mm"] / 1000, rel=1e-9
        )

    def test_priority_scoring_prefers_higher_priority_class(self, corrector):
        # At equal confidence, 'kite' (priority 2.0) beats 'sports ball' (1.5).
        dets = [
            {"bbox": [0, 0, 50, 50], "class": "sports ball", "confidence": 0.5},
            {"bbox": [1800, 1000, 1850, 1050], "class": "kite", "confidence": 0.5},
        ]
        _, _, info = corrector.get_adaptive_crosshair(dets, FW, FH, 1.0)
        assert info["target_class"] == "kite"

    def test_close_object_uses_blended_raw_position(self, corrector):
        # A box spanning much of the frame yields <500mm -> the close-object branch
        # blends the raw target center (600,500) 80/20 with the frame center.
        det = [{"bbox": [200, 100, 1000, 900], "class": "clock", "confidence": 0.9}]
        x, y, info = corrector.get_adaptive_crosshair(det, FW, FH, 1.0)
        assert info["estimated_distance_mm"] < 500
        assert (x, y) == (int(600 * 0.8 + 960 * 0.2), int(500 * 0.8 + 540 * 0.2))
        assert (x, y) == (672, 508)

    def test_smoothing_blends_toward_center_on_loss(self, corrector):
        # First lock onto an off-center target, then lose it: the returned point is
        # the smoothed blend (0.7 * last + 0.3 * center), not a hard jump to center.
        det = [{"bbox": [1700, 900, 1750, 950], "class": "kite", "confidence": 0.9}]
        fx, fy, _ = corrector.get_adaptive_crosshair(det, FW, FH, 1.0)
        nx, ny, info = corrector.get_adaptive_crosshair([], FW, FH, 1.0)
        assert info["status"] == "no_targets"
        assert nx == int(fx * 0.7 + 960 * 0.3)
        assert ny == int(fy * 0.7 + 540 * 0.3)
        # The smoothed point sits strictly between the last lock and the center.
        assert 960 < nx < fx

    def test_debug_info_offset_is_consistent(self, corrector):
        det = [{"bbox": [900, 500, 1020, 620], "class": "clock", "confidence": 0.9}]
        _, _, info = corrector.get_adaptive_crosshair(det, FW, FH, 1.0)
        ox, oy = info["original_position"]
        sx, sy = info["smoothed_position"]
        assert info["correction_offset"] == (sx - ox, sy - oy)
