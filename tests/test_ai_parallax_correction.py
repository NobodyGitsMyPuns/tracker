"""Unit tests for the pure geometry in ai_parallax_correction.

These cover the distance estimator, the parallax projection, and the
adaptive-crosshair selection logic. They are deterministic and require no
camera, GPU, or YOLO model -- only the math in AIParallaxCorrector.
"""

import math

import pytest

from ai_parallax_correction import AIParallaxCorrector


FRAME_W, FRAME_H = 1920, 1080


@pytest.fixture
def corrector():
    return AIParallaxCorrector()


# ---------------------------------------------------------------------------
# estimate_distance
# ---------------------------------------------------------------------------

def test_estimate_distance_within_clamp_range(corrector):
    d = corrector.estimate_distance(100, 100, "drone", 0.9, FRAME_W, FRAME_H)
    assert 100 <= d <= 10000


def test_estimate_distance_tiny_box_clamped_to_max(corrector):
    # A 1px box is effectively infinitely far; the estimate must clamp at 10 m.
    d = corrector.estimate_distance(1, 1, "drone", 0.9, FRAME_W, FRAME_H)
    assert d == 10000


def test_estimate_distance_huge_box_clamped_to_min(corrector):
    # A box much larger than the frame implies an extremely close object; the
    # estimate must clamp at the 10 cm floor rather than going below it.
    d = corrector.estimate_distance(4000, 4000, "drone", 0.9, FRAME_W, FRAME_H)
    assert d == 100


def test_estimate_distance_zero_bbox_does_not_crash(corrector):
    # Regression: a degenerate zero-area box used to raise ZeroDivisionError
    # (math.tan(0) == 0 -> division by zero) and crash the detection loop.
    d = corrector.estimate_distance(0, 0, "drone", 0.9, FRAME_W, FRAME_H)
    assert d == 1000.0
    assert math.isfinite(d)


def test_estimate_distance_negative_bbox_does_not_crash(corrector):
    # Defensive: malformed (negative) dimensions must also fall back, not crash.
    d = corrector.estimate_distance(-5, -5, "drone", 0.9, FRAME_W, FRAME_H)
    assert d == 1000.0


def test_estimate_distance_zero_frame_does_not_crash(corrector):
    d = corrector.estimate_distance(100, 100, "drone", 0.9, 0, 0)
    assert d == 1000.0


def test_low_confidence_pulls_estimate_toward_base(corrector):
    # The confidence weighting blends the raw estimate toward the 1 m base, so a
    # low-confidence reading must sit between the high-confidence estimate and
    # the 1000 mm base (a convex combination).
    d_high = corrector.estimate_distance(300, 300, "drone", 0.9, FRAME_W, FRAME_H)
    d_low = corrector.estimate_distance(300, 300, "drone", 0.3, FRAME_W, FRAME_H)
    assert min(d_high, 1000) <= d_low <= max(d_high, 1000)
    assert abs(d_low - 1000) <= abs(d_high - 1000)


# ---------------------------------------------------------------------------
# calculate_parallax_correction
# ---------------------------------------------------------------------------

def test_parallax_correction_returns_int_pixels_in_bounds(corrector):
    cx, cy = corrector.calculate_parallax_correction(
        2000, 1000, 600, FRAME_W, FRAME_H, zoom_factor=1.0
    )
    assert isinstance(cx, int) and isinstance(cy, int)
    assert 0 <= cx <= FRAME_W - 1
    assert 0 <= cy <= FRAME_H - 1


def test_parallax_correction_clamps_to_frame_bounds(corrector):
    # A target far off-axis at close range pushes the projection outside the
    # frame; the result must be clamped, never out of bounds.
    cx, cy = corrector.calculate_parallax_correction(
        150, FRAME_W * 4, FRAME_H * 4, FRAME_W, FRAME_H, zoom_factor=1.0
    )
    assert 0 <= cx <= FRAME_W - 1
    assert 0 <= cy <= FRAME_H - 1


# ---------------------------------------------------------------------------
# get_adaptive_crosshair
# ---------------------------------------------------------------------------

def test_adaptive_crosshair_no_detections_returns_center(corrector):
    x, y, info = corrector.get_adaptive_crosshair([], FRAME_W, FRAME_H)
    assert (x, y) == (FRAME_W // 2, FRAME_H // 2)
    assert info["status"] == "no_targets"


def test_adaptive_crosshair_filters_below_confidence_threshold(corrector):
    # Below min_confidence_drone (0.3) the detection is dropped -> no targets.
    dets = [{"bbox": [100, 100, 200, 200], "class": "clock", "confidence": 0.1}]
    x, y, info = corrector.get_adaptive_crosshair(dets, FRAME_W, FRAME_H)
    assert info["status"] == "no_targets"


def test_adaptive_crosshair_degenerate_bbox_does_not_crash(corrector):
    # End-to-end regression for the zero-area box reaching estimate_distance
    # through the public path.
    dets = [{"bbox": [500, 500, 500, 500], "class": "clock", "confidence": 0.9}]
    x, y, info = corrector.get_adaptive_crosshair(dets, FRAME_W, FRAME_H)
    assert 0 <= x <= FRAME_W - 1
    assert 0 <= y <= FRAME_H - 1
    assert info["status"] == "ai_corrected"


def test_adaptive_crosshair_picks_highest_priority_target(corrector):
    # 'clock' has a 2.0 priority multiplier vs 'cell phone' at 1.5, so even with
    # equal confidence the clock wins the target-selection scoring.
    dets = [
        {"bbox": [10, 10, 110, 110], "class": "cell phone", "confidence": 0.6},
        {"bbox": [800, 800, 900, 900], "class": "clock", "confidence": 0.6},
    ]
    _, _, info = corrector.get_adaptive_crosshair(dets, FRAME_W, FRAME_H)
    assert info["target_class"] == "clock"
