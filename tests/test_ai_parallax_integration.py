"""Integration-level tests for :mod:`ai_parallax_correction`.

The sibling ``test_ai_parallax_correction.py`` thoroughly pins the pure geometry
(distance estimation, parallax math, target selection and smoothing). This file
covers the parts that wire that math into the rest of the system and were left
unexercised:

* :func:`ai_parallax_correction.get_ai_crosshair_position` — the adapter that the
  main YOLO detection loop actually calls. It translates an Ultralytics-style
  ``results`` object (``results.boxes`` with ``.xyxy`` / ``.conf`` / ``.cls``
  tensors and a ``results.names`` id→label map) into the plain-dict detection
  format consumed by :meth:`AIParallaxCorrector.get_adaptive_crosshair`, via the
  module-level ``ai_parallax`` singleton. None of this was covered.
* The ``"no_valid_targets"`` branch of ``get_adaptive_crosshair`` (reachable when
  scoring leaves no winning detection).
* The close-object (<500 mm) blended-position branch.

No production behavior is changed by this file. The fake ``results`` object below
mimics the subset of the Ultralytics API the adapter touches (``.cpu().numpy()``
on each per-box tensor) so the suite needs neither torch nor a YOLO model.
Reference values were produced by running the shipped implementation.
"""

import numpy as np
import pytest

import ai_parallax_correction as apc
from ai_parallax_correction import AIParallaxCorrector


FW, FH = 1920, 1080  # 1080p test frame.


# ---------------------------------------------------------------------------
# Minimal fakes mimicking the Ultralytics results/box API the adapter uses.
# ---------------------------------------------------------------------------
class _FakeTensor:
    """Stands in for a torch tensor: supports ``.cpu().numpy()`` -> ndarray."""

    def __init__(self, values):
        self._arr = np.array(values)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeBox:
    """One detection: ``xyxy``/``conf``/``cls`` each index [0] to a tensor."""

    def __init__(self, xyxy, conf, cls):
        # Mirrors Ultralytics: box.xyxy[0] -> length-4 tensor, box.conf[0] /
        # box.cls[0] -> 0-dim scalar tensors (so float()/int() accept them).
        self.xyxy = [_FakeTensor(xyxy)]
        self.conf = [_FakeTensor(conf)]
        self.cls = [_FakeTensor(cls)]


class _BareBox:
    """A box object missing the xyxy/conf/cls attributes (should be skipped)."""


class _FakeResults:
    """Stands in for a YOLO ``Results`` object."""

    def __init__(self, boxes, names=None):
        self.boxes = boxes
        if names is not None:
            self.names = names


@pytest.fixture
def singleton():
    """Install a fresh corrector as the module ``ai_parallax`` singleton.

    ``get_ai_crosshair_position`` reads the module-global ``ai_parallax``; the
    shipped default is ``None``. Save/restore around each test so we never leak
    state between tests (or onto the real module global).
    """
    previous = apc.ai_parallax
    apc.ai_parallax = AIParallaxCorrector()
    try:
        yield apc.ai_parallax
    finally:
        apc.ai_parallax = previous


# ---------------------------------------------------------------------------
# get_ai_crosshair_position: the YOLO -> internal-format adapter
# ---------------------------------------------------------------------------
class TestGetAiCrosshairPosition:
    def test_valid_drone_detection_is_ai_corrected(self, singleton):
        results = _FakeResults([_FakeBox([500, 300, 600, 400], 0.85, 7)],
                               names={7: "clock"})

        x, y, info = apc.get_ai_crosshair_position(results, FW, FH, zoom_factor=2.0)

        assert info["status"] == "ai_corrected"
        # The class label is resolved through results.names, not the raw id.
        assert info["target_class"] == "clock"
        assert info["zoom_factor"] == 2.0
        assert isinstance(x, int) and isinstance(y, int)

    def test_adapter_translation_matches_direct_dict_call(self, singleton):
        """The adapter must build the exact detection dict the math expects.

        Driving the singleton through the adapter and through a hand-built dict
        with the same bbox/class/confidence must yield an identical result, which
        pins that ``xyxy``/``conf``/``cls`` are unpacked into the right fields.
        """
        bbox = [512.0, 301.0, 640.0, 455.0]
        results = _FakeResults([_FakeBox(bbox, 0.77, 7)], names={7: "clock"})

        adapter_out = apc.get_ai_crosshair_position(results, FW, FH, zoom_factor=1.0)

        # Reset smoothing history so the direct call starts from the same state.
        apc.ai_parallax.last_crosshair_x = None
        apc.ai_parallax.last_crosshair_y = None
        direct_out = apc.ai_parallax.get_adaptive_crosshair(
            [{"bbox": bbox, "class": "clock", "confidence": 0.77}], FW, FH, 1.0
        )

        assert adapter_out[0] == direct_out[0]
        assert adapter_out[1] == direct_out[1]
        assert adapter_out[2]["corrected_position"] == direct_out[2]["corrected_position"]

    def test_missing_names_falls_back_to_class_id_label(self, singleton):
        """When results has no ``names`` map, the label is ``class_<id>``.

        With drone-only filtering off, such a detection is processed (not
        dropped), so we can observe the synthesized label in the debug info.
        """
        singleton.drone_only_mode = False
        results = _FakeResults([_FakeBox([500, 300, 600, 400], 0.9, 7)])  # no names

        _, _, info = apc.get_ai_crosshair_position(results, FW, FH)

        assert info["status"] == "ai_corrected"
        assert info["target_class"] == "class_7"

    def test_boxes_none_returns_center_no_targets(self, singleton):
        results = _FakeResults(None)

        x, y, info = apc.get_ai_crosshair_position(results, FW, FH)

        assert info["status"] == "no_targets"
        assert (x, y) == (FW // 2, FH // 2)

    def test_empty_boxes_returns_center_no_targets(self, singleton):
        results = _FakeResults([])

        x, y, info = apc.get_ai_crosshair_position(results, FW, FH)

        assert info["status"] == "no_targets"
        assert (x, y) == (FW // 2, FH // 2)

    def test_boxes_missing_attribute_is_returned_when_object_lacks_boxes(self, singleton):
        """A results object without a ``boxes`` attribute yields no detections."""

        class _NoBoxes:
            pass

        _, _, info = apc.get_ai_crosshair_position(_NoBoxes(), FW, FH)
        assert info["status"] == "no_targets"

    def test_malformed_box_without_tensors_is_skipped(self, singleton):
        """A box lacking xyxy/conf/cls is skipped, leaving no usable detection."""
        results = _FakeResults([_BareBox()], names={7: "clock"})

        _, _, info = apc.get_ai_crosshair_position(results, FW, FH)

        assert info["status"] == "no_targets"

    def test_non_drone_class_is_filtered_to_center(self, singleton):
        """A confidently-detected non-drone class (e.g. 'person') is dropped."""
        results = _FakeResults([_FakeBox([10, 10, 200, 400], 0.99, 0)],
                               names={0: "person"})

        x, y, info = apc.get_ai_crosshair_position(results, FW, FH)

        assert info["status"] == "no_targets"
        assert (x, y) == (FW // 2, FH // 2)


# ---------------------------------------------------------------------------
# get_adaptive_crosshair: branches not reached by the sibling suite
# ---------------------------------------------------------------------------
class TestAdaptiveCrosshairBranches:
    def test_no_valid_targets_when_scoring_finds_no_winner(self):
        """With filtering off, a zero-confidence detection scores 0.

        ``best_score`` starts at 0 and the comparison is strictly ``>``, so a
        lone confidence-0 detection never becomes ``best_detection`` and the
        router reports ``"no_valid_targets"`` at frame center.
        """
        corrector = AIParallaxCorrector()
        corrector.drone_only_mode = False

        x, y, info = corrector.get_adaptive_crosshair(
            [{"bbox": [10, 10, 20, 20], "class": "anything", "confidence": 0.0}],
            FW, FH,
        )

        assert info["status"] == "no_valid_targets"
        assert (x, y) == (FW // 2, FH // 2)

    def test_close_object_uses_blended_raw_position(self):
        """A large, nearby object (<500 mm) blends 80% raw + 20% center.

        A fresh corrector has no smoothing history, so the smoothed result equals
        the blended corrected position exactly. bbox center is (950, 540); the
        blend toward the (960, 540) frame center gives (952, 540).
        """
        corrector = AIParallaxCorrector()
        bbox = [100, 40, 1800, 1040]  # 1700x1000 px clock => very close

        x, y, info = corrector.get_adaptive_crosshair(
            [{"bbox": bbox, "class": "clock", "confidence": 0.9}], FW, FH, 1.0
        )

        assert info["status"] == "ai_corrected"
        assert info["estimated_distance_mm"] < 500

        target_x = int((bbox[0] + bbox[2]) / 2)
        target_y = int((bbox[1] + bbox[3]) / 2)
        expected_x = int(target_x * 0.8 + (FW / 2) * 0.2)
        expected_y = int(target_y * 0.8 + (FH / 2) * 0.2)
        assert (x, y) == (expected_x, expected_y)
        # Corrected (pre-smoothing) position equals the blended value, not the
        # full parallax correction used for distant objects.
        assert info["corrected_position"] == (expected_x, expected_y)

    def test_second_valid_detection_is_smoothed_toward_previous(self):
        """On consecutive valid frames the crosshair is EMA-smoothed.

        The first detection seeds ``last_crosshair_{x,y}``; a second detection at
        a different position is blended ``0.7*previous + 0.3*new`` rather than
        snapping. This pins the ``last_crosshair_x is not None`` smoothing arm,
        which a single-frame call never reaches.
        """
        det1 = {"bbox": [200, 200, 230, 230], "class": "clock", "confidence": 0.9}
        det2 = {"bbox": [1600, 800, 1630, 830], "class": "clock", "confidence": 0.9}

        corrector = AIParallaxCorrector()
        x1, y1, _ = corrector.get_adaptive_crosshair([det1], FW, FH, 1.0)
        x2, y2, info2 = corrector.get_adaptive_crosshair([det2], FW, FH, 1.0)

        # Recover det2's unsmoothed corrected position from a history-free corrector.
        fresh = AIParallaxCorrector()
        raw_x, raw_y, _ = fresh.get_adaptive_crosshair([det2], FW, FH, 1.0)

        f = corrector.smoothing_factor  # 0.7
        assert x2 == int(x1 * f + raw_x * (1 - f))
        assert y2 == int(y1 * f + raw_y * (1 - f))
        # The smoothed point trails the raw target (it has not snapped there yet).
        assert info2["smoothed_position"] == (x2, y2)
        assert (x2, y2) != (raw_x, raw_y)
