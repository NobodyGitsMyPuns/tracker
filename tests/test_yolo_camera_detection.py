"""Unit tests for the pure-logic helpers in ``yolo_camera_detection``.

``yolo_camera_detection.py`` is the largest module in the tracker (≈800 lines)
and was entirely untested, because importing it drags in the heavy runtime
stack — ``cv2`` (OpenCV), ``torch``, and ``ultralytics`` — plus a live camera
and an ESP32 on the network. None of that is needed to exercise the module's
*deterministic* image helpers, so this suite installs lightweight stand-ins for
the heavy imports (and the usual stub ``config``, matching
``test_esp32_drone_integration.py`` / ``test_generate_arduino_config.py``)
**before** importing the module, then drives the two pure functions that have
real, regression-prone logic:

``apply_digital_zoom``
    The crop-and-rescale digital-zoom transform. The genuinely brittle part is
    the geometry: crop size is ``int(dim / zoom)``, the crop is positioned by a
    0–1 ``center`` fraction, and the origin is **clamped** so the crop never
    runs off the frame. A regression here would silently zoom into the wrong
    region or read out of bounds. ``cv2.resize`` is the only OpenCV call and is
    stubbed to capture its arguments, so the crop math is asserted directly.

``create_info_panel``
    Builds the HUD side-panel ``numpy`` image and chooses which status/target/
    debug/log lines to draw. ``cv2.putText`` is a no-op stub that records the
    text it was asked to render, so the branch logic (target present vs. absent,
    AI-debug fields, the 45-char log-line truncation, the controls footer) is
    pinned by asserting on what would have been drawn.

No production code is modified; every assertion documents current behavior. The
suite is hermetic — no OpenCV/torch/ultralytics, no camera, no network.
"""

import sys
import types
import importlib

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Stub the heavy / hardware-bound imports BEFORE importing the module so the
# import succeeds in a minimal environment (pytest + numpy only).
# ---------------------------------------------------------------------------
class _FakeCV2(types.ModuleType):
    """Minimal OpenCV stand-in that records the calls the helpers make."""

    # Constants the module references by attribute.
    FONT_HERSHEY_SIMPLEX = 0
    INTER_LINEAR = 1

    def __init__(self):
        super().__init__("cv2")
        self.resize_calls = []
        self.puttext_texts = []

    def reset(self):
        self.resize_calls.clear()
        self.puttext_texts.clear()

    def resize(self, src, dsize, interpolation=None):
        # Record the request and return the *cropped* input unchanged so the
        # caller's crop region is directly inspectable via the return value.
        self.resize_calls.append(
            {"src": src, "dsize": dsize, "interpolation": interpolation}
        )
        return src

    def putText(self, img, text, org, fontFace, fontScale, color, thickness=1,
                *args, **kwargs):
        self.puttext_texts.append(text)
        return img


def _install_module_stubs():
    fake_cv2 = _FakeCV2()
    sys.modules["cv2"] = fake_cv2

    torch_stub = types.ModuleType("torch")
    sys.modules["torch"] = torch_stub

    ultralytics_stub = types.ModuleType("ultralytics")
    ultralytics_stub.YOLO = object  # only the name needs to exist at import
    sys.modules["ultralytics"] = ultralytics_stub

    config_stub = types.ModuleType("config")

    class _Cfg:
        # Read at yolo_camera_detection import time.
        DISPLAY_QUEUE_SIZE = 2
        INFO_QUEUE_SIZE = 2
        DEFAULT_ZOOM_FACTOR = 1.0
        # Read by create_info_panel.
        INFO_PANEL_WIDTH = 600
        INFO_PANEL_HEIGHT = 400
        # Read by ESP32DroneTracker.__init__ (imported, not constructed here);
        # included so the stub is a faithful drop-in regardless of import order.
        ESP32_IP = "10.0.0.70"
        ESP32_TIMEOUT = 0.5
        COMMAND_RATE_LIMIT = 0.1

    config_stub.config = _Cfg()
    sys.modules["config"] = config_stub
    return fake_cv2


_CV2 = _install_module_stubs()
yolo = importlib.import_module("yolo_camera_detection")


@pytest.fixture(autouse=True)
def _reset_cv2_recorder():
    """Clear the recorded OpenCV calls before every test."""
    _CV2.reset()
    yield


# ---------------------------------------------------------------------------
# Helper: a frame whose every pixel encodes its (row, col) so a crop can be
# verified positionally.
# ---------------------------------------------------------------------------
def _coord_frame(height, width):
    rows = np.arange(height).reshape(height, 1)
    cols = np.arange(width).reshape(1, width)
    return rows * 1000 + cols  # frame[y, x] == y*1000 + x


# ===========================================================================
# apply_digital_zoom
# ===========================================================================
class TestApplyDigitalZoom:
    def test_zoom_at_or_below_one_returns_same_frame_untouched(self):
        frame = _coord_frame(40, 60)
        # Identity (no copy) and no resize for the no-zoom and "negative" cases.
        for zf in (1.0, 0.5, 0.0):
            result = yolo.apply_digital_zoom(frame, zf)
            assert result is frame
        assert _CV2.resize_calls == []

    def test_centered_2x_zoom_crops_the_middle_quarter_and_rescales(self):
        frame = _coord_frame(80, 100)
        result = yolo.apply_digital_zoom(frame, 2.0)  # center defaults to 0.5

        # crop_w = int(100/2)=50, crop_h = int(80/2)=40
        # crop_x = int((100-50)*0.5)=25, crop_y = int((80-40)*0.5)=20
        expected = frame[20:60, 25:75]
        assert result.shape == (40, 50)
        assert np.array_equal(result, expected)

        # The single OpenCV call must request a rescale back to the full frame.
        assert len(_CV2.resize_calls) == 1
        call = _CV2.resize_calls[0]
        assert call["dsize"] == (100, 80)            # (width, height)
        assert call["interpolation"] == _CV2.INTER_LINEAR

    def test_center_fraction_clamps_to_top_left_edge(self):
        frame = _coord_frame(80, 100)
        result = yolo.apply_digital_zoom(frame, 2.0, center_x=0.0, center_y=0.0)
        # Origin pinned at (0, 0); top-left pixel is frame[0, 0] == 0.
        assert result.shape == (40, 50)
        assert np.array_equal(result, frame[0:40, 0:50])
        assert result[0, 0] == 0

    def test_center_fraction_clamps_to_bottom_right_without_overrun(self):
        frame = _coord_frame(80, 100)
        result = yolo.apply_digital_zoom(frame, 2.0, center_x=1.0, center_y=1.0)
        # crop_x clamped to width-crop_w=50, crop_y to height-crop_h=40.
        assert result.shape == (40, 50)
        assert np.array_equal(result, frame[40:80, 50:100])
        # Bottom-right pixel is the true last pixel of the frame (no overrun).
        assert result[-1, -1] == 79 * 1000 + 99

    def test_crop_size_uses_integer_truncation_of_dimension_over_zoom(self):
        frame = _coord_frame(80, 100)
        # 100/3 -> 33, 80/3 -> 26 after int() truncation.
        result = yolo.apply_digital_zoom(frame, 3.0, center_x=0.0, center_y=0.0)
        assert result.shape == (26, 33)

    def test_zoom_preserves_color_channels(self):
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        frame[:, :, 1] = 7  # mark the green channel
        result = yolo.apply_digital_zoom(frame, 2.0)
        assert result.shape == (40, 50, 3)
        assert np.all(result[:, :, 1] == 7)


# ===========================================================================
# create_info_panel
# ===========================================================================
class TestCreateInfoPanel:
    def test_panel_has_configured_dimensions_and_dtype(self):
        panel = yolo.create_info_panel({})
        assert isinstance(panel, np.ndarray)
        assert panel.shape == (400, 600, 3)   # (INFO_PANEL_HEIGHT, WIDTH, 3)
        assert panel.dtype == np.uint8

    def test_ai_status_line_reflects_the_default_off_state(self):
        # ai_tracking_active is forced off in the module ("Always keep this off").
        yolo.create_info_panel({})
        assert any("AI TRACKING: OFF" in t for t in _CV2.puttext_texts)
        assert not any("AI TRACKING: ON" in t for t in _CV2.puttext_texts)

    def test_no_target_message_when_drone_data_absent(self):
        yolo.create_info_panel({"drone_data": None})
        assert any("NO TARGET DETECTED" in t for t in _CV2.puttext_texts)
        assert not any("TARGET DATA" in t for t in _CV2.puttext_texts)

    def test_target_data_rows_are_rendered_when_present(self):
        yolo.create_info_panel(
            {"drone_data": {"distance": "5m", "class": "drone"}}
        )
        texts = _CV2.puttext_texts
        assert any("TARGET DATA" in t for t in texts)
        assert "  distance: 5m" in texts
        assert "  class: drone" in texts
        assert not any("NO TARGET DETECTED" in t for t in texts)

    def test_ai_debug_status_class_and_confidence_lines(self):
        yolo.create_info_panel(
            {
                "ai_debug": {
                    "status": "tracking",
                    "target_class": "drone",
                    "confidence": 0.97345,
                }
            }
        )
        texts = _CV2.puttext_texts
        assert "  Status: tracking" in texts
        assert "  Class: drone" in texts
        assert "  Conf: 0.97" in texts          # formatted to 2 decimals

    def test_ai_debug_optional_fields_are_omitted_when_absent(self):
        yolo.create_info_panel({"ai_debug": {"status": "idle"}})
        texts = _CV2.puttext_texts
        assert "  Status: idle" in texts
        assert not any(t.startswith("  Class:") for t in texts)
        assert not any(t.startswith("  Conf:") for t in texts)

    def test_long_log_messages_are_truncated_to_42_chars_plus_ellipsis(self):
        long_msg = "x" * 60
        yolo.create_info_panel({"log_messages": [long_msg]})
        texts = _CV2.puttext_texts
        assert ("x" * 42 + "...") in texts
        assert long_msg not in texts            # the full 60-char line is not drawn

    def test_short_log_messages_render_verbatim_under_the_logs_header(self):
        msgs = ["boot ok", "esp32 connected", "tracking armed"]
        yolo.create_info_panel({"log_messages": msgs})
        texts = _CV2.puttext_texts
        assert any("RECENT LOGS" in t for t in texts)
        for m in msgs:
            assert m in texts

    def test_controls_footer_is_drawn_for_a_sparse_panel(self):
        yolo.create_info_panel({})
        texts = _CV2.puttext_texts
        assert any("CONTROLS" in t for t in texts)
        assert any("Fire" in t for t in texts)   # the SPACE=Fire reminder line
