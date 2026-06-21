"""Unit tests for ``config.Config`` — the centralized drone-tracker configuration.

``config.py`` is the single source of truth that every other module reads from
(``esp32_drone_integration``, ``ai_parallax_correction``, ``yolo_camera_detection``
all import ``config.config``). Each attribute is read **once, at class-definition
time**, from an environment variable with a hardcoded default and an explicit
``int``/``float``/``bool`` coercion. That makes the module a brittle contract:

* a renamed env key or a changed default silently moves a servo home angle,
  a detection threshold, or a window size;
* the ``int(...)`` / ``float(...)`` coercions raise ``ValueError`` on garbage
  input, so a typo'd ``.env`` value crashes import rather than degrading;
* ``DRONE_ONLY_MODE`` is parsed by ``.lower() == 'true'`` — only that exact
  spelling (case-insensitively) is truthy;
* ``print_config`` must never leak the WiFi password.

None of this was guarded. These tests pin it.

The tests are **hermetic**: ``dotenv.load_dotenv`` is neutralized before each
import so a stray ``.env`` on disk can never bleed in, and the module is freshly
re-imported with a controlled environment for every case. No production code is
modified; every assertion documents current behavior.
"""

import importlib
import sys

import pytest


# ---------------------------------------------------------------------------
# Helper: import (or re-import) ``config`` with a controlled environment and a
# neutralized ``load_dotenv`` so no real ``.env`` file participates.
# ---------------------------------------------------------------------------

# Every env key ``config`` consults — cleared before each fresh import so a
# default-path test sees the hardcoded fallbacks regardless of the host env.
_CONFIG_ENV_KEYS = [
    "WIFI_SSID", "WIFI_PASSWORD",
    "ESP32_IP", "ESP32_TIMEOUT",
    "CAMERA_INDEX", "CAMERA_WIDTH", "CAMERA_HEIGHT",
    "PAN_HOME", "TILT_HOME", "CENTER_ANGLE",
    "CONFIDENCE_THRESHOLD", "IOU_THRESHOLD", "MAX_DETECTIONS",
    "CAMERA_FOV_HORIZONTAL", "CAMERA_FOV_VERTICAL",
    "MANUAL_STEP_SIZE", "MANUAL_STEP_SIZE_SHIFT",
    "MAIN_WINDOW_WIDTH", "MAIN_WINDOW_HEIGHT", "MAIN_WINDOW_NAME",
    "MAIN_WINDOW_POS_X", "MAIN_WINDOW_POS_Y",
    "INFO_PANEL_WIDTH", "INFO_PANEL_HEIGHT", "INFO_PANEL_NAME",
    "INFO_PANEL_POS_X", "INFO_PANEL_POS_Y",
    "DISPLAY_QUEUE_SIZE", "INFO_QUEUE_SIZE",
    "DEFAULT_ZOOM_FACTOR", "MAX_ZOOM_FACTOR", "ZOOM_STEP",
    "CROSSHAIR_OFFSET_X", "CROSSHAIR_OFFSET_Y",
    "FONT_SCALE_SMALL", "FONT_SCALE_MEDIUM", "FONT_SCALE_LARGE",
    "MAX_LOG_MESSAGES", "LOG_MESSAGE_MAX_LENGTH",
    "MOVEMENT_THRESHOLD", "MAX_TRACKING_HISTORY",
    "DRONE_ONLY_MODE", "MIN_DRONE_CONFIDENCE", "CROSSHAIR_SMOOTHING",
    "DEBUG_FRAME_INTERVAL",
    "HTTP_TIMEOUT", "HTTP_RETRIES", "COMMAND_RATE_LIMIT",
]


def load_config(monkeypatch, env=None):
    """Freshly import ``config`` under a controlled environment.

    ``env`` maps env-var name -> value to *set*; every other recognized key is
    deleted so the module falls back to its hardcoded default. ``load_dotenv``
    is patched to a no-op first so an on-disk ``.env`` never participates.
    """
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)

    for key in _CONFIG_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)

    sys.modules.pop("config", None)
    module = importlib.import_module("config")
    return module


@pytest.fixture(autouse=True)
def _restore_config_module():
    """Drop any cached ``config`` after each test so a real import elsewhere
    in the suite is unaffected by our env-manipulated reloads."""
    yield
    sys.modules.pop("config", None)


# ---------------------------------------------------------------------------
# Defaults — the hardcoded fallbacks when nothing is set in the environment.
# ---------------------------------------------------------------------------

def test_string_defaults(monkeypatch):
    cfg = load_config(monkeypatch).config
    assert cfg.WIFI_SSID == "your_wifi_name"
    assert cfg.WIFI_PASSWORD == "your_wifi_password"
    assert cfg.ESP32_IP == "10.0.0.70"
    assert cfg.MAIN_WINDOW_NAME == "YOLO Drone Tracker - HD"
    assert cfg.INFO_PANEL_NAME == "System Info & Logs"


def test_numeric_defaults_have_correct_types_and_values(monkeypatch):
    cfg = load_config(monkeypatch).config
    # ints
    assert cfg.CAMERA_INDEX == 0 and isinstance(cfg.CAMERA_INDEX, int)
    assert cfg.CAMERA_WIDTH == 3840 and isinstance(cfg.CAMERA_WIDTH, int)
    assert cfg.CAMERA_HEIGHT == 2160
    assert cfg.PAN_HOME == 102
    assert cfg.TILT_HOME == 90
    assert cfg.CENTER_ANGLE == 90
    assert cfg.MAX_DETECTIONS == 200
    assert cfg.MOVEMENT_THRESHOLD == 25
    # floats
    assert cfg.ESP32_TIMEOUT == 0.5 and isinstance(cfg.ESP32_TIMEOUT, float)
    assert cfg.CONFIDENCE_THRESHOLD == 0.05 and isinstance(cfg.CONFIDENCE_THRESHOLD, float)
    assert cfg.IOU_THRESHOLD == 0.2
    assert cfg.DEFAULT_ZOOM_FACTOR == 1.0
    assert cfg.MAX_ZOOM_FACTOR == 5.0
    assert cfg.HTTP_TIMEOUT == 1.0
    assert cfg.COMMAND_RATE_LIMIT == 0.1


def test_default_crosshair_is_centered(monkeypatch):
    cfg = load_config(monkeypatch).config
    assert cfg.CROSSHAIR_OFFSET_X == 0.0
    assert cfg.CROSSHAIR_OFFSET_Y == 0.0


# ---------------------------------------------------------------------------
# Env overrides + type coercion — the contract a generated ``.env`` relies on.
# ---------------------------------------------------------------------------

def test_int_env_is_coerced_from_string(monkeypatch):
    cfg = load_config(monkeypatch, {"CAMERA_WIDTH": "1280", "PAN_HOME": "120"}).config
    assert cfg.CAMERA_WIDTH == 1280 and isinstance(cfg.CAMERA_WIDTH, int)
    assert cfg.PAN_HOME == 120 and isinstance(cfg.PAN_HOME, int)


def test_float_env_is_coerced_from_string(monkeypatch):
    cfg = load_config(monkeypatch, {"ESP32_TIMEOUT": "1.5", "CONFIDENCE_THRESHOLD": "0.3"}).config
    assert cfg.ESP32_TIMEOUT == 1.5 and isinstance(cfg.ESP32_TIMEOUT, float)
    assert cfg.CONFIDENCE_THRESHOLD == 0.3


def test_string_env_passes_through_unparsed(monkeypatch):
    cfg = load_config(monkeypatch, {"WIFI_SSID": "HomeNet", "ESP32_IP": "192.168.1.42"}).config
    assert cfg.WIFI_SSID == "HomeNet"
    assert cfg.ESP32_IP == "192.168.1.42"


def test_invalid_int_env_raises_on_import(monkeypatch):
    # The bare ``int(os.getenv(...))`` has no guard: a non-numeric value makes
    # importing the module fail loudly rather than silently using a default.
    with pytest.raises(ValueError):
        load_config(monkeypatch, {"CAMERA_INDEX": "not-a-number"})


def test_invalid_float_env_raises_on_import(monkeypatch):
    with pytest.raises(ValueError):
        load_config(monkeypatch, {"ESP32_TIMEOUT": "fast"})


# ---------------------------------------------------------------------------
# DRONE_ONLY_MODE — parsed via ``.lower() == 'true'``.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["true", "True", "TRUE", "tRuE"])
def test_drone_only_mode_truthy_spellings(monkeypatch, value):
    cfg = load_config(monkeypatch, {"DRONE_ONLY_MODE": value}).config
    assert cfg.DRONE_ONLY_MODE is True


@pytest.mark.parametrize("value", ["false", "False", "0", "no", "yes", "1", ""])
def test_drone_only_mode_everything_else_is_false(monkeypatch, value):
    # Only the exact word "true" (any case) is truthy; even "1"/"yes" are False.
    cfg = load_config(monkeypatch, {"DRONE_ONLY_MODE": value}).config
    assert cfg.DRONE_ONLY_MODE is False


def test_drone_only_mode_defaults_true(monkeypatch):
    cfg = load_config(monkeypatch).config
    assert cfg.DRONE_ONLY_MODE is True


# ---------------------------------------------------------------------------
# Static structures — colors and the drone-class set.
# ---------------------------------------------------------------------------

def test_bgr_colors_are_fixed(monkeypatch):
    cfg = load_config(monkeypatch).config
    # OpenCV uses BGR, so "green" is (0,255,0); regressions here mis-color the HUD.
    assert cfg.COLOR_GREEN == (0, 255, 0)
    assert cfg.COLOR_WHITE == (255, 255, 255)
    assert cfg.COLOR_RED == (0, 0, 255)
    assert cfg.COLOR_YELLOW == (0, 255, 255)
    assert cfg.COLOR_BLUE == (255, 0, 0)


def test_drone_classes_set_contents(monkeypatch):
    cfg = load_config(monkeypatch).config
    assert isinstance(cfg.DRONE_CLASSES, set)
    # These are the COCO labels YOLO tends to misclassify a drone as.
    for expected in {"bird", "kite", "frisbee", "sports ball", "cell phone", "remote"}:
        assert expected in cfg.DRONE_CLASSES
    # "drone" is deliberately NOT a member — the model has no drone class, which
    # is the whole reason this lookup set exists.
    assert "drone" not in cfg.DRONE_CLASSES
    assert len(cfg.DRONE_CLASSES) == 10


# ---------------------------------------------------------------------------
# print_config — must mask the password and never print it in the clear.
# ---------------------------------------------------------------------------

def test_print_config_masks_password(monkeypatch, capsys):
    cfg = load_config(monkeypatch, {"WIFI_PASSWORD": "hunter2", "WIFI_SSID": "HomeNet"}).config
    cfg.print_config()
    out = capsys.readouterr().out
    # The literal password must not appear; it is replaced by one '*' per char.
    assert "hunter2" not in out
    assert "*" * len("hunter2") in out
    # Non-sensitive values are shown in the clear.
    assert "HomeNet" in out
    assert "10.0.0.70" in out


def test_print_config_class_method_runs_on_class(monkeypatch, capsys):
    # It's a @classmethod, so it works off the class without an instance too.
    mod = load_config(monkeypatch)
    mod.Config.print_config()
    out = capsys.readouterr().out
    assert "Current Configuration:" in out
    assert "Servo Home:" in out
