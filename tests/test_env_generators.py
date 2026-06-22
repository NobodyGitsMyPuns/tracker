"""Unit tests for the two ``.env`` generator scripts: ``setup_env`` and
``update_config``.

These small scripts are the *producers* of the ``.env`` file that
``config.Config`` consumes. ``config`` reads every value once, at
class-definition time, through ``dotenv`` + a hardcoded ``int()`` / ``float()``
coercion (see ``config.py``). That makes the generators a brittle, unguarded
contract:

* a typo'd numeric default (``CAMERA_WIDTH=384O``) would make ``config`` crash
  on *import* — every module imports ``config`` — rather than degrade;
* ``setup_env.py`` writes inline ``# comment`` annotations after some values
  (``PAN_HOME=102 # 12 degrees right from center``); those must still load as a
  clean integer the way ``config`` reads them (``dotenv`` strips trailing
  comments), or a servo home angle silently breaks;
* ``setup_env.py`` guards an existing ``.env`` behind an interactive
  overwrite prompt — answering anything but ``y`` must leave the file
  untouched, and ``update_config.py`` deliberately has *no* such guard;
* neither script may bake a real WiFi secret into the committed template.

Nothing exercised any of that before. These tests pin the *current* behavior;
no production code is modified. Every test runs inside a per-test temp
directory (``monkeypatch.chdir``) so it can never read, write, or clobber a
real ``.env`` at the repo root.
"""

import builtins

import pytest
from dotenv import dotenv_values

import setup_env
import update_config


# ---------------------------------------------------------------------------
# The type contract that ``config.Config`` imposes on the ``.env`` values.
# Mirrors the ``int(...)`` / ``float(...)`` coercions in ``config.py`` so a
# generator that emits an un-coercible value fails here instead of at runtime.
# ---------------------------------------------------------------------------
INT_KEYS = {
    "CAMERA_INDEX", "CAMERA_WIDTH", "CAMERA_HEIGHT",
    "PAN_HOME", "TILT_HOME", "CENTER_ANGLE",
    "MAX_DETECTIONS",
    "CAMERA_FOV_HORIZONTAL", "CAMERA_FOV_VERTICAL",
    "MANUAL_STEP_SIZE", "MANUAL_STEP_SIZE_SHIFT",
    "MAIN_WINDOW_WIDTH", "MAIN_WINDOW_HEIGHT",
    "MAIN_WINDOW_POS_X", "MAIN_WINDOW_POS_Y",
    "INFO_PANEL_WIDTH", "INFO_PANEL_HEIGHT",
    "INFO_PANEL_POS_X", "INFO_PANEL_POS_Y",
    "MOVEMENT_THRESHOLD", "MAX_TRACKING_HISTORY",
    "DEBUG_FRAME_INTERVAL",
    "MAX_LOG_MESSAGES", "LOG_MESSAGE_MAX_LENGTH",
    "HTTP_RETRIES",
}
FLOAT_KEYS = {
    "ESP32_TIMEOUT",
    "CONFIDENCE_THRESHOLD", "IOU_THRESHOLD",
    "CROSSHAIR_OFFSET_X", "CROSSHAIR_OFFSET_Y",
    "DEFAULT_ZOOM_FACTOR", "MAX_ZOOM_FACTOR", "ZOOM_STEP",
    "HTTP_TIMEOUT", "COMMAND_RATE_LIMIT",
}


def _assert_config_coercions_hold(values):
    """Every key the generator emits that ``config`` coerces must coerce cleanly."""
    for key in INT_KEYS & values.keys():
        raw = values[key]
        assert raw is not None, f"{key} present but unset"
        int(raw)  # raises ValueError if the generator emitted garbage
    for key in FLOAT_KEYS & values.keys():
        raw = values[key]
        assert raw is not None, f"{key} present but unset"
        float(raw)


@pytest.fixture
def in_tmp_cwd(tmp_path, monkeypatch):
    """Run the generator inside an isolated cwd so ``.env`` writes are sandboxed."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ===========================================================================
# update_config.create_4k_env
# ===========================================================================

def test_update_config_writes_env_in_cwd(in_tmp_cwd):
    assert update_config.create_4k_env() is None
    env = in_tmp_cwd / ".env"
    assert env.is_file()
    assert env.read_text(encoding="utf-8").strip() != ""


def test_update_config_4k_specific_values(in_tmp_cwd):
    """The whole point of this script: 4K camera + centered crosshair + HD window."""
    update_config.create_4k_env()
    v = dotenv_values(str(in_tmp_cwd / ".env"))

    assert v["CAMERA_WIDTH"] == "3840"
    assert v["CAMERA_HEIGHT"] == "2160"
    # Crosshair centered (both offsets zeroed).
    assert float(v["CROSSHAIR_OFFSET_X"]) == 0.0
    assert float(v["CROSSHAIR_OFFSET_Y"]) == 0.0
    # Display window driven down to 1080p for the 4K feed.
    assert v["MAIN_WINDOW_WIDTH"] == "1920"
    assert v["MAIN_WINDOW_HEIGHT"] == "1080"
    # Movement threshold bumped for the higher-resolution frame.
    assert v["MOVEMENT_THRESHOLD"] == "25"


def test_update_config_values_satisfy_config_coercion(in_tmp_cwd):
    update_config.create_4k_env()
    v = dotenv_values(str(in_tmp_cwd / ".env"))
    _assert_config_coercions_hold(v)


def test_update_config_does_not_bake_in_a_wifi_secret(in_tmp_cwd):
    update_config.create_4k_env()
    v = dotenv_values(str(in_tmp_cwd / ".env"))
    # Both keys are present (so config doesn't fall back to placeholder defaults)
    # but blank — no real credential committed.
    assert v["WIFI_SSID"] == ""
    assert v["WIFI_PASSWORD"] == ""


def test_update_config_overwrites_unconditionally(in_tmp_cwd):
    """update_config has no overwrite guard: an existing .env is always replaced."""
    env = in_tmp_cwd / ".env"
    env.write_text("SENTINEL=keepme\n", encoding="utf-8")

    update_config.create_4k_env()

    text = env.read_text(encoding="utf-8")
    assert "SENTINEL" not in text
    assert "CAMERA_WIDTH=3840" in text


# ===========================================================================
# setup_env.setup_env_file
# ===========================================================================

def test_setup_env_writes_when_no_env_present_without_prompting(in_tmp_cwd, monkeypatch):
    """No existing .env => write straight through, never touching input()."""
    def _boom(*_a, **_k):
        raise AssertionError("input() must not be called when no .env exists")
    monkeypatch.setattr(builtins, "input", _boom)

    setup_env.setup_env_file()

    v = dotenv_values(str(in_tmp_cwd / ".env"))
    # 720p defaults distinguish this script from update_config's 4K output.
    assert v["CAMERA_WIDTH"] == "1280"
    assert v["CAMERA_HEIGHT"] == "720"
    assert v["PAN_HOME"] == "102"


def test_setup_env_inline_comments_load_as_clean_ints(in_tmp_cwd, monkeypatch):
    """``PAN_HOME=102 # ...`` must load as the int ``102`` the way config reads it."""
    monkeypatch.setattr(builtins, "input", lambda *_a, **_k: "y")
    setup_env.setup_env_file()

    v = dotenv_values(str(in_tmp_cwd / ".env"))
    # dotenv strips the trailing inline comment; the coercion config performs
    # must succeed and yield the intended value.
    assert int(v["PAN_HOME"]) == 102
    assert int(v["TILT_HOME"]) == 90
    _assert_config_coercions_hold(v)


def test_setup_env_values_satisfy_config_coercion(in_tmp_cwd):
    setup_env.setup_env_file()  # no .env present, no prompt
    v = dotenv_values(str(in_tmp_cwd / ".env"))
    _assert_config_coercions_hold(v)


def test_setup_env_does_not_bake_in_a_wifi_secret(in_tmp_cwd):
    setup_env.setup_env_file()
    v = dotenv_values(str(in_tmp_cwd / ".env"))
    assert v["WIFI_SSID"] == ""
    assert v["WIFI_PASSWORD"] == ""


@pytest.mark.parametrize("answer", ["n", "N", "", "no", "yikes"])
def test_setup_env_overwrite_guard_declines(in_tmp_cwd, monkeypatch, answer):
    """An existing .env is preserved unless the user explicitly answers 'y'."""
    env = in_tmp_cwd / ".env"
    original = "SENTINEL=preserve-me\n"
    env.write_text(original, encoding="utf-8")

    prompted = []
    monkeypatch.setattr(builtins, "input", lambda *a, **k: prompted.append(a) or answer)

    setup_env.setup_env_file()

    assert prompted, "overwrite guard must prompt when .env already exists"
    assert env.read_text(encoding="utf-8") == original  # untouched


@pytest.mark.parametrize("answer", ["y", "Y"])
def test_setup_env_overwrite_guard_accepts(in_tmp_cwd, monkeypatch, answer):
    """An explicit 'y'/'Y' overwrites the existing file."""
    env = in_tmp_cwd / ".env"
    env.write_text("SENTINEL=preserve-me\n", encoding="utf-8")
    monkeypatch.setattr(builtins, "input", lambda *_a, **_k: answer)

    setup_env.setup_env_file()

    text = env.read_text(encoding="utf-8")
    assert "SENTINEL" not in text
    assert "CAMERA_WIDTH=1280" in text
