"""Unit tests for ``generate_arduino_config.generate_arduino_config``.

This module is the bridge between the Python-side ``config`` and the firmware:
it renders an Arduino C++ header (``ESP32_OTA_Servo/config.h``) so the ESP32
sketch compiles against the *same* WiFi credentials and servo home angles the
Python tracker uses. The generator is a deterministic transform — config
attributes interpolated into a fixed header template — plus a single file
write to a hardcoded relative path.

Because the rendered ``#define``/``const char*`` lines are consumed verbatim by
a C++ compiler, the contract is unusually brittle: rename a ``config`` field,
drop the include guard, or change a hardcoded pin and the firmware either fails
to build or silently drives the wrong GPIO. Nothing guarded that before.

Following the convention established by ``test_esp32_drone_integration.py``, a
lightweight stub ``config`` module is installed *before* importing the module
under test, so the suite is hermetic: no ``python-dotenv``, no ``.env``, and
fully controlled config values. No production code is modified; every
assertion pins the generator's current behavior.
"""

import sys
import types
import importlib

import pytest


# ---------------------------------------------------------------------------
# Stub ``config`` module installed before importing the generator.
# Mirrors only the attributes ``generate_arduino_config`` reads.
# Defaults match config.py's own fallbacks so the suite reflects real values.
# ---------------------------------------------------------------------------
def _install_config_stub(**overrides):
    stub = types.ModuleType("config")

    class _Cfg:
        WIFI_SSID = "your_wifi_name"
        WIFI_PASSWORD = "your_wifi_password"
        PAN_HOME = 102
        TILT_HOME = 90
        CENTER_ANGLE = 90

    cfg = _Cfg()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    stub.config = cfg
    sys.modules["config"] = stub
    return cfg


def _load_generator(**overrides):
    """Install a fresh config stub and (re)import the generator against it."""
    _install_config_stub(**overrides)
    sys.modules.pop("generate_arduino_config", None)
    return importlib.import_module("generate_arduino_config")


def _render(tmp_path, monkeypatch, **overrides):
    """Run the generator inside an isolated CWD and return the file content.

    The generator writes to the *relative* path ``ESP32_OTA_Servo/config.h``,
    so the test provides that directory inside a throwaway working directory.
    """
    module = _load_generator(**overrides)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ESP32_OTA_Servo").mkdir()
    module.generate_arduino_config()
    return (tmp_path / "ESP32_OTA_Servo" / "config.h").read_text()


# ---------------------------------------------------------------------------
# Output location + structural contract
# ---------------------------------------------------------------------------
def test_writes_header_to_esp32_project_dir(tmp_path, monkeypatch):
    module = _load_generator()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ESP32_OTA_Servo").mkdir()

    module.generate_arduino_config()

    out = tmp_path / "ESP32_OTA_Servo" / "config.h"
    assert out.is_file(), "generator must write ESP32_OTA_Servo/config.h"
    assert out.read_text().strip(), "generated header must not be empty"


def test_missing_target_dir_raises(tmp_path, monkeypatch):
    # The generator does not create the ESP32_OTA_Servo directory; if it is
    # absent the open() for write fails. Pin that current behavior.
    module = _load_generator()
    monkeypatch.chdir(tmp_path)  # note: no ESP32_OTA_Servo dir created

    with pytest.raises((FileNotFoundError, OSError)):
        module.generate_arduino_config()


def test_has_include_guard(tmp_path, monkeypatch):
    content = _render(tmp_path, monkeypatch)
    assert "#ifndef CONFIG_H" in content
    assert "#define CONFIG_H" in content
    # The closing #endif carries the guard name as a trailing comment.
    assert "#endif // CONFIG_H" in content
    # Guard opens before it closes.
    assert content.index("#ifndef CONFIG_H") < content.index("#endif // CONFIG_H")


def test_carries_do_not_edit_banner(tmp_path, monkeypatch):
    # The header is generated; the banner tells humans to edit .env instead.
    content = _render(tmp_path, monkeypatch)
    assert "Auto-generated" in content
    assert "generate_arduino_config.py" in content


# ---------------------------------------------------------------------------
# WiFi credentials -> quoted C string literals from config
# ---------------------------------------------------------------------------
def test_wifi_credentials_emitted_as_c_strings(tmp_path, monkeypatch):
    content = _render(
        tmp_path,
        monkeypatch,
        WIFI_SSID="MyNetwork",
        WIFI_PASSWORD="hunter2",
    )
    assert 'const char* WIFI_SSID = "MyNetwork";' in content
    assert 'const char* WIFI_PASSWORD = "hunter2";' in content


def test_default_wifi_placeholders_flow_through(tmp_path, monkeypatch):
    content = _render(tmp_path, monkeypatch)
    assert 'const char* WIFI_SSID = "your_wifi_name";' in content
    assert 'const char* WIFI_PASSWORD = "your_wifi_password";' in content


def test_ssid_with_spaces_embedded_verbatim(tmp_path, monkeypatch):
    # Current behavior: the SSID is interpolated raw, no escaping. A space is
    # legal inside a C string literal, so this documents (not endorses) that
    # the generator performs no sanitisation.
    content = _render(tmp_path, monkeypatch, WIFI_SSID="Guest Network 5G")
    assert 'const char* WIFI_SSID = "Guest Network 5G";' in content


# ---------------------------------------------------------------------------
# Servo home angles -> #define lines tracking config
# ---------------------------------------------------------------------------
def test_servo_home_defines_match_config(tmp_path, monkeypatch):
    content = _render(
        tmp_path,
        monkeypatch,
        PAN_HOME=110,
        TILT_HOME=85,
        CENTER_ANGLE=95,
    )
    assert "#define PAN_HOME 110" in content
    assert "#define TILT_HOME 85" in content
    assert "#define CENTER_ANGLE 95" in content


def test_default_servo_home_defines(tmp_path, monkeypatch):
    content = _render(tmp_path, monkeypatch)
    assert "#define PAN_HOME 102" in content
    assert "#define TILT_HOME 90" in content
    assert "#define CENTER_ANGLE 90" in content


# ---------------------------------------------------------------------------
# Hardware constants are hardcoded (independent of config) and must stay put
# ---------------------------------------------------------------------------
def test_servo_pins_are_hardcoded(tmp_path, monkeypatch):
    # These are wired in hardware; the generator must not let config override
    # them. Change config and confirm the pin defines are unchanged.
    content = _render(tmp_path, monkeypatch, PAN_HOME=1, TILT_HOME=2, CENTER_ANGLE=3)
    assert "#define PAN_PIN 18" in content
    assert "#define TILT_PIN 19" in content
    assert "#define BLASTER_PIN 21" in content


def test_pwm_and_timing_constants_present(tmp_path, monkeypatch):
    content = _render(tmp_path, monkeypatch)
    for define in (
        "#define PWM_FREQ 50",
        "#define PWM_RES 16",
        "#define SERVO_MIN_US 1000",
        "#define SERVO_MAX_US 2000",
        "#define SERVO_CENTER_US 1500",
    ):
        assert define in content, f"missing hardware constant: {define}"


# ---------------------------------------------------------------------------
# Idempotency + overwrite semantics
# ---------------------------------------------------------------------------
def test_regeneration_is_idempotent(tmp_path, monkeypatch):
    module = _load_generator(WIFI_SSID="StableNet", PAN_HOME=100)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ESP32_OTA_Servo").mkdir()
    out = tmp_path / "ESP32_OTA_Servo" / "config.h"

    module.generate_arduino_config()
    first = out.read_text()
    module.generate_arduino_config()
    second = out.read_text()

    assert first == second, "same config must render byte-identical headers"


def test_overwrites_stale_header(tmp_path, monkeypatch):
    module = _load_generator(WIFI_SSID="NewNet")
    monkeypatch.chdir(tmp_path)
    target_dir = tmp_path / "ESP32_OTA_Servo"
    target_dir.mkdir()
    stale = target_dir / "config.h"
    stale.write_text('const char* WIFI_SSID = "OldNet";\n')

    module.generate_arduino_config()

    content = stale.read_text()
    assert 'const char* WIFI_SSID = "NewNet";' in content
    assert "OldNet" not in content, "stale content must be fully overwritten"
