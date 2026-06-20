"""Unit tests for ``esp32_drone_integration.ESP32DroneTracker``.

This module is the bridge between the camera/AI side and the physical ESP32
servo controller: it builds the HTTP command URLs, branches on the response
status code, rate-limits commands, and (in ``track_drone``) converts a pixel
deviation into a pan/tilt servo move. All of that is deterministic apart from
the single ``requests.get`` boundary and the ``time`` clock, both of which are
stubbed here so the suite is hermetic (no network, no ESP32, no sleeping).

The ``config`` module is replaced with a lightweight stub *before* importing
the module under test so the tests do not depend on ``python-dotenv`` or on
whatever ``.env`` happens to be present in the environment — they pin the
behavior of ``esp32_drone_integration`` itself, in isolation.

No production code is modified; every assertion documents the module's current
behavior, including a couple of quirks (see the ``step_size`` notes below).
"""

import sys
import types
import importlib

import pytest


# ---------------------------------------------------------------------------
# Install a stub ``config`` module before importing the module under test.
# Mirrors the attributes ESP32DroneTracker reads from ``config.config``.
# ---------------------------------------------------------------------------
def _install_config_stub():
    stub = types.ModuleType("config")

    class _Cfg:
        ESP32_IP = "10.0.0.70"
        ESP32_TIMEOUT = 0.5
        COMMAND_RATE_LIMIT = 0.1

    stub.config = _Cfg()
    sys.modules["config"] = stub
    return _Cfg


_CFG = _install_config_stub()
esp32 = importlib.import_module("esp32_drone_integration")


# ---------------------------------------------------------------------------
# Test doubles for the requests/time boundaries.
# ---------------------------------------------------------------------------
class FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json = json_data

    def json(self):
        return self._json


class FakeRequests:
    """Records every GET and replays a queued/constant response.

    ``raise_exc`` lets a test force the ``except`` branch.
    """

    def __init__(self, response=None, raise_exc=None):
        self.calls = []
        self._response = response if response is not None else FakeResponse(200)
        self._raise_exc = raise_exc

    def get(self, url, timeout=None):
        self.calls.append({"url": url, "timeout": timeout})
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._response

    @property
    def urls(self):
        return [c["url"] for c in self.calls]


class FakeClock:
    """Deterministic replacement for the module's ``time`` reference."""

    def __init__(self, now=1000.0):
        self.now = now
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        # Advance the clock as a real sleep would, so post-sleep reads differ.
        self.now += seconds


@pytest.fixture
def patched(monkeypatch):
    """Patch ``requests`` and ``time`` on the module, return a small helper."""

    clock = FakeClock()
    monkeypatch.setattr(esp32, "time", clock)

    state = {}

    def install(response=None, raise_exc=None):
        fake = FakeRequests(response=response, raise_exc=raise_exc)
        monkeypatch.setattr(esp32, "requests", fake)
        state["requests"] = fake
        return fake

    install()  # default: 200 OK
    state["clock"] = clock
    state["install"] = install
    return state


def make_tracker(**kwargs):
    return esp32.ESP32DroneTracker(**kwargs)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------
class TestInit:
    def test_defaults_come_from_config(self, patched):
        t = make_tracker()
        assert t.esp32_ip == _CFG.ESP32_IP
        assert t.base_url == f"http://{_CFG.ESP32_IP}"
        assert t.timeout == _CFG.ESP32_TIMEOUT
        assert t.command_rate_limit == _CFG.COMMAND_RATE_LIMIT
        assert t.last_command_time == 0
        assert t.tracking_active is False

    def test_explicit_args_override_config(self, patched):
        t = make_tracker(esp32_ip="192.168.1.5", timeout=2.0, command_rate_limit=0.25)
        assert t.esp32_ip == "192.168.1.5"
        assert t.base_url == "http://192.168.1.5"
        assert t.timeout == 2.0
        assert t.command_rate_limit == 0.25

    def test_falsy_overrides_fall_back_to_config(self, patched):
        # `x or config.X` means 0/0.0 are treated as "unset" and replaced by the
        # config default. Pin this current behavior.
        t = make_tracker(esp32_ip="", timeout=0, command_rate_limit=0)
        assert t.esp32_ip == _CFG.ESP32_IP
        assert t.timeout == _CFG.ESP32_TIMEOUT
        assert t.command_rate_limit == _CFG.COMMAND_RATE_LIMIT


# ---------------------------------------------------------------------------
# Simple one-shot GET endpoints
# ---------------------------------------------------------------------------
class TestSimpleEndpoints:
    def test_start_sweep_mode_ok(self, patched):
        fake = patched["install"](FakeResponse(200))
        t = make_tracker()
        assert t.start_sweep_mode() is True
        assert fake.urls == ["http://10.0.0.70/sweep"]
        assert fake.calls[0]["timeout"] == t.timeout

    def test_start_sweep_mode_non_200(self, patched):
        patched["install"](FakeResponse(503))
        assert make_tracker().start_sweep_mode() is False

    def test_start_sweep_mode_exception(self, patched):
        patched["install"](raise_exc=ConnectionError("boom"))
        assert make_tracker().start_sweep_mode() is False

    def test_stop_sweep_mode_ok(self, patched):
        fake = patched["install"](FakeResponse(200))
        assert make_tracker().stop_sweep_mode() is True
        assert fake.urls == ["http://10.0.0.70/stop"]

    def test_stop_sweep_mode_non_200(self, patched):
        patched["install"](FakeResponse(500))
        assert make_tracker().stop_sweep_mode() is False

    def test_test_connection_ok(self, patched):
        fake = patched["install"](FakeResponse(200))
        assert make_tracker().test_connection() is True
        assert fake.urls == ["http://10.0.0.70/status"]

    def test_test_connection_non_200_returns_false(self, patched):
        patched["install"](FakeResponse(404))
        assert make_tracker().test_connection() is False

    def test_test_connection_exception(self, patched):
        patched["install"](raise_exc=TimeoutError("no route"))
        assert make_tracker().test_connection() is False


# ---------------------------------------------------------------------------
# Endpoints that flip tracking_active as a side effect
# ---------------------------------------------------------------------------
class TestStatefulEndpoints:
    def test_center_servos_ok_clears_tracking(self, patched):
        fake = patched["install"](FakeResponse(200))
        t = make_tracker()
        t.tracking_active = True
        assert t.center_servos() is True
        assert t.tracking_active is False
        assert fake.urls == ["http://10.0.0.70/center"]

    def test_center_servos_non_200_keeps_tracking(self, patched):
        patched["install"](FakeResponse(500))
        t = make_tracker()
        t.tracking_active = True
        assert t.center_servos() is False
        # On failure the flag is not touched.
        assert t.tracking_active is True

    def test_stop_movement_ok_clears_tracking(self, patched):
        fake = patched["install"](FakeResponse(200))
        t = make_tracker()
        t.tracking_active = True
        assert t.stop_movement() is True
        assert t.tracking_active is False
        assert fake.urls == ["http://10.0.0.70/stop"]

    def test_stop_movement_exception_returns_false(self, patched):
        patched["install"](raise_exc=ConnectionError("down"))
        t = make_tracker()
        t.tracking_active = True
        assert t.stop_movement() is False
        assert t.tracking_active is True


# ---------------------------------------------------------------------------
# fire_blaster
# ---------------------------------------------------------------------------
class TestFireBlaster:
    def test_default_duration(self, patched):
        fake = patched["install"](FakeResponse(200))
        assert make_tracker().fire_blaster() is True
        assert fake.urls == ["http://10.0.0.70/fire?duration=150"]

    def test_custom_duration(self, patched):
        fake = patched["install"](FakeResponse(200))
        assert make_tracker().fire_blaster(500) is True
        assert fake.urls == ["http://10.0.0.70/fire?duration=500"]

    def test_non_200(self, patched):
        patched["install"](FakeResponse(500))
        assert make_tracker().fire_blaster() is False

    def test_exception(self, patched):
        patched["install"](raise_exc=ConnectionError("x"))
        assert make_tracker().fire_blaster() is False


# ---------------------------------------------------------------------------
# move_to_position
# ---------------------------------------------------------------------------
class TestMoveToPosition:
    def test_builds_url_and_updates_last_command_time(self, patched):
        fake = patched["install"](FakeResponse(200))
        clock = patched["clock"]
        t = make_tracker()
        assert t.move_to_position(120, 60) is True
        assert fake.urls == ["http://10.0.0.70/move?pan=120&tilt=60"]
        assert t.last_command_time == clock.now

    def test_non_200_returns_false(self, patched):
        patched["install"](FakeResponse(500))
        assert make_tracker().move_to_position(10, 20) is False

    def test_exception_returns_false(self, patched):
        patched["install"](raise_exc=ConnectionError("x"))
        assert make_tracker().move_to_position(10, 20) is False

    def test_rate_limit_sleeps_when_called_too_soon(self, patched):
        fake = patched["install"](FakeResponse(200))
        clock = patched["clock"]
        t = make_tracker()  # command_rate_limit == 0.1
        # Pretend the previous command happened "now": diff == 0 < 0.1 -> must sleep.
        t.last_command_time = clock.now
        assert t.move_to_position(90, 90) is True
        assert pytest.approx(t.command_rate_limit) == sum(clock.sleeps)
        assert fake.urls == ["http://10.0.0.70/move?pan=90&tilt=90"]

    def test_no_sleep_when_enough_time_passed(self, patched):
        patched["install"](FakeResponse(200))
        clock = patched["clock"]
        t = make_tracker()
        t.last_command_time = clock.now - 10.0  # well past the rate limit
        assert t.move_to_position(90, 90) is True
        assert clock.sleeps == []


# ---------------------------------------------------------------------------
# track_drone — pixel deviation -> servo move (the core logic)
# ---------------------------------------------------------------------------
class TestTrackDrone:
    # Geometry helper: with a 1000px frame and the default FOVs, the
    # per-pixel angle is 60/1000 = 0.06 deg/px (x) and 45/1000 = 0.045 (y).
    FW = FH = 1000

    def _tracker_ready_to_move(self, patched, response=None):
        """A tracker whose last command is far in the past so the rate
        limiter never blocks the move."""
        if response is not None:
            patched["install"](response)
        t = make_tracker()
        t.last_command_time = patched["clock"].now - 100.0
        return t

    def test_centered_target_sends_no_command(self, patched):
        fake = patched["requests"]
        t = self._tracker_ready_to_move(patched)
        # 50px -> 3.0deg x, 2.25deg y, both <= 5deg threshold.
        assert t.track_drone(50, 50, self.FW, self.FH) is True
        assert fake.calls == []
        assert t.tracking_active is False

    def test_moves_right_and_down_for_positive_deviation(self, patched):
        fake = patched["requests"]
        t = self._tracker_ready_to_move(patched)
        # 200px -> 12deg x, 9deg y. step = min(12, 9, 10) = 9.
        assert t.track_drone(200, 200, self.FW, self.FH) is True
        assert fake.urls == [
            "http://10.0.0.70/right?step=9",
            "http://10.0.0.70/down?step=9",
        ]
        assert t.tracking_active is True

    def test_moves_left_and_up_for_negative_deviation(self, patched):
        fake = patched["requests"]
        t = self._tracker_ready_to_move(patched)
        # -200px -> -12deg x, -9deg y. step = min(12, 9, 10) = 9.
        assert t.track_drone(-200, -200, self.FW, self.FH) is True
        assert fake.urls == [
            "http://10.0.0.70/left?step=9",
            "http://10.0.0.70/up?step=9",
        ]

    def test_step_size_capped_at_10(self, patched):
        fake = patched["requests"]
        t = self._tracker_ready_to_move(patched)
        # 1000px -> 60deg x, 45deg y. step = min(60, 45, 10) = 10.
        assert t.track_drone(1000, 1000, self.FW, self.FH) is True
        assert fake.urls == [
            "http://10.0.0.70/right?step=10",
            "http://10.0.0.70/down?step=10",
        ]

    def test_pan_only_when_tilt_below_threshold_uses_small_step(self, patched):
        # Quirk pinned: step_size = min(|ax|, |ay|, 10) uses |ay| even when the
        # tilt is below the move threshold. Here ax=12 (moves), ay=0, so the
        # pan command goes out with step=0 and no tilt command is sent.
        fake = patched["requests"]
        t = self._tracker_ready_to_move(patched)
        assert t.track_drone(200, 0, self.FW, self.FH) is True
        assert fake.urls == ["http://10.0.0.70/right?step=0"]

    def test_tilt_only_path_still_sends_a_pan_command(self, patched):
        # Quirk pinned: entering the movement branch via tilt alone (ax=0, ay=9)
        # still emits a pan command first (direction defaults to "left" since
        # 0 > 0 is False), then the tilt command, both with step=0.
        fake = patched["requests"]
        t = self._tracker_ready_to_move(patched)
        assert t.track_drone(0, 200, self.FW, self.FH) is True
        assert fake.urls == [
            "http://10.0.0.70/left?step=0",
            "http://10.0.0.70/down?step=0",
        ]

    def test_rate_limited_call_returns_false_without_request(self, patched):
        fake = patched["requests"]
        clock = patched["clock"]
        t = make_tracker()
        t.last_command_time = clock.now  # diff 0 < 0.1 rate limit
        assert t.track_drone(200, 200, self.FW, self.FH) is False
        assert fake.calls == []

    def test_pan_non_200_returns_false_and_skips_tilt(self, patched):
        fake = self._tracker_ready_helper(patched, FakeResponse(500))
        t = self._tracker_ready_to_move(patched)
        assert t.track_drone(200, 200, self.FW, self.FH) is False
        # Only the pan request was attempted; tilt is gated behind a 200.
        assert fake.urls == ["http://10.0.0.70/right?step=9"]

    def test_exception_returns_false(self, patched):
        patched["install"](raise_exc=ConnectionError("x"))
        t = make_tracker()
        t.last_command_time = patched["clock"].now - 100.0
        assert t.track_drone(200, 200, self.FW, self.FH) is False

    @staticmethod
    def _tracker_ready_helper(patched, response):
        # Reinstall the fake requests with a specific response, return the fake.
        return patched["install"](response)


# ---------------------------------------------------------------------------
# get_status
# ---------------------------------------------------------------------------
class TestGetStatus:
    def test_returns_json_on_200(self, patched):
        payload = {"pan": 100, "tilt": 90}
        patched["install"](FakeResponse(200, json_data=payload))
        assert make_tracker().get_status() == payload

    def test_returns_none_on_non_200(self, patched):
        patched["install"](FakeResponse(500))
        assert make_tracker().get_status() is None

    def test_returns_none_on_exception(self, patched):
        patched["install"](raise_exc=ConnectionError("x"))
        assert make_tracker().get_status() is None
