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

Every assertion documents the module's behavior. ``track_drone`` now sizes each
axis's servo step to that axis's own angular error via the pure
``plan_servo_moves`` helper (also tested directly below), fixing a prior bug
where a shared ``min(|ax|, |ay|, 10)`` step zeroed-out the moving axis whenever
the other axis was centered.
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
# plan_servo_moves — pure per-axis move planning (no network)
# ---------------------------------------------------------------------------
class TestPlanServoMoves:
    def test_centered_within_threshold_returns_no_moves(self):
        assert esp32.plan_servo_moves(3, -4) == []

    def test_each_axis_sized_to_its_own_error(self):
        # Both above threshold; pan capped at 10, tilt sized to its own 9deg.
        assert esp32.plan_servo_moves(12, 9) == [("right", 10), ("down", 9)]

    def test_pan_only_when_tilt_centered(self):
        # The regression: a centered tilt must NOT shrink the pan step to 0,
        # and must NOT emit a tilt command.
        assert esp32.plan_servo_moves(12, 0) == [("right", 10)]

    def test_tilt_only_when_pan_centered(self):
        assert esp32.plan_servo_moves(0, 9) == [("down", 9)]

    def test_negative_errors_pick_left_and_up(self):
        assert esp32.plan_servo_moves(-8, -7) == [("left", 8), ("up", 7)]

    def test_steps_capped_at_max_step(self):
        assert esp32.plan_servo_moves(60, 45) == [("right", 10), ("down", 10)]

    def test_custom_threshold_and_max_step_are_honored(self):
        # threshold=20 suppresses the 12deg pan; max_step=5 caps the tilt.
        assert esp32.plan_servo_moves(12, 30, move_threshold=20, max_step=5) == [
            ("down", 5)
        ]


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
        # 200px -> 12deg x, 9deg y. Each axis is now sized to its own error:
        # pan step = min(12, 10) = 10, tilt step = min(9, 10) = 9.
        assert t.track_drone(200, 200, self.FW, self.FH) is True
        assert fake.urls == [
            "http://10.0.0.70/right?step=10",
            "http://10.0.0.70/down?step=9",
        ]
        assert t.tracking_active is True

    def test_moves_left_and_up_for_negative_deviation(self, patched):
        fake = patched["requests"]
        t = self._tracker_ready_to_move(patched)
        # -200px -> -12deg x, -9deg y. pan step = 10, tilt step = 9.
        assert t.track_drone(-200, -200, self.FW, self.FH) is True
        assert fake.urls == [
            "http://10.0.0.70/left?step=10",
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

    def test_pan_only_error_sends_one_proportional_pan_command(self, patched):
        # Regression: a purely-horizontal error (ax=12, ay=0) now sends a single
        # pan command sized to the *pan* error (step=10) — previously the shared
        # step_size = min(|ax|, |ay|, 10) collapsed to 0 because |ay|==0, so the
        # axis that actually needed to move was told to move 0 degrees.
        fake = patched["requests"]
        t = self._tracker_ready_to_move(patched)
        assert t.track_drone(200, 0, self.FW, self.FH) is True
        assert fake.urls == ["http://10.0.0.70/right?step=10"]
        assert t.tracking_active is True

    def test_tilt_only_error_sends_no_spurious_pan_command(self, patched):
        # Regression: a purely-vertical error (ax=0, ay=9) now sends only the
        # tilt command — previously a bogus pan command (defaulting to "left",
        # step=0) was emitted first.
        fake = patched["requests"]
        t = self._tracker_ready_to_move(patched)
        assert t.track_drone(0, 200, self.FW, self.FH) is True
        assert fake.urls == ["http://10.0.0.70/down?step=9"]

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
        assert fake.urls == ["http://10.0.0.70/right?step=10"]

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


# ---------------------------------------------------------------------------
# Failure-mode matrix completion
# ---------------------------------------------------------------------------
# Every ESP32DroneTracker endpoint is meant to degrade the same way: a non-200
# reply and a thrown request both yield a falsy result and never propagate an
# exception to the caller. Most arms of that matrix are pinned above, but three
# were unreached -- and two of them guard a real safety contract: when a *stop*
# or *center* command fails (the firmware rejects it, or the network is down),
# the tracker must NOT clear ``tracking_active``. Otherwise the controller would
# believe motion had halted while the hardware never confirmed it. These tests
# close those arms and assert that invariant explicitly.
class TestFailureModeMatrix:
    def test_stop_sweep_mode_exception_returns_false(self, patched):
        # stop_sweep_mode had ok/non-200 coverage but no thrown-request arm.
        patched["install"](raise_exc=ConnectionError("link down"))
        # The except branch must swallow the error and report failure, not raise.
        assert make_tracker().stop_sweep_mode() is False

    def test_center_servos_exception_keeps_tracking(self, patched):
        # center_servos had ok/non-200 coverage but no thrown-request arm.
        patched["install"](raise_exc=TimeoutError("no ack"))
        t = make_tracker()
        t.tracking_active = True
        assert t.center_servos() is False
        # Safety: an unconfirmed center must not flip the tracking flag off.
        assert t.tracking_active is True

    def test_stop_movement_non_200_keeps_tracking(self, patched):
        # stop_movement had ok/exception coverage but no non-200 arm.
        patched["install"](FakeResponse(503))
        t = make_tracker()
        t.tracking_active = True
        assert t.stop_movement() is False
        # Safety: a rejected stop (HTTP 503) must not flip the tracking flag off.
        assert t.tracking_active is True
