"""Failure-mode coverage for the ESP32 turret *control* commands.

``esp32_drone_integration.ESP32DroneTracker`` exposes three "make the turret
stop / re-home" commands — ``stop_sweep_mode``, ``center_servos`` and
``stop_movement``. Each wraps a single ``requests.get`` in the same
three-way shape: HTTP 200 -> ``True``, any other status -> ``False``, and any
transport exception (connection refused, timeout, ...) -> ``False``. The two
state-changing commands (``center_servos`` / ``stop_movement``) additionally
clear ``tracking_active`` **only on success**.

The sibling suite (``test_esp32_drone_integration.py``) already pins the happy
path and *one* failure mode per command, but the failure matrix it leaves is
asymmetric, so three real branches in the production module are never executed:

* ``stop_sweep_mode`` — the ``except`` (transport-error) branch (module L90-92).
* ``center_servos``   — the ``except`` (transport-error) branch (module L183-185).
* ``stop_movement``   — the non-200 (HTTP-error) branch (module L211-212).

Those gaps matter because these are *safety* commands: when "stop"/"center"
fails — exactly when the network to the turret is flaky — the method must
report ``False`` **and must not** silently flip ``tracking_active`` to ``False``
(which would make the rest of the system believe the turret had been safely
parked when it had not). This suite closes the three uncovered branches and
pins that invariant uniformly across *both* failure modes for the two
state-changing commands.

The suite is hermetic — the ``config`` module is stubbed before import and the
``requests``/``time`` boundaries are replaced, so there is no network, no
ESP32, and no real sleeping (matching the sibling suite's conventions).
"""

import sys
import types
import importlib

import pytest


# ---------------------------------------------------------------------------
# Stub ``config`` before importing the module under test (mirrors the sibling
# suite so the tests pin ESP32DroneTracker in isolation, not whatever ``.env``
# happens to be present).
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


_install_config_stub()
esp32 = importlib.import_module("esp32_drone_integration")


# ---------------------------------------------------------------------------
# Minimal request/clock doubles (kept local so this file is self-contained).
# ---------------------------------------------------------------------------
class FakeResponse:
    def __init__(self, status_code=200):
        self.status_code = status_code


class FakeRequests:
    """Records every GET and either raises or replays a fixed response."""

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
    def __init__(self, now=1000.0):
        self.now = now

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def install(monkeypatch):
    """Return an installer that swaps in a fake ``requests`` for the module."""
    monkeypatch.setattr(esp32, "time", FakeClock())

    def _install(response=None, raise_exc=None):
        fake = FakeRequests(response=response, raise_exc=raise_exc)
        monkeypatch.setattr(esp32, "requests", fake)
        return fake

    return _install


def make_tracker(**kwargs):
    return esp32.ESP32DroneTracker(**kwargs)


# A representative spread of "not 200" statuses and transport exceptions.
NON_200 = [500, 503, 404, 400, 502]
TRANSPORT_ERRORS = [
    ConnectionError("connection refused"),
    TimeoutError("read timed out"),
    OSError("network is unreachable"),
    Exception("unexpected"),
]


# ===========================================================================
# stop_sweep_mode — closes the uncovered ``except`` branch (module L90-92)
# ===========================================================================
class TestStopSweepModeFailureModes:
    @pytest.mark.parametrize("exc", TRANSPORT_ERRORS)
    def test_transport_error_is_swallowed_and_returns_false(self, install, exc):
        """A transport error does not escape; the command reports ``False``."""
        fake = install(raise_exc=exc)
        # Must not raise — the method owns its error handling.
        assert make_tracker().stop_sweep_mode() is False
        # The request was still attempted against the stop endpoint.
        assert fake.urls == ["http://10.0.0.70/stop"]

    @pytest.mark.parametrize("status", NON_200)
    def test_non_200_returns_false(self, install, status):
        assert make_tracker().stop_sweep_mode() is False


# ===========================================================================
# center_servos / stop_movement — state-changing safety commands.
# Closes center_servos' ``except`` branch (L183-185) and stop_movement's
# non-200 branch (L211-212), and pins the "failure must not clear
# tracking_active" invariant across BOTH failure modes for each.
# ===========================================================================
class TestStatefulCommandFailuresPreserveTracking:
    @pytest.mark.parametrize("status", NON_200)
    def test_center_servos_non_200_keeps_tracking_active(self, install, status):
        install(FakeResponse(status))
        t = make_tracker()
        t.tracking_active = True
        assert t.center_servos() is False
        # A failed re-home must NOT report the turret as parked.
        assert t.tracking_active is True

    @pytest.mark.parametrize("exc", TRANSPORT_ERRORS)
    def test_center_servos_transport_error_keeps_tracking_active(self, install, exc):
        install(raise_exc=exc)
        t = make_tracker()
        t.tracking_active = True
        assert t.center_servos() is False
        assert t.tracking_active is True

    @pytest.mark.parametrize("status", NON_200)
    def test_stop_movement_non_200_keeps_tracking_active(self, install, status):
        install(FakeResponse(status))
        t = make_tracker()
        t.tracking_active = True
        assert t.stop_movement() is False
        # A failed "stop" must NOT report the turret as stopped.
        assert t.tracking_active is True

    @pytest.mark.parametrize("exc", TRANSPORT_ERRORS)
    def test_stop_movement_transport_error_keeps_tracking_active(self, install, exc):
        install(raise_exc=exc)
        t = make_tracker()
        t.tracking_active = True
        assert t.stop_movement() is False
        assert t.tracking_active is True

    def test_already_idle_tracker_stays_idle_on_failure(self, install):
        """Failure never *sets* tracking_active either — it is left untouched."""
        install(FakeResponse(500))
        t = make_tracker()
        t.tracking_active = False
        assert t.center_servos() is False
        assert t.stop_movement() is False
        assert t.tracking_active is False

    def test_failure_paths_still_hit_the_correct_endpoints(self, install):
        """Even on the error branch the command targets the right ESP32 route."""
        fake = install(FakeResponse(503))
        t = make_tracker()
        t.center_servos()
        t.stop_movement()
        assert fake.urls == ["http://10.0.0.70/center", "http://10.0.0.70/stop"]
        assert all(c["timeout"] == t.timeout for c in fake.calls)
