"""Acceptance tests for WP-07 ThermalMonitor and ThrottlePolicy (CC-1..CC-5).

Verifies daemon thread lifecycle, hysteresis threshold evaluations,
power ceiling protections, environment disabling, and /health telemetry.
"""

from pathlib import Path
import unittest.mock as mock
from fastapi.testclient import TestClient
import pytest

from arc_runner.core.constants import (
    MAX_CONCURRENT_SEQUENCES,
    T_CRITICAL_C,
    T_WARN_C,
    THERMAL_MONITOR_THREAD_NAME,
)
from arc_runner.devices.sysman import LevelZeroSysman, PowerState, ThermalReading
from arc_runner.engine.cb_scheduler import ContinuousBatchingScheduler
from arc_runner.engine.thermal_monitor import (
    ThermalMonitor,
    ThrottlePolicy,
    ThrottleState,
)
from arc_runner.server.app import create_app


def test_thermal_monitor_daemon_lifecycle_cc1() -> None:
    """CC-1: ThermalMonitor starts daemon thread named arc-thermal-monitor and stops within 2s."""
    mock_sysman = mock.MagicMock(spec=LevelZeroSysman)
    mock_sysman.get_temperature.return_value = ThermalReading(temp_c=55.0, valid=True)
    mock_sysman.get_power_state.return_value = PowerState(power_mw=30000.0, pl1_limit_mw=75000.0, valid=True)

    monitor = ThermalMonitor(sysman=mock_sysman, poll_interval_s=0.05)
    monitor.start()

    assert monitor._thread is not None
    assert monitor._thread.is_alive()
    assert monitor._thread.name == THERMAL_MONITOR_THREAD_NAME
    assert monitor._thread.daemon is True

    stopped = monitor.stop(timeout_s=2.0)
    assert stopped is True
    assert not monitor._thread.is_alive()


def test_throttle_policy_warn_threshold_cc2() -> None:
    """CC-2: At T >= T_WARN_C, ThrottlePolicy transitions to WARN and caps concurrency to 4."""
    scheduler = ContinuousBatchingScheduler()
    assert scheduler.concurrency_cap == MAX_CONCURRENT_SEQUENCES

    policy = ThrottlePolicy(t_warn=T_WARN_C, t_critical=T_CRITICAL_C)
    state = policy.evaluate(
        thermal=ThermalReading(temp_c=T_WARN_C + 1.0, valid=True),
        power=PowerState(power_mw=40000.0, pl1_limit_mw=75000.0, valid=True),
        scheduler=scheduler,
    )

    assert state == ThrottleState.WARN
    assert scheduler.concurrency_cap == 4


def test_throttle_policy_critical_threshold_cc3() -> None:
    """CC-3: At T >= T_CRITICAL_C, ThrottlePolicy transitions to CRITICAL and caps concurrency to 1."""
    scheduler = ContinuousBatchingScheduler()
    policy = ThrottlePolicy(t_warn=T_WARN_C, t_critical=T_CRITICAL_C)

    state = policy.evaluate(
        thermal=ThermalReading(temp_c=T_CRITICAL_C + 2.0, valid=True),
        power=PowerState(power_mw=60000.0, pl1_limit_mw=75000.0, valid=True),
        scheduler=scheduler,
    )

    assert state == ThrottleState.CRITICAL
    assert scheduler.concurrency_cap == 1


def test_sysman_disabled_by_env_cc4(monkeypatch: pytest.MonkeyPatch) -> None:
    """CC-4: ARC_SYSMAN_DISABLE=1 returns ThermalReading(valid=False) with no cap modification."""
    monkeypatch.setenv("ARC_SYSMAN_DISABLE", "1")
    sysman = LevelZeroSysman()

    reading = sysman.get_temperature()
    assert reading.valid is False
    assert reading.temp_c is None

    p_state = sysman.get_power_state()
    assert p_state.valid is False

    scheduler = ContinuousBatchingScheduler()
    policy = ThrottlePolicy()
    state = policy.evaluate(reading, p_state, scheduler=scheduler)

    assert state == ThrottleState.NORMAL
    assert scheduler.concurrency_cap == MAX_CONCURRENT_SEQUENCES


def test_health_route_thermal_fields_cc5(tmp_path: Path) -> None:
    """CC-5: /health endpoint includes thermal.temp_c and thermal.state when monitor is active."""
    app = create_app()

    mock_sysman = mock.MagicMock(spec=LevelZeroSysman)
    mock_sysman.get_temperature.return_value = ThermalReading(temp_c=68.5, valid=True)
    mock_sysman.get_power_state.return_value = PowerState(power_mw=45000.0, pl1_limit_mw=75000.0, valid=True)

    monitor = ThermalMonitor(sysman=mock_sysman)
    monitor._last_thermal = ThermalReading(temp_c=68.5, valid=True)
    monitor._last_power = PowerState(power_mw=45000.0, pl1_limit_mw=75000.0, valid=True)
    app.state.thermal_monitor = monitor

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "thermal" in data
    assert "temp_c" in data["thermal"]
    assert data["thermal"]["temp_c"] == 68.5
    assert data["thermal"]["state"] in ("NORMAL", "WARN", "CRITICAL")
    assert data["thermal"]["valid"] is True
    assert "power" in data
