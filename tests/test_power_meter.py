"""PowerMeter: package energy and average power measured around a workload via Sysman counters."""

import time
from typing import List, Optional, Tuple

import pytest

from arc_runner.benchmark.power import PowerMeter
from arc_runner.devices.sysman import LevelZeroSysman
from tests.test_level_zero_bridge import fake_env, fake_loader, requires_gcc  # noqa: F401


class _ScriptedSysman:
    """Sysman stand-in returning scripted (energy_uj, timestamp_us) snapshots."""

    def __init__(self, snapshots: List[Optional[Tuple[int, int]]], reason: str = "no counter") -> None:
        self._snapshots = list(snapshots)
        self._reason = reason

    def read_package_energy(self, device_index: int = 0) -> Optional[Tuple[int, int]]:
        return self._snapshots.pop(0)

    def get_unavailable_reason(self, metric: str) -> Optional[str]:
        return self._reason


def test_stop_reports_energy_and_average_power_from_counter_delta() -> None:
    """10 J (1e7 uJ) over 2 s (2e6 us) is 5 W; energy is delta microjoules / 1e6."""
    meter = PowerMeter(_ScriptedSysman([(1_000_000, 5_000_000), (11_000_000, 7_000_000)]))
    meter.start()
    reading = meter.stop()
    assert reading.energy_j == pytest.approx(10.0)
    assert reading.avg_power_w == pytest.approx(5.0)
    assert reading.reason is None


def test_unavailable_counter_gives_none_values_and_the_sysman_reason() -> None:
    """No counter: nothing is invented, the reason is carried."""
    meter = PowerMeter(_ScriptedSysman([None, None], reason="zesInit failed with result 0x1"))
    meter.start()
    reading = meter.stop()
    assert reading.energy_j is None and reading.avg_power_w is None
    assert "zesInit" in (reading.reason or "")


def test_counter_going_backwards_gives_none_with_reason() -> None:
    """A wrapped counter must not become a negative or huge energy."""
    meter = PowerMeter(_ScriptedSysman([(500, 1_000), (100, 2_000)]))
    meter.start()
    reading = meter.stop()
    assert reading.energy_j is None and "backwards" in (reading.reason or "")


def test_counter_one_microjoule_lower_is_still_backwards() -> None:
    """Boundary: any decrease, even 1 uJ, invalidates the interval."""
    meter = PowerMeter(_ScriptedSysman([(500, 1_000), (499, 2_000)]))
    meter.start()
    assert meter.stop().energy_j is None


def test_non_advancing_timestamp_gives_none_with_reason() -> None:
    """Zero elapsed counter time cannot define a power."""
    meter = PowerMeter(_ScriptedSysman([(500, 1_000), (900, 1_000)]))
    meter.start()
    reading = meter.stop()
    assert reading.avg_power_w is None and "timestamp" in (reading.reason or "")


def test_stop_without_start_raises() -> None:
    """Misuse is an error, not a silent None."""
    with pytest.raises(RuntimeError):
        PowerMeter(_ScriptedSysman([])).stop()


def test_context_manager_exposes_the_reading_after_exit() -> None:
    """with PowerMeter(...) as meter: ...; meter.reading is set on exit."""
    with PowerMeter(_ScriptedSysman([(0, 0), (4_000_000, 1_000_000)])) as meter:
        pass
    assert meter.reading is not None and meter.reading.avg_power_w == pytest.approx(4.0)


def test_meter_without_sysman_reports_unavailable() -> None:
    """Passing None (no Level Zero) yields an unavailable reading, never a number."""
    meter = PowerMeter(None)
    meter.start()
    reading = meter.stop()
    assert reading.energy_j is None and reading.reason


@requires_gcc
def test_meter_with_fake_loader_measures_configured_watts(fake_env: pytest.MonkeyPatch) -> None:  # noqa: F811
    """Integration: real Sysman against the fake loader whose energy counter follows FAKE_ZES_WATTS."""
    fake_env.setenv("FAKE_ZES_WATTS", "15")
    meter = PowerMeter(LevelZeroSysman())
    meter.start()
    time.sleep(0.1)
    reading = meter.stop()
    assert reading.avg_power_w == pytest.approx(15.0, rel=0.02)
    assert reading.energy_j == pytest.approx(15.0 * 0.1, rel=0.3)
