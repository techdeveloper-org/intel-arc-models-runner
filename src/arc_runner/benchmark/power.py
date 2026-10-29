"""Package energy and average power measured around a workload from Sysman energy counters."""

from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class PowerReading:
    """Energy and power over one measured interval.

    Attributes:
        energy_j: Joules consumed by the package domain, or None if it could not be measured.
        avg_power_w: Average watts over the interval, or None if it could not be measured.
        duration_s: Counter-clock duration of the interval in seconds (0.0 when unavailable).
        reason: Why the values are None; None when they were measured.
    """

    energy_j: Optional[float]
    avg_power_w: Optional[float]
    duration_s: float
    reason: Optional[str]


class PowerMeter:
    """Reads the package energy counter before and after a workload.

    The package domain covers the whole SoC (CPU and iGPU), so the figures describe the machine under
    load, not the GPU alone.
    """

    def __init__(self, sysman: Any, device_index: int = 0) -> None:
        """Bind the meter to a Sysman instance.

        Args:
            sysman: LevelZeroSysman (or an object with read_package_energy and get_unavailable_reason),
                or None when Level Zero is not available.
            device_index: GPU device index whose package domain is read.
        """
        self._sysman = sysman
        self._device_index = device_index
        self._start: Optional[Tuple[Optional[Tuple[int, int]]]] = None
        self.reading: Optional[PowerReading] = None

    def _snapshot(self) -> Optional[Tuple[int, int]]:
        """Read one (energy_uj, timestamp_us) snapshot.

        Returns:
            Optional[Tuple[int, int]]: The snapshot, or None if no counter is readable.
        """
        if self._sysman is None:
            return None
        return self._sysman.read_package_energy(self._device_index)

    def start(self) -> None:
        """Take the first snapshot."""
        self.reading = None
        self._start = (self._snapshot(),)

    def stop(self) -> PowerReading:
        """Take the second snapshot and compute energy and average power.

        Returns:
            PowerReading: Measured values, or None values with the reason.

        Raises:
            RuntimeError: If start() was not called.
        """
        if self._start is None:
            raise RuntimeError("PowerMeter.stop() called before start()")
        first = self._start[0]
        second = self._snapshot()
        self._start = None
        self.reading = self._compute(first, second)
        return self.reading

    def _compute(self, first: Optional[Tuple[int, int]], second: Optional[Tuple[int, int]]) -> PowerReading:
        """Turn two snapshots into a reading.

        Args:
            first: Snapshot at start.
            second: Snapshot at stop.

        Returns:
            PowerReading: Values or an unavailable reading with a reason.
        """
        if first is None or second is None:
            reason = "no package energy counter is readable"
            if self._sysman is not None:
                reason = self._sysman.get_unavailable_reason("package_power") or reason
            return PowerReading(None, None, 0.0, reason)
        energy_delta = second[0] - first[0]
        time_delta_us = second[1] - first[1]
        if energy_delta < 0:
            return PowerReading(None, None, 0.0, "energy counter went backwards (wrap or reset)")
        if time_delta_us <= 0:
            return PowerReading(None, None, 0.0, "energy counter timestamps did not advance")
        return PowerReading(energy_delta / 1e6, energy_delta / time_delta_us, time_delta_us / 1e6, None)

    def __enter__(self) -> "PowerMeter":
        """Start measuring.

        Returns:
            PowerMeter: This meter; read .reading after the block.
        """
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        """Stop measuring."""
        self.stop()
