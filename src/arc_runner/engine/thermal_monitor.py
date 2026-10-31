"""Closed-loop thermal monitoring and dynamic concurrency throttling policy."""

from enum import Enum
import logging
import threading
from typing import Any, Dict, Optional

from arc_runner.core.constants import (
    MAX_CONCURRENT_SEQUENCES,
    POWER_THROTTLE_RATIO,
    T_CRITICAL_C,
    T_WARN_C,
    THERMAL_HYSTERESIS_C,
    THERMAL_MONITOR_THREAD_NAME,
    THERMAL_POLL_INTERVAL_S,
)
from arc_runner.devices.sysman import LevelZeroSysman, PowerState, ThermalReading

logger = logging.getLogger(__name__)


class ThrottleState(str, Enum):
    """Operational thermal throttling states."""

    NORMAL = "NORMAL"
    WARN = "WARN"
    CRITICAL = "CRITICAL"


class ThrottlePolicy:
    """Hysteresis-based thermal and power throttling policy."""

    def __init__(
        self,
        t_warn: float = T_WARN_C,
        t_critical: float = T_CRITICAL_C,
        hysteresis: float = THERMAL_HYSTERESIS_C,
        power_ratio: float = POWER_THROTTLE_RATIO,
        max_concurrency: int = MAX_CONCURRENT_SEQUENCES,
    ) -> None:
        """Initialize throttle policy with thermal and power thresholds.

        Args:
            t_warn: Warning temperature threshold in degrees Celsius.
            t_critical: Critical temperature threshold in degrees Celsius.
            hysteresis: Hysteresis margin in degrees Celsius.
            power_ratio: Fraction of PL1 limit that triggers warning cap.
            max_concurrency: Concurrency cap for normal operating state.
        """
        self._state: ThrottleState = ThrottleState.NORMAL
        self._t_warn: float = t_warn
        self._t_critical: float = t_critical
        self._hysteresis: float = hysteresis
        self._power_ratio: float = power_ratio
        self._max_concurrency: int = max_concurrency

    @property
    def state(self) -> ThrottleState:
        """Return the current throttle state."""
        return self._state

    def evaluate(
        self,
        thermal: ThermalReading,
        power: PowerState,
        scheduler: Optional[Any] = None,
    ) -> ThrottleState:
        """Evaluate hardware readings and update scheduler concurrency cap.

        Args:
            thermal: Hardware temperature reading.
            power: Hardware power and TDP limit state.
            scheduler: Optional ContinuousBatchingScheduler instance.

        Returns:
            ThrottleState: Evaluated throttle state.
        """
        if not thermal.valid or thermal.temp_c is None:
            return self._state

        temp = thermal.temp_c
        power_exceeded = False
        if power.valid and power.power_mw is not None and power.pl1_limit_mw is not None:
            if power.pl1_limit_mw > 0 and (power.power_mw >= self._power_ratio * power.pl1_limit_mw):
                power_exceeded = True

        old_state = self._state
        if self._state == ThrottleState.NORMAL:
            if temp >= self._t_critical:
                self._state = ThrottleState.CRITICAL
            elif temp >= self._t_warn or power_exceeded:
                self._state = ThrottleState.WARN
        elif self._state == ThrottleState.WARN:
            if temp >= self._t_critical:
                self._state = ThrottleState.CRITICAL
            elif temp < (self._t_warn - self._hysteresis) and not power_exceeded:
                self._state = ThrottleState.NORMAL
        elif self._state == ThrottleState.CRITICAL:
            if temp < (self._t_critical - self._hysteresis):
                if temp < (self._t_warn - self._hysteresis) and not power_exceeded:
                    self._state = ThrottleState.NORMAL
                else:
                    self._state = ThrottleState.WARN

        if self._state != old_state:
            logger.warning(
                "Thermal state transition: %s -> %s (temp=%.1f C, power_exceeded=%s)",
                old_state.value,
                self._state.value,
                temp,
                power_exceeded,
            )

        target_cap = self._max_concurrency
        if self._state == ThrottleState.WARN:
            target_cap = 4
        elif self._state == ThrottleState.CRITICAL:
            target_cap = 1

        if scheduler is not None and hasattr(scheduler, "set_concurrency_cap"):
            try:
                scheduler.set_concurrency_cap(max(1, target_cap))
            except Exception as exc:
                logger.error("Failed to propagate concurrency cap to scheduler: %s", exc)

        return self._state


class ThermalMonitor:
    """Daemon thread for continuous hardware thermal and power surveillance."""

    def __init__(
        self,
        scheduler: Optional[Any] = None,
        sysman: Optional[LevelZeroSysman] = None,
        policy: Optional[ThrottlePolicy] = None,
        poll_interval_s: float = THERMAL_POLL_INTERVAL_S,
    ) -> None:
        """Initialize the thermal monitor.

        Args:
            scheduler: ContinuousBatchingScheduler instance to govern.
            sysman: LevelZeroSysman instance for hardware telemetry.
            policy: ThrottlePolicy instance evaluating hysteresis rules.
            poll_interval_s: Interval between telemetry polls in seconds.
        """
        self._scheduler = scheduler
        self._sysman = sysman or LevelZeroSysman()
        self._policy = policy or ThrottlePolicy()
        self._poll_interval_s = poll_interval_s
        self._stopping = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_thermal: ThermalReading = ThermalReading()
        self._last_power: PowerState = PowerState()

    def start(self) -> None:
        """Start the background thermal monitor daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stopping.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name=THERMAL_MONITOR_THREAD_NAME,
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> bool:
        """Signal monitor to terminate and await thread join.

        Args:
            timeout_s: Maximum duration in seconds to wait for thread join.

        Returns:
            bool: True if thread stopped cleanly, False if join timed out.
        """
        self._stopping.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout_s)
            return not self._thread.is_alive()
        return True

    def _poll_loop(self) -> None:
        """Poll telemetry periodically and invoke policy evaluation."""
        while not self._stopping.is_set():
            try:
                self._last_thermal = self._sysman.get_temperature()
                self._last_power = self._sysman.get_power_state()
                self._policy.evaluate(self._last_thermal, self._last_power, self._scheduler)
            except Exception as exc:
                logger.error("Thermal polling error: %s", exc, exc_info=True)

            self._stopping.wait(timeout=self._poll_interval_s)

    def get_health_telemetry(self) -> Dict[str, Any]:
        """Format current thermal and power telemetry for health endpoint.

        Returns:
            Dict[str, Any]: Formatted telemetry dictionary.
        """
        temp_val = self._last_thermal.temp_c if self._last_thermal.valid else None
        p_val = self._last_power.power_mw if self._last_power.valid else None
        pl1_val = self._last_power.pl1_limit_mw if self._last_power.valid else None

        return {
            "thermal": {
                "temp_c": temp_val,
                "power_mw": p_val,
                "state": self._policy.state.value,
                "valid": self._last_thermal.valid,
            },
            "power": {
                "mw": p_val,
                "pl1_mw": pl1_val,
                "valid": self._last_power.valid,
            },
        }
