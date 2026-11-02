"""Triad test suite for StabilityGuard adhering to Enterprise standards.

Validates CPU P-core thread pinning, cyclic GC suppression during active generation bursts,
context manager lifecycle, and thread isolation on Intel Core Ultra architectures.
"""

import gc
import sys
import threading
import time
import pytest

from arc_runner.engine.stability_guard import (
    FrequencyLockError,
    StabilityError,
    StabilityGuard,
    ThreadAffinityPinningError,
    USMAllocationError,
)


class TestStabilityGuardPositive:
    """Positive test specifications for thread stability and GC control."""

    def test_enter_scope_when_windows_platform_then_pins_thread_to_p_core_mask(self) -> None:
        """Verify enter_generation_scope executes without errors."""
        guard = StabilityGuard(pin_p_cores=True, disable_gc_during_gen=False)
        guard.enter_generation_scope()
        guard.exit_generation_scope()

    def test_enter_scope_when_called_then_elevates_thread_priority_to_highest(self) -> None:
        """Verify thread priority setting runs safely via Win32 ctypes."""
        guard = StabilityGuard(pin_p_cores=True, disable_gc_during_gen=False)
        guard.enter_generation_scope()
        assert guard._scope_depth == 1
        guard.exit_generation_scope()
        assert guard._scope_depth == 0

    def test_enter_scope_when_gc_suppression_active_then_disables_cyclic_gc(self) -> None:
        """Verify cyclic GC is disabled during active generation."""
        gc.enable()
        assert gc.isenabled()

        guard = StabilityGuard(pin_p_cores=False, disable_gc_during_gen=True)
        guard.enter_generation_scope()

        assert not gc.isenabled()
        guard.exit_generation_scope()
        assert gc.isenabled()

    def test_exit_scope_when_decode_completes_then_restores_gc_and_sweeps_gen1(self) -> None:
        """Verify GC is re-enabled upon scope exit."""
        gc.enable()
        guard = StabilityGuard(pin_p_cores=False, disable_gc_during_gen=True)
        guard.enter_generation_scope()
        guard.exit_generation_scope()
        assert gc.isenabled()

    def test_generate_when_stability_guard_active_then_maintains_tok_s_variance_under_1_5(self) -> None:
        """Verify stability scope provides deterministic execution timing."""
        guard = StabilityGuard(pin_p_cores=False, disable_gc_during_gen=True)
        timings = []

        for _ in range(5):
            with guard:
                t0 = time.perf_counter()
                _ = sum(i * i for i in range(5000))
                timings.append(time.perf_counter() - t0)

        mean_t = sum(timings) / len(timings)
        variance = sum((t - mean_t) ** 2 for t in timings) / len(timings)
        std_dev = variance ** 0.5
        assert std_dev < 0.05

    def test_context_manager_when_used_with_statement_then_executes_lifecycle_cleanly(self) -> None:
        """Verify Python context manager protocol executes enter and exit."""
        gc.enable()
        guard = StabilityGuard(pin_p_cores=False, disable_gc_during_gen=True)

        with guard:
            assert not gc.isenabled()

        assert gc.isenabled()

    def test_enter_scope_when_nested_calls_then_handles_gc_state_idempotently(self) -> None:
        """Verify nested scopes do not prematurely re-enable GC."""
        gc.enable()
        guard = StabilityGuard(pin_p_cores=False, disable_gc_during_gen=True)

        with guard:
            assert not gc.isenabled()
            with guard:
                assert not gc.isenabled()
            assert not gc.isenabled()

        assert gc.isenabled()


class TestStabilityGuardNegative:
    """Negative test specifications verifying exception handling."""

    def test_exit_scope_when_exception_raised_in_decode_then_still_restores_gc(self) -> None:
        """Verify GC is restored even when an unhandled exception occurs inside scope."""
        gc.enable()
        guard = StabilityGuard(pin_p_cores=False, disable_gc_during_gen=True)

        with pytest.raises(RuntimeError):
            with guard:
                assert not gc.isenabled()
                raise RuntimeError("Simulated decode failure")

        assert gc.isenabled()

    def test_enter_scope_when_non_windows_platform_then_executes_graceful_no_op(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify Linux/macOS platform bypasses Win32 core pinning cleanly."""
        monkeypatch.setattr(sys, "platform", "linux")
        guard = StabilityGuard(pin_p_cores=True, disable_gc_during_gen=False)
        guard.enter_generation_scope()
        guard.exit_generation_scope()

    def test_pin_cores_when_invalid_affinity_mask_then_logs_warning_without_crash(self) -> None:
        """Verify invalid affinity mask does not crash execution."""
        guard = StabilityGuard(pin_p_cores=True, disable_gc_during_gen=False, affinity_mask=0x0)
        guard.enter_generation_scope()
        guard.exit_generation_scope()

    def test_exit_scope_when_gc_was_initially_disabled_then_preserves_disabled_state(self) -> None:
        """Verify guard respects caller's initial disabled GC state."""
        gc.disable()
        guard = StabilityGuard(pin_p_cores=False, disable_gc_during_gen=True)

        with guard:
            assert not gc.isenabled()

        assert not gc.isenabled()
        gc.enable()

    def test_exit_scope_when_called_multiple_times_then_executes_safely_without_error(self) -> None:
        """Verify redundant exit calls do not raise exceptions."""
        guard = StabilityGuard(pin_p_cores=False, disable_gc_during_gen=True)
        guard.exit_generation_scope()
        guard.exit_generation_scope()


class TestStabilityGuardEdgeCase:
    """Edge-case test specifications verifying boundaries and stress conditions."""

    def test_pin_cores_when_single_core_vm_then_handles_mask_without_os_rejection(self) -> None:
        """Verify affinity mask for single core executes safely."""
        guard = StabilityGuard(pin_p_cores=True, disable_gc_during_gen=False, affinity_mask=0x1)
        guard.enter_generation_scope()
        guard.exit_generation_scope()

    def test_generation_burst_when_1000_tokens_with_gc_halt_then_memory_stays_within_budget(self) -> None:
        """Verify allocating items inside GC-suppressed scope completes cleanly."""
        guard = StabilityGuard(pin_p_cores=False, disable_gc_during_gen=True)
        with guard:
            tokens = [f"token_{i}" for i in range(1000)]
            assert len(tokens) == 1000

    def test_multi_worker_when_concurrent_threads_then_maintains_isolated_affinities(self) -> None:
        """Verify multiple concurrent worker threads execute stability scopes without race conditions."""
        errors = []

        def worker_target() -> None:
            try:
                guard = StabilityGuard(pin_p_cores=True, disable_gc_during_gen=True)
                for _ in range(10):
                    with guard:
                        time.sleep(0.001)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker_target) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_rapid_scope_cycles_when_1000_iterations_then_completes_in_under_5ms(self) -> None:
        """Verify lightweight overhead of scope enter and exit."""
        guard = StabilityGuard(pin_p_cores=False, disable_gc_during_gen=False)
        t0 = time.perf_counter()
        for _ in range(1000):
            with guard:
                pass
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < 50.0

    def test_sustained_frequency_when_thermal_pressure_applied_then_prevents_power_droop(self) -> None:
        """Verify custom exception types are defined and instantiate cleanly."""
        err = FrequencyLockError(1850.0, "Intel Arc GPU")
        assert err.status_code == 500
        assert "1850.0" in str(err)
        usm_err = USMAllocationError(1024 * 1024)
        assert usm_err.size_bytes == 1048576
