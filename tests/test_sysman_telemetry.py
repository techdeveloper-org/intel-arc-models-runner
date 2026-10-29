"""Sysman power, frequency and device-memory telemetry against a fake Level Zero Sysman loader."""

import pytest

from arc_runner.devices.sysman import LevelZeroSysman
from tests.test_level_zero_bridge import fake_env, fake_loader, requires_gcc  # noqa: F401

GIB = 1024**3


@pytest.fixture
def sysman(fake_env: pytest.MonkeyPatch) -> LevelZeroSysman:  # noqa: F811
    """Sysman bound to the fake loader with every Sysman control cleared."""
    for name in ("FAKE_ZES_WATTS", "FAKE_ZES_UNSUPPORTED", "FAKE_ZES_MEM_LOC", "FAKE_ZES_COUNTER_WRAP", "FAKE_ZES_FREQ", "FAKE_ZES_FAIL", "FAKE_ZES_NO_PACKAGE"):
        fake_env.delenv(name, raising=False)
    return LevelZeroSysman()


@requires_gcc
def test_query_package_power_derives_watts_from_energy_counter_delta(sysman: LevelZeroSysman, fake_env: pytest.MonkeyPatch) -> None:
    """Power is delta-energy over delta-timestamp of the package domain, not a constant."""
    fake_env.setenv("FAKE_ZES_WATTS", "12.5")
    watts = sysman.query_package_power(0, sample_interval_s=0.05)
    assert watts is not None
    assert watts == pytest.approx(12.5, rel=0.01)
    fake_env.setenv("FAKE_ZES_WATTS", "30")
    assert sysman.query_package_power(0, sample_interval_s=0.05) == pytest.approx(30.0, rel=0.01)


@requires_gcc
def test_query_package_power_ignores_subdevice_domains(sysman: LevelZeroSysman, fake_env: pytest.MonkeyPatch) -> None:
    """The fake lists a half-power subdevice domain first; only the package domain counts."""
    fake_env.setenv("FAKE_ZES_WATTS", "20")
    assert sysman.query_package_power(0, sample_interval_s=0.05) == pytest.approx(20.0, rel=0.01)


@requires_gcc
def test_query_package_power_without_package_domain_is_none_with_reason(sysman: LevelZeroSysman, fake_env: pytest.MonkeyPatch) -> None:
    """Only subdevice domains: no package reading is invented."""
    fake_env.setenv("FAKE_ZES_NO_PACKAGE", "1")
    assert sysman.query_package_power(0, sample_interval_s=0.01) is None
    assert "package" in (sysman.get_unavailable_reason("package_power") or "")


@requires_gcc
def test_query_package_power_unsupported_feature_reports_result_code(sysman: LevelZeroSysman, fake_env: pytest.MonkeyPatch) -> None:
    """A driver that does not expose power gives None and the hex result code in the reason."""
    fake_env.setenv("FAKE_ZES_UNSUPPORTED", "power")
    assert sysman.query_package_power(0, sample_interval_s=0.01) is None
    assert "0x78000003" in (sysman.get_unavailable_reason("package_power") or "")


@requires_gcc
def test_query_package_power_counter_going_backwards_is_none(sysman: LevelZeroSysman, fake_env: pytest.MonkeyPatch) -> None:
    """A wrapped or reset energy counter must not produce a negative or huge value."""
    fake_env.setenv("FAKE_ZES_COUNTER_WRAP", "1")
    assert sysman.query_package_power(0, sample_interval_s=0.01) is None
    assert "counter" in (sysman.get_unavailable_reason("package_power") or "")


@requires_gcc
def test_query_package_power_out_of_range_device_is_none(sysman: LevelZeroSysman) -> None:
    """Index validation."""
    assert sysman.query_package_power(5, sample_interval_s=0.01) is None


@requires_gcc
def test_query_gpu_frequency_reports_actual_mhz(sysman: LevelZeroSysman, fake_env: pytest.MonkeyPatch) -> None:
    """The resolved GPU-core frequency is returned as measured."""
    fake_env.setenv("FAKE_ZES_FREQ", "1150.5")
    assert sysman.query_gpu_frequency_mhz(0) == pytest.approx(1150.5)


@requires_gcc
def test_query_gpu_frequency_unknown_value_is_none_with_reason(sysman: LevelZeroSysman, fake_env: pytest.MonkeyPatch) -> None:
    """The driver reports unknown frequency as a negative number; that is not a reading."""
    fake_env.setenv("FAKE_ZES_FREQ", "-1")
    assert sysman.query_gpu_frequency_mhz(0) is None
    assert sysman.get_unavailable_reason("gpu_frequency")


@requires_gcc
def test_query_gpu_frequency_unsupported_reports_result_code(sysman: LevelZeroSysman, fake_env: pytest.MonkeyPatch) -> None:
    """No frequency domains: None and a reason."""
    fake_env.setenv("FAKE_ZES_UNSUPPORTED", "freq")
    assert sysman.query_gpu_frequency_mhz(0) is None
    assert "0x78000003" in (sysman.get_unavailable_reason("gpu_frequency") or "")


@requires_gcc
def test_query_gpu_memory_reports_free_from_device_memory_module(sysman: LevelZeroSysman) -> None:
    """With an on-board memory module, free memory comes from zesMemoryGetState."""
    total_mb, free_mb = sysman.query_gpu_memory(0)
    assert total_mb == 8 * 1024
    assert free_mb == 5 * 1024


@requires_gcc
def test_query_gpu_memory_system_memory_module_gives_no_free_value(sysman: LevelZeroSysman, fake_env: pytest.MonkeyPatch) -> None:
    """An integrated GPU's system-memory module is not GPU free memory."""
    fake_env.setenv("FAKE_ZES_MEM_LOC", "0")
    total_mb, free_mb = sysman.query_gpu_memory(0)
    assert total_mb == 8 * 1024
    assert free_mb is None
    assert "system" in (sysman.get_unavailable_reason("gpu_free_memory") or "")


@requires_gcc
def test_query_gpu_memory_sysman_unsupported_gives_no_free_value(sysman: LevelZeroSysman, fake_env: pytest.MonkeyPatch) -> None:
    """No memory modules exposed: total still reported, free None with the result code."""
    fake_env.setenv("FAKE_ZES_UNSUPPORTED", "mem")
    total_mb, free_mb = sysman.query_gpu_memory(0)
    assert total_mb == 8 * 1024 and free_mb is None
    assert "0x78000003" in (sysman.get_unavailable_reason("gpu_free_memory") or "")


@requires_gcc
def test_sysman_init_failure_disables_all_sysman_metrics_with_reason(sysman: LevelZeroSysman, fake_env: pytest.MonkeyPatch) -> None:
    """zesInit failing yields None for power and frequency with an explanation."""
    fake_env.setenv("FAKE_ZES_FAIL", "zesInit")
    assert sysman.query_package_power(0, sample_interval_s=0.01) is None
    assert sysman.query_gpu_frequency_mhz(0) is None
    assert "zesInit" in (sysman.get_unavailable_reason("package_power") or "")
