"""Red-first tests for WP-02 measurement support items 02a, 02e, 02f.

02a: schema v1 power_mode and external_artifacts fields + capture_environment()
02e: model-directory safety guard before benchmark/engine load
02f: probe_hardware.py --out <file> option
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_SCRIPT = REPO_ROOT / "scripts" / "probe_hardware.py"
SCRIPTS_DIR = REPO_ROOT / "scripts"


# ---------------------------------------------------------------------------
# 02f: probe_hardware.py --out <file>
# ---------------------------------------------------------------------------

class TestProbeHardwareOutFlag:
    """probe_hardware.py must write its JSON to a file when --out is given."""

    def test_probe_hardware_out_flag_writes_json_file(self, tmp_path):
        """probe_hardware.py --out <path> creates the file and it is valid JSON."""
        out_path = tmp_path / "probe.json"
        result = subprocess.run(
            [sys.executable, str(PROBE_SCRIPT), "--out", str(out_path)],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, f"probe_hardware.py exited non-zero: {result.stderr}"
        assert out_path.exists(), "--out file was not created"
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data.get("schema") == "probe_hardware.v1"

    def test_probe_hardware_out_flag_still_prints_to_stdout(self, tmp_path):
        """When --out is given, JSON is also printed to stdout (no regression)."""
        out_path = tmp_path / "probe.json"
        result = subprocess.run(
            [sys.executable, str(PROBE_SCRIPT), "--out", str(out_path)],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data.get("schema") == "probe_hardware.v1"

    def test_probe_hardware_no_out_flag_still_works(self):
        """Original behaviour (no --out) remains intact."""
        result = subprocess.run(
            [sys.executable, str(PROBE_SCRIPT)],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data.get("schema") == "probe_hardware.v1"

    def test_probe_hardware_out_creates_parent_dirs(self, tmp_path):
        """--out creates intermediate directories if they do not exist."""
        out_path = tmp_path / "sub" / "deep" / "probe.json"
        result = subprocess.run(
            [sys.executable, str(PROBE_SCRIPT), "--out", str(out_path)],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0
        assert out_path.exists()


# ---------------------------------------------------------------------------
# 02a: schema v1 — power_mode and external_artifacts
# ---------------------------------------------------------------------------

class TestCaptureEnvironmentPowerMode:
    """capture_environment() must include a power_mode field."""

    def test_capture_environment_includes_power_mode_key(self):
        """capture_environment() returns a dict that includes power_mode."""
        from arc_runner.benchmark.results import capture_environment

        env = capture_environment(REPO_ROOT)
        assert "power_mode" in env, "power_mode missing from capture_environment output"

    def test_capture_environment_power_mode_is_string_or_none(self):
        """power_mode is either a string ('ac'/'battery'/'unknown') or None."""
        from arc_runner.benchmark.results import capture_environment

        env = capture_environment(REPO_ROOT)
        value = env["power_mode"]
        assert value is None or isinstance(value, str)

    def test_capture_environment_accepts_explicit_power_mode(self):
        """capture_environment() forwards an explicit power_mode argument."""
        from arc_runner.benchmark.results import capture_environment

        env = capture_environment(REPO_ROOT, power_mode="ac")
        assert env["power_mode"] == "ac"


class TestCaptureEnvironmentExternalArtifacts:
    """capture_environment() must include an external_artifacts field."""

    def test_capture_environment_includes_external_artifacts_key(self):
        """capture_environment() returns a dict with external_artifacts."""
        from arc_runner.benchmark.results import capture_environment

        env = capture_environment(REPO_ROOT)
        assert "external_artifacts" in env

    def test_capture_environment_external_artifacts_is_list(self):
        """external_artifacts is an empty list by default."""
        from arc_runner.benchmark.results import capture_environment

        env = capture_environment(REPO_ROOT)
        assert isinstance(env["external_artifacts"], list)

    def test_capture_environment_accepts_external_artifacts_list(self):
        """capture_environment() stores a supplied external_artifacts list."""
        from arc_runner.benchmark.results import capture_environment

        artifact = {"path": "/models/model.xml", "sha256": "abc123", "tool": "optimum-intel", "version": "1.2.3"}
        env = capture_environment(REPO_ROOT, external_artifacts=[artifact])
        assert env["external_artifacts"] == [artifact]


class TestSchemaV1NewFields:
    """Schema JSON file must declare power_mode and external_artifacts as optional."""

    def _load_schema(self) -> dict:
        schema_path = REPO_ROOT / "docs" / "benchmarks" / "schema" / "bench_result.v1.json"
        return json.loads(schema_path.read_text(encoding="utf-8"))

    def test_schema_environment_allows_power_mode(self):
        """bench_result.v1.json environment.properties contains power_mode."""
        schema = self._load_schema()
        env_props = schema["properties"]["environment"]["properties"]
        assert "power_mode" in env_props, "power_mode not in schema environment.properties"

    def test_schema_environment_allows_external_artifacts(self):
        """bench_result.v1.json environment.properties contains external_artifacts."""
        schema = self._load_schema()
        env_props = schema["properties"]["environment"]["properties"]
        assert "external_artifacts" in env_props, "external_artifacts not in schema environment.properties"

    def test_schema_external_artifacts_items_have_required_keys(self):
        """external_artifacts items schema requires path, sha256, tool, version."""
        schema = self._load_schema()
        env_props = schema["properties"]["environment"]["properties"]
        items = env_props["external_artifacts"].get("items", {})
        required = set(items.get("required", []))
        assert required >= {"path", "sha256", "tool", "version"}


# ---------------------------------------------------------------------------
# 02e: model-directory safety guard
# ---------------------------------------------------------------------------

class TestModelDirGuard:
    """validate_model_dir() must raise before any engine or benchmark load."""

    def test_validate_model_dir_raises_when_path_does_not_exist(self, tmp_path):
        """Raises ModelNotFoundError when the directory does not exist."""
        from arc_runner.core.exceptions import ModelNotFoundError
        from arc_runner.benchmark.results import validate_model_dir

        missing = tmp_path / "no_such_model"
        with pytest.raises(ModelNotFoundError):
            validate_model_dir(missing)

    def test_validate_model_dir_raises_when_path_is_file(self, tmp_path):
        """Raises ModelNotFoundError when the path is a file, not a directory."""
        from arc_runner.core.exceptions import ModelNotFoundError
        from arc_runner.benchmark.results import validate_model_dir

        file_path = tmp_path / "weights.bin"
        file_path.touch()
        with pytest.raises(ModelNotFoundError):
            validate_model_dir(file_path)

    def test_validate_model_dir_raises_when_no_ir_files(self, tmp_path):
        """Raises ModelNotFoundError when directory has no .xml or .bin files."""
        from arc_runner.core.exceptions import ModelNotFoundError
        from arc_runner.benchmark.results import validate_model_dir

        model_dir = tmp_path / "empty_model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ModelNotFoundError):
            validate_model_dir(model_dir)

    def test_validate_model_dir_passes_when_xml_and_bin_exist(self, tmp_path):
        """No exception when directory contains both .xml and .bin files."""
        from arc_runner.benchmark.results import validate_model_dir

        model_dir = tmp_path / "valid_model"
        model_dir.mkdir()
        (model_dir / "openvino_model.xml").touch()
        (model_dir / "openvino_model.bin").touch()
        validate_model_dir(model_dir)

    def test_validate_model_dir_passes_when_only_xml_exists(self, tmp_path):
        """No exception when directory has at least an .xml file (bin may be embedded)."""
        from arc_runner.benchmark.results import validate_model_dir

        model_dir = tmp_path / "xml_only_model"
        model_dir.mkdir()
        (model_dir / "openvino_model.xml").touch()
        validate_model_dir(model_dir)

    def test_validate_model_dir_accepts_string_path(self, tmp_path):
        """validate_model_dir() accepts a plain string, not only a Path."""
        from arc_runner.benchmark.results import validate_model_dir

        model_dir = tmp_path / "str_model"
        model_dir.mkdir()
        (model_dir / "model.xml").touch()
        validate_model_dir(str(model_dir))
