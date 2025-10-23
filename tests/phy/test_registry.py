import tempfile
from pathlib import Path

import pytest
import yaml

from spacelink.phy.performance import ErrorMetric, ModePerformanceThreshold
from spacelink.phy.registry import (
    DuplicateRegistryEntryError,
    NoRegistryFilesError,
    Registry,
)

# Test constants
STANDARD_CURVE_POINTS = [[0.0, 1e-1], [1.0, 1e-2]]
STANDARD_THRESHOLD = {"ebn0": 5.0, "error_rate": 1.0e-7}


# Fixtures
@pytest.fixture
def temp_dirs():
    """Create temporary directories for mode and performance data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        modes_dir = temp_path / "modes"
        perf_dir = temp_path / "perf"
        modes_dir.mkdir()
        perf_dir.mkdir()
        yield modes_dir, perf_dir


@pytest.fixture
def basic_mode():
    """Return basic mode data dictionary."""
    return {
        "id": "TEST_MODE",
        "modulation": {"name": "BPSK", "bits_per_symbol": 1},
        "coding": {"codes": []},
    }


def create_mode_yaml(
    modes_dir: Path, mode_data: list[dict], filename: str = "modes.yaml"
):
    """Create a mode YAML file."""
    with open(modes_dir / filename, "w") as f:
        yaml.dump(mode_data, f)


def create_curve_yaml(
    perf_dir: Path,
    mode_ids: list[str],
    metric: str = "bit error rate",
    points: list | None = None,
    filename: str = "perf.yaml",
):
    """Create a curve performance YAML file."""
    if points is None:
        points = STANDARD_CURVE_POINTS
    data = {"mode_ids": mode_ids, "metric": metric, "points": points}
    with open(perf_dir / filename, "w") as f:
        yaml.dump(data, f)


def create_threshold_yaml(
    perf_dir: Path,
    mode_ids: list[str],
    metric: str = "frame error rate",
    threshold: dict | None = None,
    filename: str = "threshold.yaml",
    ref: str = "",
):
    """Create a threshold performance YAML file."""
    if threshold is None:
        threshold = STANDARD_THRESHOLD
    data = {
        "mode_ids": mode_ids,
        "metric": metric,
        "threshold": threshold,
    }
    if ref:
        data["ref"] = ref
    with open(perf_dir / filename, "w") as f:
        yaml.dump(data, f)


class TestBasicLoading:
    def test_load_with_valid_data(self, temp_dirs, basic_mode):
        modes_dir, perf_dir = temp_dirs

        mode_data = [
            basic_mode,
            {
                "id": "TEST_QPSK_CODED",
                "modulation": {"name": "QPSK", "bits_per_symbol": 2},
                "coding": {"codes": [{"name": "Test Code", "rate": "1/2"}]},
            },
        ]
        create_mode_yaml(modes_dir, mode_data)
        create_curve_yaml(perf_dir, ["TEST_MODE"], points=STANDARD_CURVE_POINTS)

        registry = Registry()
        registry.load(modes_dir, perf_dir)

        assert len(registry.modes) == 2
        assert "TEST_MODE" in registry.modes
        assert "TEST_QPSK_CODED" in registry.modes

        assert len(registry.curve_index) == 1
        assert ("TEST_MODE", ErrorMetric.BER) in registry.curve_index

        perf = registry.curve_index[("TEST_MODE", ErrorMetric.BER)]
        assert perf.metric == ErrorMetric.BER

    def test_load_with_multiple_mode_files(self, temp_dirs):
        modes_dir, perf_dir = temp_dirs

        mode_data1 = [
            {
                "id": "MODE1",
                "modulation": {"name": "BPSK", "bits_per_symbol": 1},
                "coding": {"codes": []},
            }
        ]
        mode_data2 = [
            {
                "id": "MODE2",
                "modulation": {"name": "QPSK", "bits_per_symbol": 2},
                "coding": {"codes": []},
            }
        ]

        create_mode_yaml(modes_dir, mode_data1, "modes1.yaml")
        create_mode_yaml(modes_dir, mode_data2, "modes2.yaml")

        registry = Registry()
        registry.load(modes_dir, perf_dir=None)

        assert len(registry.modes) == 2
        assert "MODE1" in registry.modes
        assert "MODE2" in registry.modes

    def test_load_with_perf_dir_none(self, temp_dirs, basic_mode):
        modes_dir, perf_dir = temp_dirs

        create_mode_yaml(modes_dir, [basic_mode])

        registry = Registry()
        registry.load(modes_dir, perf_dir=None)

        assert len(registry.modes) == 1
        assert "TEST_MODE" in registry.modes
        assert len(registry.curves) == 0
        assert len(registry.thresholds) == 0
        assert len(registry.curve_index) == 0
        assert len(registry.threshold_index) == 0


class TestCurveLoading:
    def test_get_performance_curve_multiple_metrics(self, temp_dirs, basic_mode):
        modes_dir, perf_dir = temp_dirs

        create_mode_yaml(modes_dir, [basic_mode])
        create_curve_yaml(perf_dir, ["TEST_MODE"], "bit error rate", None, "ber.yaml")
        create_curve_yaml(perf_dir, ["TEST_MODE"], "frame error rate", None, "fer.yaml")

        registry = Registry()
        registry.load(modes_dir, perf_dir)

        ber_perf = registry.get_performance_curve("TEST_MODE", ErrorMetric.BER)
        fer_perf = registry.get_performance_curve("TEST_MODE", ErrorMetric.FER)

        assert ber_perf is not fer_perf
        assert ber_perf.metric == ErrorMetric.BER
        assert fer_perf.metric == ErrorMetric.FER

    def test_get_performance_curve_key_error(self):
        registry = Registry()
        with pytest.raises(KeyError):
            registry.get_performance_curve("NONEXISTENT_MODE", ErrorMetric.BER)


class TestThresholdLoading:
    def test_load_threshold_data(self, temp_dirs):
        modes_dir, perf_dir = temp_dirs

        mode_data = [
            {
                "id": "DVB_S2_QPSK_1_4",
                "modulation": {"name": "QPSK", "bits_per_symbol": 2},
                "coding": {"codes": [{"name": "LDPC", "rate": "1/4"}]},
            }
        ]
        create_mode_yaml(modes_dir, mode_data)
        create_threshold_yaml(perf_dir, ["DVB_S2_QPSK_1_4"], ref="ETSI EN 302 307-1")

        registry = Registry()
        registry.load(modes_dir, perf_dir)

        assert len(registry.thresholds) == 1
        assert len(registry.threshold_index) == 1

        threshold = registry.get_performance_threshold(
            "DVB_S2_QPSK_1_4", ErrorMetric.FER
        )
        assert isinstance(threshold, ModePerformanceThreshold)
        assert threshold.ebn0 == 5.0
        assert threshold.error_rate == 1.0e-7

    def test_get_performance_threshold_key_error(self):
        registry = Registry()
        with pytest.raises(KeyError):
            registry.get_performance_threshold("NONEXISTENT_MODE", ErrorMetric.FER)


class TestErrorHandling:
    def test_load_performance_with_missing_mode_id(self, temp_dirs, basic_mode):
        modes_dir, perf_dir = temp_dirs

        create_mode_yaml(modes_dir, [basic_mode])
        create_curve_yaml(perf_dir, ["NONEXISTENT_MODE"])

        registry = Registry()
        with pytest.raises(KeyError):
            registry.load(modes_dir, perf_dir)

    def test_load_with_duplicate_mode_ids(self, temp_dirs):
        modes_dir, perf_dir = temp_dirs

        duplicate_mode = {
            "id": "DUPLICATE_MODE",
            "modulation": {"name": "BPSK", "bits_per_symbol": 1},
            "coding": {"codes": []},
        }
        create_mode_yaml(modes_dir, [duplicate_mode], "modes1.yaml")
        create_mode_yaml(modes_dir, [duplicate_mode], "modes2.yaml")

        registry = Registry()
        with pytest.raises(DuplicateRegistryEntryError):
            registry.load(modes_dir, perf_dir)

    def test_load_with_duplicate_performance_entries(self, temp_dirs, basic_mode):
        modes_dir, perf_dir = temp_dirs

        create_mode_yaml(modes_dir, [basic_mode])
        create_curve_yaml(perf_dir, ["TEST_MODE"], filename="perf1.yaml")
        create_curve_yaml(perf_dir, ["TEST_MODE"], filename="perf2.yaml")

        registry = Registry()
        with pytest.raises(DuplicateRegistryEntryError):
            registry.load(modes_dir, perf_dir)

    def test_load_with_no_mode_files(self, temp_dirs):
        modes_dir, perf_dir = temp_dirs

        create_curve_yaml(perf_dir, ["TEST_MODE"])

        registry = Registry()
        with pytest.raises(NoRegistryFilesError):
            registry.load(modes_dir, perf_dir)

    def test_load_with_no_performance_files(self, temp_dirs, basic_mode):
        modes_dir, perf_dir = temp_dirs

        create_mode_yaml(modes_dir, [basic_mode])

        registry = Registry()
        with pytest.raises(NoRegistryFilesError):
            registry.load(modes_dir, perf_dir)


class TestYAMLValidation:
    def test_yaml_with_both_points_and_threshold(self, temp_dirs, basic_mode):
        modes_dir, perf_dir = temp_dirs

        create_mode_yaml(modes_dir, [basic_mode])

        invalid_data = {
            "mode_ids": ["TEST_MODE"],
            "metric": "bit error rate",
            "points": STANDARD_CURVE_POINTS,
            "threshold": STANDARD_THRESHOLD,
        }
        with open(perf_dir / "invalid.yaml", "w") as f:
            yaml.dump(invalid_data, f)

        registry = Registry()
        with pytest.raises(ValueError):
            registry.load(modes_dir, perf_dir)

    def test_yaml_with_neither_points_nor_threshold(self, temp_dirs, basic_mode):
        modes_dir, perf_dir = temp_dirs

        create_mode_yaml(modes_dir, [basic_mode])

        invalid_data = {
            "mode_ids": ["TEST_MODE"],
            "metric": "bit error rate",
        }
        with open(perf_dir / "invalid.yaml", "w") as f:
            yaml.dump(invalid_data, f)

        registry = Registry()
        with pytest.raises(ValueError):
            registry.load(modes_dir, perf_dir)
