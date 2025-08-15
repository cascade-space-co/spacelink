from pathlib import Path
import pytest
import tempfile
import yaml

from spacelink.phy.registry import (
    Registry,
    DuplicateRegistryEntryError,
    NoRegistryFilesError,
)
from spacelink.phy.performance import ErrorMetric


class TestRegistry:

    def test_load_with_valid_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            modes_dir = temp_path / "modes"
            perf_dir = temp_path / "perf"
            modes_dir.mkdir()
            perf_dir.mkdir()

            # Create a test mode file
            mode_data = [
                {
                    "id": "TEST_BPSK_UNCODED",
                    "modulation": {"name": "BPSK", "bits_per_symbol": 1},
                    "coding": {"codes": []},
                    "ref": "Test Reference",
                },
                {
                    "id": "TEST_QPSK_CODED",
                    "modulation": {"name": "QPSK", "bits_per_symbol": 2},
                    "coding": {"codes": [{"name": "Test Code", "rate": "1/2"}]},
                },
            ]

            mode_file = modes_dir / "test_modes.yaml"
            with open(mode_file, "w") as f:
                yaml.dump(mode_data, f)

            # Create a test performance file
            perf_data = {
                "mode_ids": ["TEST_BPSK_UNCODED"],
                "metric": "bit error rate",
                "ref": "Test Performance Reference",
                "points": [[0.0, 1e-1], [1.0, 3e-2], [2.0, 5e-3], [3.0, 1e-4]],
            }

            perf_file = perf_dir / "test_performance.yaml"
            with open(perf_file, "w") as f:
                yaml.dump(perf_data, f)

            # Test loading
            registry = Registry()
            registry.load(modes_dir, perf_dir)

            # Test the Registry's core logic: loading and indexing
            assert len(registry.modes) == 2
            assert "TEST_BPSK_UNCODED" in registry.modes
            assert "TEST_QPSK_CODED" in registry.modes

            # Verify performance index was built correctly (the key Registry logic)
            assert len(registry.perf_index) == 1
            assert ("TEST_BPSK_UNCODED", ErrorMetric.BER) in registry.perf_index

            # Test that we can retrieve the performance data
            perf = registry.perf_index[("TEST_BPSK_UNCODED", ErrorMetric.BER)]
            assert perf.metric == ErrorMetric.BER

    def test_get_performance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            modes_dir = temp_path / "modes"
            perf_dir = temp_path / "perf"
            modes_dir.mkdir()
            perf_dir.mkdir()

            # Create test data
            mode_data = [
                {
                    "id": "TEST_MODE",
                    "modulation": {"name": "BPSK", "bits_per_symbol": 1},
                    "coding": {"codes": []},
                }
            ]

            mode_file = modes_dir / "test_modes.yaml"
            with open(mode_file, "w") as f:
                yaml.dump(mode_data, f)

            # Create performance data for multiple metrics
            ber_data = {
                "mode_ids": ["TEST_MODE"],
                "metric": "bit error rate",
                "points": [[0.0, 1e-1], [1.0, 1e-2]],
            }

            fer_data = {
                "mode_ids": ["TEST_MODE"],
                "metric": "frame error rate",
                "points": [[0.0, 2e-1], [1.0, 2e-2]],
            }

            ber_file = perf_dir / "test_ber.yaml"
            with open(ber_file, "w") as f:
                yaml.dump(ber_data, f)

            fer_file = perf_dir / "test_fer.yaml"
            with open(fer_file, "w") as f:
                yaml.dump(fer_data, f)

            # Load and test
            registry = Registry()
            registry.load(modes_dir, perf_dir)

            # Test get_performance retrieves correct objects
            ber_perf = registry.get_performance("TEST_MODE", ErrorMetric.BER)
            fer_perf = registry.get_performance("TEST_MODE", ErrorMetric.FER)

            # Verify they are different objects with correct metrics
            assert ber_perf is not fer_perf
            assert ber_perf.metric == ErrorMetric.BER
            assert fer_perf.metric == ErrorMetric.FER

    def test_get_performance_key_error(self):
        registry = Registry()
        with pytest.raises(KeyError):
            registry.get_performance("NONEXISTENT_MODE", ErrorMetric.BER)

    def test_load_with_multiple_mode_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            modes_dir = temp_path / "modes"
            perf_dir = temp_path / "perf"
            modes_dir.mkdir()
            perf_dir.mkdir()

            # Create multiple mode files
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

            with open(modes_dir / "modes1.yaml", "w") as f:
                yaml.dump(mode_data1, f)
            with open(modes_dir / "modes2.yaml", "w") as f:
                yaml.dump(mode_data2, f)

            registry = Registry()
            registry.load(modes_dir, perf_dir=None)

            # Test that Registry correctly loads from multiple files
            assert len(registry.modes) == 2
            assert "MODE1" in registry.modes
            assert "MODE2" in registry.modes

    def test_load_performance_with_missing_mode_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            modes_dir = temp_path / "modes"
            perf_dir = temp_path / "perf"
            modes_dir.mkdir()
            perf_dir.mkdir()

            # Create mode file with one mode
            mode_data = [
                {
                    "id": "EXISTING_MODE",
                    "modulation": {"name": "BPSK", "bits_per_symbol": 1},
                    "coding": {"codes": []},
                }
            ]
            with open(modes_dir / "modes.yaml", "w") as f:
                yaml.dump(mode_data, f)

            # Create performance file referencing non-existent mode
            perf_data = {
                "mode_ids": ["NONEXISTENT_MODE"],
                "metric": "bit error rate",
                "points": [[0.0, 1e-1]],
            }
            with open(perf_dir / "perf.yaml", "w") as f:
                yaml.dump(perf_data, f)

            registry = Registry()

            with pytest.raises(KeyError):
                registry.load(modes_dir, perf_dir)

    def test_load_with_duplicate_mode_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            modes_dir = temp_path / "modes"
            perf_dir = temp_path / "perf"
            modes_dir.mkdir()
            perf_dir.mkdir()

            # Create two mode files with the same mode ID
            mode_data1 = [
                {
                    "id": "DUPLICATE_MODE",
                    "modulation": {"name": "BPSK", "bits_per_symbol": 1},
                    "coding": {"codes": []},
                }
            ]
            mode_data2 = [
                {
                    "id": "DUPLICATE_MODE",  # Same ID as above
                    "modulation": {"name": "QPSK", "bits_per_symbol": 2},
                    "coding": {"codes": []},
                }
            ]

            with open(modes_dir / "modes1.yaml", "w") as f:
                yaml.dump(mode_data1, f)
            with open(modes_dir / "modes2.yaml", "w") as f:
                yaml.dump(mode_data2, f)

            registry = Registry()

            with pytest.raises(DuplicateRegistryEntryError):
                registry.load(modes_dir, perf_dir)

    def test_load_with_duplicate_performance_entries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            modes_dir = temp_path / "modes"
            perf_dir = temp_path / "perf"
            modes_dir.mkdir()
            perf_dir.mkdir()

            # Create a mode file
            mode_data = [
                {
                    "id": "TEST_MODE",
                    "modulation": {"name": "BPSK", "bits_per_symbol": 1},
                    "coding": {"codes": []},
                }
            ]
            with open(modes_dir / "modes.yaml", "w") as f:
                yaml.dump(mode_data, f)

            # Create two performance files with the same mode_id and metric
            perf_data1 = {
                "mode_ids": ["TEST_MODE"],
                "metric": "bit error rate",
                "points": [[0.0, 1e-1], [1.0, 1e-2]],
            }
            perf_data2 = {
                "mode_ids": ["TEST_MODE"],
                "metric": "bit error rate",  # Same mode and metric as above
                "points": [[0.0, 2e-1], [1.0, 2e-2]],
            }

            with open(perf_dir / "perf1.yaml", "w") as f:
                yaml.dump(perf_data1, f)
            with open(perf_dir / "perf2.yaml", "w") as f:
                yaml.dump(perf_data2, f)

            registry = Registry()

            with pytest.raises(DuplicateRegistryEntryError):
                registry.load(modes_dir, perf_dir)

    def test_load_with_no_mode_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            modes_dir = temp_path / "modes"
            perf_dir = temp_path / "perf"
            modes_dir.mkdir()
            perf_dir.mkdir()

            # Create only a performance file, no mode files
            perf_data = {
                "mode_ids": ["TEST_MODE"],
                "metric": "bit error rate",
                "points": [[0.0, 1e-1]],
            }
            with open(perf_dir / "perf.yaml", "w") as f:
                yaml.dump(perf_data, f)

            registry = Registry()

            with pytest.raises(NoRegistryFilesError):
                registry.load(modes_dir, perf_dir)

    def test_load_with_no_performance_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            modes_dir = temp_path / "modes"
            perf_dir = temp_path / "perf"
            modes_dir.mkdir()
            perf_dir.mkdir()

            # Create only a mode file, no performance files
            mode_data = [
                {
                    "id": "TEST_MODE",
                    "modulation": {"name": "BPSK", "bits_per_symbol": 1},
                    "coding": {"codes": []},
                }
            ]
            with open(modes_dir / "modes.yaml", "w") as f:
                yaml.dump(mode_data, f)

            registry = Registry()

            with pytest.raises(NoRegistryFilesError):
                registry.load(modes_dir, perf_dir)

    def test_load_with_perf_dir_none(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            modes_dir = temp_path / "modes"
            modes_dir.mkdir()

            # Create only a mode file
            mode_data = [
                {
                    "id": "TEST_MODE",
                    "modulation": {"name": "BPSK", "bits_per_symbol": 1},
                    "coding": {"codes": []},
                }
            ]
            with open(modes_dir / "modes.yaml", "w") as f:
                yaml.dump(mode_data, f)

            registry = Registry()
            registry.load(modes_dir, perf_dir=None)

            # Should load modes but not performance data
            assert len(registry.modes) == 1
            assert "TEST_MODE" in registry.modes
            assert len(registry.perfs) == 0
            assert len(registry.perf_index) == 0
