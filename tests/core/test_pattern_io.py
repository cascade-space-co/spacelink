"""Tests for pattern I/O functionality."""

import io
import pathlib
import tempfile
import pytest
import numpy as np
import astropy.units as u

from spacelink.core.antenna import RadiationPattern
from spacelink.core import pattern_io


def test_radiation_pattern_npz_roundtrip():
    """Test saving and loading RadiationPattern with NPZ format."""
    # Create a simple test pattern
    theta = np.linspace(0, np.pi, 10) * u.rad
    phi = np.linspace(0, 2 * np.pi, 15, endpoint=False) * u.rad
    e_theta = (0.8 + 0.3j) * np.ones((10, 15)) * u.dimensionless
    e_phi = (0.2 - 0.5j) * np.ones((10, 15)) * u.dimensionless
    rad_efficiency = 0.75 * u.dimensionless

    original = RadiationPattern(theta, phi, e_theta, e_phi, rad_efficiency)

    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = pathlib.Path(tmpdir) / "test_pattern.npz"
        pattern_io.save_radiation_pattern_npz(original, npz_path)
        assert npz_path.exists()
        loaded = pattern_io.load_radiation_pattern_npz(npz_path)

        np.testing.assert_array_equal(loaded.theta, original.theta)
        np.testing.assert_array_equal(loaded.phi, original.phi)
        np.testing.assert_array_equal(loaded.e_theta, original.e_theta)
        np.testing.assert_array_equal(loaded.e_phi, original.e_phi)
        np.testing.assert_array_equal(loaded.rad_efficiency, original.rad_efficiency)


def test_radiation_pattern_from_npz_missing_file():
    nonexistent_path = pathlib.Path("/nonexistent/path/missing.npz")
    with pytest.raises(FileNotFoundError):
        pattern_io.load_radiation_pattern_npz(nonexistent_path)


def test_radiation_pattern_from_npz_missing_keys():
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = pathlib.Path(tmpdir) / "incomplete.npz"
        np.savez_compressed(
            npz_path,
            theta=np.array([0, 1, 2]),
            phi=np.array([0, 1, 2]),
            # Missing e_theta, e_phi, rad_efficiency
        )

        with pytest.raises(KeyError):
            pattern_io.load_radiation_pattern_npz(npz_path)


def test_radiation_pattern_npz_roundtrip_filelike():
    """Test saving and loading RadiationPattern with NPZ format using file-like objects."""
    # Create a simple test pattern
    theta = np.linspace(0, np.pi, 8) * u.rad
    phi = np.linspace(0, 2 * np.pi, 12, endpoint=False) * u.rad
    e_theta = (0.7 + 0.4j) * np.ones((8, 12)) * u.dimensionless
    e_phi = (0.3 - 0.6j) * np.ones((8, 12)) * u.dimensionless
    rad_efficiency = 0.85 * u.dimensionless

    original = RadiationPattern(theta, phi, e_theta, e_phi, rad_efficiency)

    # Test with BytesIO (simulates database BLOB storage)
    buffer = io.BytesIO()

    # Save to buffer
    pattern_io.save_radiation_pattern_npz(original, buffer)

    # Reset buffer position for reading
    buffer.seek(0)

    # Load from buffer
    loaded = pattern_io.load_radiation_pattern_npz(buffer)

    # Verify the data matches
    np.testing.assert_array_equal(loaded.theta, original.theta)
    np.testing.assert_array_equal(loaded.phi, original.phi)
    np.testing.assert_array_equal(loaded.e_theta, original.e_theta)
    np.testing.assert_array_equal(loaded.e_phi, original.e_phi)
    np.testing.assert_array_equal(loaded.rad_efficiency, original.rad_efficiency)
