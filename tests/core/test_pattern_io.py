"""Tests for pattern I/O functionality."""

import io
import pathlib
import tempfile
import textwrap

import astropy.units as u
import numpy as np
import pytest

from spacelink.core.antenna import RadiationPattern, Polarization, Handedness
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
        assert loaded.default_polarization is None


def test_radiation_pattern_npz_roundtrip_with_polarization():
    """Test saving and loading RadiationPattern with default_polarization set."""
    # Create a test pattern with a default polarization
    theta = np.linspace(0, np.pi, 8) * u.rad
    phi = np.linspace(0, 2 * np.pi, 12, endpoint=False) * u.rad
    e_theta = (0.6 + 0.4j) * np.ones((8, 12)) * u.dimensionless
    e_phi = (0.3 - 0.2j) * np.ones((8, 12)) * u.dimensionless
    rad_efficiency = 0.9 * u.dimensionless

    # Create a custom polarization
    default_pol = Polarization(
        tilt_angle=30 * u.deg,
        axial_ratio=2.5 * u.dimensionless,
        handedness=Handedness.LEFT,
    )

    original = RadiationPattern(
        theta, phi, e_theta, e_phi, rad_efficiency, default_polarization=default_pol
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = pathlib.Path(tmpdir) / "test_pattern_with_pol.npz"
        pattern_io.save_radiation_pattern_npz(original, npz_path)
        assert npz_path.exists()
        loaded = pattern_io.load_radiation_pattern_npz(npz_path)

        # Test basic arrays
        np.testing.assert_array_equal(loaded.theta, original.theta)
        np.testing.assert_array_equal(loaded.phi, original.phi)
        np.testing.assert_array_equal(loaded.e_theta, original.e_theta)
        np.testing.assert_array_equal(loaded.e_phi, original.e_phi)
        np.testing.assert_array_equal(loaded.rad_efficiency, original.rad_efficiency)

        # Test polarization was preserved
        assert loaded.default_polarization is not None
        assert (
            loaded.default_polarization.tilt_angle
            == original.default_polarization.tilt_angle
        )
        assert (
            loaded.default_polarization.axial_ratio
            == original.default_polarization.axial_ratio
        )
        assert (
            loaded.default_polarization.handedness
            == original.default_polarization.handedness
        )


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
    assert loaded.default_polarization is None


class TestRadiationPatternHFSSImport:

    def test_hfss_csv_import_success(self, tmp_path):
        """Test successful import of HFSS CSV file."""
        # Create a simple test CSV file with HFSS format
        csv_content = textwrap.dedent(
            """\
            Freq [GHz],Theta [deg],Phi [deg],"""
            """dB(RealizedGainRHCP) [],dB(RealizedGainLHCP) [],"""
            """ang_deg(rELHCP) [deg],ang_deg(rERHCP) [deg]
            2.4,20,0,-5.0,-25.0,0,90
            2.4,20,90,-6.0,-26.0,45,135
            2.4,20,180,-7.0,-27.0,90,180
            2.4,20,270,-8.0,-28.0,135,225
            2.4,40,0,-5.5,-25.5,15,105
            2.4,40,90,-6.5,-26.5,60,150
            2.4,40,180,-7.5,-27.5,105,195
            2.4,40,270,-8.5,-28.5,150,240
            2.4,60,0,-6.0,-26.0,30,120
            2.4,60,90,-7.0,-27.0,75,165
            2.4,60,180,-8.0,-28.0,120,210
            2.4,60,270,-9.0,-29.0,165,255
            2.4,80,0,-6.5,-26.5,45,135
            2.4,80,90,-7.5,-27.5,90,180
            2.4,80,180,-8.5,-28.5,135,225
            2.4,80,270,-9.5,-29.5,180,270
            """
        )
        csv_file = tmp_path / "test_pattern.csv"
        csv_file.write_text(csv_content)

        # Import pattern at 2.4 GHz
        pattern = pattern_io.import_hfss_csv(
            csv_file,
            carrier_frequency=2.4 * u.GHz,
            rad_efficiency=0.8 * u.dimensionless,
        )

        # Verify basic properties
        assert pattern.rad_efficiency == 0.8 * u.dimensionless
        assert len(pattern.theta) == 4  # 20, 40, 60, 80 degrees
        assert len(pattern.phi) == 4  # 0, 90, 180, and 270 degrees
        assert pattern.theta[0] == 20 * u.deg
        assert pattern.theta[1] == 40 * u.deg
        assert pattern.theta[2] == 60 * u.deg
        assert pattern.theta[3] == 80 * u.deg
        assert pattern.phi[0] == 0 * u.deg
        assert pattern.phi[1] == 90 * u.deg
        assert pattern.phi[2] == 180 * u.deg
        assert pattern.phi[3] == 270 * u.deg

        # Test basic functionality - just verify it works without crashing
        lhcp_pol = Polarization.lhcp()
        rhcp_pol = Polarization.rhcp()

        # Check gain at a test point
        gain_lhcp = pattern.gain(40 * u.deg, 90 * u.deg, lhcp_pol)
        gain_rhcp = pattern.gain(40 * u.deg, 90 * u.deg, rhcp_pol)

        # Just verify they're reasonable negative dB values
        assert gain_lhcp < 0 * u.dB
        assert gain_rhcp < 0 * u.dB

    def test_hfss_csv_phi_360_handling(self, tmp_path):
        """Test handling of redundant phi=360 degree values."""
        # Create CSV with phi from 0 to 360 (redundant last value)
        csv_content = textwrap.dedent(
            """\
            Freq [GHz],Theta [deg],Phi [deg],"""
            """dB(RealizedGainRHCP) [],dB(RealizedGainLHCP) [],"""
            """ang_deg(rELHCP) [deg],ang_deg(rERHCP) [deg]
            2.4,20,0,-5.0,-25.0,0,90
            2.4,20,90,-6.0,-26.0,45,135
            2.4,20,180,-7.0,-27.0,90,180
            2.4,20,270,-6.5,-25.5,135,225
            2.4,20,360,-5.0,-25.0,0,90
            2.4,40,0,-5.5,-25.5,15,105
            2.4,40,90,-6.5,-26.5,60,150
            2.4,40,180,-7.5,-27.5,105,195
            2.4,40,270,-7.0,-26.0,150,240
            2.4,40,360,-5.5,-25.5,15,105
            2.4,60,0,-6.0,-26.0,30,120
            2.4,60,90,-7.0,-27.0,75,165
            2.4,60,180,-8.0,-28.0,120,210
            2.4,60,270,-7.5,-26.5,165,255
            2.4,60,360,-6.0,-26.0,30,120
            2.4,80,0,-6.5,-26.5,45,135
            2.4,80,90,-7.5,-27.5,90,180
            2.4,80,180,-8.5,-28.5,135,225
            2.4,80,270,-8.0,-27.0,180,270
            2.4,80,360,-6.5,-26.5,45,135
            """
        )
        csv_file = tmp_path / "test_360.csv"
        csv_file.write_text(csv_content)

        pattern = pattern_io.import_hfss_csv(
            csv_file,
            carrier_frequency=2.4 * u.GHz,
            rad_efficiency=1.0 * u.dimensionless,
        )

        # Should have 4 phi values (0, 90, 180, 270), redundant 360 removed
        assert len(pattern.phi) == 4
        assert pattern.phi[0] == 0 * u.deg
        assert pattern.phi[1] == 90 * u.deg
        assert pattern.phi[2] == 180 * u.deg
        assert pattern.phi[3] == 270 * u.deg

    def test_hfss_csv_frequency_not_found(self, tmp_path):
        """Test error when requested frequency is not in CSV."""
        csv_content = textwrap.dedent(
            """\
            Freq [GHz],Theta [deg],Phi [deg],"""
            """dB(RealizedGainRHCP) [],dB(RealizedGainLHCP) [],"""
            """ang_deg(rELHCP) [deg],ang_deg(rERHCP) [deg]
            2.4,20,0,2.0,-15.0,0,90
            2.4,20,90,1.0,-16.0,45,135
            2.4,40,0,1.5,-15.2,22,112
            2.4,40,90,0.5,-16.2,67,157
            2.4,60,0,1.0,-15.5,45,135
            2.4,60,90,0.0,-16.5,90,180
            """
        )
        csv_file = tmp_path / "test_missing_freq.csv"
        csv_file.write_text(csv_content)

        # Request frequency not in file - should raise an error
        with pytest.raises(ValueError):
            pattern_io.import_hfss_csv(
                csv_file,
                carrier_frequency=5.8 * u.GHz,
                rad_efficiency=1.0 * u.dimensionless,
            )

    def test_hfss_csv_file_not_found(self, tmp_path):
        """Test error when CSV file doesn't exist."""
        nonexistent_file = tmp_path / "does_not_exist.csv"

        with pytest.raises(FileNotFoundError):
            pattern_io.import_hfss_csv(
                nonexistent_file,
                carrier_frequency=2.4 * u.GHz,
                rad_efficiency=1.0 * u.dimensionless,
            )

    def test_hfss_csv_missing_columns(self, tmp_path):
        """Test error when required columns are missing."""
        # CSV missing the LHCP gain column
        csv_content = textwrap.dedent(
            """\
            Freq [GHz],Theta [deg],Phi [deg],dB(RealizedGainRHCP) [],"""
            """ang_deg(rELHCP) [deg],ang_deg(rERHCP) [deg]
            2.4,20,0,2.0,0,90
            2.4,20,90,1.0,45,135
            2.4,40,0,1.5,22,112
            2.4,40,90,0.5,67,157
            2.4,60,0,1.0,45,135
            2.4,60,90,0.0,90,180
            """
        )
        csv_file = tmp_path / "test_missing_column.csv"
        csv_file.write_text(csv_content)

        with pytest.raises(KeyError):
            pattern_io.import_hfss_csv(
                csv_file,
                carrier_frequency=2.4 * u.GHz,
                rad_efficiency=1.0 * u.dimensionless,
            )
