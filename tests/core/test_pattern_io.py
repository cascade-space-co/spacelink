"""Tests for pattern I/O functionality."""

import io
import pathlib
import tempfile
import textwrap

import astropy.units as u
import numpy as np
import pytest
from astropy.tests.helper import assert_quantity_allclose

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

    original = RadiationPattern(theta, phi, None, e_theta, e_phi, rad_efficiency)

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
        theta,
        phi,
        None,
        e_theta,
        e_phi,
        rad_efficiency,
        default_polarization=default_pol,
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

    original = RadiationPattern(theta, phi, None, e_theta, e_phi, rad_efficiency)

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
    assert loaded.frequency is None
    assert loaded.default_frequency is None


def test_radiation_pattern_npz_roundtrip_3d():
    """Test saving and loading 3D RadiationPattern with frequency."""
    # Create a 3D test pattern
    theta = np.linspace(0, np.pi, 6) * u.rad
    phi = np.linspace(0, 2 * np.pi, 8, endpoint=False) * u.rad
    frequency = np.array([2.4, 5.8, 10.0]) * u.GHz
    e_theta = (0.8 + 0.3j) * np.ones((6, 8, 3)) * u.dimensionless
    e_phi = (0.2 - 0.5j) * np.ones((6, 8, 3)) * u.dimensionless
    rad_efficiency = 0.75 * u.dimensionless
    default_frequency = 2.4 * u.GHz

    original = RadiationPattern(
        theta,
        phi,
        frequency,
        e_theta,
        e_phi,
        rad_efficiency,
        default_frequency=default_frequency,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = pathlib.Path(tmpdir) / "test_pattern_3d.npz"
        pattern_io.save_radiation_pattern_npz(original, npz_path)
        assert npz_path.exists()
        loaded = pattern_io.load_radiation_pattern_npz(npz_path)

        np.testing.assert_array_equal(loaded.theta, original.theta)
        np.testing.assert_array_equal(loaded.phi, original.phi)
        np.testing.assert_array_equal(loaded.frequency, original.frequency)
        np.testing.assert_array_equal(loaded.e_theta, original.e_theta)
        np.testing.assert_array_equal(loaded.e_phi, original.e_phi)
        np.testing.assert_array_equal(loaded.rad_efficiency, original.rad_efficiency)
        np.testing.assert_array_equal(
            loaded.default_frequency, original.default_frequency
        )
        assert loaded.default_polarization is None


def test_radiation_pattern_npz_roundtrip_3d_with_polarization():
    """Test saving and loading 3D RadiationPattern with frequency and polarization."""
    # Create a 3D test pattern with default polarization
    theta = np.linspace(0, np.pi, 5) * u.rad
    phi = np.linspace(0, 2 * np.pi, 7, endpoint=False) * u.rad
    frequency = np.array([1.0, 2.0]) * u.GHz
    e_theta = (0.6 + 0.4j) * np.ones((5, 7, 2)) * u.dimensionless
    e_phi = (0.3 - 0.2j) * np.ones((5, 7, 2)) * u.dimensionless
    rad_efficiency = 0.9 * u.dimensionless

    # Create a custom polarization
    default_pol = Polarization(
        tilt_angle=45 * u.deg,
        axial_ratio=1.5 * u.dimensionless,
        handedness=Handedness.RIGHT,
    )

    original = RadiationPattern(
        theta,
        phi,
        frequency,
        e_theta,
        e_phi,
        rad_efficiency,
        default_polarization=default_pol,
        default_frequency=1.5 * u.GHz,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = pathlib.Path(tmpdir) / "test_pattern_3d_pol.npz"
        pattern_io.save_radiation_pattern_npz(original, npz_path)
        assert npz_path.exists()
        loaded = pattern_io.load_radiation_pattern_npz(npz_path)

        # Test basic arrays
        np.testing.assert_array_equal(loaded.theta, original.theta)
        np.testing.assert_array_equal(loaded.phi, original.phi)
        np.testing.assert_array_equal(loaded.frequency, original.frequency)
        np.testing.assert_array_equal(loaded.e_theta, original.e_theta)
        np.testing.assert_array_equal(loaded.e_phi, original.e_phi)
        np.testing.assert_array_equal(loaded.rad_efficiency, original.rad_efficiency)
        np.testing.assert_array_equal(
            loaded.default_frequency, original.default_frequency
        )

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


class TestRadiationPatternHFSSImport:

    def test_single_freq(self, tmp_path):
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

        pattern = pattern_io.import_hfss_csv(
            csv_file,
            rad_efficiency=0.8 * u.dimensionless,
        )

        # Verify basic properties
        assert pattern.rad_efficiency == 0.8 * u.dimensionless
        assert len(pattern.theta) == 4  # 20, 40, 60, 80 degrees
        assert len(pattern.phi) == 4  # 0, 90, 180, and 270 degrees
        assert pattern.frequency is None
        assert pattern.theta[0] == 20 * u.deg
        assert pattern.theta[1] == 40 * u.deg
        assert pattern.theta[2] == 60 * u.deg
        assert pattern.theta[3] == 80 * u.deg
        assert pattern.phi[0] == 0 * u.deg
        assert pattern.phi[1] == 90 * u.deg
        assert pattern.phi[2] == 180 * u.deg
        assert pattern.phi[3] == 270 * u.deg

        lhcp_pol = Polarization.lhcp()
        rhcp_pol = Polarization.rhcp()

        # Exact-value checks at grid points (should match CSV dB values)
        assert_quantity_allclose(
            pattern.gain(20 * u.deg, 0 * u.deg, polarization=lhcp_pol),
            -25.0 * u.dB,
        )
        assert_quantity_allclose(
            pattern.gain(20 * u.deg, 0 * u.deg, polarization=rhcp_pol),
            -5.0 * u.dB,
        )
        assert_quantity_allclose(
            pattern.gain(40 * u.deg, 90 * u.deg, polarization=lhcp_pol),
            -26.5 * u.dB,
        )
        assert_quantity_allclose(
            pattern.gain(40 * u.deg, 90 * u.deg, polarization=rhcp_pol),
            -6.5 * u.dB,
        )

    def test_multi_freq(self, tmp_path):
        csv_content = textwrap.dedent(
            """\
            Freq [GHz],Theta [deg],Phi [deg],"""
            """dB(RealizedGainRHCP) [],dB(RealizedGainLHCP) [],"""
            """ang_deg(rELHCP) [deg],ang_deg(rERHCP) [deg]
            2.4,20,0,-5.0,-25.0,0,90
            2.4,20,90,-6.0,-26.0,45,135
            2.4,40,0,-5.5,-25.5,15,105
            2.4,40,90,-6.5,-26.5,60,150
            5.8,20,0,-4.0,-24.0,10,100
            5.8,20,90,-5.0,-25.0,55,145
            5.8,40,0,-4.5,-24.5,25,115
            5.8,40,90,-5.5,-25.5,70,160
            """
        )
        csv_file = tmp_path / "test_multi_freq.csv"
        csv_file.write_text(csv_content)

        pattern = pattern_io.import_hfss_csv(
            csv_file,
            rad_efficiency=0.9 * u.dimensionless,
        )

        # Should create 3D pattern with frequency axis
        assert pattern.frequency is not None
        assert len(pattern.frequency) == 2
        assert pattern.frequency[0] == 2.4 * u.GHz
        assert pattern.frequency[1] == 5.8 * u.GHz
        assert len(pattern.theta) == 2  # 20, 40 degrees
        assert len(pattern.phi) == 2  # 0, 90 degrees

        # Exact LHCP gains at a shared grid point, per frequency
        lhcp_pol = Polarization.lhcp()
        # Use exact frequency values from pattern to avoid precision issues
        freq_24 = pattern.frequency[0]
        freq_58 = pattern.frequency[1]
        assert_quantity_allclose(
            pattern.gain(
                20 * u.deg, 0 * u.deg, frequency=freq_24, polarization=lhcp_pol
            ),
            -25.0 * u.dB,
        )
        assert_quantity_allclose(
            pattern.gain(
                20 * u.deg, 0 * u.deg, frequency=freq_58, polarization=lhcp_pol
            ),
            -24.0 * u.dB,
        )

    def test_file_not_found(self, tmp_path):
        """Test error when CSV file doesn't exist."""
        nonexistent_file = tmp_path / "does_not_exist.csv"

        with pytest.raises(FileNotFoundError):
            pattern_io.import_hfss_csv(
                nonexistent_file,
                rad_efficiency=1.0 * u.dimensionless,
            )

    @pytest.mark.parametrize(
        "drop_col",
        [
            "dB(RealizedGainLHCP) []",
            "dB(RealizedGainRHCP) []",
            "ang_deg(rELHCP) [deg]",
            "ang_deg(rERHCP) [deg]",
        ],
    )
    def test_missing_columns(self, tmp_path, drop_col):
        """Parametrized: each missing required column should raise KeyError."""
        cols = [
            "Freq [GHz]",
            "Theta [deg]",
            "Phi [deg]",
            "dB(RealizedGainRHCP) []",
            "dB(RealizedGainLHCP) []",
            "ang_deg(rELHCP) [deg]",
            "ang_deg(rERHCP) [deg]",
        ]
        cols = [c for c in cols if c != drop_col]
        header = ",".join(cols)
        # Single row with values where present
        values_map = {
            "Freq [GHz]": "2.4",
            "Theta [deg]": "20",
            "Phi [deg]": "0",
            "dB(RealizedGainRHCP) []": "2.0",
            "dB(RealizedGainLHCP) []": "-15.0",
            "ang_deg(rELHCP) [deg]": "0",
            "ang_deg(rERHCP) [deg]": "90",
        }
        row = ",".join(values_map[c] for c in cols)
        csv_content = f"{header}\n{row}\n"
        csv_file = tmp_path / "test_missing_column.csv"
        csv_file.write_text(csv_content)

        with pytest.raises(KeyError):
            pattern_io.import_hfss_csv(
                csv_file,
                rad_efficiency=1.0 * u.dimensionless,
            )

    def test_duplicates(self, tmp_path):
        """Duplicate (Freq,Theta,Phi) rows should raise ValueError."""
        csv_content = textwrap.dedent(
            """\
            Freq [GHz],Theta [deg],Phi [deg],"""
            """dB(RealizedGainRHCP) [],dB(RealizedGainLHCP) [],"""
            """ang_deg(rELHCP) [deg],ang_deg(rERHCP) [deg]
            2.4,20,0,-5.0,-25.0,0,90
            2.4,20,0,-5.0,-25.0,0,90
            """
        )
        csv_file = tmp_path / "dup.csv"
        csv_file.write_text(csv_content)
        with pytest.raises(ValueError):
            pattern_io.import_hfss_csv(csv_file, rad_efficiency=1.0 * u.dimensionless)

    def test_irregular_grid(self, tmp_path):
        """Missing a single (theta,phi) grid cell should raise ValueError."""
        csv_content = textwrap.dedent(
            """\
            Freq [GHz],Theta [deg],Phi [deg],"""
            """dB(RealizedGainRHCP) [],dB(RealizedGainLHCP) [],"""
            """ang_deg(rELHCP) [deg],ang_deg(rERHCP) [deg]
            2.4,20,0,-3.0,-13.0,0,0
            2.4,20,90,-6.0,-16.0,0,0
            2.4,40,90,-12.0,-22.0,0,0
            """
        )
        csv_file = tmp_path / "irregular.csv"
        csv_file.write_text(csv_content)
        with pytest.raises(ValueError):
            pattern_io.import_hfss_csv(csv_file, rad_efficiency=1.0 * u.dimensionless)

    def test_extra_columns_and_any_order(self, tmp_path):
        """Shuffled headers and extra columns should be tolerated."""
        csv_content = textwrap.dedent(
            """\
            ang_deg(rERHCP) [deg],dB(RealizedGainRHCP) [],Phi [deg],"""
            """Freq [GHz],Theta [deg],EXTRA,"""
            """dB(RealizedGainLHCP) [],ang_deg(rELHCP) [deg]
            90,-3.0,0,2.4,20,foo,-13.0,0
            180,-6.0,90,2.4,20,bar,-16.0,0
            """
        )
        csv_file = tmp_path / "extra.csv"
        csv_file.write_text(csv_content)
        pat = pattern_io.import_hfss_csv(csv_file, rad_efficiency=1.0 * u.dimensionless)
        assert pat.frequency is None
        assert pat.theta.size == 1 and pat.phi.size == 2
        lhcp_pol = Polarization.lhcp()
        rhcp_pol = Polarization.rhcp()
        assert_quantity_allclose(
            pat.gain(20 * u.deg, 0 * u.deg, polarization=lhcp_pol), -13.0 * u.dB
        )
        assert_quantity_allclose(
            pat.gain(20 * u.deg, 0 * u.deg, polarization=rhcp_pol), -3.0 * u.dB
        )
        assert_quantity_allclose(
            pat.gain(20 * u.deg, 90 * u.deg, polarization=lhcp_pol), -16.0 * u.dB
        )
        assert_quantity_allclose(
            pat.gain(20 * u.deg, 90 * u.deg, polarization=rhcp_pol), -6.0 * u.dB
        )
