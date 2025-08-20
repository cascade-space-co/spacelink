"""Tests for pattern I/O functionality."""

import io
import json
import pathlib
import textwrap

import astropy.units as u
import numpy as np
import pytest
from astropy.tests.helper import assert_quantity_allclose

from spacelink.core.antenna import RadiationPattern, Polarization, Handedness
from spacelink.core import pattern_io


def _assert_npz_metadata(data: np.lib.npyio.NpzFile) -> None:
    # Required metadata keys should exist
    assert "format_name" in data.files
    assert "format_version" in data.files
    assert "conventions" in data.files
    assert "producer" in data.files
    assert "created_utc" in data.files
    assert "dtype_info" in data.files

    # Basic value checks (no strict message comparisons)
    format_name = str(np.asarray(data["format_name"]).item())
    assert format_name == "spacelink.RadiationPattern"
    format_version = int(np.asarray(data["format_version"]))
    assert format_version == 1
    # Producer should include package name
    producer = str(np.asarray(data["producer"]).item())
    assert producer.startswith("spacelink ")
    # dtype_info should be valid JSON and include expected keys
    dtype_info = json.loads(str(np.asarray(data["dtype_info"]).item()))
    for key in ["theta", "phi", "e_theta", "e_phi", "rad_efficiency"]:
        assert key in dtype_info


def _assert_pattern_equal(actual: RadiationPattern, expected: RadiationPattern) -> None:
    np.testing.assert_array_equal(actual.theta, expected.theta)
    np.testing.assert_array_equal(actual.phi, expected.phi)
    np.testing.assert_array_equal(actual.e_theta, expected.e_theta)
    np.testing.assert_array_equal(actual.e_phi, expected.e_phi)
    np.testing.assert_array_equal(actual.rad_efficiency, expected.rad_efficiency)

    # Optional axes/metadata
    if expected.frequency is None:
        assert actual.frequency is None
    else:
        np.testing.assert_array_equal(actual.frequency, expected.frequency)

    if expected.default_frequency is None:
        assert actual.default_frequency is None
    else:
        np.testing.assert_array_equal(
            actual.default_frequency, expected.default_frequency
        )

    if expected.default_polarization is None:
        assert actual.default_polarization is None
    else:
        assert actual.default_polarization is not None
        assert (
            actual.default_polarization.tilt_angle
            == expected.default_polarization.tilt_angle
        )
        assert (
            actual.default_polarization.axial_ratio
            == expected.default_polarization.axial_ratio
        )
        assert (
            actual.default_polarization.handedness
            == expected.default_polarization.handedness
        )


class TestRadiationPatternNPZ:

    @pytest.mark.parametrize("dest_type", ["path", "filelike"])
    @pytest.mark.parametrize("dim", ["2d", "3d"])
    @pytest.mark.parametrize("with_pol", [False, True])
    @pytest.mark.parametrize("with_default_freq", [False, True])
    def test_roundtrip_npz(self, tmp_path, dest_type, dim, with_pol, with_default_freq):
        default_pol = None
        if with_pol:
            default_pol = Polarization(
                tilt_angle=45 * u.deg,
                axial_ratio=1.5 * u.dimensionless,
                handedness=Handedness.RIGHT,
            )

        if dim == "2d":
            original = RadiationPattern(
                theta=np.linspace(0, np.pi, 10) * u.rad,
                phi=np.linspace(0, 2 * np.pi, 15, endpoint=False) * u.rad,
                frequency=None,
                e_theta=(0.8 + 0.3j) * np.ones((10, 15)) * u.dimensionless,
                e_phi=(0.2 - 0.5j) * np.ones((10, 15)) * u.dimensionless,
                rad_efficiency=0.75 * u.dimensionless,
                default_frequency=(2.4 * u.GHz if with_default_freq else None),
                default_polarization=default_pol,
            )
        else:
            original = RadiationPattern(
                theta=np.linspace(0, np.pi, 6) * u.rad,
                phi=np.linspace(0, 2 * np.pi, 8, endpoint=False) * u.rad,
                frequency=(np.array([2.4, 5.8, 10.0]) * u.GHz),
                e_theta=(0.8 + 0.3j) * np.ones((6, 8, 3)) * u.dimensionless,
                e_phi=(0.2 - 0.5j) * np.ones((6, 8, 3)) * u.dimensionless,
                rad_efficiency=0.75 * u.dimensionless,
                default_frequency=(2.4 * u.GHz if with_default_freq else None),
                default_polarization=default_pol,
            )

        if dest_type == "path":
            npz_path = tmp_path / "pattern.npz"
            pattern_io.save_radiation_pattern_npz(original, npz_path)
            source = npz_path
        else:
            buffer = io.BytesIO()
            pattern_io.save_radiation_pattern_npz(original, buffer)
            source = io.BytesIO(buffer.getvalue())

        # Verify metadata directly from saved artifact
        _assert_npz_metadata(np.load(source))

        # Load for functional roundtrip assertions
        if dest_type == "path":
            loaded = pattern_io.load_radiation_pattern_npz(npz_path)
        else:
            buffer.seek(0)
            loaded = pattern_io.load_radiation_pattern_npz(buffer)

        _assert_pattern_equal(loaded, original)

    def test_missing_file(self):
        nonexistent_path = pathlib.Path("/nonexistent/path/missing.npz")
        with pytest.raises(FileNotFoundError):
            pattern_io.load_radiation_pattern_npz(nonexistent_path)

    def test_missing_keys(self, tmp_path):
        npz_path = tmp_path / "incomplete.npz"
        np.savez_compressed(
            npz_path,
            theta=np.array([0, 1, 2]),
            phi=np.array([0, 1, 2]),
            # Missing e_theta, e_phi, rad_efficiency
        )

        with pytest.raises(KeyError):
            pattern_io.load_radiation_pattern_npz(npz_path)

    @pytest.mark.parametrize(
        "format_name,format_version",
        [
            ("not.spacelink", 1),
            ("spacelink.RadiationPattern", 999),
        ],
    )
    def test_invalid_format_identity(self, tmp_path, format_name, format_version):
        # Construct a minimal but complete NPZ overriding identity fields
        theta = np.array([0.0])
        phi = np.array([0.0])
        e_theta = np.ones((1, 1), dtype=np.complex128)
        e_phi = 1j * np.ones((1, 1), dtype=np.complex128)
        rad_eff = np.array(1.0)
        npz_path = tmp_path / "bad_identity.npz"
        np.savez_compressed(
            npz_path,
            format_name=format_name,
            format_version=format_version,
            theta=theta,
            phi=phi,
            e_theta=e_theta,
            e_phi=e_phi,
            rad_efficiency=rad_eff,
        )
        with pytest.raises(ValueError):
            pattern_io.load_radiation_pattern_npz(npz_path)


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
