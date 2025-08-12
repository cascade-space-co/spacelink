"""Tests for core antenna calculation functions."""

import math
import pytest
import numpy as np
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose
from dataclasses import dataclass

from spacelink.core.antenna import (
    polarization_loss,
    dish_gain,
    Polarization,
    Handedness,
    RadiationPattern,
    SphericalInterpolator,
    gain_from_g_over_t,
    temperature_from_g_over_t,
)
from spacelink.core.units import (
    Dimensionless,
    Length,
    Frequency,
    Decibels,
    to_dB,
    to_linear,
)


"""
This site was used to generate the following test cases:
https://phillipmfeldman.org/Engineering/pol_mismatch_loss.html
"""


@pytest.mark.parametrize(
    "ar_tx_db, ar_rx_db, expected_loss_db, tol",
    [
        (0 * u.dB, 0 * u.dB, 0.0 * u.dB, 0.01 * u.dB),  # matchy matchy
        (0 * u.dB, 60 * u.dB, 3.002 * u.dB, 0.01 * u.dB),  # Circular to linear
        (60 * u.dB, 0 * u.dB, 3.002 * u.dB, 0.01 * u.dB),  # Linear to circular
        (1 * u.dB, 1 * u.dB, 0.057 * u.dB, 0.01 * u.dB),  # Small mismatch
        (3 * u.dB, 3 * u.dB, 0.508 * u.dB, 0.01 * u.dB),  # Small mismatch
        (1 * u.dB, 2 * u.dB, 0.128 * u.dB, 0.01 * u.dB),  # Small mismatch
        (2 * u.dB, 1 * u.dB, 0.128 * u.dB, 0.01 * u.dB),  # Small mismatch
        (2 * u.dB, 2 * u.dB, 0.228 * u.dB, 0.01 * u.dB),  # Small mismatch
        (2 * u.dB, 3 * u.dB, 0.354 * u.dB, 0.01 * u.dB),  # Small mismatch
        (3 * u.dB, 2 * u.dB, 0.354 * u.dB, 0.01 * u.dB),  # Small mismatch
        (3 * u.dB, 3 * u.dB, 0.508 * u.dB, 0.01 * u.dB),  # Small mismatch
        (3 * u.dB, 4 * u.dB, 0.684 * u.dB, 0.01 * u.dB),  # Small mismatch
        (4 * u.dB, 3 * u.dB, 0.684 * u.dB, 0.01 * u.dB),  # Small mismatch
        (4 * u.dB, 4 * u.dB, 0.890 * u.dB, 0.01 * u.dB),  # Small mismatch
        (4 * u.dB, 5 * u.dB, 1.114 * u.dB, 0.01 * u.dB),  # Small mismatch
        (5 * u.dB, 4 * u.dB, 1.114 * u.dB, 0.01 * u.dB),  # Small mismatch
        (5 * u.dB, 5 * u.dB, 1.366 * u.dB, 0.01 * u.dB),  # Small mismatch
    ],
)
def test_polarization_loss_calculation(ar_tx_db, ar_rx_db, expected_loss_db, tol):
    """
    Test the polarization loss calculation based on axial ratios.
    """
    assert_quantity_allclose(
        polarization_loss(ar_tx_db, ar_rx_db), expected_loss_db, atol=tol
    )


# Unvalidated
@pytest.mark.parametrize(
    "diameter, frequency, efficiency, expected_gain_db, tol",
    [
        (20.0 * u.m, 8.4 * u.GHz, 0.65 * u.dimensionless, 63.04 * u.dB, 0.01 * u.dB),
        (1.0 * u.m, 2.4 * u.GHz, 0.65 * u.dimensionless, 26.14 * u.dB, 0.01 * u.dB),
        (1.0 * u.m, 2.4 * u.GHz, 0.5 * u.dimensionless, 25.00 * u.dB, 0.01 * u.dB),
        (20.0 * u.m, 2.4 * u.GHz, 0.65 * u.dimensionless, 52.16 * u.dB, 0.01 * u.dB),
        (1.0 * u.m, 5.8 * u.GHz, 0.65 * u.dimensionless, 33.80 * u.dB, 0.01 * u.dB),
        (1.0 * u.m, 900 * u.MHz, 0.65 * u.dimensionless, 17.62 * u.dB, 0.01 * u.dB),
    ],
)
def test_dish_gain_calculation(
    diameter: Length,
    frequency: Frequency,
    efficiency: Dimensionless,
    expected_gain_db: Decibels,
    tol: Decibels,
):
    """Test the core dish_gain calculation function with various parameters."""
    assert_quantity_allclose(
        dish_gain(diameter, frequency, efficiency), expected_gain_db, atol=tol
    )


def test_dish_gain_invalid_frequency():
    """Test that dish_gain raises error for invalid frequency."""
    # Test invalid frequency
    with pytest.raises(ValueError, match="Frequency must be positive"):
        dish_gain(1.0 * u.m, 0 * u.Hz, 0.65 * u.dimensionless)
    with pytest.raises(ValueError, match="Frequency must be positive"):
        dish_gain(1.0 * u.m, -1 * u.GHz, 0.65 * u.dimensionless)


class TestPolarization:
    """Tests for the Polarization class."""

    @pytest.mark.parametrize(
        "tilt_angle, axial_ratio, phase_difference, expected_jones",
        [
            # Linear polarization along theta
            (
                0 * u.rad,
                np.inf * u.dimensionless,
                Handedness.LEFT,
                np.array([1.0, 0.0]),
            ),
            # Linear polarization along phi
            (
                np.pi / 2 * u.rad,
                np.inf * u.dimensionless,
                Handedness.LEFT,
                np.array([0.0, 1.0]),
            ),
            # Linear polarization at 45 degrees
            (
                np.pi / 4 * u.rad,
                np.inf * u.dimensionless,
                Handedness.LEFT,
                np.array([1.0, 1.0]) / np.sqrt(2),
            ),
            # Left-hand circular polarization
            (
                np.pi / 4 * u.rad,
                1 * u.dimensionless,
                Handedness.LEFT,
                np.array([1.0, 1.0j]) / np.sqrt(2),
            ),
            # Right-hand circular polarization
            (
                np.pi / 4 * u.rad,
                1 * u.dimensionless,
                Handedness.RIGHT,
                np.array([1.0, -1.0j]) / np.sqrt(2),
            ),
            # Elliptical polarization
            (
                0 * u.rad,
                2 * u.dimensionless,
                Handedness.LEFT,
                np.array([1.0, 0.5j]) / np.sqrt(1.25),
            ),
        ],
    )
    def test_polarization_jones_vector(
        self, tilt_angle, axial_ratio, phase_difference, expected_jones
    ):
        pol = Polarization(tilt_angle, axial_ratio, phase_difference)
        np.testing.assert_allclose(pol.jones_vector, expected_jones, atol=1e-10)

    def test_polarization_factories(self):
        lhcp = Polarization.lhcp()
        rhcp = Polarization.rhcp()

        np.testing.assert_allclose(
            lhcp.jones_vector, np.array([1.0, 1.0j]) / np.sqrt(2)
        )
        np.testing.assert_allclose(
            rhcp.jones_vector, np.array([1.0, -1.0j]) / np.sqrt(2)
        )

        # Orthogonal states should have zero inner product
        inner_product = np.dot(lhcp.jones_vector.conj(), rhcp.jones_vector)
        assert abs(inner_product) == pytest.approx(0.0)


@dataclass
class RadiationPatternExpectedResults:
    """Expected results for radiation pattern tests."""

    lhcp_gain: Dimensionless
    lhcp_directivity: Dimensionless
    rhcp_gain: Dimensionless
    rhcp_directivity: Dimensionless
    theta_directivity: Dimensionless
    phi_directivity: Dimensionless
    axial_ratio: Decibels


@dataclass
class RadiationPatternTestCase:
    """Data structure for radiation pattern test cases."""

    name: str
    pattern: RadiationPattern
    expected_results: RadiationPatternExpectedResults


def create_antenna_pattern_test_cases():
    """Create test patterns for different antenna configurations."""

    return [
        RadiationPatternTestCase(
            name="isotropic_theta",
            pattern=RadiationPattern(
                theta=np.linspace(0, np.pi, 40) * u.rad,
                phi=np.linspace(0, 2 * np.pi, 50, endpoint=False) * u.rad,
                e_theta=np.ones((40, 50)) * u.dimensionless,
                e_phi=np.zeros((40, 50)) * u.dimensionless,
                rad_efficiency=0.8 * u.dimensionless,
            ),
            expected_results=RadiationPatternExpectedResults(
                lhcp_gain=0.8 * 0.5 * u.dimensionless,
                lhcp_directivity=0.5 * u.dimensionless,
                rhcp_gain=0.8 * 0.5 * u.dimensionless,
                rhcp_directivity=0.5 * u.dimensionless,
                theta_directivity=1.0 * u.dimensionless,
                phi_directivity=0.0 * u.dimensionless,
                axial_ratio=np.inf * u.dB,
            ),
        ),
        RadiationPatternTestCase(
            name="isotropic_phi",
            pattern=RadiationPattern(
                theta=np.linspace(0, np.pi, 30) * u.rad,
                phi=np.linspace(0, 2 * np.pi, 40, endpoint=False) * u.rad,
                e_theta=np.zeros((30, 40)) * u.dimensionless,
                e_phi=np.ones((30, 40)) * u.dimensionless,
                rad_efficiency=1.0 * u.dimensionless,
            ),
            expected_results=RadiationPatternExpectedResults(
                lhcp_gain=0.5 * u.dimensionless,
                lhcp_directivity=0.5 * u.dimensionless,
                rhcp_gain=0.5 * u.dimensionless,
                rhcp_directivity=0.5 * u.dimensionless,
                theta_directivity=0.0 * u.dimensionless,
                phi_directivity=1.0 * u.dimensionless,
                axial_ratio=np.inf * u.dB,
            ),
        ),
        RadiationPatternTestCase(
            name="isotropic_lhcp",
            pattern=RadiationPattern(
                theta=np.linspace(0, np.pi, 50) * u.rad,
                phi=np.linspace(0, 2 * np.pi, 60, endpoint=False) * u.rad,
                e_theta=(1.0 + 0.0j) / np.sqrt(2) * np.ones((50, 60)) * u.dimensionless,
                e_phi=(0.0 + 1.0j) / np.sqrt(2) * np.ones((50, 60)) * u.dimensionless,
                rad_efficiency=1.0 * u.dimensionless,
            ),
            expected_results=RadiationPatternExpectedResults(
                lhcp_gain=1.0 * u.dimensionless,
                lhcp_directivity=1.0 * u.dimensionless,
                rhcp_gain=0.0 * u.dimensionless,
                rhcp_directivity=0.0 * u.dimensionless,
                theta_directivity=0.5 * u.dimensionless,
                phi_directivity=0.5 * u.dimensionless,
                axial_ratio=0.0 * u.dB,
            ),
        ),
        RadiationPatternTestCase(
            name="isotropic_elliptical",
            pattern=RadiationPattern(
                theta=np.linspace(0, np.pi, 25) * u.rad,
                phi=np.linspace(0, 2 * np.pi, 17, endpoint=False) * u.rad,
                e_theta=1.0 / np.sqrt(5 / 4) * np.ones((25, 17)) * u.dimensionless,
                e_phi=0.5j / np.sqrt(5 / 4) * np.ones((25, 17)) * u.dimensionless,
                rad_efficiency=0.7 * u.dimensionless,
            ),
            expected_results=RadiationPatternExpectedResults(
                lhcp_gain=0.7 * 0.9 * u.dimensionless,
                lhcp_directivity=0.9 * u.dimensionless,
                rhcp_gain=0.7 * 0.1 * u.dimensionless,
                rhcp_directivity=0.1 * u.dimensionless,
                theta_directivity=4 / 5 * u.dimensionless,
                phi_directivity=1 / 5 * u.dimensionless,
                axial_ratio=10 * math.log10(2) * u.dB,
            ),
        ),
    ]


@pytest.mark.parametrize("test_case", create_antenna_pattern_test_cases())
def test_antenna_pattern_calculations(test_case):

    shape_interp = (100, 200)
    theta_interp = np.linspace(0, np.pi, shape_interp[0]) * u.rad
    phi_interp = np.linspace(0, 2 * np.pi, shape_interp[1]) * u.rad

    pol_theta = Polarization(0 * u.rad, np.inf * u.dimensionless, Handedness.LEFT)
    pol_phi = Polarization(np.pi / 2 * u.rad, np.inf * u.dimensionless, Handedness.LEFT)

    assert_quantity_allclose(
        to_linear(
            test_case.pattern.gain(
                theta_interp[:, np.newaxis], phi_interp, Polarization.lhcp()
            )
        ),
        test_case.expected_results.lhcp_gain,
        atol=1e-10 * u.dimensionless,
    )

    assert_quantity_allclose(
        to_linear(
            test_case.pattern.directivity(
                theta_interp[:, np.newaxis], phi_interp, Polarization.lhcp()
            )
        ),
        test_case.expected_results.lhcp_directivity,
        atol=1e-10 * u.dimensionless,
    )

    assert_quantity_allclose(
        to_linear(
            test_case.pattern.gain(
                theta_interp[:, np.newaxis], phi_interp, Polarization.rhcp()
            )
        ),
        test_case.expected_results.rhcp_gain,
        atol=1e-10 * u.dimensionless,
    )

    assert_quantity_allclose(
        to_linear(
            test_case.pattern.directivity(
                theta_interp[:, np.newaxis], phi_interp, Polarization.rhcp()
            )
        ),
        test_case.expected_results.rhcp_directivity,
        atol=1e-10 * u.dimensionless,
    )

    assert_quantity_allclose(
        to_linear(
            test_case.pattern.directivity(
                theta_interp[:, np.newaxis], phi_interp, pol_theta
            )
        ),
        test_case.expected_results.theta_directivity,
        atol=1e-10 * u.dimensionless,
    )

    assert_quantity_allclose(
        to_linear(
            test_case.pattern.directivity(
                theta_interp[:, np.newaxis], phi_interp, pol_phi
            )
        ),
        test_case.expected_results.phi_directivity,
        atol=1e-10 * u.dimensionless,
    )

    assert_quantity_allclose(
        test_case.pattern.axial_ratio(theta_interp[:, np.newaxis], phi_interp),
        test_case.expected_results.axial_ratio,
        atol=1e-10 * u.dB,
    )


class TestSphericalInterpolator:
    """Tests for the SphericalInterpolator class."""

    def test_interpolation(self):
        N = 150
        M = 200
        downsample = 5
        periods = 7  # Number of periods in phi
        peak_gain = 50.0  # Gain varies between -50 and +50 dBi

        theta = np.linspace(0, np.pi, N) * u.rad
        phi = np.linspace(0, 2 * np.pi, M, endpoint=False) * u.rad

        gain_db_expected = (
            peak_gain
            * np.sin(theta.value)[:, np.newaxis]  # Taper to 0 dB at poles
            * np.sin(periods * phi.value)
            * u.dB
        )

        theta_decim = np.linspace(0, np.pi, N // downsample) * u.rad
        phi_decim = np.linspace(0, 2 * np.pi, M // downsample, endpoint=False) * u.rad
        gain_decim_db = (
            peak_gain
            * np.sin(theta_decim.value)[:, np.newaxis]  # Taper to 0 dB at poles
            * np.sin(periods * phi_decim.value)
            * u.dB
        )
        gain_decim = to_linear(gain_decim_db)

        interpolator = SphericalInterpolator(theta_decim, phi_decim, gain_decim)

        result = interpolator(theta[:, np.newaxis], phi)

        assert result.shape == (N, M)
        assert result.unit == u.dimensionless
        np.testing.assert_allclose(
            to_dB(result).value, gain_db_expected.value, atol=0.3
        )

    def test_subset_sphere_interpolation(self):
        N = 80
        M = 100
        downsample = 4
        periods = 5  # Number of periods in phi
        peak_gain = 40.0  # Gain varies between -40 and +40 dBi

        # Create subset of sphere: theta from 30° to 120°, phi from 45° to 270°
        theta = np.linspace(np.pi / 6, 2 * np.pi / 3, N) * u.rad
        phi = np.linspace(np.pi / 4, 3 * np.pi / 2, M, endpoint=False) * u.rad

        gain_db = (
            peak_gain
            * np.sin(theta.value)[:, np.newaxis]
            * np.sin(periods * phi.value)
            * u.dB
        )
        gain = to_linear(gain_db)

        interpolator = SphericalInterpolator(
            theta[::downsample],
            phi[::downsample],
            gain[::downsample, ::downsample],
        )

        result = interpolator(theta[:-downsample, np.newaxis], phi[:-downsample])

        assert result.shape == (N - downsample, M - downsample)
        assert result.unit == u.dimensionless

        # Loose tolerance because interpolation performance degrades at the edges when
        # the grid does not span the full circle.
        np.testing.assert_allclose(
            to_dB(result).value,
            gain_db[:-downsample, :-downsample].value,
            atol=1.2,
        )

    def test_with_zeros(self):
        """Test interpolation when input contains zeros (should use floor value)."""
        unit = u.K  # Arbitrary unit
        floor = -100 * u.dB
        theta = np.linspace(0, np.pi, 20) * u.rad
        phi = np.linspace(0, 2 * np.pi, 30, endpoint=False) * u.rad
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

        # Pattern with zeros at theta > π/2
        values = (
            np.where(
                theta_grid.value < np.pi / 2,
                np.sin(theta_grid.value) * np.exp(1j * phi_grid.value),
                0,
            )
            * unit
        )

        interpolator = SphericalInterpolator(theta, phi, values, floor=floor)

        # Test upper hemisphere
        test_theta = np.pi / 4 * u.rad
        test_phi = np.linspace(0, 2 * np.pi, 100) * u.rad
        result = interpolator(test_theta, test_phi)
        assert result.shape == (100,)
        assert result.unit == unit
        expected = np.sin(test_theta.value) * np.exp(1j * test_phi.value)
        np.testing.assert_allclose(result.value, expected, atol=1e-2)

        # Test lower hemisphere
        test_theta = 3 * np.pi / 4 * u.rad
        test_phi = np.linspace(0, 2 * np.pi, 100) * u.rad
        result = interpolator(test_theta, test_phi)
        assert result.shape == (100,)
        assert result.unit == unit
        expected = 10 ** (floor.value / 10)
        np.testing.assert_allclose(result.value, expected, atol=1e-10)

    def test_phase_continuity(self):
        theta = np.linspace(0, np.pi, 11) * u.rad
        phi = np.linspace(0, 2 * np.pi, 21, endpoint=False) * u.rad

        # Create a pattern with phase equal to 2*phi. This will cause two wraps from
        # 2π back to 0, one in the middle of the grid where phi == π and another at the
        # extreme edges of the grid where phi == 0 and phi == 2π. We expect the
        # interpolator to handle both of these correctly.
        values = (
            np.sin(theta[:, np.newaxis]) * np.exp(1j * 2 * phi.value) * u.dimensionless
        )

        interpolator = SphericalInterpolator(theta, phi, values)

        test_theta = np.linspace(0.03, np.pi - 0.03, 100) * u.rad
        test_phi = np.linspace(0, 2 * np.pi, 100) * u.rad
        result = interpolator(test_theta[:, np.newaxis], test_phi)

        assert result.shape == (100, 100)
        assert result.unit == u.dimensionless

        # Conjugate product avoids problems when comparing angles near 0 and 2π.
        phase_diff = np.angle(result.value * np.conj(np.exp(1j * 2 * test_phi.value)))
        np.testing.assert_allclose(
            phase_diff,
            0.0,
            atol=1e-4,
        )


@pytest.mark.parametrize(
    "theta, phi, values, expected_result, tol",
    [
        # Test 1: Constant function f(θ,φ) = 1
        # Expected: ∫∫ 1 sin(θ) dθ dφ = 4π
        (
            np.linspace(0, np.pi, 100) * u.rad,
            np.linspace(0, 2 * np.pi, 200) * u.rad,
            np.ones((100, 200)) * u.dimensionless,
            4 * np.pi * u.sr,
            1e-10 * u.sr,
        ),
        # Test 2: Function f(θ,φ) = sin(θ)
        # Expected: ∫∫ sin(θ) sin(θ) dθ dφ = ∫∫ sin²(θ) dθ dφ = π²
        (
            np.linspace(0, np.pi, 100) * u.rad,
            np.linspace(0, 2 * np.pi, 100) * u.rad,
            np.sin(np.linspace(0, np.pi, 100))[:, np.newaxis]
            * np.ones(100)
            * u.dimensionless,
            np.pi**2 * u.sr,
            1e-10 * u.sr,
        ),
        # Test 3: Function f(θ,φ) = cos²(θ) (dipole pattern)
        # Expected: ∫∫ cos²(θ) sin(θ) dθ dφ = 4π/3
        (
            np.linspace(0, np.pi, 100) * u.rad,
            np.linspace(0, 2 * np.pi, 200) * u.rad,
            (np.cos(np.linspace(0, np.pi, 100)) ** 2)[:, np.newaxis]
            * np.ones(200)
            * u.dimensionless,
            4 * np.pi / 3 * u.sr,
            1e-5 * u.sr,
        ),
        # Test 4: Function f(θ,φ) = cos(θ)
        # Expected: ∫∫ cos(θ) sin(θ) dθ dφ = 0
        (
            np.linspace(0, np.pi, 127) * u.rad,
            np.linspace(0, 2 * np.pi, 173) * u.rad,
            np.cos(np.linspace(0, np.pi, 127))[:, np.newaxis]
            * np.ones(173)
            * u.dimensionless,
            0 * u.sr,
            1e-10 * u.sr,
        ),
        # Test 5: Constant function f(θ,φ) = 1 over a section of the sphere
        # θ ∈ [0, π/2], φ ∈ [0, π/2] - quarter sphere
        # Expected: ∫∫ 1 sin(θ) dθ dφ = π/2
        (
            np.linspace(0, np.pi / 2, 50) * u.rad,
            np.linspace(0, np.pi / 2, 50) * u.rad,
            np.ones((50, 50)) * u.dimensionless,
            np.pi / 2 * u.sr,
            1e-10 * u.sr,
        ),
    ],
)
def test_surface_integral(theta, phi, values, expected_result, tol):
    """
    Test the _surface_integral function with known analytical results.

    The surface integral is defined as:
    ∫∫ f(θ,φ) sin(θ) dθ dφ

    This test uses simple functions with known analytical results.
    """
    from spacelink.core.antenna import _surface_integral

    result = _surface_integral(theta, phi, values)

    assert result.unit == expected_result.unit
    assert_quantity_allclose(result, expected_result, atol=tol)


class TestRadiationPatternValidation:
    """Tests for RadiationPattern constructor validation checks."""

    def test_theta_range_validation(self):
        """Test that theta values must be in [0, pi]."""
        phi = np.linspace(0, 2 * np.pi, 10, endpoint=False) * u.rad
        e_theta = np.ones((5, 10)) * u.dimensionless
        e_phi = np.zeros((5, 10)) * u.dimensionless
        rad_efficiency = 1.0 * u.dimensionless

        # Test theta < 0
        theta_negative = np.linspace(-0.1, np.pi, 5) * u.rad
        with pytest.raises(ValueError):
            RadiationPattern(theta_negative, phi, e_theta, e_phi, rad_efficiency)

        # Test theta > pi
        theta_too_large = np.linspace(0, np.pi + 0.1, 5) * u.rad
        with pytest.raises(ValueError):
            RadiationPattern(theta_too_large, phi, e_theta, e_phi, rad_efficiency)

    def test_theta_sorting_validation(self):
        """Test that theta must be sorted in strictly increasing order."""
        phi = np.linspace(0, 2 * np.pi, 10, endpoint=False) * u.rad
        e_theta = np.ones((5, 10)) * u.dimensionless
        e_phi = np.zeros((5, 10)) * u.dimensionless
        rad_efficiency = 1.0 * u.dimensionless

        # Test unsorted theta
        theta_unsorted = np.array([0, 0.5, 0.3, 1.0, 1.5]) * u.rad
        with pytest.raises(ValueError):
            RadiationPattern(theta_unsorted, phi, e_theta, e_phi, rad_efficiency)

        # Test repeated theta values
        theta_repeated = np.array([0, 0.5, 0.5, 1.0, 1.5]) * u.rad
        with pytest.raises(ValueError):
            RadiationPattern(theta_repeated, phi, e_theta, e_phi, rad_efficiency)

    def test_phi_sorting_validation(self):
        """Test that phi must be sorted in strictly increasing order."""
        theta = np.linspace(0, np.pi, 5) * u.rad
        e_theta = np.ones((5, 5)) * u.dimensionless
        e_phi = np.zeros((5, 5)) * u.dimensionless
        rad_efficiency = 1.0 * u.dimensionless

        # Test unsorted phi
        phi_unsorted = np.array([0, 1.0, 0.5, 2.0, 3.0]) * u.rad
        with pytest.raises(ValueError):
            RadiationPattern(theta, phi_unsorted, e_theta, e_phi, rad_efficiency)

        # Test repeated phi values
        phi_repeated = np.array([0, 1.0, 1.0, 2.0, 3.0]) * u.rad
        with pytest.raises(ValueError):
            RadiationPattern(theta, phi_repeated, e_theta, e_phi, rad_efficiency)

    def test_theta_spacing_validation(self):
        """Test that theta must be equally spaced."""
        phi = np.linspace(0, 2 * np.pi, 10, endpoint=False) * u.rad
        e_theta = np.ones((5, 10)) * u.dimensionless
        e_phi = np.zeros((5, 10)) * u.dimensionless
        rad_efficiency = 1.0 * u.dimensionless

        # Test unequally spaced theta
        theta_unequal = np.array([0, 0.5, 1.2, 2.0, 3.0]) * u.rad
        with pytest.raises(ValueError):
            RadiationPattern(theta_unequal, phi, e_theta, e_phi, rad_efficiency)

    def test_phi_spacing_validation(self):
        """Test that phi must be equally spaced."""
        theta = np.linspace(0, np.pi, 5) * u.rad
        e_theta = np.ones((5, 5)) * u.dimensionless
        e_phi = np.zeros((5, 5)) * u.dimensionless
        rad_efficiency = 1.0 * u.dimensionless

        # Test unequally spaced phi
        phi_unequal = np.array([0, 0.5, 1.2, 2.0, 3.0]) * u.rad
        with pytest.raises(ValueError):
            RadiationPattern(theta, phi_unequal, e_theta, e_phi, rad_efficiency)

    def test_phi_coverage_validation(self):
        """Test that phi must cover less than 2π radians."""
        theta = np.linspace(0, np.pi, 5) * u.rad
        e_theta = np.ones((5, 10)) * u.dimensionless
        e_phi = np.zeros((5, 10)) * u.dimensionless
        rad_efficiency = 1.0 * u.dimensionless

        # Test phi covering exactly 2π
        phi_full_circle = np.linspace(0, 2 * np.pi, 10) * u.rad
        with pytest.raises(ValueError):
            RadiationPattern(theta, phi_full_circle, e_theta, e_phi, rad_efficiency)

        # Test phi covering more than 2π
        phi_too_much = np.linspace(0, 2.5 * np.pi, 10) * u.rad
        with pytest.raises(ValueError):
            RadiationPattern(theta, phi_too_much, e_theta, e_phi, rad_efficiency)


def test_gain_from_g_over_t():
    """ where temp is exectly 1 K """
    gain = gain_from_g_over_t(10 * u.dB, 1 * u.K)
    assert_quantity_allclose(gain, 10 * u.dB)

    """ test where temp is greater than 1 K """ 
    gain_100_K = gain_from_g_over_t(10 * u.dB, 100 * u.K)
    assert_quantity_allclose(gain_100_K, 30 * u.dB)

    """ test for negative temperature """ 
    with pytest.raises(ValueError):
        gain_from_g_over_t(10 * u.dB, -1 * u.K)


def test_temperature_from_g_over_t():
    temperature = temperature_from_g_over_t(10 * u.dB, 20 * u.dB)
    assert_quantity_allclose(temperature, 10 * u.K)