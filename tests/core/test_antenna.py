"""Tests for core antenna calculation functions."""

import math
from dataclasses import dataclass

import astropy.units as u
import numpy as np
import pytest
from astropy.tests.helper import assert_quantity_allclose

from spacelink.core.antenna import (
    Handedness,
    Polarization,
    RadiationPattern,
    _ComplexInterpolator,
    dish_gain,
    gain_from_g_over_t,
    polarization_loss,
    temperature_from_g_over_t,
)
from spacelink.core.units import (
    Decibels,
    Dimensionless,
    Frequency,
    Length,
    to_linear,
)


def _assert_lin_gain_equals_eta_times_directivity(
    pattern: RadiationPattern,
    theta: u.Quantity,
    phi: u.Quantity,
    *,
    polarization: Polarization,
    eta: u.Quantity,
) -> None:
    dir_lin = to_linear(pattern.directivity(theta, phi, polarization=polarization))
    gain_lin = to_linear(pattern.gain(theta, phi, polarization=polarization))
    assert_quantity_allclose(gain_lin, eta * dir_lin)


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

    def test_invalid_axial_ratio_raises(self):
        with pytest.raises(ValueError):
            Polarization(0 * u.rad, 0.5 * u.dimensionless, Handedness.LEFT)


class TestComplexInterpolator:
    """Tests for the ComplexInterpolator class."""

    def test_exactness_at_grid_nodes_2d(self):
        """Test that 2D interpolation is exact at the provided grid points."""
        N, M = 9, 13
        theta = np.linspace(0.1, np.pi - 0.1, N) * u.rad
        phi = np.linspace(0, 2 * np.pi, M, endpoint=False) * u.rad

        # Use random complex values to ensure we're not relying on special structure
        np.random.seed(42)
        values = (np.random.randn(N, M) + 1j * np.random.randn(N, M)) * u.dimensionless

        interpolator = _ComplexInterpolator(theta, phi, None, values)
        result = interpolator(theta[:, np.newaxis], phi)

        # Should be exact (within machine precision)
        assert_quantity_allclose(result, values, atol=1e-12 * u.dimensionless)

    def test_exactness_at_grid_nodes_3d(self):
        """Test that 3D interpolation is exact at the provided grid points."""
        N, M, K = 9, 13, 5
        theta = np.linspace(0.1, np.pi - 0.1, N) * u.rad
        phi = np.linspace(0, 2 * np.pi, M, endpoint=False) * u.rad
        frequency = (np.arange(K) + 1) * u.GHz

        # Use random complex values to ensure we're not relying on special structure
        np.random.seed(42)
        values = (
            np.random.randn(N, M, K) + 1j * np.random.randn(N, M, K)
        ) * u.dimensionless

        interpolator = _ComplexInterpolator(theta, phi, frequency, values)
        result = interpolator(
            theta[:, np.newaxis, np.newaxis],
            phi[np.newaxis, :, np.newaxis],
            frequency[np.newaxis, np.newaxis, :],
        )

        # Should be exact (within machine precision)
        assert_quantity_allclose(result, values, atol=1e-12 * u.dimensionless)

    def test_phi_periodicity_and_rotation_invariance_2d(self):
        """Test φ modulo invariance and rotation invariance for 2D patterns."""
        N, M = 12, 16
        theta = np.linspace(0.1, np.pi - 0.1, N) * u.rad
        phi = np.linspace(0, 2 * np.pi, M, endpoint=False) * u.rad

        np.random.seed(123)
        values = (np.random.randn(N, M) + 1j * np.random.randn(N, M)) * u.dimensionless
        interpolator = _ComplexInterpolator(theta, phi, None, values)

        # Test φ modulo invariance: f(θ, φ) == f(θ, φ + 2π)
        result1 = interpolator(theta[:, np.newaxis], phi)
        result2 = interpolator(theta[:, np.newaxis], phi + 2 * np.pi * u.rad)
        assert_quantity_allclose(result1, result2, atol=1e-12 * u.dimensionless)

        # Test φ rotation invariance: shifting grid and data should give same results
        shift = 3
        phi_shifted = np.roll(phi.value, shift) * u.rad
        values_shifted = np.roll(values.value, shift, axis=1) * u.dimensionless
        interpolator_shifted = _ComplexInterpolator(
            theta, phi_shifted, None, values_shifted
        )

        # Test at a different set of points to avoid testing only at grid nodes
        eval_theta = np.linspace(0.2, np.pi - 0.2, 8) * u.rad
        eval_phi = np.linspace(0.1, 2 * np.pi - 0.1, 10) * u.rad
        result_orig = interpolator(eval_theta[:, np.newaxis], eval_phi)
        result_shifted = interpolator_shifted(eval_theta[:, np.newaxis], eval_phi)
        assert_quantity_allclose(
            result_orig, result_shifted, atol=1e-12 * u.dimensionless
        )

    def test_phi_periodicity_and_rotation_invariance_3d(self):
        """Test φ modulo invariance and rotation invariance for 3D patterns."""
        N, M, K = 12, 16, 4
        theta = np.linspace(0.1, np.pi - 0.1, N) * u.rad
        phi = np.linspace(0, 2 * np.pi, M, endpoint=False) * u.rad
        frequency = (np.arange(K) + 1) * u.GHz

        np.random.seed(123)
        values = (
            np.random.randn(N, M, K) + 1j * np.random.randn(N, M, K)
        ) * u.dimensionless
        interpolator = _ComplexInterpolator(theta, phi, frequency, values)

        # Test φ modulo invariance: f(θ, φ, f) == f(θ, φ + 2π, f)
        result1 = interpolator(theta[:, np.newaxis], phi, frequency[0])
        result2 = interpolator(
            theta[:, np.newaxis], phi + 2 * np.pi * u.rad, frequency[0]
        )
        assert_quantity_allclose(result1, result2, atol=1e-12 * u.dimensionless)

        # Test φ rotation invariance: shifting grid and data should give same results
        shift = 3
        phi_shifted = np.roll(phi.value, shift) * u.rad
        values_shifted = np.roll(values.value, shift, axis=1) * u.dimensionless
        interpolator_shifted = _ComplexInterpolator(
            theta, phi_shifted, frequency, values_shifted
        )

        # Test at evaluation points
        eval_theta = np.linspace(0.2, np.pi - 0.2, 6) * u.rad
        eval_phi = np.linspace(0.1, 2 * np.pi - 0.1, 8) * u.rad
        result_orig = interpolator(eval_theta[:, np.newaxis], eval_phi, frequency[1])
        result_shifted = interpolator_shifted(
            eval_theta[:, np.newaxis], eval_phi, frequency[1]
        )
        assert_quantity_allclose(
            result_orig, result_shifted, atol=1e-12 * u.dimensionless
        )

    def test_complex_equivariance_properties_2d(self):
        """Test scaling and conjugation equivariance properties for 2D patterns."""
        N, M = 8, 12
        theta = np.linspace(0.2, np.pi - 0.2, N) * u.rad
        phi = np.linspace(0, 2 * np.pi, M, endpoint=False) * u.rad

        np.random.seed(456)
        values = (np.random.randn(N, M) + 1j * np.random.randn(N, M)) * u.dimensionless
        c = 1.2 - 0.7j
        test_theta = np.linspace(0.3, np.pi - 0.3, 5) * u.rad
        test_phi = np.linspace(0.1, 2 * np.pi - 0.1, 7) * u.rad

        # Test complex scaling: Interp(c * V) == c * Interp(V)
        interpolator_orig = _ComplexInterpolator(theta, phi, None, values)
        interpolator_scaled = _ComplexInterpolator(theta, phi, None, c * values)

        result_scaled = interpolator_scaled(test_theta[:, np.newaxis], test_phi)
        expected_scaled = c * interpolator_orig(test_theta[:, np.newaxis], test_phi)
        assert_quantity_allclose(
            result_scaled, expected_scaled, atol=1e-12 * u.dimensionless
        )

        # Test conjugation: Interp(conj(V)) == conj(Interp(V))
        interpolator_conj = _ComplexInterpolator(theta, phi, None, np.conj(values))

        result_conj = interpolator_conj(test_theta[:, np.newaxis], test_phi)
        expected_conj = np.conj(interpolator_orig(test_theta[:, np.newaxis], test_phi))
        assert_quantity_allclose(
            result_conj, expected_conj, atol=1e-12 * u.dimensionless
        )

    def test_complex_equivariance_properties_3d(self):
        """Test scaling and conjugation equivariance properties for 3D patterns."""
        N, M, K = 8, 12, 3
        theta = np.linspace(0.2, np.pi - 0.2, N) * u.rad
        phi = np.linspace(0, 2 * np.pi, M, endpoint=False) * u.rad
        frequency = (np.arange(K) + 1) * u.GHz

        np.random.seed(456)
        values = (
            np.random.randn(N, M, K) + 1j * np.random.randn(N, M, K)
        ) * u.dimensionless
        c = 1.2 - 0.7j
        test_theta = np.linspace(0.3, np.pi - 0.3, 5) * u.rad
        test_phi = np.linspace(0.1, 2 * np.pi - 0.1, 7) * u.rad

        interpolator_orig = _ComplexInterpolator(theta, phi, frequency, values)
        interpolator_scaled = _ComplexInterpolator(theta, phi, frequency, c * values)
        interpolator_conj = _ComplexInterpolator(theta, phi, frequency, np.conj(values))

        # Test complex scaling: Interp(c * V) == c * Interp(V)
        result_scaled = interpolator_scaled(
            test_theta[:, np.newaxis], test_phi, frequency[1]
        )
        expected_scaled = c * interpolator_orig(
            test_theta[:, np.newaxis], test_phi, frequency[1]
        )
        assert_quantity_allclose(
            result_scaled, expected_scaled, atol=1e-12 * u.dimensionless
        )

        # Test conjugation: Interp(conj(V)) == conj(Interp(V))
        result_conj = interpolator_conj(
            test_theta[:, np.newaxis], test_phi, frequency[1]
        )
        expected_conj = np.conj(
            interpolator_orig(test_theta[:, np.newaxis], test_phi, frequency[1])
        )
        assert_quantity_allclose(
            result_conj, expected_conj, atol=1e-12 * u.dimensionless
        )

    def test_2d_3d_consistency_with_single_frequency(self):
        """Test that 2D and 3D interpolators agree when K=1."""
        N, M = 8, 12
        theta = np.linspace(0.1, np.pi - 0.1, N) * u.rad
        phi = np.linspace(0, 2 * np.pi, M, endpoint=False) * u.rad

        np.random.seed(789)
        values_2d = (
            np.random.randn(N, M) + 1j * np.random.randn(N, M)
        ) * u.dimensionless
        frequency_single = np.array([1.5]) * u.GHz
        values_3d = values_2d[..., np.newaxis]

        interpolator_2d = _ComplexInterpolator(theta, phi, None, values_2d)
        interpolator_3d = _ComplexInterpolator(theta, phi, frequency_single, values_3d)

        # Test at evaluation points
        eval_theta = np.linspace(0.2, np.pi - 0.2, 7) * u.rad
        eval_phi = np.linspace(0.1, 2 * np.pi - 0.1, 9) * u.rad

        result_2d = interpolator_2d(eval_theta[:, np.newaxis], eval_phi)
        result_3d = interpolator_3d(
            eval_theta[:, np.newaxis], eval_phi, frequency_single[0]
        )

        assert_quantity_allclose(result_2d, result_3d, atol=1e-12 * u.dimensionless)

    def test_frequency_endpoint_exactness(self):
        """Test that interpolation is exact at frequency endpoints."""
        N, M, K = 7, 9, 4
        theta = np.linspace(0.1, np.pi - 0.1, N) * u.rad
        phi = np.linspace(0, 2 * np.pi, M, endpoint=False) * u.rad
        frequency = (np.arange(K) + 1) * u.GHz

        np.random.seed(101112)
        values = (
            np.random.randn(N, M, K) + 1j * np.random.randn(N, M, K)
        ) * u.dimensionless

        interpolator = _ComplexInterpolator(theta, phi, frequency, values)

        # Test exactness at frequency endpoints for arbitrary theta, phi
        eval_theta = np.linspace(0.15, np.pi - 0.15, 6) * u.rad
        eval_phi = np.linspace(0.05, 2 * np.pi - 0.05, 8) * u.rad

        # At minimum frequency
        result_min = interpolator(eval_theta[:, np.newaxis], eval_phi, frequency[0])
        expected_min = _ComplexInterpolator(theta, phi, None, values[..., 0])(
            eval_theta[:, np.newaxis], eval_phi
        )
        assert_quantity_allclose(result_min, expected_min, atol=1e-12 * u.dimensionless)

        # At maximum frequency
        result_max = interpolator(eval_theta[:, np.newaxis], eval_phi, frequency[-1])
        expected_max = _ComplexInterpolator(theta, phi, None, values[..., -1])(
            eval_theta[:, np.newaxis], eval_phi
        )
        assert_quantity_allclose(result_max, expected_max, atol=1e-12 * u.dimensionless)

    def test_convergence_under_refinement(self):
        """Test that interpolation error decreases monotonically with refinement."""

        def smooth_pattern(theta, phi):
            """Smooth 2π-periodic complex function for testing convergence."""
            th = theta.to_value(u.rad)
            ph = phi.to_value(u.rad)
            amplitude = 1 + 0.2 * np.sin(th) + 0.1 * np.sin(2 * ph)
            phase = 0.3 * th + 0.4 * np.sin(ph)
            return amplitude * np.exp(1j * phase) * u.dimensionless

        # Create evaluation points (avoiding boundaries for robustness)
        eval_theta = np.linspace(0.1, np.pi - 0.1, 15) * u.rad
        eval_phi = np.linspace(0.05, 2 * np.pi - 0.05, 20) * u.rad
        true_values = smooth_pattern(eval_theta[:, np.newaxis], eval_phi)

        errors = []
        grid_sizes = [(12, 16), (24, 32)]

        for N, M in grid_sizes:
            theta = np.linspace(0.05, np.pi - 0.05, N) * u.rad
            phi = np.linspace(0, 2 * np.pi, M, endpoint=False) * u.rad
            values = smooth_pattern(theta[:, np.newaxis], phi)

            interpolator = _ComplexInterpolator(theta, phi, None, values)
            result = interpolator(eval_theta[:, np.newaxis], eval_phi)

            error = np.abs(result - true_values).to_value(u.dimensionless).mean()
            errors.append(error)

        # Error should decrease with refinement (no specific rate assumed)
        assert errors[1] < errors[0], (
            f"Error did not decrease: {errors[0]:.2e} -> {errors[1]:.2e}"
        )

    def test_phase_continuity(self):
        theta = np.linspace(0, np.pi, 11) * u.rad
        phi = np.linspace(0, 2 * np.pi, 21, endpoint=False) * u.rad

        # Create a pattern with phase equal to 2*phi. This will cause two wraps from
        # 2π back to 0, one in the middle of the grid where phi == π and another at the
        # extreme edges of the grid where phi == 0 and phi == 2π. We expect the
        # interpolator to handle both of these correctly.
        values = (
            np.ones_like(theta.value[:, np.newaxis])
            * np.exp(1j * 2 * phi.value)
            * u.dimensionless
        )

        interpolator = _ComplexInterpolator(theta, phi, None, values)

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
            atol=1e-2,
        )

    def test_out_of_bounds_theta_phi_raise(self):
        # Create subset grids so bounds checking is meaningful
        theta = np.linspace(0.2 * np.pi, 0.8 * np.pi, 10) * u.rad
        phi = np.linspace(0.25 * np.pi, 1.25 * np.pi, 12, endpoint=False) * u.rad
        values = np.ones((10, 12)) * u.dimensionless

        interpolator = _ComplexInterpolator(theta, phi, None, values)

        # phi below range (after modulo) → error
        with pytest.raises(ValueError):
            interpolator(theta[0], (0.0 * u.rad))

        # theta above range → error
        with pytest.raises(ValueError):
            interpolator(0.95 * np.pi * u.rad, phi[0])

    def test_3d_interpolator_requires_frequency(self):
        """Test that 3D interpolator raises error when called without frequency."""
        theta = np.linspace(0, np.pi, 10) * u.rad
        phi = np.linspace(0, 2 * np.pi, 12, endpoint=False) * u.rad
        frequency = np.array([1.0, 2.0]) * u.GHz
        values = np.ones((10, 12, 2)) * u.dimensionless

        interpolator = _ComplexInterpolator(theta, phi, frequency, values)

        with pytest.raises(ValueError):
            interpolator(theta[0], phi[0])

    @pytest.mark.parametrize("f_query", [0.5 * u.GHz, 2.5 * u.GHz])
    def test_3d_frequency_out_of_bounds_raises(self, f_query):
        theta = np.linspace(0, np.pi, 6) * u.rad
        phi = np.linspace(0, 2 * np.pi, 8, endpoint=False) * u.rad
        frequency = np.array([1.0, 2.0]) * u.GHz
        values = np.ones((6, 8, 2)) * u.dimensionless

        interpolator = _ComplexInterpolator(theta, phi, frequency, values)
        with pytest.raises(ValueError):
            interpolator(theta[0], phi[0], f_query)

    def test_phi_wrapping_negative_to_positive_range(self):
        """Test phi ranging from -180° to +180° works."""
        theta = np.linspace(0, np.pi, 20) * u.rad
        phi = np.linspace(-180, 180, 50) * u.deg
        values = np.ones((theta.size, phi.size)) * u.dimensionless

        interpolator = _ComplexInterpolator(theta, phi, None, values)

        test_theta = np.pi / 2 * u.rad
        test_phi = np.array([-90, 0, 90]) * u.deg
        result = interpolator(test_theta, test_phi)

        expected = np.ones(3) * u.dimensionless
        assert_quantity_allclose(result, expected)


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
                frequency=None,
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
                frequency=None,
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
                frequency=None,
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
                frequency=None,
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
    shape_interp = (40, 80)
    theta_interp = np.linspace(0, np.pi, shape_interp[0]) * u.rad
    phi_interp = np.linspace(0, 2 * np.pi, shape_interp[1]) * u.rad

    pol_theta = Polarization(0 * u.rad, np.inf * u.dimensionless, Handedness.LEFT)
    pol_phi = Polarization(np.pi / 2 * u.rad, np.inf * u.dimensionless, Handedness.LEFT)

    assert_quantity_allclose(
        to_linear(
            test_case.pattern.gain(
                theta_interp[:, np.newaxis],
                phi_interp,
                polarization=Polarization.lhcp(),
            )
        ),
        test_case.expected_results.lhcp_gain,
        atol=1e-10 * u.dimensionless,
    )

    assert_quantity_allclose(
        to_linear(
            test_case.pattern.directivity(
                theta_interp[:, np.newaxis],
                phi_interp,
                polarization=Polarization.lhcp(),
            )
        ),
        test_case.expected_results.lhcp_directivity,
        atol=1e-10 * u.dimensionless,
    )

    assert_quantity_allclose(
        to_linear(
            test_case.pattern.gain(
                theta_interp[:, np.newaxis],
                phi_interp,
                polarization=Polarization.rhcp(),
            )
        ),
        test_case.expected_results.rhcp_gain,
        atol=1e-10 * u.dimensionless,
    )

    assert_quantity_allclose(
        to_linear(
            test_case.pattern.directivity(
                theta_interp[:, np.newaxis],
                phi_interp,
                polarization=Polarization.rhcp(),
            )
        ),
        test_case.expected_results.rhcp_directivity,
        atol=1e-10 * u.dimensionless,
    )

    assert_quantity_allclose(
        to_linear(
            test_case.pattern.directivity(
                theta_interp[:, np.newaxis], phi_interp, polarization=pol_theta
            )
        ),
        test_case.expected_results.theta_directivity,
        atol=1e-10 * u.dimensionless,
    )

    assert_quantity_allclose(
        to_linear(
            test_case.pattern.directivity(
                theta_interp[:, np.newaxis], phi_interp, polarization=pol_phi
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


class TestRadiationPatternDefaults:
    def _simple_pattern(self, default_pol=None) -> RadiationPattern:
        theta = np.linspace(0, np.pi, 10) * u.rad
        phi = np.linspace(0, 2 * np.pi, 12, endpoint=False) * u.rad
        e_theta = np.ones((10, 12)) * u.dimensionless
        e_phi = np.zeros((10, 12)) * u.dimensionless
        return RadiationPattern(
            theta=theta,
            phi=phi,
            frequency=None,
            e_theta=e_theta,
            e_phi=e_phi,
            rad_efficiency=0.8 * u.dimensionless,
            default_polarization=default_pol,
        )

    def test_methods_use_default_polarization_when_none(self):
        pattern = self._simple_pattern(default_pol=Polarization.lhcp())
        theta = np.linspace(0, np.pi, 8) * u.rad
        phi = np.linspace(0, 2 * np.pi, 9, endpoint=False) * u.rad

        # Compare explicit vs default for e_field, directivity, gain
        e_explicit = pattern.e_field(
            theta[:, np.newaxis], phi, polarization=Polarization.lhcp()
        )
        e_default = pattern.e_field(theta[:, np.newaxis], phi)
        assert_quantity_allclose(e_default, e_explicit)

        d_explicit = pattern.directivity(
            theta[:, np.newaxis], phi, polarization=Polarization.lhcp()
        )
        d_default = pattern.directivity(theta[:, np.newaxis], phi)
        assert_quantity_allclose(d_default, d_explicit)

        g_explicit = pattern.gain(
            theta[:, np.newaxis], phi, polarization=Polarization.lhcp()
        )
        g_default = pattern.gain(theta[:, np.newaxis], phi)
        assert_quantity_allclose(g_default, g_explicit)

    def test_missing_polarization_raises_when_no_default(self):
        pattern = self._simple_pattern(default_pol=None)
        theta = np.linspace(0, np.pi, 3) * u.rad
        phi = np.linspace(0, 2 * np.pi, 4, endpoint=False) * u.rad

        with pytest.raises(ValueError):
            pattern.e_field(theta[:, np.newaxis], phi)
        with pytest.raises(ValueError):
            pattern.directivity(theta[:, np.newaxis], phi)
        with pytest.raises(ValueError):
            pattern.gain(theta[:, np.newaxis], phi)

    def test_factories_forward_default_polarization(self):
        theta = np.linspace(0, np.pi, 6) * u.rad
        phi = np.linspace(0, 2 * np.pi, 7, endpoint=False) * u.rad

        # Circular factory
        e_lhcp = (1.0 + 0.0j) * np.ones((6, 7)) * u.dimensionless
        e_rhcp = (0.0 + 0.0j) * np.ones((6, 7)) * u.dimensionless
        pat_circ = RadiationPattern.from_circular_e_field(
            theta=theta,
            phi=phi,
            frequency=None,
            e_lhcp=e_lhcp,
            e_rhcp=e_rhcp,
            rad_efficiency=1.0 * u.dimensionless,
            default_polarization=Polarization.lhcp(),
        )
        # Works without explicit polarization
        _ = pat_circ.gain(theta[:, np.newaxis], phi)

        # Linear factory
        gain_theta = np.ones((6, 7)) * u.dimensionless
        gain_phi = np.zeros((6, 7)) * u.dimensionless
        phase_theta = np.zeros((6, 7)) * u.rad
        phase_phi = np.zeros((6, 7)) * u.rad
        pat_lin = RadiationPattern.from_linear_gain(
            theta=theta,
            phi=phi,
            frequency=None,
            gain_theta=gain_theta,
            gain_phi=gain_phi,
            phase_theta=phase_theta,
            phase_phi=phase_phi,
            rad_efficiency=0.9 * u.dimensionless,
            default_polarization=Polarization.lhcp(),
        )
        # Works without explicit polarization
        _ = pat_lin.directivity(theta[:, np.newaxis], phi)

    @pytest.mark.parametrize("with_default", [False, True])
    @pytest.mark.parametrize(
        "method", ["e_field", "directivity", "gain", "axial_ratio"]
    )
    def test_default_frequency_semantics_3d(self, with_default, method):
        theta = np.linspace(0, np.pi, 5) * u.rad
        phi = np.linspace(0, 2 * np.pi, 6, endpoint=False) * u.rad
        freq = np.array([1.0, 2.0]) * u.GHz
        e_theta = np.ones((5, 6, 2)) * u.dimensionless
        e_phi = np.zeros((5, 6, 2)) * u.dimensionless

        kwargs = {}
        if with_default:
            kwargs["default_frequency"] = 2.0 * u.GHz

        pat = RadiationPattern(
            theta=theta,
            phi=phi,
            frequency=freq,
            e_theta=e_theta,
            e_phi=e_phi,
            rad_efficiency=1.0 * u.dimensionless,
            **kwargs,
        )

        pol = Polarization.lhcp()
        func = getattr(pat, method)

        if method == "axial_ratio":
            if with_default:
                _ = func(theta[:, np.newaxis], phi)
            else:
                with pytest.raises(ValueError):
                    _ = func(theta[:, np.newaxis], phi)
        else:
            if with_default:
                # Should not raise when frequency omitted
                _ = func(theta[:, np.newaxis], phi, polarization=pol)
            else:
                with pytest.raises(ValueError):
                    _ = func(theta[:, np.newaxis], phi, polarization=pol)


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
            RadiationPattern(theta_negative, phi, None, e_theta, e_phi, rad_efficiency)

        # Test theta > pi
        theta_too_large = np.linspace(0, np.pi + 0.1, 5) * u.rad
        with pytest.raises(ValueError):
            RadiationPattern(theta_too_large, phi, None, e_theta, e_phi, rad_efficiency)

    @pytest.mark.parametrize(
        "axis,case,values",
        [
            ("theta", "unsorted", np.array([0, 0.5, 0.3, 1.0, 1.5])),
            ("theta", "duplicate", np.array([0, 0.5, 0.5, 1.0, 1.5])),
            ("theta", "unequal", np.array([0, 0.5, 1.2, 2.0, 3.0])),
            ("phi", "unsorted", np.array([0, 1.0, 0.5, 2.0, 3.0])),
            ("phi", "duplicate", np.array([0, 1.0, 1.0, 2.0, 3.0])),
            ("phi", "unequal", np.array([0, 0.5, 1.2, 2.0, 3.0])),
        ],
    )
    def test_axis_order_and_spacing_validation(self, axis, case, values):
        """Parametrized validation for axis ordering and equal spacing."""
        rad_efficiency = 1.0 * u.dimensionless
        if axis == "theta":
            theta = values * u.rad
            phi = np.linspace(0, 2 * np.pi, 10, endpoint=False) * u.rad
            e_theta = np.ones((values.size, phi.size)) * u.dimensionless
            e_phi = np.zeros((values.size, phi.size)) * u.dimensionless
            with pytest.raises(ValueError):
                RadiationPattern(theta, phi, None, e_theta, e_phi, rad_efficiency)
        else:
            theta = np.linspace(0, np.pi, values.size) * u.rad
            phi = values * u.rad
            e_theta = np.ones((theta.size, values.size)) * u.dimensionless
            e_phi = np.zeros((theta.size, values.size)) * u.dimensionless
            with pytest.raises(ValueError):
                RadiationPattern(theta, phi, None, e_theta, e_phi, rad_efficiency)

    def test_phi_coverage_validation(self):
        """Test that phi must cover 2π or less radians."""
        theta = np.linspace(0, np.pi, 5) * u.rad
        e_theta = np.ones((5, 10)) * u.dimensionless
        e_phi = np.zeros((5, 10)) * u.dimensionless
        rad_efficiency = 1.0 * u.dimensionless

        # Test phi covering more than the allowed threshold (2π + tolerance)
        phi_just_over_limit = np.linspace(0, 2.2 * np.pi, 10) * u.rad
        with pytest.raises(ValueError):
            RadiationPattern(
                theta, phi_just_over_limit, None, e_theta, e_phi, rad_efficiency
            )

        # Test phi covering more than 2π
        phi_too_much = np.linspace(0, 2.5 * np.pi, 10) * u.rad
        with pytest.raises(ValueError):
            RadiationPattern(theta, phi_too_much, None, e_theta, e_phi, rad_efficiency)

    @pytest.mark.parametrize(
        "eta",
        [
            0.0 * u.dimensionless,  # boundary
            1.1 * u.dimensionless,  # > 1
            np.array([0.8, 0.9]) * u.dimensionless,  # non-scalar
        ],
    )
    def test_rad_efficiency_validation(self, eta):
        theta = np.linspace(0, np.pi, 3) * u.rad
        phi = np.linspace(0, 2 * np.pi, 4, endpoint=False) * u.rad
        e_theta = np.ones((3, 4)) * u.dimensionless
        e_phi = np.zeros((3, 4)) * u.dimensionless
        with pytest.raises(ValueError):
            RadiationPattern(theta, phi, None, e_theta, e_phi, eta)

    def test_surface_integral_exceeds_4pi_raises(self):
        # Construct fields whose directivity integrates to > 4π
        theta = np.linspace(0, np.pi, 10) * u.rad
        phi = np.linspace(0, 2 * np.pi, 12, endpoint=False) * u.rad
        scale = np.sqrt(2.0)
        e_theta = scale * np.ones((10, 12)) * u.dimensionless
        e_phi = np.zeros((10, 12)) * u.dimensionless
        with pytest.raises(ValueError):
            RadiationPattern(theta, phi, None, e_theta, e_phi, 1.0 * u.dimensionless)

    def test_surface_integral_validation_3d_exceeds_4pi(self):
        """Test surface integral validation for 3D patterns that exceed 4π."""
        theta = np.linspace(0, np.pi, 6) * u.rad
        phi = np.linspace(0, 2 * np.pi, 8, endpoint=False) * u.rad
        frequency = np.array([1.0, 2.0]) * u.GHz

        # Create 3D fields whose directivity exceeds 4π at some frequency
        scale = np.sqrt(2.0)
        e_theta_3d = scale * np.ones((6, 8, 2)) * u.dimensionless
        e_phi_3d = np.zeros((6, 8, 2)) * u.dimensionless

        with pytest.raises(ValueError):
            RadiationPattern(
                theta, phi, frequency, e_theta_3d, e_phi_3d, 1.0 * u.dimensionless
            )

    def test_3d_frequency_validation(self):
        """Test validation errors for 3D frequency-dependent patterns."""
        theta = np.linspace(0, np.pi, 5) * u.rad
        phi = np.linspace(0, 2 * np.pi, 6, endpoint=False) * u.rad
        e_theta_3d = np.ones((5, 6, 3)) * u.dimensionless
        e_phi_3d = np.zeros((5, 6, 3)) * u.dimensionless
        rad_efficiency = 1.0 * u.dimensionless

        # Empty frequency array
        with pytest.raises(ValueError):
            RadiationPattern(
                theta, phi, np.array([]) * u.GHz, e_theta_3d, e_phi_3d, rad_efficiency
            )

        # Non-increasing frequency values
        frequency_decreasing = np.array([3.0, 2.0, 1.0]) * u.GHz
        with pytest.raises(ValueError):
            RadiationPattern(
                theta, phi, frequency_decreasing, e_theta_3d, e_phi_3d, rad_efficiency
            )

        # Wrong 3D array shape
        frequency = np.array([1.0, 2.0, 3.0]) * u.GHz
        e_theta_wrong_shape = (
            np.ones((5, 6, 2)) * u.dimensionless
        )  # Wrong frequency dimension
        with pytest.raises(ValueError):
            RadiationPattern(
                theta, phi, frequency, e_theta_wrong_shape, e_phi_3d, rad_efficiency
            )

    def test_2d_array_shape_validation(self):
        """Test validation error for wrong 2D array shapes."""
        theta = np.linspace(0, np.pi, 5) * u.rad
        phi = np.linspace(0, 2 * np.pi, 6, endpoint=False) * u.rad
        rad_efficiency = 1.0 * u.dimensionless

        # Wrong 2D array shape
        e_theta_wrong = np.ones((4, 6)) * u.dimensionless  # Wrong theta dimension
        e_phi = np.zeros((5, 6)) * u.dimensionless
        with pytest.raises(ValueError):
            RadiationPattern(theta, phi, None, e_theta_wrong, e_phi, rad_efficiency)

    def test_factories_negative_gain_raise(self):
        theta = np.linspace(0, np.pi, 6) * u.rad
        phi = np.linspace(0, 2 * np.pi, 7, endpoint=False) * u.rad

        # Circular gain negative
        gain_l = np.ones((6, 7)) * u.dimensionless
        gain_r = np.ones((6, 7)) * u.dimensionless
        gain_l[0, 0] = -1 * u.dimensionless
        with pytest.raises(ValueError):
            RadiationPattern.from_circular_gain(
                theta=theta,
                phi=phi,
                frequency=None,
                gain_lhcp=gain_l,
                gain_rhcp=gain_r,
                phase_lhcp=np.zeros((6, 7)) * u.rad,
                phase_rhcp=np.zeros((6, 7)) * u.rad,
                rad_efficiency=1.0 * u.dimensionless,
            )

        # Linear gain negative
        g_th = np.ones((6, 7)) * u.dimensionless
        g_ph = np.ones((6, 7)) * u.dimensionless
        g_ph[0, 0] = -1 * u.dimensionless
        with pytest.raises(ValueError):
            RadiationPattern.from_linear_gain(
                theta=theta,
                phi=phi,
                frequency=None,
                gain_theta=g_th,
                gain_phi=g_ph,
                phase_theta=np.zeros((6, 7)) * u.rad,
                phase_phi=np.zeros((6, 7)) * u.rad,
                rad_efficiency=1.0 * u.dimensionless,
            )

    def test_default_frequency_3d_validation(self):
        frequency = np.array([1.0, 2.0]) * u.GHz
        base_kwargs = {
            "theta": np.linspace(0, np.pi, 4) * u.rad,
            "phi": np.linspace(0, 2 * np.pi, 5, endpoint=False) * u.rad,
            "frequency": frequency,
            "e_theta": np.ones((4, 5, 2)) * u.dimensionless,
            "e_phi": np.zeros((4, 5, 2)) * u.dimensionless,
            "rad_efficiency": 1.0 * u.dimensionless,
        }

        # Accept: min, max, and mid-point
        for good in [frequency.min(), frequency.max(), 1.5 * u.GHz]:
            _ = RadiationPattern(**base_kwargs, default_frequency=good)
            pat = RadiationPattern(**base_kwargs)
            pat.default_frequency = good

        # Reject: out-of-bounds and non-scalar
        for bad in [0.5 * u.GHz, 2.5 * u.GHz, np.array([1.0, 1.5]) * u.GHz]:
            with pytest.raises(ValueError):
                _ = RadiationPattern(**base_kwargs, default_frequency=bad)
            pat = RadiationPattern(**base_kwargs)
            with pytest.raises(ValueError):
                pat.default_frequency = bad


class TestRadiationPatternFactoryConstructors:
    def _grid(self):
        # Avoid endpoints to match interpolator constraints
        N, M = 12, 16
        theta = np.linspace(0.1, np.pi - 0.1, N) * u.rad
        phi = np.linspace(0, 2 * np.pi, M, endpoint=False) * u.rad
        return theta, phi

    def test_from_circular_e_field(self):
        theta, phi = self._grid()
        # Non-trivial phase across phi, constant over theta
        e_lhcp = (
            np.exp(1j * phi.value)[np.newaxis, :] * np.ones((theta.size, 1))
        ) * u.dimensionless
        e_rhcp = np.zeros((theta.size, phi.size)) * u.dimensionless

        pat = RadiationPattern.from_circular_e_field(
            theta=theta,
            phi=phi,
            frequency=None,
            e_lhcp=e_lhcp,
            e_rhcp=e_rhcp,
            rad_efficiency=0.65 * u.dimensionless,
        )

        pol_lhcp = Polarization.lhcp()
        pol_rhcp = Polarization.rhcp()
        pol_theta = Polarization(0 * u.rad, np.inf * u.dimensionless, Handedness.LEFT)
        pol_phi = Polarization(
            np.pi / 2 * u.rad, np.inf * u.dimensionless, Handedness.LEFT
        )

        # LHCP directivity should be 1, RHCP should be 0, linear components 0.5 and 0.5
        dir_lhcp = to_linear(
            pat.directivity(theta[:, np.newaxis], phi, polarization=pol_lhcp)
        )
        dir_rhcp = to_linear(
            pat.directivity(theta[:, np.newaxis], phi, polarization=pol_rhcp)
        )
        dir_theta = to_linear(
            pat.directivity(theta[:, np.newaxis], phi, polarization=pol_theta)
        )
        dir_phi = to_linear(
            pat.directivity(theta[:, np.newaxis], phi, polarization=pol_phi)
        )

        assert_quantity_allclose(dir_lhcp, 1.0 * u.dimensionless)
        assert_quantity_allclose(
            dir_rhcp, 0.0 * u.dimensionless, atol=1e-10 * u.dimensionless
        )
        assert_quantity_allclose(dir_theta, 0.5 * u.dimensionless)
        assert_quantity_allclose(dir_phi, 0.5 * u.dimensionless)

        # Gain should be eta * directivity (in linear)
        _assert_lin_gain_equals_eta_times_directivity(
            pat,
            theta[:, np.newaxis],
            phi,
            polarization=pol_lhcp,
            eta=0.65 * u.dimensionless,
        )

        # Phase of LHCP e-field should match exp(1j*phi)
        e_l = pat.e_field(theta[:, np.newaxis], phi, polarization=pol_lhcp)
        phase_err = np.angle(e_l.value * np.exp(-1j * phi.value))
        np.testing.assert_allclose(phase_err, 0.0, atol=1e-10)

    def test_from_circular_gain(self):
        theta, phi = self._grid()
        gain_lhcp = 0.25 * np.ones((theta.size, phi.size)) * u.dimensionless
        gain_rhcp = np.zeros((theta.size, phi.size)) * u.dimensionless
        # Give LHCP a varying phase; directivity is phase-independent
        phase_lhcp = (
            np.linspace(0, 2 * np.pi, phi.size)[np.newaxis, :]
            * np.ones((theta.size, 1))
            * u.rad
        )
        phase_rhcp = np.zeros((theta.size, phi.size)) * u.rad

        pat = RadiationPattern.from_circular_gain(
            theta=theta,
            phi=phi,
            frequency=None,
            gain_lhcp=gain_lhcp,
            gain_rhcp=gain_rhcp,
            phase_lhcp=phase_lhcp,
            phase_rhcp=phase_rhcp,
            rad_efficiency=0.75 * u.dimensionless,
        )

        pol_lhcp = Polarization.lhcp()
        pol_rhcp = Polarization.rhcp()
        dir_lhcp = to_linear(
            pat.directivity(theta[:, np.newaxis], phi, polarization=pol_lhcp)
        )
        dir_rhcp = to_linear(
            pat.directivity(theta[:, np.newaxis], phi, polarization=pol_rhcp)
        )
        assert_quantity_allclose(dir_lhcp, (0.25 / 0.75) * u.dimensionless)
        assert_quantity_allclose(
            dir_rhcp, 0.0 * u.dimensionless, atol=1e-10 * u.dimensionless
        )
        _assert_lin_gain_equals_eta_times_directivity(
            pat,
            theta[:, np.newaxis],
            phi,
            polarization=pol_lhcp,
            eta=0.75 * u.dimensionless,
        )

        # Phase of LHCP e-field should match provided phase_lhcp
        e_l = pat.e_field(theta[:, np.newaxis], phi, polarization=pol_lhcp)
        phase_err = np.angle(e_l.value * np.exp(-1j * phase_lhcp.value))
        np.testing.assert_allclose(phase_err, 0.0, atol=1e-10)

    def test_from_linear_gain(self):
        theta, phi = self._grid()
        # Choose gain equal to eta so resulting directivity is 1
        gain_theta = 0.6 * np.ones((theta.size, phi.size)) * u.dimensionless
        gain_phi = np.zeros((theta.size, phi.size)) * u.dimensionless
        # Non-trivial phase that varies in theta
        phase_theta = theta[:, np.newaxis]
        phase_phi = np.zeros((theta.size, phi.size)) * u.rad

        pat = RadiationPattern.from_linear_gain(
            theta=theta,
            phi=phi,
            frequency=None,
            gain_theta=gain_theta,
            gain_phi=gain_phi,
            phase_theta=phase_theta,
            phase_phi=phase_phi,
            rad_efficiency=0.6 * u.dimensionless,
        )

        pol_theta = Polarization(0 * u.rad, np.inf * u.dimensionless, Handedness.LEFT)
        pol_phi = Polarization(
            np.pi / 2 * u.rad, np.inf * u.dimensionless, Handedness.LEFT
        )
        pol_lhcp = Polarization.lhcp()
        dir_theta = to_linear(
            pat.directivity(theta[:, np.newaxis], phi, polarization=pol_theta)
        )
        dir_phi = to_linear(
            pat.directivity(theta[:, np.newaxis], phi, polarization=pol_phi)
        )
        dir_lhcp = to_linear(
            pat.directivity(theta[:, np.newaxis], phi, polarization=pol_lhcp)
        )
        assert_quantity_allclose(dir_theta, 1.0 * u.dimensionless)
        assert_quantity_allclose(
            dir_phi, 0.0 * u.dimensionless, atol=1e-10 * u.dimensionless
        )
        assert_quantity_allclose(dir_lhcp, 0.5 * u.dimensionless)
        _assert_lin_gain_equals_eta_times_directivity(
            pat,
            theta[:, np.newaxis],
            phi,
            polarization=pol_theta,
            eta=0.6 * u.dimensionless,
        )
        _assert_lin_gain_equals_eta_times_directivity(
            pat,
            theta[:, np.newaxis],
            phi,
            polarization=pol_lhcp,
            eta=0.6 * u.dimensionless,
        )

        # Phase of theta-polarized e-field should match phase_theta
        e_th = pat.e_field(theta[:, np.newaxis], phi, polarization=pol_theta)
        phase_err = np.angle(e_th.value * np.exp(-1j * phase_theta.value))
        np.testing.assert_allclose(phase_err, 0.0, atol=1e-10)


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


def test_gain_from_g_over_t():
    # Test where temp is exactly 1 K
    gain = gain_from_g_over_t(10 * u.dB_per_K, 1 * u.K)
    assert_quantity_allclose(gain, 10 * u.dB)

    # Test where temp is greater than 1 K
    gain_100_K = gain_from_g_over_t(10 * u.dB_per_K, 100 * u.K)
    assert_quantity_allclose(gain_100_K, 30 * u.dB)

    # Test for negative temperature
    with pytest.raises(ValueError):
        gain_from_g_over_t(10 * u.dB_per_K, -1 * u.K)


def test_temperature_from_g_over_t():
    # Test temperature calculation from G/T ratio
    temperature = temperature_from_g_over_t(10 * u.dB_per_K, 20 * u.dB)
    assert_quantity_allclose(temperature, 10 * u.K)
