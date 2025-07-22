"""Tests for core antenna calculation functions."""

import pytest
import numpy as np
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose

from spacelink.core.antenna import (
    polarization_loss,
    dish_gain,
    Polarization,
    Handedness,
)
from spacelink.core.units import Dimensionless, Length, Frequency, Decibels

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
    tilt_angle, axial_ratio, phase_difference, expected_jones
):
    pol = Polarization(tilt_angle, axial_ratio, phase_difference)
    np.testing.assert_allclose(pol.jones_vector, expected_jones, atol=1e-10)


def test_polarization_factories():
    lhcp = Polarization.lhcp()
    rhcp = Polarization.rhcp()

    np.testing.assert_allclose(lhcp.jones_vector, np.array([1.0, 1.0j]) / np.sqrt(2))
    np.testing.assert_allclose(rhcp.jones_vector, np.array([1.0, -1.0j]) / np.sqrt(2))

    # Orthogonal states should have zero inner product
    inner_product = np.dot(lhcp.jones_vector.conj(), rhcp.jones_vector)
    assert abs(inner_product) == pytest.approx(0.0)
