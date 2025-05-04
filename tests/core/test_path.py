"""
Tests for path loss calculations.
"""

import pytest
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose

# Updated imports
from spacelink.core.path import free_space_path_loss, spreading_loss, aperture_loss

# from spacelink.core import units # Unused


# DO NOT MODIFY THIS TEST CASE
@pytest.mark.parametrize(
    "distance, frequency, expected_loss",
    [
        (1.0 * u.km, 1.0 * u.GHz, 92.448 * u.dB),
        (1000 * u.km, 1.0 * u.GHz, 152.448 * u.dB),
        (40_000 * u.km, 1.0 * u.GHz, 184.489 * u.dB),
        (400_000 * u.km, 1.0 * u.GHz, 204.48 * u.dB),
        (400_000 * u.km, 10.0 * u.GHz, 224.48 * u.dB),
        (1_000_000 * u.km, 1.0 * u.GHz, 212.44 * u.dB),
        (1_000_000 * u.km, 10.0 * u.GHz, 232.44 * u.dB),
        (2_000_000 * u.km, 1.0 * u.GHz, 218.46 * u.dB),
        (2_000_000 * u.km, 2.0 * u.GHz, 224.48 * u.dB),
        (2_000_000 * u.km, 2.3 * u.GHz, 225.695 * u.dB),
        (2_000_000 * u.km, 8.4 * u.GHz, 236.946 * u.dB),
        (2_000_000 * u.km, 10.0 * u.GHz, 238.46 * u.dB),
    ],
)
def test_free_space_path_loss(distance, frequency, expected_loss):
    """
    -VALIDATED-
    """
    path_loss = free_space_path_loss(distance, frequency)
    assert_quantity_allclose(path_loss, expected_loss, atol=0.01 * u.dB)


@pytest.mark.parametrize(
    "distance, expected_loss",
    [
        (36000 * u.km, 162.12 * u.dB),  # GEO satellite distance
        (2000 * u.km, 137.01 * u.dB),  # LEO satellite distance
    ],
)
def test_spreading_loss(distance, expected_loss):
    """
    TODO: validate
    """
    assert_quantity_allclose(spreading_loss(distance), expected_loss, atol=0.01 * u.dB)


@pytest.mark.parametrize(
    "frequency, expected_loss",
    [
        (12 * u.GHz, 43.04 * u.dB),  # High frequency case
        (2.25 * u.GHz, 28.50 * u.dB),  # Lower frequency case
    ],
)
def test_aperture_loss(frequency, expected_loss):
    """
    TODO: validate
    """
    assert_quantity_allclose(aperture_loss(frequency), expected_loss, atol=0.01 * u.dB)


@pytest.mark.parametrize(
    "distance, frequency, error_msg",
    [
        (-1000 * u.m, 1 * u.GHz, "Distance must be positive"),
        (1000 * u.m, -1 * u.GHz, "Frequency must be positive"),
        (0 * u.m, 1 * u.GHz, "Distance must be positive"),
        (1000 * u.m, 0 * u.GHz, "Frequency must be positive"),
    ],
)
def test_invalid_parameters(distance, frequency, error_msg):
    with pytest.raises(ValueError, match=error_msg):
        free_space_path_loss(distance, frequency)
