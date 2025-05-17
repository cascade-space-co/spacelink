"""
Tests for path loss calculations.
"""

import pytest
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose

# Updated imports
from spacelink.core.path import free_space_path_loss, spreading_loss, aperture_loss

# from spacelink.core import units # Unused

def assert_decibel_equal(actual, expected, atol=1e-2):
    """
    Assert that two decibel quantities are equal within a tolerance, comparing value and unit string.
    Accepts dB, dB(1), Decibel, etc. Does not use .to() for conversion.
    """
    # Accept both Quantity and Decibel types
    actual_val = actual.value if hasattr(actual, 'value') else actual
    expected_val = expected.value if hasattr(expected, 'value') else expected
    actual_unit = str(actual.unit) if hasattr(actual, 'unit') else str(actual)
    expected_unit = str(expected.unit) if hasattr(expected, 'unit') else str(expected)
    # Accept atol as float or Quantity
    if hasattr(atol, 'value'):
        atol = float(atol.value)
    assert actual_unit.startswith('dB') and expected_unit.startswith('dB'), f"Units must be dB-like, got {actual_unit} and {expected_unit}"
    assert abs(actual_val - expected_val) <= atol, f"Values differ: {actual_val} vs {expected_val} (atol={atol})"
    assert actual_unit == expected_unit or actual_unit.startswith('dB') and expected_unit.startswith('dB'), f"Unit strings differ: {actual_unit} vs {expected_unit}"

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
    assert_decibel_equal(path_loss, expected_loss)


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
    assert_decibel_equal(spreading_loss(distance), expected_loss)


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
    assert_decibel_equal(aperture_loss(frequency), expected_loss)


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
