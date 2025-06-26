"""Tests for the noise module."""

import pytest
from typing import Tuple, List
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose
from spacelink.core import noise
from spacelink.core.units import Dimensionless, Decibels, Temperature

# Define type aliases for clarity
LinearStage = Tuple[Dimensionless, Dimensionless]
DbStage = Tuple[Decibels, Decibels]
# from spacelink.core import units # Unused

def assert_decibel_equal(actual, expected, atol=1e-2):
    """
    Assert that two decibel quantities are equal within a tolerance, comparing value and unit string.
    Accepts dB, dB(1), Decibel, etc. Does not use .to() for conversion.
    """
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

def test_thermal_noise_power():
    """Test thermal noise power calculation."""
    # Test with standard room temperature (290K)
    expected = 1.380649e-23 * 290.0 * 1e6
    assert_quantity_allclose(noise.noise_power(1.0 * u.MHz), expected * u.W)

    # Test with different temperature
    expected = 1.380649e-23 * 100.0 * 1e6
    assert_quantity_allclose(noise.noise_power(1.0 * u.MHz, 100 * u.K), expected * u.W)

    # Test with zero bandwidth
    assert_quantity_allclose(noise.noise_power(0.0 * u.MHz), 0.0 * u.W)

    # Test with negative bandwidth
    with pytest.raises(ValueError):
        noise.noise_power(-1.0 * u.MHz)


# DO NOT MODIFY - This test uses validated reference values
@pytest.mark.parametrize(
    "temperature, bandwidth, expected_noise_dBW",
    [
        (50 * u.K, 10 * u.Hz, -201.61 * u.dBW),
        (100 * u.K, 10 * u.Hz, -198.60 * u.dBW),
        (150 * u.K, 10 * u.Hz, -196.84 * u.dBW),
        (200 * u.K, 10 * u.Hz, -195.59 * u.dBW),
        (290 * u.K, 10 * u.Hz, -193.98 * u.dBW),
    ],
)
def test_noise_dBW_conversion(temperature, bandwidth, expected_noise_dBW):
    """
    -VALIDATED-
    """
    noise_w = noise.noise_power(bandwidth, temperature)
    assert_quantity_allclose(
        noise_w.to(u.dB(u.W)), expected_noise_dBW, atol=0.01 * u.dB(u.W)
    )


# Parameterized test with only the validated case
@pytest.mark.parametrize(
    "temperature, expected_noise_figure",
    [
        (290 * u.K, 3.0103 * u.dB),  # Standard room temperature - Validated
        # Add more validated cases here if available
    ],
)
def test_temperature_to_noise_figure(temperature, expected_noise_figure):
    """Test temperature to noise figure conversion."""  # Updated docstring
    nf = noise.temperature_to_noise_figure(temperature)
    assert_decibel_equal(nf, expected_noise_figure)


 