"""Tests for the noise module."""

import pytest
from spacelink.core import noise
import astropy.units as u
# from astropy.units import Quantity # Unused
from astropy.tests.helper import assert_quantity_allclose

# Updated imports
# from spacelink.core import units # Unused


def test_thermal_noise_power():
    """Test thermal noise power calculation."""
    # Test with standard room temperature (290K)
    expected = 1.380649e-23 * 290.0 * 1e6
    assert_quantity_allclose(noise.power(1.0 * u.MHz), expected * u.W)

    # Test with different temperature
    expected = 1.380649e-23 * 100.0 * 1e6
    assert_quantity_allclose(noise.power(1.0 * u.MHz, 100 * u.K), expected * u.W)

    # Test with zero bandwidth
    assert_quantity_allclose(noise.power(0.0 * u.MHz), 0.0 * u.W)

    # Test with negative bandwidth
    with pytest.raises(ValueError):
        noise.power(-1.0 * u.MHz)

# DO NOT MODIFY - This test uses validated reference values
@pytest.mark.parametrize(
    "temperature, bandwidth, expected_noise_dBW",
    [
        (50 * u.K, 10 * u.Hz, -201.61 * u.dB(u.W)),
        (100 * u.K, 10 * u.Hz, -198.60 * u.dB(u.W)),
        (150 * u.K, 10 * u.Hz, -196.84 * u.dB(u.W)),
        (200 * u.K, 10 * u.Hz, -195.59 * u.dB(u.W)),
        (290 * u.K, 10 * u.Hz, -193.98 * u.dB(u.W)),
    ],
)
def test_noise_dBW_conversion(temperature, bandwidth, expected_noise_dBW):
    """
    -VALIDATED-
    """
    noise_w = noise.power(bandwidth, temperature)
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
    """Test temperature to noise figure conversion.""" # Updated docstring
    nf = noise.temperature_to_noise_figure(temperature)
    assert_quantity_allclose(nf, expected_noise_figure, atol=0.01 * u.dB)


# TODO: test cascaded noise values
