"""Tests for the noise module."""

import pytest
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose
from spacelink.core import noise


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
    ],
)
def test_temperature_to_noise_figure(temperature, expected_noise_figure):
    """Test temperature to noise figure conversion."""  # Updated docstring
    nf = noise.temperature_to_noise_figure(temperature)
    assert_quantity_allclose(nf, expected_noise_figure, atol=0.01 * u.dB)
