"""Tests for the noise module."""

import astropy.units as u
import pytest
from astropy.tests.helper import assert_quantity_allclose

from spacelink.core import noise


def test_thermal_noise_power():
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


def test_noise_power_density():
    expected = 1.380649e-23 * 290.0
    assert_quantity_allclose(
        noise.noise_power_density(290 * u.K), expected * u.W / u.Hz
    )


def test_temperature_to_noise_factor():
    temp = 290 * u.K
    expected_factor = 2.0
    result = noise.temperature_to_noise_factor(temp)
    assert_quantity_allclose(result, expected_factor * u.dimensionless)

    temp = 0 * u.K
    expected_factor = 1.0
    result = noise.temperature_to_noise_factor(temp)
    assert_quantity_allclose(result, expected_factor * u.dimensionless)

    temp = 580 * u.K  # 2 * T0
    expected_factor = 3.0
    result = noise.temperature_to_noise_factor(temp)
    assert_quantity_allclose(result, expected_factor * u.dimensionless)


def test_noise_factor_to_temperature():
    factor = 2.0 * u.dimensionless
    expected_temp = 290 * u.K
    result = noise.noise_factor_to_temperature(factor)
    assert_quantity_allclose(result, expected_temp)

    factor = 1.0 * u.dimensionless
    expected_temp = 0 * u.K
    result = noise.noise_factor_to_temperature(factor)
    assert_quantity_allclose(result, expected_temp)

    # Factor < 1 should raise ValueError
    with pytest.raises(ValueError):
        noise.noise_factor_to_temperature(0.5 * u.dimensionless)


def test_noise_figure_to_temperature():
    noise_figure = 3.0 * u.dB  # Noise factor ~2.0
    expected_temp = 290 * u.K
    result = noise.noise_figure_to_temperature(noise_figure)
    assert_quantity_allclose(result, expected_temp, rtol=1e-2)

    noise_figure = 0.0 * u.dB
    expected_temp = 0 * u.K
    result = noise.noise_figure_to_temperature(noise_figure)
    assert_quantity_allclose(result, expected_temp)


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
    """ """
    noise_w = noise.noise_power(bandwidth, temperature)
    assert_quantity_allclose(
        noise_w.to(u.dB(u.W)), expected_noise_dBW, atol=0.01 * u.dB(u.W)
    )


# Parameterized test with only the validated case
@pytest.mark.parametrize(
    "temperature, expected_noise_figure",
    [
        (290 * u.K, 3.0103 * u.dB),
    ],
)
def test_temperature_to_noise_figure(temperature, expected_noise_figure):
    """Test temperature to noise figure conversion."""
    nf = noise.temperature_to_noise_figure(temperature)
    assert_quantity_allclose(nf, expected_noise_figure, atol=0.01 * u.dB)


@pytest.mark.parametrize(
    "ebn0, bitrate, cn0",
    [
        (10.0 * u.dB(1), 1000 * u.Hz, 40.0 * u.dBHz),  # 10 dB + 30 dBHz = 40 dBHz
        (5.0 * u.dB(1), 1e6 * u.Hz, 65.0 * u.dBHz),  # 5 dB + 60 dBHz = 65 dBHz
        (0.0 * u.dB(1), 100 * u.Hz, 20.0 * u.dBHz),  # 0 dB + 20 dBHz = 20 dBHz
    ],
)
def test_ebn0_cn0_conversions(ebn0, bitrate, cn0):
    """Test Eb/N0 to/from C/N0 conversion functions."""
    cn0_result = noise.ebn0_to_cn0(ebn0, bitrate)
    assert_quantity_allclose(cn0_result, cn0)
    ebn0_result = noise.cn0_to_ebn0(cn0, bitrate)
    assert_quantity_allclose(ebn0_result, ebn0)
