"""Tests for the noise module."""

import pytest
import spacelink.noise as noise
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose


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


def test_noise_dBW_conversion():
    """Test noise power in dBW.

    Temperature	 Bandwidth	    Noise Power	dBW	dBm	dBm/Hz
    50	         10	6.90E-21	-201.61	-171.61	-181.61
    100	         10	1.38E-20	-198.60	-168.60	-178.60
    150	         10	2.07E-20	-196.84	-166.84	-176.84
    200	         10	2.76E-20	-195.59	-165.59	-175.59
    290	         10	4.00E-20	-193.98	-163.98	-173.98
    """
    # Calculate noise power for 10 Hz bandwidth at 290K
    noise_w = noise.power(10.0 * u.Hz)
    # Check the dBW conversion
    assert_quantity_allclose(
        noise_w.to(u.dB(u.W)), -193.98 * u.dB(u.W), atol=0.01 * u.dB(u.W)
    )


# TODO: test cascaded noise values
