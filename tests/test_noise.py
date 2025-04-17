"""Tests for the noise module."""

import pytest
import pyradio.noise as noise
from pyradio.units import MHz, K

def test_thermal_noise_power():
    """Test thermal noise power calculation."""
    # Test with standard room temperature (290K)
    expected = 1.380649e-23 * 290.0 * 1e6
    assert noise.power(1.0 * MHz).magnitude == pytest.approx(expected)

    # Test with different temperature
    expected = 1.380649e-23 * 100.0 * 1e6
    assert noise.power(1.0 * MHz, 100 * K).magnitude == pytest.approx(expected)

    # Test with zero bandwidth
    assert noise.power(0.0 * MHz).magnitude == 0.0

    # Test with negative bandwidth
    with pytest.raises(ValueError):
        noise.power(-1.0 * MHz)
