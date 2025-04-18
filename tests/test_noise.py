"""Tests for the noise module."""

import pytest
import pyradio.noise as noise
from pyradio.units import MHz, K, dBW, W

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

def test_noise_dBW_conversion():
    """Test noise power in dBW."""
    # Calculate noise power for 1 MHz bandwidth at 290K
    noise_w = noise.power(1.0 * MHz)
    
    # Convert to dBW
    noise_dbw = noise_w.to(dBW)
    
    # Expected value is approximately -114 dBW for 1 MHz at 290K
    expected_dbw_mag = 1.380649e-23 * 290 * 1e6
    import math
    expected_dbw_log = 10 * math.log10(expected_dbw_mag)
    
    # Verify the calculation directly
    assert noise_w.magnitude == pytest.approx(expected_dbw_mag)
    
    # Check the dBW conversion
    assert noise_dbw.magnitude == pytest.approx(expected_dbw_log, abs=0.1)
