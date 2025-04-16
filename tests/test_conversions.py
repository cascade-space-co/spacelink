"""
Tests for the conversions module.

This module contains pytest-style tests for the dB to linear conversion functions.
"""

import pytest
import numpy as np
from pyradio.conversions import (
    db, db2linear, wavelength,
    ghz, mhz, khz, kilometers
)


def test_db2linear_common_values():
    """Test db2linear with common values."""
    assert db2linear(0) == pytest.approx(1.0)
    assert db2linear(10) == pytest.approx(10.0)
    assert db2linear(-10) == pytest.approx(0.1)


def test_db2linear_fractional():
    """Test db2linear with fractional values."""
    assert db2linear(3) == pytest.approx(1.9952623149688795)
    assert db2linear(-3) == pytest.approx(0.5011872336272722)


def test_db2linear_extreme():
    """Test db2linear with extreme values."""
    assert db2linear(100) == pytest.approx(10000000000.0)
    assert db2linear(-100) == pytest.approx(1e-10)


def test_db_common_values():
    """Test db with common values."""
    assert db(1) == pytest.approx(0.0)
    assert db(10) == pytest.approx(10.0)
    assert db(0.1) == pytest.approx(-10.0)


def test_db_fractional():
    """Test db with fractional values."""
    assert db(2) == pytest.approx(3.0102999566398125)
    assert db(0.5) == pytest.approx(-3.0102999566398125)


def test_db_invalid_values():
    """Test db with invalid values that should raise exceptions."""
    with pytest.raises(ValueError):
        db(0)

    with pytest.raises(ValueError):
        db(-1)


@pytest.mark.parametrize("db_val", [-20, -10, -3, 0, 3, 10, 20])
def test_roundtrip_conversion(db_val):
    """Test that converting from dB to linear and back gives original value."""
    linear = db2linear(db_val)
    db_result = db(linear)
    assert db_val == pytest.approx(db_result)


def test_wavelength():
    """Test wavelength calculations."""
    # Test common frequencies
    assert wavelength(1e9) == pytest.approx(0.299792458)  # 1 GHz
    assert wavelength(2.4e9) == pytest.approx(0.124913524)  # WiFi 2.4 GHz
    assert wavelength(5.8e9) == pytest.approx(0.051688355)  # WiFi 5.8 GHz
    
    # Test with very high and low frequencies
    assert wavelength(1e12) == pytest.approx(0.299792458e-3)  # 1 THz
    assert wavelength(1e6) == pytest.approx(299.792458)  # 1 MHz


def test_wavelength_invalid_values():
    """Test wavelength with invalid values that should raise exceptions."""
    with pytest.raises(ValueError):
        wavelength(0)
    
    with pytest.raises(ValueError):
        wavelength(-1)


def test_ghz():
    """Test GHz to Hz conversion."""
    assert ghz(1.0) == pytest.approx(1e9)
    assert ghz(2.4) == pytest.approx(2.4e9)
    assert ghz(0.5) == pytest.approx(0.5e9)
    assert ghz(10) == pytest.approx(10e9)


def test_mhz():
    """Test MHz to Hz conversion."""
    assert mhz(1.0) == pytest.approx(1e6)
    assert mhz(88.5) == pytest.approx(88.5e6)
    assert mhz(0.5) == pytest.approx(0.5e6)
    assert mhz(100) == pytest.approx(100e6)


def test_khz():
    """Test kHz to Hz conversion."""
    assert khz(1.0) == pytest.approx(1e3)
    assert khz(455.0) == pytest.approx(455e3)
    assert khz(0.5) == pytest.approx(0.5e3)
    assert khz(100) == pytest.approx(100e3)


def test_kilometers():
    """Test kilometers to meters conversion."""
    assert kilometers(1.0) == pytest.approx(1000)
    assert kilometers(36e3) == pytest.approx(36e6)
    assert kilometers(0.5) == pytest.approx(500)
    assert kilometers(100) == pytest.approx(100000)
    
    # Test with negative values
    assert kilometers(-1.0) == pytest.approx(-1000)
    assert kilometers(-0.5) == pytest.approx(-500)
