"""
Tests for the conversions module.

This module contains pytest-style tests for the dB to linear conversion functions.
"""

import pytest
import numpy as np
from pyradio.conversions import db, db2linear, wavelength

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