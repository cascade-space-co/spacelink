"""
Tests for path loss calculations.
"""

import pytest
from pint import Quantity
from pyradio.path import free_space_path_loss, spreading_loss, aperture_loss
from pyradio.units import Q_, m, km, GHz, MHz, Hz, db


def test_free_space_path_loss():
    """Test free space path loss calculations."""
    # Test with GEO satellite parameters
    distance = Q_(36000, km)
    frequency = Q_(12, GHz)  # 12 GHz
    path_loss = free_space_path_loss(distance, frequency)
    assert path_loss.to('dB').magnitude == pytest.approx(-205.16, abs=0.01)

    # Test with LEO satellite parameters
    distance = Q_(2000, km)  # 2,000 km
    frequency = Q_(2.25, GHz)  # 2.25 GHz
    path_loss = free_space_path_loss(distance, frequency)
    assert path_loss.to('dB').magnitude == pytest.approx(-165.51, abs=0.01)
    
    # Test with very short distance and low frequency (to test the range condition)
    distance = Q_(10, m) 
    frequency = Q_(100, MHz)
    path_loss = free_space_path_loss(distance, frequency)
    # Expected value based on the implementation
    # The formula is spreading_loss(distance) * aperture_loss(frequency)
    # For 10m and 100MHz, this is approximately -32.45 dB
    assert path_loss.to('dB').magnitude == pytest.approx(-32.45, abs=0.01)


def test_spreading_loss():
    """Test spreading loss calculations."""
    assert db(spreading_loss(Q_(36000, km)).magnitude) == pytest.approx(-162.12, abs=0.01)
    assert db(spreading_loss(Q_(2000, km)).magnitude) == pytest.approx(-137.01, abs=0.01)


def test_aperture_loss():
    """Test aperture loss calculations."""
    assert db(aperture_loss(Q_(12, GHz)).magnitude) == pytest.approx(-43.04, abs=0.01)
    assert db(aperture_loss(Q_(2.25, GHz)).magnitude) == pytest.approx(-28.50, abs=0.01)


def test_invalid_parameters():
    """Test that invalid parameters raise appropriate errors."""
    # Test with negative distance
    with pytest.raises(ValueError):
        free_space_path_loss(Q_(-1000, m), Q_(1, GHz))

    # Test with negative frequency
    with pytest.raises(ValueError):
        free_space_path_loss(Q_(1000, m), Q_(-1, GHz))

    # Test with zero distance
    with pytest.raises(ValueError):
        free_space_path_loss(Q_(0, m), Q_(1, GHz))

    # Test with zero frequency
    with pytest.raises(ValueError):
        free_space_path_loss(Q_(1000, m), Q_(0, GHz))
