"""
Tests for path loss calculations.
"""

import pytest
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose
from spacelink.path import free_space_path_loss, spreading_loss, aperture_loss


def test_free_space_path_loss():
    """Test free space path loss calculations."""
    # Test with GEO satellite parameters
    distance = 36000 * u.km
    frequency = 12 * u.GHz  # 12 GHz
    path_loss = free_space_path_loss(distance, frequency)
    # Expect positive dB loss
    assert_quantity_allclose(path_loss, 205.16 * u.dB, atol=0.01 * u.dB)

    # Test with LEO satellite parameters
    distance = 2000 * u.km  # 2,000 km
    frequency = 2.25 * u.GHz  # 2.25 GHz
    path_loss = free_space_path_loss(distance, frequency)
    assert_quantity_allclose(path_loss, 165.51 * u.dB, atol=0.01 * u.dB)

    # Test with very short distance and low frequency (to test the range condition)
    distance = 10 * u.m
    frequency = 100 * u.MHz
    path_loss = free_space_path_loss(distance, frequency)
    # Expect positive dB loss
    assert_quantity_allclose(path_loss, 32.45 * u.dB, atol=0.01 * u.dB)


def test_spreading_loss():
    """Test spreading loss calculations."""
    # Spreading loss in positive dB
    assert_quantity_allclose(spreading_loss(36000 * u.km), 162.12 * u.dB, atol=0.01 * u.dB)
    assert_quantity_allclose(spreading_loss(2000 * u.km), 137.01 * u.dB, atol=0.01 * u.dB)


def test_aperture_loss():
    """Test aperture loss calculations."""
    # Aperture loss in positive dB
    assert_quantity_allclose(aperture_loss(12 * u.GHz), 43.04 * u.dB, atol=0.01 * u.dB)
    assert_quantity_allclose(aperture_loss(2.25 * u.GHz), 28.50 * u.dB, atol=0.01 * u.dB)


def test_invalid_parameters():
    """Test that invalid parameters raise appropriate errors."""
    # Test with negative distance
    with pytest.raises(ValueError):
        free_space_path_loss(-1000 * u.m, 1 * u.GHz)

    # Test with negative frequency
    with pytest.raises(ValueError):
        free_space_path_loss(1000 * u.m, -1 * u.GHz)

    # Test with zero distance
    with pytest.raises(ValueError):
        free_space_path_loss(0 * u.m, 1 * u.GHz)

    # Test with zero frequency
    with pytest.raises(ValueError):
        free_space_path_loss(1000 * u.m, 0 * u.GHz)