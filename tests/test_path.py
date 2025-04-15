"""
Tests for the path loss module.

This module contains pytest-style tests for path loss calculations.
"""

import pytest
from pyradio.path import (
    free_space_path_loss_db,
    spreading_loss,
    aperture_loss
)
from pyradio.conversions import ghz, kilometers

def test_free_space_path_loss():
    """Test free space path loss calculation.
    """

    assert free_space_path_loss_db(kilometers(1000), ghz(2.2)) == pytest.approx(159.3, rel=0.01)


def test_free_space_path_loss_invalid():
    """Test free space path loss with invalid inputs."""
    # Test zero distance
    with pytest.raises(ValueError, match="Distance must be positive"):
        free_space_path_loss_db(0, 2.4e9)
    
    # Test negative distance
    with pytest.raises(ValueError, match="Distance must be positive"):
        free_space_path_loss_db(-1000, 2.4e9)
    
    # Test zero frequency
    with pytest.raises(ValueError, match="Frequency must be positive"):
        free_space_path_loss_db(1000, 0)
    
    # Test negative frequency
    with pytest.raises(ValueError, match="Frequency must be positive"):
        free_space_path_loss_db(1000, -2.4e9)

def test_spreading_loss():
    """Test spreading loss calculation."""
    assert spreading_loss(kilometers(1000)) == pytest.approx(130.99, rel=0.01)

def test_spreading_loss_invalid():
    """Test spreading loss with invalid inputs."""
    # Test zero distance
    with pytest.raises(ValueError, match="Distance must be positive"):
        spreading_loss(0)
    
    # Test negative distance
    with pytest.raises(ValueError, match="Distance must be positive"):
        spreading_loss(-1000)

def test_aperture_loss():
    """Test aperture loss calculation."""
    # Test with typical parameters
    actual = aperture_loss(ghz(1.0))
    assert actual == pytest.approx(21.45, rel=0.01)
    

def test_aperture_loss_invalid():
    """Test aperture loss with invalid inputs."""
    # Test zero frequency
    with pytest.raises(ValueError, match="Frequency must be positive"):
        aperture_loss(0)
    
    # Test negative frequency
    with pytest.raises(ValueError, match="Frequency must be positive"):
        aperture_loss(-2.4e9) 