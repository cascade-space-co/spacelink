"""Tests for the units module."""

# Always import from spacelink.units first to ensure custom units are available
from spacelink import units
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose
import pytest
from functools import wraps
from inspect import signature

from spacelink.units import (
    Decibels,
    DecibelWatts,
    Frequency,
    Wavelength,
    Linear,
    return_loss_to_vswr,
    vswr_to_return_loss,
    wavelength,
    frequency,
    mismatch_loss,
    enforce_units,
)

def test_wavelength_calculation():
    """Test wavelength calculation from frequency."""
    # Test at different frequencies
    assert_quantity_allclose(wavelength(1 * u.GHz).to(u.m), 0.299792458 * u.m)
    assert_quantity_allclose(wavelength(300 * u.MHz).to(u.m), 0.999308193 * u.m)
    assert_quantity_allclose(wavelength(30 * u.kHz).to(u.m), 9993.08193 * u.m)

    # Test unit conversion in result
    wavelength_result = wavelength(2.4 * u.GHz)
    assert_quantity_allclose(wavelength_result.to(u.m), 0.12491352416667 * u.m)


def test_frequency_calculation():
    """Test frequency calculation from wavelength."""
    # Test at different wavelengths
    assert_quantity_allclose(frequency(1 * u.m).to(u.MHz), 299.792458 * u.MHz)
    assert_quantity_allclose(frequency(10 * u.m).to(u.MHz), 29.9792458 * u.MHz)
    assert_quantity_allclose(frequency(0.1 * u.m).to(u.GHz), 2.99792458 * u.GHz)


def test_db_conversion():
    """Test dB conversion function."""
    power_db = 20 * u.dBW
    gain_db = 30.0 * u.dB
    assert_quantity_allclose(power_db.to(u.W), 100 * u.W)
    # For dB to linear conversion, we need to use the proper conversion
    assert_quantity_allclose(10**(gain_db.value/10), 1000.0)
    assert_quantity_allclose(power_db + gain_db, 50 * u.dBW)


def test_invalid_inputs():
    """Test the functions raise errors with invalid inputs."""
    # Wavelength with non-frequency input
    with pytest.raises(Exception):
        wavelength(1 * u.m)

    with pytest.raises(Exception):
        wavelength(1.0 * u.m)
    # Frequency with non-length input
    with pytest.raises(Exception):
        frequency(1 * u.Hz)


@pytest.mark.parametrize(
    "vswr,gamma,return_loss",
    [
        (1.1, 0.0476, 26.44),
        (1.2, 0.0909, 20.83),
        (1.3, 0.1304, 17.69),
        (1.4, 0.1667, 15.56),
        (1.5, 0.2000, 13.98),
        (1.6, 0.2308, 12.74),
        (1.7, 0.2593, 11.73),
        (1.8, 0.2857, 10.88),
        (1.9, 0.3103, 10.16),
        (2.0, 0.3333, 9.54),
    ],
)
def test_vswr(vswr, gamma, return_loss):
    """Test VSWR and return loss conversions."""
    # Test return loss to VSWR conversion
    vswr_result = return_loss_to_vswr(return_loss * u.dB)
    assert_quantity_allclose(
        vswr_result,
        vswr * u.linear,
        atol=0.01 * u.linear
    )
    
    # Test VSWR to return loss conversion
    return_loss_result = vswr_to_return_loss(vswr * u.linear)
    assert_quantity_allclose(
        return_loss_result,
        return_loss * u.dB,
        atol=0.01 * u.dB
    )


def test_mismatch_loss():
    """Test mismatch loss calculation."""
    vswr = 2.0 * u.linear
    return_loss = vswr_to_return_loss(vswr)
    assert_quantity_allclose(mismatch_loss(return_loss), 0.5115 * u.dB, atol=0.01 * u.dB)


def test_enforce_units_decorator():
    """Test that enforce_units decorator raises the correct astropy exception with incompatible units."""
    # Test that wavelength() raises UnitConversionError with incorrect units
    with pytest.raises(u.UnitConversionError):
        # Passing length units to a function expecting frequency
        wavelength(1.0 * u.m)
        
    # Test that frequency() raises UnitConversionError with incorrect units
    with pytest.raises(u.UnitConversionError):
        # Passing frequency units to a function expecting length
        frequency(1.0 * u.Hz)
