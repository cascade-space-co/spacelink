"""Tests for the Demodulator class."""

import pytest
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose

from spacelink.demodulator import Demodulator
from spacelink.signal import Signal


def test_demodulator_creation():
    """Test creating a Demodulator with valid parameters."""
    demodulator = Demodulator(conversion_loss=6 * u.dB, noise_temperature=290 * u.K)
    assert_quantity_allclose(demodulator.conversion_loss, 6 * u.dB)
    assert_quantity_allclose(demodulator.noise_temperature, 290 * u.K)


def test_demodulator_process_input():
    """Test that a Demodulator processes input correctly."""
    demodulator = Demodulator(conversion_loss=6 * u.dB, noise_temperature=290 * u.K)
    
    # Set up input signal
    def input_signal():
        return Signal(power=1 * u.W, noise_temperature=290 * u.K)
    demodulator.input = input_signal
    
    # Process input
    demodulator.process_input(1 * u.GHz)
    
    # Check processed signal
    assert hasattr(demodulator, '_processed_signal')
    assert isinstance(demodulator._processed_signal, Signal)
    assert_quantity_allclose(demodulator._processed_signal.power, 0.25 * u.W, rtol=1e-5)  # 6 dB loss = 1/4 power
    assert_quantity_allclose(demodulator._processed_signal.noise_temperature, 580 * u.K)


def test_demodulator_no_input():
    """Test that Demodulator raises ValueError when input is not set."""
    demodulator = Demodulator(conversion_loss=6 * u.dB, noise_temperature=290 * u.K)
    with pytest.raises(ValueError, match="Demodulator input must be set before processing"):
        demodulator.process_input(1 * u.GHz)


def test_demodulator_unit_conversion():
    """Test that Demodulator handles unit conversions correctly."""
    # Create demodulator with different conversion loss unit
    demodulator = Demodulator(conversion_loss=6 * u.dB, noise_temperature=290 * u.K)
    assert_quantity_allclose(demodulator.conversion_loss, 6 * u.dB)

    # Create demodulator with different temperature unit
    demodulator = Demodulator(conversion_loss=6 * u.dB, noise_temperature=273.15 * u.K)  # 0°C in Kelvin
    assert_quantity_allclose(demodulator.noise_temperature, 273.15 * u.K)


def test_demodulator_invalid_loss_type():
    """Test that creating a Demodulator with non-Quantity conversion loss raises TypeError."""
    with pytest.raises(TypeError, match="Conversion loss must be a Quantity with dB units"):
        Demodulator(conversion_loss=6.0, noise_temperature=290 * u.K)


def test_demodulator_invalid_temperature_type():
    """Test that creating a Demodulator with non-Quantity temperature raises TypeError."""
    with pytest.raises(TypeError, match="Noise temperature must be a Quantity with temperature units"):
        Demodulator(conversion_loss=6 * u.dB, noise_temperature=290.0)


def test_demodulator_invalid_loss_unit():
    """Test that creating a Demodulator with wrong conversion loss unit raises UnitConversionError."""
    with pytest.raises(u.UnitConversionError):
        Demodulator(conversion_loss=6 * u.W, noise_temperature=290 * u.K)


def test_demodulator_invalid_temperature_unit():
    """Test that creating a Demodulator with wrong temperature unit raises UnitConversionError."""
    with pytest.raises(u.UnitConversionError):
        Demodulator(conversion_loss=6 * u.dB, noise_temperature=290 * u.m) 