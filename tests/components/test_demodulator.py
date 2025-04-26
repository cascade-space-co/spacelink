"""Tests for the Demodulator module."""

import pytest
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose

from spacelink.components.demodulator import Demodulator
from spacelink.components.signal import Signal
from spacelink.components.source import Source


class MockSource(Source):
    """Concrete implementation of Source for testing."""

    def __init__(self):
        super().__init__()
        self._power = 1 * u.W
        self.noise_temperature = 290 * u.K

    def output(self, frequency):
        return Signal(power=self._power, noise_temperature=self.noise_temperature)


def test_demodulator_creation():
    """Test creating a Demodulator with valid parameters."""
    demodulator = Demodulator(conversion_loss=6 * u.dB, noise_temperature=290 * u.K)
    assert_quantity_allclose(demodulator.conversion_loss, 6 * u.dB)
    assert_quantity_allclose(demodulator.noise_temperature, 290 * u.K)


def test_demodulator_process_input():
    """Test that a Demodulator processes input correctly."""
    demodulator = Demodulator(conversion_loss=6 * u.dB, noise_temperature=290 * u.K)

    # Set up input signal
    source = MockSource()
    demodulator.input = source

    # Process input
    demodulator.process_input(1 * u.GHz)

    # Check that output power is reduced by conversion loss
    # 6 dB conversion loss means power is reduced by a factor of 10^(-6/10) ≈ 0.251188643150958
    processed_signal = demodulator.get_processed_signal()
    assert processed_signal is not None
    assert_quantity_allclose(processed_signal.power, 0.251188643150958 * u.W)

    # Check that output noise temperature is the sum of input and demodulator temperatures
    assert_quantity_allclose(processed_signal.noise_temperature, 580 * u.K)


def test_demodulator_no_input():
    """Test that Demodulator raises ValueError when input is not set."""
    demodulator = Demodulator(conversion_loss=6 * u.dB, noise_temperature=290 * u.K)

    with pytest.raises(
        ValueError, match="Demodulator input must be set before processing"
    ):
        demodulator.process_input(1 * u.GHz)


def test_demodulator_unit_conversion():
    """Test that Demodulator handles unit conversions correctly."""
    # Create demodulator with different conversion loss unit
    demodulator = Demodulator(conversion_loss=6 * u.dB, noise_temperature=290 * u.K)
    assert_quantity_allclose(demodulator.conversion_loss, 6 * u.dB)

    # Create demodulator with different temperature unit
    demodulator = Demodulator(
        conversion_loss=6 * u.dB, noise_temperature=273.15 * u.K
    )  # 0°C in Kelvin
    assert_quantity_allclose(demodulator.noise_temperature, 273.15 * u.K)


def test_demodulator_invalid_loss_type():
    """Test that creating a Demodulator with non-Quantity conversion loss raises TypeError."""
    with pytest.raises(
        TypeError, match="Conversion loss must be a Quantity with dB units"
    ):
        Demodulator(conversion_loss=6.0, noise_temperature=290 * u.K)


def test_demodulator_invalid_temperature_type():
    """Test that creating a Demodulator with non-Quantity temperature raises TypeError."""
    with pytest.raises(
        TypeError, match="Noise temperature must be a Quantity with temperature units"
    ):
        Demodulator(conversion_loss=6 * u.dB, noise_temperature=290.0)


def test_demodulator_invalid_loss_unit():
    """Test that creating a Demodulator with wrong conversion loss unit raises UnitConversionError."""
    with pytest.raises(u.UnitConversionError):
        Demodulator(conversion_loss=6 * u.W, noise_temperature=290 * u.K)


def test_demodulator_invalid_temperature_unit():
    """Test that creating a Demodulator with wrong temperature unit raises UnitConversionError."""
    with pytest.raises(u.UnitConversionError):
        Demodulator(conversion_loss=6 * u.dB, noise_temperature=290 * u.m)


def test_demodulator_negative_values():
    """Test that Demodulator raises error for negative values."""
    with pytest.raises(ValueError, match="Conversion loss must be non-negative"):
        Demodulator(conversion_loss=-6 * u.dB, noise_temperature=290 * u.K)

    with pytest.raises(
        ValueError, match="Noise temperature must be non-negative"
    ):
        Demodulator(conversion_loss=6 * u.dB, noise_temperature=-290 * u.K)


def test_demodulator_type_checks():
    """Test that Demodulator raises error for invalid types."""
    with pytest.raises(
        TypeError, match="Conversion loss must be a Quantity with dB units"
    ):
        Demodulator(conversion_loss=6, noise_temperature=290 * u.K)

    with pytest.raises(
        TypeError, match="Noise temperature must be a Quantity with temperature units"
    ):
        Demodulator(conversion_loss=6 * u.dB, noise_temperature=290)
