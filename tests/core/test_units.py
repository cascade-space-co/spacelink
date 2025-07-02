"""Tests for the units module."""

import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose
import pytest
import numpy as np

from spacelink.core import units
from spacelink.core.units import (
    return_loss_to_vswr,
    vswr_to_return_loss,
    wavelength,
    frequency,
    to_dB,
    enforce_units,
    Angle,
    Frequency,
    Temperature,
)


@pytest.mark.parametrize(
    "frequency, expected_wavelength",
    [
        (1 * u.GHz, 0.299792458 * u.m),
        (300 * u.MHz, 0.999308193 * u.m),
        (30 * u.kHz, 9993.08193 * u.m),
    ],
)
def test_wavelength_calculation(frequency, expected_wavelength):
    """
    TODO: validate
    """
    assert_quantity_allclose(wavelength(frequency).to(u.m), expected_wavelength)


# DO NOT MODIFY
@pytest.mark.parametrize(
    "wavelength, expected_frequency",
    [
        (1 * u.m, 299.792458 * u.MHz),
        (10 * u.m, 29.9792458 * u.MHz),
        (0.1 * u.m, 2.99792458 * u.GHz),
    ],
)
def test_frequency_calculation(wavelength, expected_frequency):
    """
    TODO: validate
    """
    assert_quantity_allclose(
        frequency(wavelength).to(expected_frequency.unit), expected_frequency
    )


@pytest.mark.parametrize(
    "func,invalid_input,error_type",
    [
        (wavelength, 1 * u.m, u.UnitConversionError),
        (wavelength, 1.0 * u.m, u.UnitConversionError),
        (frequency, 1 * u.Hz, u.UnitConversionError),
    ],
)
def test_invalid_inputs(func, invalid_input, error_type):
    with pytest.raises(error_type):
        func(invalid_input)


def assert_decibel_equal(actual, expected, atol=0.01):
    # Compare value
    assert np.isclose(
        actual.to_value(expected.unit), expected.value, atol=atol
    ), f"{actual} != {expected}"
    # Compare unit string (should both be 'dB')
    assert str(actual.unit) == str(
        expected.unit
    ), f"Units differ: {actual.unit} != {expected.unit}"


# DO NOT MODIFY
@pytest.mark.parametrize(
    "vswr,return_loss",
    [
        (1.0, float("inf")),
        (1.1, 26.44),
        (1.2, 20.83),
        (1.3, 17.69),
        (1.4, 15.56),
        (1.5, 13.98),
        (1.6, 12.74),
        (1.7, 11.73),
        (1.8, 10.88),
        (1.9, 10.16),
        (2.0, 9.54),
    ],
)
def test_vswr_return_loss_conversions(vswr, return_loss):
    """Test VSWR to return loss and return loss to VSWR conversions."""
    vswr_result = return_loss_to_vswr(return_loss * u.dB)
    assert_quantity_allclose(
        vswr_result, vswr * u.dimensionless, atol=0.01 * u.dimensionless
    )

    return_loss_result = vswr_to_return_loss(vswr * u.dimensionless)
    assert_decibel_equal(return_loss_result, return_loss * u.dB, atol=0.01)


@pytest.mark.parametrize(
    "input_value, factor, expected",
    [
        (100 * u.dimensionless, 10, 20 * u.dB),  # dB for ratios
        (1000 * u.dimensionless, 10, 30 * u.dB),
        (10 * u.dimensionless, 20, 20 * u.dB),
    ],
)
def test_to_dB(input_value, factor, expected):
    assert_decibel_equal(units.to_dB(input_value, factor=factor), expected, atol=0.01)


@pytest.mark.parametrize(
    "input_value, factor, expected",
    [
        (20 * u.dB, 10, 100 * u.dimensionless),
        (30 * u.dB, 10, 1000 * u.dimensionless),
        (20 * u.dB, 20, 10 * u.dimensionless),
    ],
)
def test_to_linear(input_value, factor, expected):
    assert_quantity_allclose(units.to_linear(input_value, factor=factor), expected)


@pytest.mark.parametrize(
    "return_loss, vswr",
    [
        (20 * u.dB, 1.222 * u.dimensionless),
        (float("inf") * u.dB, 1.0 * u.dimensionless),
    ],
)
def test_vswr_return_loss_conversions(return_loss, vswr):
    vswr_result = units.return_loss_to_vswr(return_loss)
    assert_quantity_allclose(vswr_result, vswr, atol=0.01 * u.dimensionless)

    # Use safe_negate for dB quantities
    gamma = (vswr - 1) / (vswr + 1)
    return_loss_result = safe_negate(to_dB(gamma, factor=20))
    assert_decibel_equal(return_loss_result, return_loss, atol=0.01)


def test_return_loss_to_vswr_invalid_input():
    with pytest.raises(ValueError):
        units.return_loss_to_vswr(-1 * u.dB)


def test_vswr_to_return_loss_invalid_input():
    with pytest.raises(ValueError):
        vswr_to_return_loss(0.5 * u.dimensionless)


@pytest.mark.parametrize(
    "input_value, factor, expected",
    [
        (1 * u.W, 10, 0 * u.dBW),
        (10 * u.W, 10, 10 * u.dBW),
        (1 * u.K, 10, 0 * u.dBK),
        (100 * u.K, 10, 20 * u.dBK),
        (1 * u.Hz, 10, 0 * u.dBHz),
        (1000 * u.Hz, 10, 30 * u.dBHz),
        (100 * u.dimensionless, 10, 20 * u.dB),
        (10 * u.dimensionless, 20, 20 * u.dB),
    ],
)
def safe_negate(quantity):
    # Astropy does not allow -quantity for function units, so use multiplication
    return (-1) * quantity


def test_enforce_units_conversion():
    """Test that enforce_units decorator properly converts units."""

    @enforce_units
    def test_angle_function(angle: Angle) -> Angle:
        """Function that expects angle in radians."""
        # This function expects the input to be converted to radians
        # If the decorator works correctly, angle.unit should be u.rad
        assert angle.unit == u.rad, f"Expected radians, got {angle.unit}"
        return angle * 2

    @enforce_units
    def test_frequency_function(freq: Frequency) -> Frequency:
        """Function that expects frequency in Hz."""
        # This function expects the input to be converted to Hz
        assert freq.unit == u.Hz, f"Expected Hz, got {freq.unit}"
        return freq * 2

    @enforce_units
    def test_temperature_function(temp: Temperature) -> Temperature:
        """Function that expects temperature in Kelvin."""
        # This function expects the input to be converted to Kelvin
        assert temp.unit == u.K, f"Expected K, got {temp.unit}"
        return temp + (10 * u.K)

    # Test angle conversion: degrees should be converted to radians
    input_angle = 180 * u.deg  # 180 degrees = π radians
    result_angle = test_angle_function(input_angle)
    expected_radians = np.pi * u.rad
    assert_quantity_allclose(input_angle.to(u.rad), expected_radians)
    assert_quantity_allclose(result_angle, 2 * expected_radians)

    # Test frequency conversion: MHz should be converted to Hz
    input_freq = 1000 * u.MHz  # 1000 MHz = 1e9 Hz
    result_freq = test_frequency_function(input_freq)
    expected_hz = 1e9 * u.Hz
    assert_quantity_allclose(result_freq, 2 * expected_hz)

    # Test temperature conversion: Celsius should be converted to Kelvin
    input_temp = 0 * u.deg_C  # 0°C = 273.15 K
    result_temp = test_temperature_function(input_temp)
    expected_k = 273.15 * u.K
    assert_quantity_allclose(result_temp, expected_k + (10 * u.K), rtol=1e-10)


def test_enforce_units_rejects_incompatible_units():
    """Test that enforce_units decorator rejects incompatible units."""

    @enforce_units
    def test_angle_function(angle: Angle) -> Angle:
        return angle * 2

    # Should raise UnitConversionError for incompatible units
    with pytest.raises(u.UnitConversionError):
        test_angle_function(5 * u.m)  # Length instead of angle

    # Should raise TypeError for raw numbers
    with pytest.raises(TypeError, match="must be provided as an astropy Quantity"):
        test_angle_function(45)  # Raw number instead of Quantity

        
def test_custom_units_exist():
    """Test that custom units are added to astropy.units"""
    assert hasattr(u, "dBHz")
    assert hasattr(u, "dBW")
    assert hasattr(u, "dBm")
    assert hasattr(u, "dBK")
    assert hasattr(u, "dimensionless")
