"""Tests for the units module."""

import astropy.units as u
import numpy as np
import pytest
from astropy.tests.helper import assert_quantity_allclose

from spacelink.core import units
from spacelink.core.units import (
    Angle,
    Decibels,
    Dimensionless,
    Frequency,
    Temperature,
    enforce_units,
    frequency,
    return_loss_to_vswr,
    vswr_to_return_loss,
    wavelength,
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
    assert_quantity_allclose(return_loss_result, return_loss * u.dB, atol=0.01 * u.dB)


@pytest.mark.parametrize(
    "input_value, factor, expected",
    [
        (100 * u.dimensionless, 10, 20 * u.dB),  # dB for ratios
        (1000 * u.dimensionless, 10, 30 * u.dB),
        (10 * u.dimensionless, 20, 20 * u.dB),
    ],
)
def test_to_dB(input_value, factor, expected):
    assert_quantity_allclose(
        units.to_dB(input_value, factor=factor), expected, atol=0.01 * u.dB
    )


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


def test_enforce_units_optional_parameters():
    """Test that enforce_units decorator handles optional parameters correctly."""

    @enforce_units
    def test_optional_frequency(freq: Frequency | None = None) -> str:
        """Function with optional frequency parameter."""
        if freq is None:
            return "no frequency"
        # Verify the frequency is converted to Hz
        assert freq.unit == u.Hz
        return f"frequency: {freq.value} Hz"

    @enforce_units
    def test_optional_angle(angle: Angle | None = None, scale: float = 1.0) -> str:
        """Function with optional angle parameter and non-unit parameter."""
        if angle is None:
            return "no angle"
        # Verify the angle is converted to radians
        assert angle.unit == u.rad
        return f"angle: {angle.value * scale} rad"

    # Test with None values - should work without enforcement
    result = test_optional_frequency(None)
    assert result == "no frequency"

    result = test_optional_frequency()  # Default None
    assert result == "no frequency"

    # Test with valid Quantity values - should be converted properly
    result = test_optional_frequency(1000 * u.MHz)
    assert result == "frequency: 1000000000.0 Hz"

    result = test_optional_angle(180 * u.deg)
    expected_radians = np.pi
    assert f"angle: {expected_radians} rad" == result

    # Test with multiple parameters where one is optional
    result = test_optional_angle(90 * u.deg, scale=2.0)
    expected_radians = np.pi / 2 * 2.0
    assert f"angle: {expected_radians} rad" == result

    result = test_optional_angle(None, scale=2.0)
    assert result == "no angle"


def test_enforce_units_optional_parameters_invalid_units():
    """Test that enforce_units rejects invalid units for optional parameters."""

    @enforce_units
    def test_optional_frequency(freq: Frequency | None = None) -> str:
        return "ok"

    # Should raise UnitConversionError for incompatible units
    with pytest.raises(u.UnitConversionError):
        test_optional_frequency(5 * u.m)  # Length instead of frequency

    # Should raise TypeError for raw numbers
    with pytest.raises(TypeError, match="must be provided as an astropy Quantity"):
        test_optional_frequency(1000)  # Raw number instead of Quantity


def test_enforce_units_multiple_optional_parameters():
    """Test enforce_units with multiple optional parameters."""

    @enforce_units
    def test_function(
        freq: Frequency | None = None,
        angle: Angle | None = None,
        temp: Temperature | None = None,
    ) -> str:
        results = []
        if freq is not None:
            assert freq.unit == u.Hz
            results.append(f"freq: {freq.value}")
        if angle is not None:
            assert angle.unit == u.rad
            results.append(f"angle: {angle.value}")
        if temp is not None:
            assert temp.unit == u.K
            results.append(f"temp: {temp.value}")
        return ", ".join(results) if results else "all none"

    # Test all None
    assert test_function() == "all none"

    # Test mixed values
    result = test_function(freq=1 * u.GHz, angle=None, temp=300 * u.K)
    assert "freq: 1000000000.0" in result
    assert "temp: 300.0" in result
    assert "angle:" not in result

    # Test temperature conversion (Celsius to Kelvin)
    result = test_function(temp=0 * u.deg_C)  # 0°C = 273.15 K
    assert "temp: 273.15" in result


def test_custom_units_exist():
    """Test that custom units are added to astropy.units"""
    assert hasattr(u, "dBHz")
    assert hasattr(u, "dBW")
    assert hasattr(u, "dBm")
    assert hasattr(u, "dBK")
    assert hasattr(u, "dimensionless")


def test_enforce_units_dataclass_support():
    """Test that enforce_units decorator works correctly on dataclasses."""
    from dataclasses import dataclass

    @enforce_units
    @dataclass(frozen=True)
    class TestDataclass:
        frequency: Frequency
        angle: Angle

    # Test valid units - should work and convert
    obj = TestDataclass(frequency=1000 * u.MHz, angle=180 * u.deg)

    # Verify units were converted correctly
    assert_quantity_allclose(obj.frequency, 1e9 * u.Hz)
    assert_quantity_allclose(obj.angle, np.pi * u.rad)

    # Test invalid units - should raise UnitConversionError
    with pytest.raises(u.UnitConversionError):
        TestDataclass(
            frequency=5 * u.m, angle=180 * u.deg
        )  # Length instead of frequency

    # Test raw numbers - should raise TypeError
    with pytest.raises(TypeError):
        TestDataclass(frequency=1000, angle=180 * u.deg)


def test_enforce_units_rejects_regular_classes():
    """Test that enforce_units decorator rejects regular classes with helpful error."""

    with pytest.raises(TypeError):

        @enforce_units
        class RegularClass:
            def __init__(self, frequency: Frequency):
                self.frequency = frequency

    # Error message should include helpful guidance
    with pytest.raises(TypeError):

        @enforce_units
        class AnotherRegularClass:
            pass


def test_enforce_units_return_value_strict_mode():
    """Test that enforce_units validates return value units in strict mode."""

    @enforce_units
    def correct_return_units() -> Frequency:
        """Function that returns correct units."""
        return 1000.0 * u.Hz

    @enforce_units
    def wrong_return_units() -> Frequency:
        """Function that returns wrong units."""
        return 1.0 * u.MHz  # MHz instead of Hz

    @enforce_units
    def non_quantity_return() -> Frequency:
        """Function that returns non-Quantity."""
        return 1000.0  # Raw number instead of Quantity

    @enforce_units
    def optional_return_none() -> Frequency | None:
        """Function that can return None."""
        return None

    @enforce_units
    def optional_return_wrong_none() -> Frequency:
        """Function that returns None but not annotated as optional."""
        return None

    # Test correct return units - should pass
    result = correct_return_units()
    assert result == 1000.0 * u.Hz

    # Test wrong return units - should fail in strict mode
    with pytest.raises(u.UnitConversionError):
        wrong_return_units()

    # Test non-Quantity return - should fail
    with pytest.raises(TypeError):
        non_quantity_return()

    # Test optional return None - should pass
    result = optional_return_none()
    assert result is None

    # Test non-optional return None - should fail
    with pytest.raises(TypeError):
        optional_return_wrong_none()


def test_enforce_units_return_value_disabled():
    """Test that return value checking can be disabled."""

    @enforce_units
    def wrong_return_units() -> Frequency:
        """Function that returns wrong units."""
        return 1.0 * u.MHz  # MHz instead of Hz

    # Temporarily disable strict checking
    original_value = units._RETURN_UNITS_CHECK_STRICT
    try:
        units._RETURN_UNITS_CHECK_STRICT = False

        # Should not raise when checking is disabled
        result = wrong_return_units()
        assert result == 1.0 * u.MHz

    finally:
        units._RETURN_UNITS_CHECK_STRICT = original_value


def test_enforce_units_return_value_complex_types():
    """Test return value checking with complex unit types."""

    @enforce_units
    def return_decibels() -> Decibels:
        """Function that returns decibels."""
        return 10.0 * u.dB

    @enforce_units
    def return_wrong_decibels() -> Decibels:
        """Function that returns wrong decibel unit."""
        return 10.0 * u.dBW  # dBW instead of dB

    @enforce_units
    def return_dimensionless() -> Dimensionless:
        """Function that returns dimensionless quantity."""
        return 1.5 * u.dimensionless

    # Test correct decibel return
    result = return_decibels()
    assert result == 10.0 * u.dB

    # Test wrong decibel type - should fail
    with pytest.raises(u.UnitConversionError):
        return_wrong_decibels()

    # Test dimensionless return
    result = return_dimensionless()
    assert result == 1.5 * u.dimensionless
