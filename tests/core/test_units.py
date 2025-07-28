"""Tests for the units module."""

from dataclasses import dataclass

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
    Power,
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
    vswr_result = return_loss_to_vswr(return_loss * u.dB(1))
    assert_quantity_allclose(
        vswr_result, vswr * u.dimensionless, atol=0.01 * u.dimensionless
    )

    return_loss_result = vswr_to_return_loss(vswr * u.dimensionless)
    assert_quantity_allclose(
        return_loss_result, return_loss * u.dB(1), atol=0.01 * u.dB(1)
    )


@pytest.mark.parametrize(
    "input_value, factor, expected",
    [
        (20 * u.dB(1), 10, 100 * u.dimensionless),
        (30 * u.dB(1), 10, 1000 * u.dimensionless),
        (20 * u.dB(1), 20, 10 * u.dimensionless),
    ],
)
def test_to_linear(input_value, factor, expected):
    assert_quantity_allclose(units.to_linear(input_value, factor=factor), expected)


def test_return_loss_to_vswr_invalid_input():
    with pytest.raises(ValueError):
        units.return_loss_to_vswr(-1 * u.dB(1))


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
        (100 * u.dimensionless, 10, 20 * u.dB(1)),
        (10 * u.dimensionless, 20, 20 * u.dB(1)),
    ],
)
def safe_negate(quantity):
    # Astropy does not allow -quantity for function units, so use multiplication
    return (-1) * quantity


def test_custom_units_exist():
    """Test that custom units are added to astropy.units"""
    assert hasattr(u, "dBHz")
    assert hasattr(u, "dBW")
    assert hasattr(u, "dBm")
    assert hasattr(u, "dBK")
    assert hasattr(u, "dimensionless")


class TestEnforceUnitsArgs:
    """Test enforce_units decorator argument processing."""

    def test_conversion(self):
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
            # This function expects the input to be converted to K
            assert temp.unit == u.K, f"Expected K, got {temp.unit}"
            return temp * 2

        # Test angle conversion
        result = test_angle_function(np.pi / 4 * u.rad)
        assert_quantity_allclose(result, np.pi / 2 * u.rad)

        # Test conversion from degrees to radians
        result = test_angle_function(45 * u.deg)  # Should be converted to radians
        assert_quantity_allclose(result, np.pi / 2 * u.rad)

        # Test frequency conversion
        result = test_frequency_function(1000 * u.Hz)
        assert_quantity_allclose(result, 2000 * u.Hz)

        # Test conversion from MHz to Hz
        result = test_frequency_function(1 * u.MHz)
        assert_quantity_allclose(result, 2000000 * u.Hz)

        # Test temperature conversion (identity for Kelvin)
        result = test_temperature_function(300 * u.K)
        assert_quantity_allclose(result, 600 * u.K)

        # Test temperature conversion from Celsius to Kelvin
        result = test_temperature_function(0 * u.deg_C)  # 0°C = 273.15 K
        assert_quantity_allclose(result, 2 * 273.15 * u.K)

    def test_rejects_incompatible_units(self):
        """Test that enforce_units decorator rejects incompatible units."""

        @enforce_units
        def test_angle_function(angle: Angle) -> Angle:
            return angle * 2

        # Should raise UnitConversionError for incompatible units
        with pytest.raises(u.UnitConversionError):
            test_angle_function(5 * u.m)  # Length instead of angle

        # Should raise TypeError for raw numbers
        with pytest.raises(TypeError):
            test_angle_function(45)  # Raw number instead of Quantity

    def test_optional_parameters(self):
        """Test that enforce_units decorator handles optional parameters correctly."""

        @enforce_units
        def test_optional_frequency(freq: Frequency | None = None) -> str:
            """Function with optional frequency parameter."""
            if freq is None:
                return "no frequency"
            else:
                return f"frequency: {freq.value} Hz"

        @enforce_units
        def test_optional_angle(angle: Angle | None = None, scale: float = 1.0) -> str:
            """Function with optional angle parameter and non-unit parameter."""
            results = []
            if angle is not None:
                results.append(f"angle: {angle.value}")
            if scale != 1.0:
                results.append(f"scale: {scale}")
            return ", ".join(results) if results else "defaults"

        # Test with None values
        assert test_optional_frequency() == "no frequency"
        assert test_optional_frequency(None) == "no frequency"
        assert test_optional_angle() == "defaults"
        assert test_optional_angle(None) == "defaults"

        # Test with actual values
        result = test_optional_frequency(1000 * u.Hz)
        assert result == "frequency: 1000.0 Hz"

        # Test angle conversion with optional
        result = test_optional_angle(45 * u.deg)  # Should convert to radians
        expected_rad = (45 * u.deg).to(u.rad).value
        assert f"angle: {expected_rad}" in result

        # Test with non-unit parameter
        result = test_optional_angle(scale=2.0)
        assert "scale: 2.0" in result

    def test_optional_parameters_invalid_units(self):
        """Test that enforce_units rejects invalid units for optional parameters."""

        @enforce_units
        def test_optional_frequency(freq: Frequency | None = None) -> str:
            return "ok"

        # Should raise UnitConversionError for incompatible units
        with pytest.raises(u.UnitConversionError):
            test_optional_frequency(5 * u.m)  # Length instead of frequency

        # Should raise TypeError for raw numbers
        with pytest.raises(TypeError):
            test_optional_frequency(1000)  # Raw number instead of Quantity

    def test_multiple_optional_parameters(self):
        """Test enforce_units with multiple optional parameters."""

        @enforce_units
        def test_function(
            freq: Frequency | None = None,
            angle: Angle | None = None,
            temp: Temperature | None = None,
        ) -> str:
            """Function with multiple optional unit parameters."""
            results = []
            if freq is not None:
                results.append(f"freq: {freq.value}")
            if angle is not None:
                results.append(f"angle: {angle.value}")
            if temp is not None:
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


class TestEnforceUnitsDataclass:
    def test_dataclass_support(self):
        """Test that enforce_units decorator works correctly on dataclasses."""

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

    def test_rejects_regular_classes(self):
        """Test that enforce_units rejects regular classes."""

        with pytest.raises(TypeError):

            @enforce_units
            class RegularClass:
                def __init__(self, frequency: Frequency):
                    self.frequency = frequency

        with pytest.raises(TypeError):

            @enforce_units
            class AnotherRegularClass:
                pass


class TestEnforceUnitsReturns:
    """Test enforce_units decorator return value validation."""

    def test_return_value_strict_mode(self):
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

    def test_return_value_complex_types(self):
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

    @pytest.mark.parametrize(
        "return_type, wrong_value, expected_value",
        [
            ("single", lambda: 1.0 * u.MHz, 1.0 * u.MHz),
            ("tuple", lambda: (1.0 * u.MHz, 10.0 * u.W), (1.0 * u.MHz, 10.0 * u.W)),
        ],
    )
    def test_return_value_disabled(self, return_type, wrong_value, expected_value):
        """Test that return value checking can be disabled."""

        if return_type == "single":

            @enforce_units
            def wrong_return_units() -> Frequency:
                """Function that returns wrong units."""
                return wrong_value()
        else:  # tuple

            @enforce_units
            def wrong_return_units() -> tuple[Frequency, Power]:
                """Function that returns wrong units in tuple."""
                return wrong_value()

        # Temporarily disable strict checking
        original_value = units._RETURN_UNITS_CHECK_ENABLED
        try:
            units._RETURN_UNITS_CHECK_ENABLED = False

            # Should not raise when checking is disabled
            result = wrong_return_units()
            assert result == expected_value

        finally:
            units._RETURN_UNITS_CHECK_ENABLED = original_value

    def test_tuple_return_values(self):
        """Test that enforce_units validates tuple return values correctly."""

        @enforce_units
        def correct_tuple() -> tuple[Frequency, Power]:
            """Function that returns correct tuple units."""
            return 1000.0 * u.Hz, 10.0 * u.W

        @enforce_units
        def wrong_tuple_units() -> tuple[Frequency, Power]:
            """Function that returns wrong units in tuple."""
            return 1.0 * u.MHz, 10.0 * u.W  # MHz instead of Hz

        @enforce_units
        def wrong_tuple_length() -> tuple[Frequency, Power]:
            """Function that returns wrong tuple length."""
            return (1000.0 * u.Hz,)  # Missing second element

        @enforce_units
        def mixed_tuple() -> tuple[str, Frequency, int]:
            """Function with mixed annotated and non-annotated tuple elements."""
            return "hello", 1000.0 * u.Hz, 42

        @enforce_units
        def wrong_mixed_tuple() -> tuple[str, Frequency, int]:
            """Function with wrong units in mixed tuple."""
            return "hello", 1.0 * u.MHz, 42  # Wrong frequency unit

        @enforce_units
        def non_quantity_in_tuple() -> tuple[Frequency, Power]:
            """Function that returns non-Quantity in tuple."""
            return 1000.0, 10.0 * u.W  # Raw number instead of Quantity

        # Test correct tuple - should pass
        result = correct_tuple()
        assert result == (1000.0 * u.Hz, 10.0 * u.W)

        # Test wrong units in tuple - should fail
        with pytest.raises(u.UnitConversionError):
            wrong_tuple_units()

        # Test wrong tuple length - should fail
        with pytest.raises(TypeError):
            wrong_tuple_length()

        # Test mixed tuple with correct units - should pass
        result = mixed_tuple()
        assert result == ("hello", 1000.0 * u.Hz, 42)

        # Test mixed tuple with wrong units - should fail
        with pytest.raises(u.UnitConversionError):
            wrong_mixed_tuple()

        # Test non-Quantity in tuple - should fail
        with pytest.raises(TypeError):
            non_quantity_in_tuple()

        @enforce_units
        def optional_tuple_correct() -> tuple[Frequency | None, Power]:
            """Function that returns tuple with valid optional element."""
            return None, 10.0 * u.W

        @enforce_units
        def optional_tuple_wrong_none() -> tuple[Frequency, Power]:
            """Function that returns None in non-optional tuple element."""
            return None, 10.0 * u.W  # None not allowed for Frequency

        # Test optional element with None - should pass
        result = optional_tuple_correct()
        assert result == (None, 10.0 * u.W)

        # Test None in non-optional element - should fail
        with pytest.raises(TypeError):
            optional_tuple_wrong_none()
