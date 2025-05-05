"""Tests for the antenna component classes."""

import pytest
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose

# Updated import
from spacelink.components.antenna import Antenna, Dish, FixedGain


# TODO: parameterize tests


def test_antenna_is_abstract():
    """Test that Antenna class cannot be instantiated."""
    with pytest.raises(TypeError):
        Antenna()


def test_fixed_gain_initializes_with_valid_gain():
    """Test FixedGain initialization with valid gain."""
    antenna = FixedGain(10.0 * u.dB)
    assert_quantity_allclose(antenna._gain, 10.0 * u.dB, atol=0.01 * u.dB)

    antenna = FixedGain(0.0 * u.dB)
    assert_quantity_allclose(antenna._gain, 0.0 * u.dB, atol=0.01 * u.dB)


def test_fixed_gain_returns_constant_gain():
    """Test FixedGain returns constant gain regardless of frequency."""
    antenna = FixedGain(10.0 * u.dB)
    assert_quantity_allclose(antenna.gain(8.4 * u.GHz), 10.0 * u.dB, atol=0.01 * u.dB)
    assert_quantity_allclose(antenna.gain(2.4 * u.GHz), 10.0 * u.dB, atol=0.01 * u.dB)
    assert_quantity_allclose(antenna.gain(100 * u.MHz), 10.0 * u.dB, atol=0.01 * u.dB)


def test_fixed_gain_with_axial_ratio():
    """Test FixedGain with axial ratio parameter."""
    antenna = FixedGain(10.0 * u.dB, axial_ratio=3.0 * u.dB)
    assert_quantity_allclose(antenna._gain, 10.0 * u.dB, atol=0.01 * u.dB)
    assert_quantity_allclose(antenna.axial_ratio, 3.0 * u.dB, atol=0.01 * u.dB)


def test_dish_initialization():
    """Test Dish initialization with various parameters."""
    # Default efficiency
    dish = Dish(1.0 * u.m)
    assert_quantity_allclose(dish.diameter, 1.0 * u.m, atol=0.01 * u.m)
    assert_quantity_allclose(
        dish.efficiency, 0.65 * u.dimensionless, atol=0.01 * u.dimensionless
    )
    assert_quantity_allclose(dish.axial_ratio, 0.0 * u.dB, atol=0.01 * u.dB)

    # Custom efficiency and axial ratio
    dish = Dish(2.0 * u.m, efficiency=0.5 * u.dimensionless, axial_ratio=1.5 * u.dB)
    assert_quantity_allclose(dish.diameter, 2.0 * u.m, atol=0.01 * u.m)
    assert_quantity_allclose(
        dish.efficiency, 0.5 * u.dimensionless, atol=0.01 * u.dimensionless
    )
    assert_quantity_allclose(dish.axial_ratio, 1.5 * u.dB, atol=0.01 * u.dB)


def test_dish_raises_on_invalid_diameter():
    """Test Dish raises ValueError for invalid diameter."""
    with pytest.raises(ValueError, match="Dish diameter must be positive"):
        Dish(-1.0 * u.m)
    with pytest.raises(ValueError, match="Dish diameter must be positive"):
        Dish(0.0 * u.m)


def test_dish_raises_on_invalid_efficiency():
    """Test Dish raises ValueError for invalid efficiency."""
    with pytest.raises(ValueError, match="Efficiency must be between 0 and 1"):
        Dish(1.0 * u.m, efficiency=-0.1 * u.dimensionless)
    with pytest.raises(ValueError, match="Efficiency must be between 0 and 1"):
        Dish(1.0 * u.m, efficiency=1.1 * u.dimensionless)


def test_dish_calculates_gain():
    """Test Dish calculates gain correctly at various frequencies."""
    dish = Dish(20.0 * u.m)
    assert_quantity_allclose(dish.gain(8.4 * u.GHz), 63.04 * u.dB, atol=0.01 * u.dB)
    assert_quantity_allclose(dish.gain(2.4 * u.GHz), 52.16 * u.dB, atol=0.01 * u.dB)

    # Test with smaller dish
    dish = Dish(1.0 * u.m)
    assert_quantity_allclose(dish.gain(2.4 * u.GHz), 26.14 * u.dB, atol=0.01 * u.dB)
    assert_quantity_allclose(dish.gain(5.8 * u.GHz), 33.80 * u.dB, atol=0.01 * u.dB)
    assert_quantity_allclose(dish.gain(900 * u.MHz), 17.62 * u.dB, atol=0.01 * u.dB)

    # Test with different efficiency
    dish = Dish(1.0 * u.m, efficiency=0.5 * u.dimensionless)
    assert_quantity_allclose(dish.gain(2.4 * u.GHz), 25.00 * u.dB, atol=0.01 * u.dB)


def test_antenna_axial_ratio_validation():
    """Test the antenna axial ratio validation."""
    # Axial ratio must be non-negative
    with pytest.raises(ValueError, match="Axial ratio must be non-negative"):
        Dish(1.0 * u.m, axial_ratio=-0.1 * u.dB)

    with pytest.raises(ValueError, match="Axial ratio must be non-negative"):
        FixedGain(10.0 * u.dB, axial_ratio=-1.0 * u.dB)
