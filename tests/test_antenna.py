"""Tests for the antenna module."""

import pytest
from pyradio.antenna import Antenna, Dish, FixedGain, polarization_loss
from pyradio.units import GHz, MHz, m


def test_antenna_is_abstract():
    """Test that Antenna class cannot be instantiated."""
    with pytest.raises(TypeError):
        Antenna()


def test_fixed_gain_initializes_with_valid_gain():
    """Test FixedGain initialization with valid gain."""
    antenna = FixedGain(10.0)
    assert antenna.gain_ == pytest.approx(10.0, abs=0.01)

    antenna = FixedGain(0.0)
    assert antenna.gain_ == pytest.approx(0.0, abs=0.01)


def test_fixed_gain_accepts_negative_gain():
    """Test FixedGain accepts negative gain values."""
    antenna = FixedGain(-3.0)
    assert antenna.gain_ == pytest.approx(-3.0, abs=0.01)


def test_fixed_gain_returns_constant_gain():
    """Test FixedGain returns constant gain regardless of frequency."""
    antenna = FixedGain(10.0)
    assert antenna.gain(8.4 * GHz) == pytest.approx(10.0, abs=0.01)
    assert antenna.gain(2.4 * GHz) == pytest.approx(10.0, abs=0.01)
    assert antenna.gain(100 * MHz) == pytest.approx(10.0, abs=0.01)


def test_fixed_gain_with_axial_ratio():
    """Test FixedGain with axial ratio parameter."""
    antenna = FixedGain(10.0, axial_ratio=3.0)
    assert antenna.gain_ == pytest.approx(10.0, abs=0.01)
    assert antenna.axial_ratio == pytest.approx(3.0, abs=0.01)


def test_dish_initialization():
    """Test Dish initialization with various parameters."""
    # Default efficiency
    dish = Dish(1.0 * m)
    assert dish.diameter.magnitude == pytest.approx(1.0, abs=0.01)
    assert dish.efficiency == pytest.approx(0.65, abs=0.01)
    assert dish.axial_ratio == pytest.approx(0.0, abs=0.01)

    # Custom efficiency and axial ratio
    dish = Dish(2.0 * m, efficiency=0.5, axial_ratio=1.5)
    assert dish.diameter.magnitude == pytest.approx(2.0, abs=0.01)
    assert dish.efficiency == pytest.approx(0.5, abs=0.01)
    assert dish.axial_ratio == pytest.approx(1.5, abs=0.01)


def test_dish_raises_on_invalid_diameter():
    """Test Dish raises ValueError for invalid diameter."""
    with pytest.raises(ValueError, match="Dish diameter must be positive"):
        Dish(-1.0 * m)
    with pytest.raises(ValueError, match="Dish diameter must be positive"):
        Dish(0.0 * m)


def test_dish_raises_on_invalid_efficiency():
    """Test Dish raises ValueError for invalid efficiency."""
    with pytest.raises(ValueError, match="Efficiency must be between 0 and 1"):
        Dish(1.0 * m, efficiency=-0.1)
    with pytest.raises(ValueError, match="Efficiency must be between 0 and 1"):
        Dish(1.0 * m, efficiency=0.0)
    with pytest.raises(ValueError, match="Efficiency must be between 0 and 1"):
        Dish(1.0 * m, efficiency=1.1)


def test_dish_calculates_gain():
    """Test Dish calculates gain correctly at various frequencies."""
    dish = Dish(20.0 * m)
    assert dish.gain(8.4 * GHz) == pytest.approx(63.04, abs=0.01)
    assert dish.gain(2.4 * GHz) == pytest.approx(52.16, abs=0.01)

    # Test with smaller dish
    dish = Dish(1.0 * m)
    assert dish.gain(2.4 * GHz) == pytest.approx(26.14, abs=0.01)
    assert dish.gain(5.8 * GHz) == pytest.approx(33.80, abs=0.01)
    assert dish.gain(900 * MHz) == pytest.approx(17.62, abs=0.01)

    # Test with different efficiency
    dish = Dish(1.0 * m, efficiency=0.5)
    assert dish.gain(2.4 * GHz) == pytest.approx(25.00, abs=0.01)


def test_antenna_axial_ratio_validation():
    """Test the antenna axial ratio validation."""
    # Axial ratio must be non-negative
    with pytest.raises(ValueError, match="Axial ratio must be non-negative"):
        Dish(1.0 * m, axial_ratio=-0.1)

    with pytest.raises(ValueError, match="Axial ratio must be non-negative"):
        FixedGain(10.0, axial_ratio=-1.0)


def test_polarization_loss_calculation():
    """Test the polarization loss calculation based on axial ratios.

    Only testing for circular polarization with fairly tight axial ratios
    """
    # Perfect circular to perfect circular should have 0 dB loss
    assert polarization_loss(0.0, 0.0) == pytest.approx(0.0, abs=0.01)

    # Two antennas with identical moderate axial ratios
    assert polarization_loss(3.0, 3.0) == pytest.approx(0.51, abs=0.01)

    # Common axial ratio combinations from the table (positive dB values)
    assert polarization_loss(0.5, 3.0) == pytest.approx(0.26, abs=0.01)
    assert polarization_loss(1.0, 3.0) == pytest.approx(0.28, abs=0.01)
    assert polarization_loss(2.0, 3.0) == pytest.approx(0.37, abs=0.01)
    assert polarization_loss(3.0, 4.0) == pytest.approx(0.70, abs=0.01)
