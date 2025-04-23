"""Tests for the antenna module."""

import pytest
from spacelink.antenna import Antenna, Dish, FixedGain, polarization_loss
from spacelink.units import GHz, MHz, m, dB, Q_
from pint.testing import assert_allclose


# TODO: parameterize tests


def test_antenna_is_abstract():
    """Test that Antenna class cannot be instantiated."""
    with pytest.raises(TypeError):
        Antenna()


def test_fixed_gain_initializes_with_valid_gain():
    """Test FixedGain initialization with valid gain."""
    antenna = FixedGain(Q_(10.0, dB))
    assert_allclose(antenna.gain_, Q_(10.0, dB), atol=0.01)

    antenna = FixedGain(Q_(0.0, dB))
    assert_allclose(antenna.gain_, Q_(0.0, dB), atol=0.01)


def test_fixed_gain_accepts_negative_gain():
    """Test FixedGain accepts negative gain values."""
    antenna = FixedGain(Q_(-3.0, dB))
    assert_allclose(antenna.gain_, Q_(-3.0, dB), atol=0.01)


def test_fixed_gain_returns_constant_gain():
    """Test FixedGain returns constant gain regardless of frequency."""
    antenna = FixedGain(Q_(10.0, dB))
    assert_allclose(antenna.gain(Q_(8.4, GHz)), Q_(10.0, dB), atol=0.01)
    assert_allclose(antenna.gain(Q_(2.4, GHz)), Q_(10.0, dB), atol=0.01)
    assert_allclose(antenna.gain(Q_(100, MHz)), Q_(10.0, dB), atol=0.01)


def test_fixed_gain_with_axial_ratio():
    """Test FixedGain with axial ratio parameter."""
    antenna = FixedGain(Q_(10.0, dB), axial_ratio=Q_(3.0, dB))
    assert_allclose(antenna.gain_, Q_(10.0, dB), atol=0.01)
    assert_allclose(antenna.axial_ratio, Q_(3.0, dB), atol=0.01)


def test_dish_initialization():
    """Test Dish initialization with various parameters."""
    # Default efficiency
    dish = Dish(Q_(1.0, m))
    assert_allclose(dish.diameter, Q_(1.0, m), atol=0.01)
    assert_allclose(dish.efficiency, Q_(0.65, ""), atol=0.01)
    assert_allclose(dish.axial_ratio, Q_(0.0, dB), atol=0.01)

    # Custom efficiency and axial ratio
    dish = Dish(Q_(2.0, m), efficiency=Q_(0.5, ""), axial_ratio=Q_(1.5, dB))
    assert_allclose(dish.diameter, Q_(2.0, m), atol=0.01)
    assert_allclose(dish.efficiency, Q_(0.5, ""), atol=0.01)
    assert_allclose(dish.axial_ratio, Q_(1.5, dB), atol=0.01)


def test_dish_raises_on_invalid_diameter():
    """Test Dish raises ValueError for invalid diameter."""
    with pytest.raises(ValueError, match="Dish diameter must be positive"):
        Dish(Q_(-1.0, m))
    with pytest.raises(ValueError, match="Dish diameter must be positive"):
        Dish(Q_(0.0, m))


def test_dish_raises_on_invalid_efficiency():
    """Test Dish raises ValueError for invalid efficiency."""
    with pytest.raises(ValueError, match="Efficiency must be between 0 and 1"):
        Dish(Q_(1.0, m), efficiency=Q_(-0.1, ""))
    with pytest.raises(ValueError, match="Efficiency must be between 0 and 1"):
        Dish(Q_(1.0, m), efficiency=Q_(1.1, ""))


def test_dish_calculates_gain():
    """Test Dish calculates gain correctly at various frequencies."""
    dish = Dish(Q_(20.0, m))
    assert_allclose(dish.gain(Q_(8.4, GHz)), Q_(63.04, dB), atol=0.01)
    assert_allclose(dish.gain(Q_(2.4, GHz)), Q_(52.16, dB), atol=0.01)

    # Test with smaller dish
    dish = Dish(Q_(1.0, m))
    assert_allclose(dish.gain(Q_(2.4, GHz)), Q_(26.14, dB), atol=0.01)
    assert_allclose(dish.gain(Q_(5.8, GHz)), Q_(33.80, dB), atol=0.01)
    assert_allclose(dish.gain(Q_(900, MHz)), Q_(17.62, dB), atol=0.01)

    # Test with different efficiency
    dish = Dish(Q_(1.0, m), efficiency=Q_(0.5, ""))
    assert_allclose(dish.gain(Q_(2.4, GHz)), Q_(25.00, dB), atol=0.01)


def test_antenna_axial_ratio_validation():
    """Test the antenna axial ratio validation."""
    # Axial ratio must be non-negative
    with pytest.raises(ValueError, match="Axial ratio must be non-negative"):
        Dish(Q_(1.0, m), axial_ratio=Q_(-0.1, dB))

    with pytest.raises(ValueError, match="Axial ratio must be non-negative"):
        FixedGain(Q_(10.0, dB), axial_ratio=Q_(-1.0, dB))


@pytest.mark.parametrize(
    "ar_tx_db, ar_rx_db, expected_loss_db, tol",
    [
        (Q_(0, dB),   Q_(0, dB),   0.0,        0.01),  # mathcy matchy
        (Q_(0, dB),   Q_(60, dB),  3.002,      0.01),  # also matchy
        (Q_(60, dB),  Q_(0, dB),   3.002,      0.01),  # Big mismatch
        (Q_(10, dB),  Q_(30, dB),  0.332,      0.01),  # Small mismatch
    ]
)
def test_polarization_loss_calculation(ar_tx_db, ar_rx_db, expected_loss_db, tol):
    # Test the polarization loss calculation based on axial ratios.
    assert_allclose(polarization_loss(ar_tx_db, ar_rx_db), expected_loss_db, atol=tol)
