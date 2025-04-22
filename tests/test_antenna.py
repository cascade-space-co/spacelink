"""Tests for the antenna module."""

import pytest
from spacelink.antenna import Antenna, Dish, FixedGain, polarization_loss
from spacelink.units import GHz, MHz, m, dB, Q_

# TODO: parameterize tests


def test_antenna_is_abstract():
    """Test that Antenna class cannot be instantiated."""
    with pytest.raises(TypeError):
        Antenna()


def test_fixed_gain_initializes_with_valid_gain():
    """Test FixedGain initialization with valid gain."""
    antenna = FixedGain(Q_(10.0, dB))
    assert antenna.gain_.magnitude == pytest.approx(10.0, abs=0.01)

    antenna = FixedGain(Q_(0.0, dB))
    assert antenna.gain_.magnitude == pytest.approx(0.0, abs=0.01)


def test_fixed_gain_accepts_negative_gain():
    """Test FixedGain accepts negative gain values."""
    antenna = FixedGain(Q_(-3.0, dB))
    assert antenna.gain_.magnitude == pytest.approx(-3.0, abs=0.01)


def test_fixed_gain_returns_constant_gain():
    """Test FixedGain returns constant gain regardless of frequency."""
    antenna = FixedGain(Q_(10.0, dB))
    assert antenna.gain(Q_(8.4, GHz)).magnitude == pytest.approx(10.0, abs=0.01)
    assert antenna.gain(Q_(2.4, GHz)).magnitude == pytest.approx(10.0, abs=0.01)
    assert antenna.gain(Q_(100, MHz)).magnitude == pytest.approx(10.0, abs=0.01)


def test_fixed_gain_with_axial_ratio():
    """Test FixedGain with axial ratio parameter."""
    antenna = FixedGain(Q_(10.0, dB), axial_ratio=Q_(3.0, dB))
    assert antenna.gain_.magnitude == pytest.approx(10.0, abs=0.01)
    assert antenna.axial_ratio.magnitude == pytest.approx(3.0, abs=0.01)


def test_dish_initialization():
    """Test Dish initialization with various parameters."""
    # Default efficiency
    dish = Dish(Q_(1.0, m))
    assert dish.diameter.magnitude == pytest.approx(1.0, abs=0.01)
    assert dish.efficiency == pytest.approx(0.65, abs=0.01)
    assert dish.axial_ratio.magnitude == pytest.approx(0.0, abs=0.01)

    # Custom efficiency and axial ratio
    dish = Dish(Q_(2.0, m), efficiency=0.5, axial_ratio=Q_(1.5, dB))
    assert dish.diameter.magnitude == pytest.approx(2.0, abs=0.01)
    assert dish.efficiency == pytest.approx(0.5, abs=0.01)
    assert dish.axial_ratio.magnitude == pytest.approx(1.5, abs=0.01)


def test_dish_raises_on_invalid_diameter():
    """Test Dish raises ValueError for invalid diameter."""
    with pytest.raises(ValueError, match="Dish diameter must be positive"):
        Dish(Q_(-1.0, m))
    with pytest.raises(ValueError, match="Dish diameter must be positive"):
        Dish(Q_(0.0, m))


def test_dish_raises_on_invalid_efficiency():
    """Test Dish raises ValueError for invalid efficiency."""
    with pytest.raises(ValueError, match="Efficiency must be between 0 and 1"):
        Dish(Q_(1.0, m), efficiency=-0.1)
    with pytest.raises(ValueError, match="Efficiency must be between 0 and 1"):
        Dish(Q_(1.0, m), efficiency=0.0)
    with pytest.raises(ValueError, match="Efficiency must be between 0 and 1"):
        Dish(Q_(1.0, m), efficiency=1.1)


def test_dish_calculates_gain():
    """Test Dish calculates gain correctly at various frequencies."""
    dish = Dish(Q_(20.0, m))
    assert dish.gain(Q_(8.4, GHz)).magnitude == pytest.approx(63.04, abs=0.01)
    assert dish.gain(Q_(2.4, GHz)).magnitude == pytest.approx(52.16, abs=0.01)

    # Test with smaller dish
    dish = Dish(Q_(1.0, m))
    assert dish.gain(Q_(2.4, GHz)).magnitude == pytest.approx(26.14, abs=0.01)
    assert dish.gain(Q_(5.8, GHz)).magnitude == pytest.approx(33.80, abs=0.01)
    assert dish.gain(Q_(900, MHz)).magnitude == pytest.approx(17.62, abs=0.01)

    # Test with different efficiency
    dish = Dish(Q_(1.0, m), efficiency=0.5)
    assert dish.gain(Q_(2.4, GHz)).magnitude == pytest.approx(25.00, abs=0.01)


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
        (Q_(0, dB),   Q_(0, dB),   0.0,        0.01), # mathcy matchy
        (Q_(0, dB),   Q_(60, dB),  3.002,      0.01), # also matchy
        (Q_(60, dB),  Q_(0, dB),   3.002,      0.01), # Big mismatch
        (Q_(10, dB),  Q_(30, dB),  0.332,      0.01), # Small mismatch
    ]
)
def test_polarization_loss_calculation(ar_tx_db, ar_rx_db, expected_loss_db, tol):
    """Test the polarization loss calculation based on axial ratios."""
    assert polarization_loss(ar_tx_db, ar_rx_db) == pytest.approx(expected_loss_db, abs=tol)
