"""Tests for the antenna module."""

import pytest
from pyradio.antenna import Antenna, Dish, FixedGain, dish_3db_beamwidth, polarization_loss
from pyradio.conversions import ghz, mhz, kilometers

def test_antenna_is_abstract():
    """Test that Antenna class cannot be instantiated."""
    with pytest.raises(TypeError): Antenna()

def test_fixed_gain_initializes_with_valid_gain():
    """Test FixedGain initialization with valid gain."""
    antenna = FixedGain(10.0)
    assert antenna.gain_db == pytest.approx(10.0, rel=0.01)
    
    antenna = FixedGain(0.0)
    assert antenna.gain_db == pytest.approx(0.0, rel=0.01)

def test_fixed_gain_accepts_negative_gain():
    """Test FixedGain accepts negative gain values."""
    antenna = FixedGain(-3.0)
    assert antenna.gain_db == pytest.approx(-3.0, abs=0.01)

def test_fixed_gain_returns_constant_gain():
    """Test FixedGain returns constant gain regardless of frequency."""
    antenna = FixedGain(10.0)
    assert antenna.gain(ghz(8.4)) == pytest.approx(10.0, rel=0.01)
    assert antenna.gain(ghz(2.4)) == pytest.approx(10.0, rel=0.01)
    assert antenna.gain(mhz(100)) == pytest.approx(10.0, rel=0.01)

def test_dish_raises_on_invalid_diameter():
    """Test Dish raises ValueError for invalid diameter."""
    with pytest.raises(ValueError, match="Dish diameter must be positive"): Dish(-1.0)
    with pytest.raises(ValueError, match="Dish diameter must be positive"): Dish(0.0)

def test_dish_raises_on_invalid_efficiency():
    """Test Dish raises ValueError for invalid efficiency."""
    with pytest.raises(ValueError, match="Efficiency must be between 0 and 1"): Dish(1.0, efficiency=-0.1)
    with pytest.raises(ValueError, match="Efficiency must be between 0 and 1"): Dish(1.0, efficiency=1.1)

def test_dish_calculates_gain():
    """Test Dish calculates gain correctly at various frequencies."""
    dish = Dish(20.0)
    assert dish.gain(ghz(8.4)) == pytest.approx(63.04, abs=0.01)
    assert dish.gain(ghz(2.4)) == pytest.approx(52.16, abs=0.01)
    

def test_dish_raises_on_invalid_frequency():
    """Test Dish raises ValueError for invalid frequency."""
    dish = Dish(1.0)
    with pytest.raises(ValueError, match="Frequency must be positive"): dish.gain(-1.0)
    with pytest.raises(ValueError, match="Frequency must be positive"): dish.gain(0.0)

def test_dish_calculates_beamwidth():
    """Test dish calculates 3dB beamwidth correctly."""
    assert dish_3db_beamwidth(20.0, ghz(8.4)) == pytest.approx(0.12, abs=0.01)
    assert dish_3db_beamwidth(20.0, ghz(2.4)) == pytest.approx(0.44, rel=0.01)
    assert dish_3db_beamwidth(10.0, ghz(2.4)) == pytest.approx(0.88, rel=0.01)

def test_dish_raises_on_invalid_beamwidth_diameter():
    """Test dish raises ValueError for invalid beamwidth diameter."""
    with pytest.raises(ValueError, match="Diameter must be positive"): dish_3db_beamwidth(-1.0, 2.4e9)
    with pytest.raises(ValueError, match="Diameter must be positive"): dish_3db_beamwidth(0.0, 2.4e9)

def test_dish_raises_on_invalid_beamwidth_frequency():
    """Test dish raises ValueError for invalid beamwidth frequency."""
    with pytest.raises(ValueError, match="Frequency must be positive"): dish_3db_beamwidth(1.0, -1.0)
    with pytest.raises(ValueError, match="Frequency must be positive"): dish_3db_beamwidth(1.0, 0.0)

def test_fixed_gain_antenna():
    """Test the FixedGain antenna class."""
    # Create a fixed gain antenna with gain of 10 dB and axial ratio of 3 dB
    antenna = FixedGain(10.0, axial_ratio=3.0)
    
    # Gain should be fixed regardless of frequency
    assert antenna.gain(ghz(2.4)) == pytest.approx(10.0, abs=0.01)
    assert antenna.gain(ghz(5.8)) == pytest.approx(10.0, abs=0.01)
    assert antenna.gain(mhz(900)) == pytest.approx(10.0, abs=0.01)
    
    # Axial ratio should be stored correctly
    assert antenna.axial_ratio == pytest.approx(3.0, abs=0.01)

def test_dish_antenna():
    """Test the Dish antenna class."""
    # Create a dish antenna with diameter of 1 meter, default efficiency (0.65), and axial ratio of 1.5 dB
    antenna = Dish(1.0, axial_ratio=1.5)
    
    # Check default efficiency and axial ratio
    assert antenna.efficiency == pytest.approx(0.65, abs=0.01)
    assert antenna.axial_ratio == pytest.approx(1.5, abs=0.01)
    
    # Check gain calculation at different frequencies
    assert antenna.gain(ghz(2.4)) == pytest.approx(26.14, abs=0.01)
    assert antenna.gain(ghz(5.8)) == pytest.approx(33.80, abs=0.01)
    assert antenna.gain(mhz(900)) == pytest.approx(17.62, abs=0.01)
    
    # Create a dish antenna with custom efficiency and axial ratio
    antenna = Dish(1.0, efficiency=0.5, axial_ratio=0.0)
    assert antenna.efficiency == pytest.approx(0.5, abs=0.01)
    assert antenna.axial_ratio == pytest.approx(0.0, abs=0.01)
    assert antenna.gain(ghz(2.4)) == pytest.approx(25.00, abs=0.01)

def test_antenna_axial_ratio_validation():
    """Test the antenna axial ratio validation."""
    # Axial ratio must be non-negative
    with pytest.raises(ValueError, match="Axial ratio must be non-negative"):
        Dish(1.0, axial_ratio=-0.1)
    
    with pytest.raises(ValueError, match="Axial ratio must be non-negative"):
        FixedGain(10.0, axial_ratio=-1.0)

def test_polarization_loss_calculation():
    """Test the polarization loss calculation based on axial ratios."""
    # Perfect circular to perfect circular should have 0 dB loss
    assert polarization_loss(0.0, 0.0) == pytest.approx(0.0, abs=0.01)
    
    # Linear polarization with same orientation (both have high axial ratio)
    assert polarization_loss(40.0, 40.0) == pytest.approx(33.98, abs=0.01)
    
    # Circular to linear should have ~17 dB loss with this formula
    assert polarization_loss(0.0, 40.0) == pytest.approx(16.99, abs=0.01)
    
    # Two antennas with moderate axial ratios
    assert polarization_loss(3.0, 6.0) == pytest.approx(1.22, abs=0.01)
