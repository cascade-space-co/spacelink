"""Unit tests for the Dish class."""

import pytest
import numpy as np
from pyradio.dish import Dish

def test_dish_initialization():
    """Test dish initialization with valid parameters."""
    dish = Dish(diameter=1.0, efficiency=0.7, lna_gain=30.0, 
                system_temperature=290.0, sky_temperature=20.0)
    assert dish.diameter == 1.0
    assert dish.efficiency == 0.7
    assert dish.lna_gain == 30.0
    assert dish.system_temperature == 290.0
    assert dish.sky_temperature == 20.0

def test_dish_initialization_defaults():
    """Test dish initialization with default parameters."""
    dish = Dish(diameter=1.0)
    assert dish.diameter == 1.0
    assert dish.efficiency == 0.65
    assert dish.lna_gain == 0.0
    assert dish.system_temperature == 290.0
    assert dish.sky_temperature == 20.0

def test_dish_initialization_validation():
    """Test dish initialization validation."""
    with pytest.raises(ValueError, match="Diameter must be positive"):
        Dish(diameter=-1.0)
    
    with pytest.raises(ValueError, match="Efficiency must be between 0 and 1"):
        Dish(diameter=1.0, efficiency=1.5)
    
    with pytest.raises(ValueError, match="System temperature must be positive"):
        Dish(diameter=1.0, system_temperature=-10.0)
    
    with pytest.raises(ValueError, match="Sky temperature must be positive"):
        Dish(diameter=1.0, sky_temperature=-5.0)

def test_dish_property_setters():
    """Test property setters with validation."""
    dish = Dish(diameter=1.0)
    
    with pytest.raises(ValueError, match="Diameter must be positive"):
        dish.diameter = -1.0
    
    with pytest.raises(ValueError, match="Efficiency must be between 0 and 1"):
        dish.efficiency = 1.5
    
    with pytest.raises(ValueError, match="System temperature must be positive"):
        dish.system_temperature = -10.0
    
    with pytest.raises(ValueError, match="Sky temperature must be positive"):
        dish.sky_temperature = -5.0
    
    # Test valid property assignments
    dish.diameter = 2.0
    dish.efficiency = 0.8
    dish.lna_gain = 40.0
    dish.system_temperature = 300.0
    dish.sky_temperature = 25.0
    
    assert dish.diameter == 2.0
    assert dish.efficiency == 0.8
    assert dish.lna_gain == 40.0
    assert dish.system_temperature == 300.0
    assert dish.sky_temperature == 25.0

def test_dish_gain_in_db():
    """Test gain calculation at various frequencies.
    
    Test values generated from: https://www.keysight.com/used/us/en/knowledge/calculators/antenna-gain-calculator
    """
    dish = Dish(diameter=20, efficiency=0.65)
    
    # Test at X-band (8.4 GHz)
    frequency = 8.4e9
    assert dish.dish_gain(frequency) == pytest.approx(63.04, abs=0.01)
    assert dish.half_power_beamwidth(frequency) == pytest.approx(0.12, abs=0.01)
    
    # Test at S-band (2.4 GHz)
    frequency = 2.4e9
    assert dish.dish_gain(frequency) == pytest.approx(52.16, abs=0.01)
    assert dish.half_power_beamwidth(frequency) == pytest.approx(0.44, abs=0.01)

def test_total_gain_in_db():
    """Test total gain calculation (dish + LNA) in dB."""
    dish = Dish(diameter=20.0, efficiency=0.65, lna_gain=30.0)
    frequency = 2.4e9  # 2.4 GHz
    
    
    assert dish.gain(frequency) == pytest.approx(52.16 + 30.0, abs=0.01)

def test_gt_ratio():
    """Test G/T ratio calculation."""
    dish = Dish(diameter=20.0, efficiency=0.65, lna_gain=0.0,
                system_temperature=100, sky_temperature=50)
    frequency = 2.4e9  # 2.4 GHz
    
    # Calculate expected G/T
    assert dish.gt_ratio(frequency) == pytest.approx(30.39, abs=0.01)
