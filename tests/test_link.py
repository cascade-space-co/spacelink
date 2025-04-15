"""
Tests for the link budget module.

This module contains pytest-style tests for link budget calculations.
"""

import pytest
from pyradio.link import Link
from pyradio.antenna import FixedGain, Dish
from pyradio.conversions import ghz, mhz, kilometers

def test_link_initialization():
    """Test Link initialization with valid parameters."""
    # Create antennas
    tx_antenna = FixedGain(10.0)
    rx_antenna = FixedGain(20.0)
    
    # Create link with valid parameters
    link = Link(tx_power_dbw=10.0,
                tx_antenna=tx_antenna,
                rx_antenna=rx_antenna,
                rx_system_noise_temp_k=290.0,
                rx_antenna_noise_temp_k=100.0,
                distance_fn=lambda: kilometers(1000.0),
                frequency_hz=ghz(2.4),
                bandwidth_hz=mhz(1.0),
                required_ebno_db=10.0)
                
    # Check parameters
    assert link.tx_power_dbw == pytest.approx(10.0, rel=0.01)
    assert link.tx_antenna == tx_antenna
    assert link.rx_antenna == rx_antenna
    assert link.rx_system_noise_temp_k == pytest.approx(290.0, rel=0.01)
    assert link.rx_antenna_noise_temp_k == pytest.approx(100.0, rel=0.01)
    assert link.distance_m == pytest.approx(1000000.0, rel=0.01)
    assert link.frequency_hz == pytest.approx(ghz(2.4), rel=0.01)
    assert link.bandwidth_hz == pytest.approx(mhz(1.0), rel=0.01)
    assert link.implementation_loss_db == pytest.approx(0.0, rel=0.01)
    assert link.polarization_loss_db == pytest.approx(0.0, rel=0.01)
    assert link.pointing_loss_db == pytest.approx(0.0, rel=0.01)
    assert link.atmospheric_loss_db == pytest.approx(0.0, rel=0.01)
    assert link.required_ebno_db == pytest.approx(10.0, rel=0.01)

def test_link_initialization_invalid():
    """Test Link initialization with invalid parameters."""
    # Create valid antennas for testing
    tx_antenna = FixedGain(10.0)
    rx_antenna = FixedGain(20.0)
    
    # Test invalid transmitter power
    with pytest.raises(ValueError, match="Transmitter power must be positive"):
        Link(tx_power_dbw=0.0,
             tx_antenna=tx_antenna,
             rx_antenna=rx_antenna,
             rx_system_noise_temp_k=290.0,
             rx_antenna_noise_temp_k=100.0,
             distance_fn=lambda: kilometers(1000.0),
             frequency_hz=ghz(2.4),
             bandwidth_hz=mhz(1.0),
             required_ebno_db=10.0)
             
    # Test invalid antenna types
    with pytest.raises(ValueError, match="tx_antenna must be an Antenna instance"):
        Link(tx_power_dbw=10.0,
             tx_antenna=10.0,  # Not an Antenna instance
             rx_antenna=rx_antenna,
             rx_system_noise_temp_k=290.0,
             rx_antenna_noise_temp_k=100.0,
             distance_fn=lambda: kilometers(1000.0),
             frequency_hz=ghz(2.4),
             bandwidth_hz=mhz(1.0),
             required_ebno_db=10.0)
             
    with pytest.raises(ValueError, match="rx_antenna must be an Antenna instance"):
        Link(tx_power_dbw=10.0,
             tx_antenna=tx_antenna,
             rx_antenna=20.0,  # Not an Antenna instance
             rx_system_noise_temp_k=290.0,
             rx_antenna_noise_temp_k=100.0,
             distance_fn=lambda: kilometers(1000.0),
             frequency_hz=ghz(2.4),
             bandwidth_hz=mhz(1.0),
             required_ebno_db=10.0)
             
    # Test invalid system noise temperature
    with pytest.raises(ValueError, match="System noise temperature must be positive"):
        Link(tx_power_dbw=10.0,
             tx_antenna=tx_antenna,
             rx_antenna=rx_antenna,
             rx_system_noise_temp_k=0.0,
             rx_antenna_noise_temp_k=100.0,
             distance_fn=lambda: kilometers(1000.0),
             frequency_hz=ghz(2.4),
             bandwidth_hz=mhz(1.0),
             required_ebno_db=10.0)
             
    # Test invalid antenna noise temperature
    with pytest.raises(ValueError, match="Antenna noise temperature cannot be negative"):
        Link(tx_power_dbw=10.0,
             tx_antenna=tx_antenna,
             rx_antenna=rx_antenna,
             rx_system_noise_temp_k=290.0,
             rx_antenna_noise_temp_k=-1.0,
             distance_fn=lambda: kilometers(1000.0),
             frequency_hz=ghz(2.4),
             bandwidth_hz=mhz(1.0),
             required_ebno_db=10.0)
             
    # Test invalid distance function
    with pytest.raises(ValueError, match="distance_fn must be a callable"):
        Link(tx_power_dbw=10.0,
             tx_antenna=tx_antenna,
             rx_antenna=rx_antenna,
             rx_system_noise_temp_k=290.0,
             rx_antenna_noise_temp_k=100.0,
             distance_fn=1000.0,  # Not a callable
             frequency_hz=ghz(2.4),
             bandwidth_hz=mhz(1.0),
             required_ebno_db=10.0)
             
    # Test invalid frequency
    with pytest.raises(ValueError, match="Frequency must be positive"):
        Link(tx_power_dbw=10.0,
             tx_antenna=tx_antenna,
             rx_antenna=rx_antenna,
             rx_system_noise_temp_k=290.0,
             rx_antenna_noise_temp_k=100.0,
             distance_fn=lambda: kilometers(1000.0),
             frequency_hz=0.0,
             bandwidth_hz=mhz(1.0),
             required_ebno_db=10.0)
             
    # Test invalid bandwidth
    with pytest.raises(ValueError, match="Bandwidth must be positive"):
        Link(tx_power_dbw=10.0,
             tx_antenna=tx_antenna,
             rx_antenna=rx_antenna,
             rx_system_noise_temp_k=290.0,
             rx_antenna_noise_temp_k=100.0,
             distance_fn=lambda: kilometers(1000.0),
             frequency_hz=ghz(2.4),
             bandwidth_hz=0.0,
             required_ebno_db=10.0)
             
    # Test invalid losses
    with pytest.raises(ValueError, match="Implementation loss cannot be negative"):
        Link(tx_power_dbw=10.0,
             tx_antenna=tx_antenna,
             rx_antenna=rx_antenna,
             rx_system_noise_temp_k=290.0,
             rx_antenna_noise_temp_k=100.0,
             distance_fn=lambda: kilometers(1000.0),
             frequency_hz=ghz(2.4),
             bandwidth_hz=mhz(1.0),
             required_ebno_db=10.0,
             implementation_loss_db=-1.0)
             
    with pytest.raises(ValueError, match="Polarization loss cannot be negative"):
        Link(tx_power_dbw=10.0,
             tx_antenna=tx_antenna,
             rx_antenna=rx_antenna,
             rx_system_noise_temp_k=290.0,
             rx_antenna_noise_temp_k=100.0,
             distance_fn=lambda: kilometers(1000.0),
             frequency_hz=ghz(2.4),
             bandwidth_hz=mhz(1.0),
             required_ebno_db=10.0,
             polarization_loss_db=-1.0)
             
    with pytest.raises(ValueError, match="Pointing loss cannot be negative"):
        Link(tx_power_dbw=10.0,
             tx_antenna=tx_antenna,
             rx_antenna=rx_antenna,
             rx_system_noise_temp_k=290.0,
             rx_antenna_noise_temp_k=100.0,
             distance_fn=lambda: kilometers(1000.0),
             frequency_hz=ghz(2.4),
             bandwidth_hz=mhz(1.0),
             required_ebno_db=10.0,
             pointing_loss_db=-1.0)
             
    with pytest.raises(ValueError, match="Atmospheric loss cannot be negative"):
        Link(tx_power_dbw=10.0,
             tx_antenna=tx_antenna,
             rx_antenna=rx_antenna,
             rx_system_noise_temp_k=290.0,
             rx_antenna_noise_temp_k=100.0,
             distance_fn=lambda: kilometers(1000.0),
             frequency_hz=ghz(2.4),
             bandwidth_hz=mhz(1.0),
             required_ebno_db=10.0,
             atmospheric_loss_db=-1.0)
             
    with pytest.raises(ValueError, match="Required Eb/N0 cannot be negative"):
        Link(tx_power_dbw=10.0,
             tx_antenna=tx_antenna,
             rx_antenna=rx_antenna,
             rx_system_noise_temp_k=290.0,
             rx_antenna_noise_temp_k=100.0,
             distance_fn=lambda: kilometers(1000.0),
             frequency_hz=ghz(2.4),
             bandwidth_hz=mhz(1.0),
             required_ebno_db=-1.0)

def test_link_calculations():
    """Test link budget calculations."""
    
    # Create link with typical parameters
    link = Link(tx_power_dbw=10.0,
                tx_antenna=Dish(20.0),
                rx_antenna=FixedGain(5.0),
                rx_system_noise_temp_k=100.0,
                rx_antenna_noise_temp_k=100.0,
                distance_fn=lambda: kilometers(1000.0),
                frequency_hz=ghz(2.4),
                bandwidth_hz=mhz(1.0),
                required_ebno_db=10.0)
                
    # Check EIRP calculation
    assert link.eirp == pytest.approx(62.16, abs=0.01)  # 10 dBW + 10 dB - 0 dB
    
    # Check path loss calculation
    assert link.path_loss_db == pytest.approx(160.05, abs=0.01)  # Free space path loss + 0 dB
    
    # Check received power calculation
    assert link.received_power_dbw == pytest.approx(-92.89, abs=0.01)  # 20 dBW - 122.58 dB + 20 dB - 0 dB
    
    # Check system noise temperature calculation
    assert link.system_noise_temperature_k == pytest.approx(200.0, abs=0.01)  # 100 K + 290 K
    
    # Check noise power calculation
    assert link.noise_power_dbw == pytest.approx(-145.59, abs=0.01)  # k * T * B
    
    # Check carrier-to-noise ratio calculation
    assert link.carrier_to_noise_ratio_db == pytest.approx(52.7, abs=0.01)  # -82.58 dBW - (-138.0 dBW)
    
    # Check Eb/N0 calculation
    assert link.ebno_db() == pytest.approx(52.7, abs=0.01)  # 55.42 dB + 30 dB
    
    # Check link margin calculation
    assert link.link_margin_db() == pytest.approx(42.7, abs=0.01)  
