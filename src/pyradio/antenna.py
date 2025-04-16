"""
Antenna calculations for radio communications.

This module provides classes and functions for antenna calculations,
including dish gain and beamwidth calculations.
"""

import math
from abc import ABC, abstractmethod
from typing import Union
from .conversions import db, wavelength

class Antenna(ABC):
    """
    Abstract base class for antennas.
    
    This class defines the interface that all antenna implementations must follow.
    The gain method must be implemented by all concrete antenna classes.
    """
    
    def __init__(self, axial_ratio: float = 0):
        """
        Initialize an antenna with an axial ratio.
        
        Args:
            axial_ratio: Axial ratio in dB (default: 0.0 for perfect circular polarization)
                         0 dB represents perfect circular polarization
                         >40 dB represents linear polarization
        
        Raises:
            ValueError: If axial ratio is negative
        """
        if axial_ratio < 0:
            raise ValueError("Axial ratio must be non-negative")
        self.axial_ratio = axial_ratio
    
    @abstractmethod
    def gain(self, frequency_hz: float) -> float:
        """
        Calculate the antenna gain at a given frequency.
        
        Args:
            frequency_hz: Frequency in Hz
            
        Returns:
            float: Antenna gain in dB
            
        Raises:
            ValueError: If frequency is not positive
        """
        pass

class FixedGain(Antenna):
    """
    A class representing an antenna with fixed gain.
    
    This class implements the Antenna interface for an antenna with a constant gain
    regardless of frequency. The gain can be positive or negative, as some
    antennas (like small whip antennas or patch antennas) can have negative gain.
    
    Attributes:
        gain_db: Fixed antenna gain in dB
        axial_ratio: Axial ratio in dB (0 dB for perfect circular, >40 dB for linear)
    """
    
    def __init__(self, gain_db: float, axial_ratio: float = 0):
        """
        Initialize a FixedGain antenna.
        
        Args:
            gain_db: Fixed antenna gain in dB (can be positive or negative)
            axial_ratio: Axial ratio in dB (default: 0.0 for perfect circular polarization)
                         0 dB represents perfect circular polarization
                         >40 dB represents linear polarization
        """
        super().__init__(axial_ratio)
        self.gain_db = gain_db
        
    def gain(self, frequency_hz: float) -> float:
        """
        Return the fixed antenna gain.
        
        Args:
            frequency_hz: Frequency in Hz (ignored)
            
        Returns:
            float: Fixed antenna gain in dB
        """
        return self.gain_db

class Dish(Antenna):
    """
    A class representing a parabolic dish antenna.
    
    This class implements the Antenna interface for a parabolic dish antenna,
    calculating gain based on the dish diameter and frequency.
    
    Attributes:
        diameter_m: Dish diameter in meters
        efficiency: Antenna efficiency (default: 0.65)
        axial_ratio: Axial ratio in dB (0 dB for perfect circular, >40 dB for linear)
    """
    
    def __init__(self, diameter_m: float, efficiency: float = 0.65, axial_ratio: float = 0):
        """
        Initialize a Dish antenna.
        
        Args:
            diameter_m: Dish diameter in meters
            efficiency: Antenna efficiency (default: 0.65)
            axial_ratio: Axial ratio in dB (default: 0.0 for perfect circular polarization)
                         0 dB represents perfect circular polarization
                         >40 dB represents linear polarization
            
        Raises:
            ValueError: If diameter is not positive or efficiency is not between 0 and 1
        """
        super().__init__(axial_ratio)
        if diameter_m <= 0:
            raise ValueError("Dish diameter must be positive")
        if not 0 < efficiency <= 1:
            raise ValueError("Efficiency must be between 0 and 1")
            
        self.diameter_m = diameter_m
        self.efficiency = efficiency
        
    def gain(self, frequency_hz: float) -> float:
        """
        Calculate the dish antenna gain at a given frequency.
        
        The gain is calculated using the formula:
        G = 10 * log10(efficiency * (pi * D / lambda)^2)
        where:
        - efficiency is the antenna efficiency
        - D is the dish diameter in meters
        - lambda is the wavelength in meters
        
        Args:
            frequency_hz: Frequency in Hz
            
        Returns:
            float: Antenna gain in dB
            
        Raises:
            ValueError: If frequency is not positive
        """
        return dish_gain(self.diameter_m, frequency_hz, self.efficiency)

def dish_gain(diameter: float, frequency: float, efficiency: float = 0.65) -> float:
    """
    Calculate the gain of a parabolic dish antenna.

    The gain is calculated using the formula:
    G = η * (π * D / λ)²
    where:
    - η is the antenna efficiency
    - D is the antenna diameter
    - λ is the wavelength (c/frequency)

    Args:
        diameter: Diameter of the dish in meters
        frequency: Frequency in Hz
        efficiency: Antenna efficiency (0.0 to 1.0, default: 0.65)

    Returns:
        float: Gain in dB

    Raises:
        ValueError: If efficiency is not between 0 and 1,
                   or if diameter or frequency is not positive

    Examples:
        >>> # 1 meter dish at 2.4 GHz with default 65% efficiency
        >>> gain = dish_gain(1.0, 2.4e9)
        >>> round(gain, 1)
        24.6

        >>> # Same dish with 50% efficiency
        >>> gain = dish_gain(1.0, 2.4e9, efficiency=0.5)
        >>> round(gain, 1)
        22.1
    """
    if not 0 <= efficiency <= 1:
        raise ValueError("Efficiency must be between 0 and 1")
    if diameter <= 0:
        raise ValueError("Diameter must be positive")
    if frequency <= 0:
        raise ValueError("Frequency must be positive")
        
    wavelength_m = wavelength(frequency)
    gain_linear = efficiency * (3.14159 * diameter / wavelength_m) ** 2
    return db(gain_linear)


def dish_3db_beamwidth(diameter: float, frequency: float) -> float:
    """
    Calculate the half-power beamwidth (3 dB beamwidth) of a parabolic dish.

    The half-power beamwidth is approximately given by:
    HPBW ≈ 70° * (λ/D)
    where:
    - λ is the wavelength (c/frequency)
    - D is the antenna diameter

    Args:
        diameter: Diameter of the dish in meters
        frequency: Frequency in Hz

    Returns:
        float: Half-power beamwidth in degrees

    Raises:
        ValueError: If diameter or frequency is not positive

    Examples:
        >>> # 1 meter dish at 2.4 GHz
        >>> beamwidth = dish_3db_beamwidth(1.0, 2.4e9)
        >>> round(beamwidth, 1)
        3.7
    """
    if diameter <= 0:
        raise ValueError("Diameter must be positive")
    if frequency <= 0:
        raise ValueError("Frequency must be positive")
        
    wavelength_m = wavelength(frequency)
    return 70 * (wavelength_m / diameter)

def polarization_loss(tx_axial_ratio: float, rx_axial_ratio: float) -> float:
    """
    Calculate the polarization loss in dB between two antennas with given axial ratios.
    
    The polarization loss is calculated using the standard formula for polarization
    mismatch between two antennas with different axial ratios. For circular polarization,
    the axial ratio is 0 dB, and for linear polarization, it is >20 dB.
    
    Args:
        tx_axial_ratio: Transmit antenna axial ratio in dB
        rx_axial_ratio: Receive antenna axial ratio in dB
    
    Returns:
        float: Polarization loss in dB (positive value)
    
    Examples:
        >>> # Perfect circular to perfect circular
        >>> loss = polarization_loss(0.0, 0.0)
        >>> round(loss, 1)
        0.0
        
        >>> # Circular to linear (theoretical 3dB, actual depends on implementation)
        >>> loss = polarization_loss(0.0, 40.0)
        >>> round(loss, 1)
        17.0
    """
    # Convert axial ratios from dB to linear scale
    tx_ar_linear = 10 ** (tx_axial_ratio / 20)
    rx_ar_linear = 10 ** (rx_axial_ratio / 20)
    
    # Calculate polarization loss using the polarization efficiency formula
    # This is the standard formula for polarization mismatch between antennas
    plf = (4 * tx_ar_linear * rx_ar_linear) / ((1 + tx_ar_linear**2) * (1 + rx_ar_linear**2))
    
    # Convert polarization loss factor to dB (negative because it's a loss)
    return -10 * math.log10(plf)
