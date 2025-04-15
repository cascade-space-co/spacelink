"""
Antenna calculations for radio communications.

This module provides classes and functions for antenna calculations,
including dish gain and beamwidth calculations.
"""

from abc import ABC, abstractmethod
from typing import Union
from .conversions import db, wavelength

class Antenna(ABC):
    """
    Abstract base class for antennas.
    
    This class defines the interface that all antenna implementations must follow.
    The gain method must be implemented by all concrete antenna classes.
    """
    
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
    regardless of frequency.
    
    Attributes:
        gain_db: Fixed antenna gain in dB
    """
    
    def __init__(self, gain_db: float):
        """
        Initialize a FixedGain antenna.
        
        Args:
            gain_db: Fixed antenna gain in dB
            
        Raises:
            ValueError: If gain is negative
        """
        if gain_db < 0:
            raise ValueError("Gain cannot be negative")
            
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
    """
    
    def __init__(self, diameter_m: float, efficiency: float = 0.65):
        """
        Initialize a Dish antenna.
        
        Args:
            diameter_m: Dish diameter in meters
            efficiency: Antenna efficiency (default: 0.65)
            
        Raises:
            ValueError: If diameter is not positive or efficiency is not between 0 and 1
        """
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
