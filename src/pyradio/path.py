"""
Functions for calculating path loss in radio communications.

This module provides utilities for calculating various types of path loss,
including free space path loss and other propagation models.
"""

import math
from typing import Union
from .conversions import db, wavelength


def spreading_loss(distance_m: Union[float, int]) -> float:
    """
    Calculate spreading loss in dB.
    The spreading loss is given by:
    Spreading Loss(dB) = 20 * log10(4πd)
    where d is the distance in meters
    
    Args:
        distance_m: Distance between transmitter and receiver in meters
        
    Returns:
        float: Spreading loss in dB
        
    Raises:
        ValueError: If distance is less than or equal to 0
        
    Examples:
        >>> spreading_loss(1000)  # 1 km
        71.5
        >>> spreading_loss(36000000)  # Geostationary satellite distance
        144.2
    """
    if distance_m <= 0:
        raise ValueError("Distance must be positive")
        
    return db(4 * math.pi * distance_m**2)

def aperture_loss(frequency_hz: Union[float, int]) -> float:
    """
    Calculate aperture loss in dB.
    The aperture loss is given by:
    Aperture Loss = λ^2/(4π)
    where:
    - λ is the wavelength in meters
    
    Args:
        frequency_hz: Frequency in Hz
        
    Returns:
        float: Aperture loss in dB
        
    Raises:
        ValueError: If frequency is less than or equal to 0
        
    Examples:
        >>> aperture_loss(2.4e9)  # WiFi frequency
        20.0
        >>> aperture_loss(12e9)  # Ku-band
        32.0
    """
    if frequency_hz <= 0:
        raise ValueError("Frequency must be positive")
    
    wavelength_m = wavelength(frequency_hz)
    return db((4*math.pi)/wavelength_m**2)

def free_space_path_loss_db(distance_m: Union[float, int], 
                           frequency_hz: Union[float, int]) -> float:
    """
    Calculate free space path loss in dB.
    
    The free space path loss is given by:
    FSPL = (λ/4πR)^2
    where:
    - R is the distance in meters
    - λ is the wavelength in meters
    
    Args:
        distance_m: Distance between transmitter and receiver in meters
        frequency_hz: Frequency in Hz
        
    Returns:
        float: Free space path loss in dB
        
    Raises:
        ValueError: If distance or frequency is less than or equal to 0
        
    Examples:
        >>> free_space_path_loss_db(1000, 2.4e9)  # 1 km at WiFi frequency
        91.5
        >>> free_space_path_loss_db(36000000, 12e9)  # Geostationary satellite at Ku-band
        176.2
    """
    return spreading_loss(distance_m) + aperture_loss(frequency_hz) 