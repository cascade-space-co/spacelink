"""
Functions for converting between decibels (dB) and linear scale values.

This module provides utility functions for converting values between decibel
and linear scales, commonly used in radio, acoustics, and signal processing.
"""

import math
from typing import Union

# Speed of light in meters per second
SPEED_OF_LIGHT = 299792458


def db2linear(db_value: float) -> float:
    """
    Convert a value from decibels (dB) to linear scale.

    Args:
        db_value: The value in decibels to convert

    Returns:
        float: The equivalent value in linear scale

    Examples:
        >>> db2linear(0)
        1.0
        >>> round(db2linear(3), 4)
        1.9953
        >>> round(db2linear(-6), 4)
        0.5012
    """
    return 10 ** (db_value / 10)


def db(linear_value: float) -> float:
    """
    Convert a value from linear scale to decibels (dB).

    Args:
        linear_value: The linear value to convert (must be positive)

    Returns:
        float: The equivalent value in decibels

    Raises:
        ValueError: If linear_value is less than or equal to 0

    Examples:
        >>> db(1)
        0.0
        >>> round(db(2), 4)
        3.0103
        >>> round(db(0.5), 4)
        -3.0103
    """
    if linear_value <= 0:
        raise ValueError("Linear value must be positive")
    return 10 * math.log10(linear_value)


def wavelength(frequency: float) -> float:
    """
    Calculate the wavelength from a frequency.

    Args:
        frequency: The frequency in Hertz (Hz)

    Returns:
        float: The wavelength in meters

    Raises:
        ValueError: If frequency is less than or equal to 0

    Examples:
        >>> wavelength(1000000)  # 1 MHz
        299.792458
        >>> round(wavelength(2450000000), 6)  # 2.45 GHz (microwave)
        0.122365
        >>> round(wavelength(5.8e9), 6)  # 5.8 GHz (WiFi)
        0.051688
    """
    if frequency <= 0:
        raise ValueError("Frequency must be positive")
    return SPEED_OF_LIGHT / frequency


def ghz(frequency: float) -> float:
    """
    Convert a frequency from GHz to Hz.
    
    Args:
        frequency: The frequency in GHz
        
    Returns:
        float: The frequency in Hz
        
    Examples:
        >>> ghz(1.0)
        1000000000.0
        >>> ghz(2.4)  # WiFi frequency
        2400000000.0
    """
    return frequency * 1e9


def mhz(frequency: float) -> float:
    """
    Convert a frequency from MHz to Hz.
    
    Args:
        frequency: The frequency in MHz
        
    Returns:
        float: The frequency in Hz
        
    Examples:
        >>> mhz(1.0)
        1000000.0
        >>> mhz(88.5)  # FM radio frequency
        88500000.0
    """
    return frequency * 1e6


def khz(frequency: float) -> float:
    """
    Convert a frequency from kHz to Hz.
    
    Args:
        frequency: The frequency in kHz
        
    Returns:
        float: The frequency in Hz
        
    Examples:
        >>> khz(1.0)
        1000.0
        >>> khz(455.0)  # AM radio IF frequency
        455000.0
    """
    return frequency * 1e3


def kilometers(distance: float) -> float:
    """
    Convert a distance from kilometers to meters.
    
    Args:
        distance: The distance in kilometers
        
    Returns:
        float: The distance in meters
        
    Examples:
        >>> kilometers(1.0)
        1000.0
        >>> kilometers(36e3)  # Geostationary orbit altitude
        36000000.0
    """
    return distance * 1000
