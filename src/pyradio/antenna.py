"""
Functions for antenna calculations.

This module provides utilities for calculating antenna parameters,
including dish gain and beamwidth.
"""

import numpy as np
from .conversions import db, wavelength


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
    gain_linear = efficiency * (np.pi * diameter / wavelength_m) ** 2
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
