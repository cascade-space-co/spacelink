"""
Path loss calculations for radio communications.

This module provides functions for calculating various types of path loss,
including free space path loss, spreading loss, and aperture loss.
"""

import astropy.units as u
from spacelink.units import (
    wavelength,
    Frequency,
    Decibels,
    Distance,
    enforce_units,
    to_dB,
)
import numpy as np


@enforce_units
def spreading_loss(distance: Distance) -> Decibels:
    """
    Calculate the spreading loss in decibels (positive value).

    Spreading loss (dB) = 10 * log10(4 * pi * r^2)
    where r is the distance in meters.

    Args:
        distance: Distance between transmitter and receiver

    Returns:
        Spreading loss in dB (positive value)
    """
    # Validate input
    if distance <= 0 * u.m:
        raise ValueError("Distance must be positive")

    # We have to strip the unit here
    r = distance.to(u.m).value
    return to_dB(4.0 * np.pi * r**2 * u.dimensionless)


@enforce_units
def aperture_loss(frequency: Frequency) -> Decibels:
    """
    Calculate the aperture loss in decibels (positive value).

    Aperture loss (dB) = 10 * log10(4 * pi / lambda^2)
    where lambda is the wavelength in meters.

    Args:
        frequency: Carrier frequency

    Returns:
        Aperture loss in dB (positive value)
    """
    # Validate input
    if frequency <= 0 * u.Hz:
        raise ValueError("Frequency must be positive")

    # Again stripping the unit here
    lam = wavelength(frequency).to(u.m).value
    return to_dB(4.0 * np.pi / (lam**2) * u.dimensionless)


@enforce_units
def free_space_path_loss(distance: Distance, frequency: Frequency) -> Decibels:
    """
    Calculate the free space path loss

    Sum of spreading loss and aperture loss
    Args:
        distance: Distance between transmitter and receiver
        frequency: Carrier frequency

    Returns:
        Path loss in dB (positive value)
    """
    return spreading_loss(distance) + aperture_loss(frequency)
