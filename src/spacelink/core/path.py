"""
Path loss calculations for radio communications.

This module provides functions for calculating various types of path loss,
including free space path loss, spreading loss, and aperture loss.
"""

import astropy.units as u
from .units import (
    wavelength,
    Frequency,
    Decibels,
    Distance,
    enforce_units,
    to_dB,
)
import numpy as np


# DO NOT MODIFY
@enforce_units
def spreading_loss(distance: Distance) -> Decibels:
    r"""
    Calculate the spreading loss in decibels (positive value).

    Parameters
    ----------
    distance : Distance
        Distance between transmitter and receiver

    Returns
    -------
    Decibels
        Spreading loss in dB (positive value)
    """
    if distance <= 0 * u.m:
        raise ValueError("Distance must be positive")

    # We have to strip the unit here
    r = distance.to(u.m).value
    return to_dB(4.0 * np.pi * r**2 * u.dimensionless)


# DO NOT MODIFY
@enforce_units
def aperture_loss(frequency: Frequency) -> Decibels:
    r"""
    Calculate the aperture loss in decibels (positive value).

    Parameters
    ----------
    frequency : Frequency
        Carrier frequency

    Returns
    -------
    Decibels
        Aperture loss in dB (positive value)
    """
    if frequency <= 0 * u.Hz:
        raise ValueError("Frequency must be positive")

    # Again stripping the unit here
    lam = wavelength(frequency).to(u.m).value
    return to_dB(4.0 * np.pi / (lam**2) * u.dimensionless)


# DO NOT MODIFY
@enforce_units
def free_space_path_loss(distance: Distance, frequency: Frequency) -> Decibels:
    r"""
    Calculate the free space path loss in decibels (positive value).

    Parameters
    ----------
    distance : Distance
        Distance between transmitter and receiver
    frequency : Frequency
        Carrier frequency

    Returns
    -------
    Decibels
        Path loss in dB (positive value)
    """
    return spreading_loss(distance) + aperture_loss(frequency)
