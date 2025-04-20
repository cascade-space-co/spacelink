"""
Path loss calculations for radio communications.

This module provides functions for calculating various types of path loss,
including free space path loss, spreading loss, and aperture loss.
"""

import math
from pint import Quantity
from spacelink.units import wavelength, m, Hz


def spreading_loss(distance: Quantity) -> Quantity:
    """Calculate the spreading loss.
        The spreading loss is loss simply due to spherical spreading of the wave
    Args:
        distance: Distance between transmitter and receiver

    Returns:
        Spreading loss
    """
    """
    Calculate the spreading loss in decibels (positive value).

    Spreading loss (dB) = 10 * log10(4 * pi * r^2)
    where r is the distance in meters.
    """
    # Validate input
    if distance.to(m).magnitude <= 0:
        raise ValueError("Distance must be positive")
    # Compute loss
    r = distance.to(m).magnitude
    loss_db = 10.0 * math.log10(4.0 * math.pi * r**2)
    return loss_db


def aperture_loss(frequency: Quantity) -> Quantity:
    """Calculate the aperture loss.

    The aperture loss is the loss due to the effective area of the antenna
    being less than the physical area.

    Args:
        frequency: Carrier frequency

    Returns:
        Aperture loss
    """
    """
    Calculate the aperture loss in decibels (positive value).

    Aperture loss (dB) = 10 * log10(4 * pi / lambda^2)
    where lambda is the wavelength in meters.
    """
    # Validate input
    if frequency.to(Hz).magnitude <= 0:
        raise ValueError("Frequency must be positive")
    # Compute loss
    lam = wavelength(frequency).to(m).magnitude
    loss_db = 10.0 * math.log10(4.0 * math.pi / (lam**2))
    return loss_db


def free_space_path_loss(distance: Quantity, frequency: Quantity) -> float:
    """
    Calculate the free space path loss in decibels (positive value).

    Path loss (dB) = spreading_loss(d) + aperture_loss(f)
    = 20*log10(4*pi*d/lambda)

    Args:
        distance: Distance between transmitter and receiver
        frequency: Carrier frequency

    Returns:
        Path loss in dB (positive value)
    """
    # The individual functions validate inputs
    sl = spreading_loss(distance)
    al = aperture_loss(frequency)
    return sl + al
