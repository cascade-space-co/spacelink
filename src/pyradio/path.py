"""
Path loss calculations for radio communications.

This module provides functions for calculating various types of path loss,
including free space path loss, spreading loss, and aperture loss.
"""

import math
from pint import Quantity
from pyradio.units import Q_, wavelength, m, Hz


def spreading_loss(distance: Quantity) -> Quantity:
    """Calculate the spreading loss.
        The spreading loss is loss simply due to spherical spreading of the wave
    Args:
        distance: Distance between transmitter and receiver

    Returns:
        Spreading loss
    """
    if distance.magnitude <= 0:
        raise ValueError("Distance must be positive")

    return 1/(4 * math.pi * distance.to(m)**2)


def aperture_loss(frequency: Quantity) -> Quantity:
    """Calculate the aperture loss.

    The aperture loss is the loss due to the effective area of the antenna
    being less than the physical area.

    Args:
        frequency: Carrier frequency

    Returns:
        Aperture loss
    """
    if frequency.magnitude <= 0:
        raise ValueError("Frequency must be positive")

    return wavelength(frequency)**2 / (4 * math.pi)


def free_space_path_loss(distance: Quantity, frequency: Quantity) -> float:
    """Calculate the free space path loss.
    This is the basic Friis equation for free space path loss
    Args:
        distance: Distance between transmitter and receiver
        frequency: Carrier frequency

    Returns:
        Free space path loss in dB
    """
    return spreading_loss(distance) * aperture_loss(frequency)
