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

    The spreading loss represents the attenuation due to the spherical spreading
    of electromagnetic waves and is given by:

    .. math::
        L_{spreading} = 10 \log_{10}(4\pi d^2)

    where :math:`d` is the distance in meters.

    Parameters
    ----------
    distance : Quantity
        Distance between transmitter and receiver

    Returns
    -------
    Quantity
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

    The aperture loss represents the loss due to the wavelength-dependent effective
    aperture of an ideal isotropic antenna and is given by:

    .. math::
        L_{aperture} = 10 \log_{10}\left(\frac{4\pi}{\lambda^2}\right)

    where :math:`\lambda` is the wavelength in meters.

    This can also be expressed in terms of frequency:

    .. math::
        L_{aperture} = 20 \log_{10}(f) - 20 \log_{10}(c) + 20 \log_{10}(4\pi)

    where :math:`f` is the frequency in Hz and :math:`c` is the speed of light.

    Parameters
    ----------
    frequency : Quantity
        Carrier frequency

    Returns
    -------
    Quantity
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

    The free space path loss (FSPL) represents the attenuation of radio energy
    between two points in free space and is given by:

    .. math::
        FSPL = \left(\frac{4\pi d}{\lambda}\right)^2

    where :math:`d` is the distance and :math:`\lambda` is the wavelength.

    In decibels:

    .. math::
        FSPL_{dB} = 20\log_{10}\left(\frac{4\pi d}{\lambda}\right)

    In SpaceLink, this is calculated as the sum of two components:

    .. math::
        FSPL_{dB} = L_{spreading} + L_{aperture}

    where :math:`L_{spreading}` is the spreading loss and :math:`L_{aperture}` is the aperture loss.

    Parameters
    ----------
    distance : Quantity
        Distance between transmitter and receiver
    frequency : Quantity
        Carrier frequency

    Returns
    -------
    Quantity
        Path loss in dB (positive value)
    """
    return spreading_loss(distance) + aperture_loss(frequency)
