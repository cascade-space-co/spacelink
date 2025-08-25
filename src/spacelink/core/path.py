r"""
The Friis Transmission Equation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The complete Friis transmission equation is:

.. math::

   P_r = P_t G_t G_r \left(\frac{\lambda}{4\pi d}\right)^2

where:

* :math:`P_r` is the received power
* :math:`P_t` is the transmitted power
* :math:`G_t` is the gain of the transmitting antenna
* :math:`G_r` is the gain of the receiving antenna
* :math:`\lambda` is the wavelength
* :math:`d` is the distance between antennas

Loss Components
^^^^^^^^^^^^^^^

.. math::

   \text{FSPL} = \underbrace{(4\pi d^2)}_{\text{Spreading Loss}} \times\
   \underbrace{\left(\frac{4\pi}{\lambda^2}\right)}_{\text{Aperture Loss}}


Spreading Loss
^^^^^^^^^^^^^^
The spreading loss is the loss due to spherical spreading of the plane wave.

Aperture Loss
^^^^^^^^^^^^^
The aperture loss is the loss due to the effective aperture of the antenna.
The aperture loss term is actually the effective aperture of an ideal isotropic antenna.
"""

import astropy.units as u
import numpy as np

from .units import (
    Decibels,
    Distance,
    Frequency,
    enforce_units,
    to_dB,
    wavelength,
)


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
