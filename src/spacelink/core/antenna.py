"""
Core antenna calculation functions.
"""

import astropy.units as u
import numpy as np

from .units import (
    Decibels,
    Dimensionless,
    Frequency,
    Length,
    wavelength,
    enforce_units,
    to_dB,
    to_linear,
    safe_negate,
)


@enforce_units
def polarization_loss(ar1: Decibels, ar2: Decibels) -> Decibels:
    r"""
    Calculate the polarization loss in dB between two antennas with given axial ratios.

    Parameters
    ----------
    ar1 : Quantity
        First antenna axial ratio in dB (amplitude ratio)
    ar2 : Quantity
        Second antenna axial ratio in dB (amplitude ratio)

    Returns
    -------
    Quantity
        Polarization loss in dB (positive value)
    """
    # Polarization mismatch angle is omitted (assumed to be 90 degrees)
    # https://www.microwaves101.com/encyclopedias/polarization-mismatch-between-antennas
    gamma1 = to_linear(ar1, factor=20)
    gamma2 = to_linear(ar2, factor=20)

    numerator = 4 * gamma1 * gamma2 - (1 - gamma1**2) * (1 - gamma2**2)
    denominator = (1 + gamma1**2) * (1 + gamma2**2)

    # Calculate polarization loss factor
    plf = 0.5 + 0.5 * (numerator / denominator)
    return safe_negate(to_dB(plf))


@enforce_units
def dish_gain(
    diameter: Length, frequency: Frequency, efficiency: Dimensionless
) -> Decibels:
    r"""
    Calculate the gain in dB of a parabolic dish antenna.

    Parameters
    ----------
    diameter : Quantity
        Dish diameter
    frequency : Quantity
        Frequency
    efficiency : Quantity
        Antenna efficiency (dimensionless)

    Returns
    -------
    Quantity
        Gain in decibels (dB)

    Raises
    ------
    ValueError
        If frequency is not positive
    """
    if frequency <= 0 * u.Hz:
        raise ValueError("Frequency must be positive")

    wl = wavelength(frequency)
    gain_linear = efficiency * (np.pi * diameter.to(u.m) / wl) ** 2
    return to_dB(gain_linear)
