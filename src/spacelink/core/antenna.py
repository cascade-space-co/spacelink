"""
Core antenna calculation functions.
"""

import astropy.units as u # Added import
import numpy as np # Added import

# Assuming units.py is now in the same core directory
from .units import (
    Decibels,
    Dimensionless,
    Frequency,
    Length,
    wavelength, # Need wavelength
    enforce_units,
    to_dB,
    to_linear,
)


@enforce_units
def polarization_loss(ar1: Decibels, ar2: Decibels) -> Decibels:
    """
    -VALIDATED-
    Calculate the polarization loss in dB between two antennas with given axial ratios.

    The polarization loss is calculated using the standard formula for polarization
    mismatch between two antennas with different axial ratios. For circular polarization,
    the axial ratio is 0 dB, and for linear polarization, it is >40 dB.

    Args:
        ar1: First antenna axial ratio in dB (amplitude ratio)
        ar2: Second antenna axial ratio in dB (amplitude ratio)

    Returns:
        Polarization loss in dB (positive value)

    Examples:
        >>> polarization_loss(0 * u.dB, 0 * u.dB)  # Both circular
        <Quantity(0., 'dB')>
        >>> polarization_loss(0 * u.dB, 60 * u.dB)  # Circular to linear
        <Quantity(3.01, 'dB')>
        >>> polarization_loss(3 * u.dB, 3 * u.dB)  # Same elliptical polarization
        <Quantity(0.51, 'dB')>

    """
    # Polarization mismatch angle is omitted (assumed to be 0 degrees)
    # https://www.microwaves101.com/encyclopedias/polarization-mismatch-between-antennas
    gamma1 = to_linear(ar1, factor=20)
    gamma2 = to_linear(ar2, factor=20)

    numerator = 4 * gamma1 * gamma2 - (1 - gamma1**2) * (1 - gamma2**2)
    denominator = (1 + gamma1**2) * (1 + gamma2**2)

    # Calculate polarization loss factor
    plf = 0.5 + 0.5 * (numerator / denominator)

    # Convert to decibels and make positive (loss)
    return -to_dB(plf)


@enforce_units
def dish_gain(
    diameter: Length,
    frequency: Frequency,
    efficiency: Dimensionless
) -> Decibels:
    """
    -VALIDATED-
    Calculate the gain in dB of a parabolic dish antenna.

    Args:
        diameter: Dish diameter.
        frequency: Frequency.
        efficiency: Antenna efficiency (dimensionless).

    Returns:
        Gain in decibels (dB).

    Raises:
        ValueError: If frequency is not positive.
    """
    # Added check back
    if frequency <= 0 * u.Hz:
        raise ValueError("Frequency must be positive")

    wl = wavelength(frequency)
    gain_linear = efficiency * (np.pi * diameter / wl) ** 2
    return to_dB(gain_linear)
