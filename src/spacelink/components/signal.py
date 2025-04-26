"""
Signal module for radio signal representation.

This module provides the Signal class that represents a radio signal with its
properties such as power and noise temperature.
"""

# import astropy.units as u # Re-removed
from typing import Optional

from ..core.units import (
    Watts,
    Temperature,
    Decibels,
)


class Signal:
    """
    A class representing a radio signal with its properties.

    Attributes:
        power: Signal power in Watts
        noise_temperature: Noise temperature in Kelvin
        axial_ratio: Axial ratio in dB (None if not specified)
                     0 dB represents perfect circular polarization
                     >40 dB represents linear polarization
    """

    def __init__(
        self,
        power: Watts,
        noise_temperature: Temperature,
        axial_ratio: Optional[Decibels] = None,
    ):
        """
        Initialize a Signal.

        Args:
            power: Signal power in Watts
            noise_temperature: Noise temperature in Kelvin
            axial_ratio: Axial ratio in dB (None if not specified)
                        0 dB represents perfect circular polarization
                        >40 dB represents linear polarization
        """
        self.power = power
        self.noise_temperature = noise_temperature
        self.axial_ratio = axial_ratio

    def __eq__(self, other: object) -> bool:
        """Compare two signals for equality."""
        if not isinstance(other, Signal):
            return NotImplemented
        return (
            self.power == other.power
            and self.noise_temperature == other.noise_temperature
            and self.axial_ratio == other.axial_ratio
        )

    def __repr__(self) -> str:
        """Return a string representation of the signal."""
        return (
            f"Signal(power={self.power}, "
            f"noise_temperature={self.noise_temperature}, "
            f"axial_ratio={self.axial_ratio})"
        )
