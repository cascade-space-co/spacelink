"""
Transmitter module for RF transmitters.

This module provides the Transmitter class that represents a complete RF transmitter
chain, including power amplifiers, filters, and other components.
"""

import astropy.units as u

from spacelink.source import Source
from spacelink.signal import Signal
from spacelink.units import Frequency, Watts, Temperature, enforce_units


class Transmitter(Source):
    """
    A class representing an RF transmitter.

    A transmitter is a source that generates signals with specific power and
    noise characteristics. It can include components like power amplifiers,
    filters, and other RF stages.

    Attributes:
        power: Output power in Watts
        noise_temperature: Noise temperature in Kelvin
    """
    @enforce_units
    def __init__(self, power: Watts, noise_temperature: Temperature):
        """
        Initialize a Transmitter.

        Args:
            power: Output power in Watts
            noise_temperature: Noise temperature in Kelvin
        """
        self.power = power
        self.noise_temperature = noise_temperature

    @enforce_units
    def output(self, frequency: Frequency) -> Signal:
        """
        Calculate the output signal for this transmitter.

        Args:
            frequency: The frequency at which to calculate the output

        Returns:
            Signal: The output signal from this transmitter
        """
        return Signal(power=self.power, noise_temperature=self.noise_temperature) 