"""
Source module for RF signal sources.

This module provides the Source base class that represents components that generate
signals without requiring an input, such as oscillators or signal generators.
"""

from abc import ABC, abstractmethod
import astropy.units as u

from .signal import Signal
from ..core.units import (
    Frequency,
    Temperature,
    # Decibels, # Unused
    enforce_units,
)


class Source(ABC):
    """
    Base class for signal sources.

    A source represents a component that generates signals without requiring an input,
    such as an oscillator or signal generator. Sources are the starting points of
    RF chains.
    """

    def __init__(self):
        self._noise_temperature: Temperature = 290 * u.K

    @property
    def noise_temperature(self) -> Temperature:
        """
        Get the noise temperature of this source.

        Returns:
            Temperature: The noise temperature in Kelvin
        """
        return self._noise_temperature

    @noise_temperature.setter
    @enforce_units
    def noise_temperature(self, value: Temperature) -> None:
        """
        Set the noise temperature of this source.

        Args:
            value: The noise temperature in Kelvin
        """
        self._noise_temperature = value

    @abstractmethod
    def output(self, frequency: Frequency) -> Signal:
        """
        Calculate the output signal for this source.

        Args:
            frequency: The frequency at which to calculate the output

        Returns:
            Signal: The output signal from this source

        Raises:
            NotImplementedError: If the source does not implement this method
        """
        pass
