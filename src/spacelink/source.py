"""
Source module for RF signal sources.

This module provides the Source base class that represents components that generate
signals without requiring an input, such as oscillators or signal generators.
"""

import astropy.units as u

from spacelink.signal import Signal
from spacelink.units import Frequency, Temperature, Decibels, enforce_units
from spacelink.stage import Stage


class Source(Stage):
    """
    Base class for signal sources.

    A source represents a component that generates signals without requiring an input,
    such as an oscillator or signal generator. Sources are the starting points of
    RF chains.
    """
    def __init__(self):
        super().__init__()
        self._noise_temperature: Temperature = 290 * u.K

    @property
    def input(self):
        """
        Sources do not have inputs.

        Raises:
            AttributeError: Always, since sources do not have inputs
        """
        raise AttributeError("Source objects do not have inputs")

    @input.setter
    def input(self, value):
        """
        Sources do not have inputs.

        Raises:
            AttributeError: Always, since sources do not have inputs
        """
        raise AttributeError("Source objects do not have inputs")

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

    @property
    def cascaded_gain(self) -> Decibels:
        """
        Get the cascaded gain up to this point.

        For a source, this is 0 dB since it's the start of the chain.

        Returns:
            Decibels: The cascaded gain in dB
        """
        return 0 * u.dB

    @property
    def cascaded_noise_figure(self) -> Decibels:
        """
        Get the cascaded noise figure up to this point.

        For a source, this is 0 dB since it's the start of the chain.

        Returns:
            Decibels: The cascaded noise figure in dB
        """
        return 0 * u.dB

    @property
    def gain(self) -> Decibels:
        """
        Get the gain of this source.

        For a source, this is 0 dB since it's the start of the chain.

        Returns:
            Decibels: The gain in dB
        """
        return 0 * u.dB

    @property
    def noise_figure(self) -> Decibels:
        """
        Get the noise figure of this source.

        For a source, this is 0 dB since it's the start of the chain.

        Returns:
            Decibels: The noise figure in dB
        """
        return 0 * u.dB

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
        raise NotImplementedError("Source must implement output method") 