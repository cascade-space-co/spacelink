"""
Receiver module for RF receivers.

This module provides the Receiver class that represents a complete RF receiver
chain, including low-noise amplifiers, filters, and other components.
"""

# from abc import ABC, abstractmethod # Unused
# from typing import Optional, Callable # Unused
import astropy.units as u
from astropy.units import Quantity, UnitConversionError

from .stage import Stage
from .signal import Signal
from ..core.units import (
    Frequency,
    # Watts, # Unused
    # Temperature, # Unused
    Decibels,
    enforce_units,
)
from ..core.noise import temperature_to_noise_figure


class Receiver(Stage):
    """
    A class representing an RF receiver.

    A receiver is a stage that processes input signals, typically including
    components like low-noise amplifiers, filters, and other RF stages.

    Attributes:
        gain: Gain in dB
        noise_temperature: Noise temperature in Kelvin
    """

    def __init__(self, gain: Quantity, noise_temperature: Quantity):
        """
        Initialize a Receiver.

        Args:
            gain: The gain of the receiver in dB.
            noise_temperature: The noise temperature of the receiver in Kelvin.

        Raises:
            TypeError: If gain or noise_temperature is not a Quantity
            UnitConversionError: If gain or noise_temperature has wrong units
            ValueError: If gain or noise_temperature is negative
        """
        super().__init__()

        # Validate gain
        if not isinstance(gain, Quantity):
            raise TypeError("Gain must be a Quantity with dB units")
        if not gain.unit.is_equivalent(u.dB):
            raise UnitConversionError("Gain must have dB units")
        if gain < 0 * u.dB:
            raise ValueError("Gain must be non-negative")
        self._gain = gain

        # Validate noise temperature
        if not isinstance(noise_temperature, Quantity):
            raise TypeError(
                "Noise temperature must be a Quantity with temperature units"
            )
        try:
            noise_temperature = noise_temperature.to(u.K, equivalencies=u.temperature())
        except UnitConversionError:
            raise UnitConversionError("Noise temperature must have temperature units")
        if noise_temperature < 0 * u.K:
            raise ValueError("Noise temperature must be non-negative")
        self._noise_temperature = noise_temperature

        # Initialize processed signal
        self._processed_signal = None

    @enforce_units
    def gain(self, frequency: Frequency) -> Decibels:
        """Get the gain at a specific frequency.

        Args:
            frequency: The frequency at which to get the gain (not used)

        Returns:
            The gain in dB
        """
        return self._gain

    def noise_figure(self, frequency: Frequency) -> Decibels:
        """Get the noise figure at a specific frequency."""
        return temperature_to_noise_figure(self._noise_temperature)

    @property
    def noise_temperature(self) -> Quantity:
        """Get the noise temperature of the receiver.

        Returns:
            The noise temperature in Kelvin.
        """
        return self._noise_temperature

    @noise_temperature.setter
    def noise_temperature(self, value: Quantity) -> None:
        """Set the noise temperature in Kelvin."""
        if not isinstance(value, Quantity):
            raise TypeError(
                "Noise temperature must be a Quantity with temperature units"
            )
        if not value.unit.is_equivalent(u.K):
            raise UnitConversionError("Noise temperature must have temperature units")
        if value < 0 * u.K:
            raise ValueError("Noise temperature must be non-negative")
        self._noise_temperature = value.to(u.K)

    def process_input(self, frequency: Quantity) -> None:
        """Process the input signal at the given frequency.

        Args:
            frequency: The frequency at which to process the signal.

        Raises:
            ValueError: If input is not set
            TypeError: If frequency is not a Quantity or input does not provide a Signal
        """
        if not isinstance(frequency, Quantity):
            raise TypeError("Frequency must be a Quantity with frequency units")
        if not frequency.unit.is_equivalent(u.Hz):
            raise UnitConversionError("Frequency must have frequency units")
        if frequency <= 0 * u.Hz:
            raise ValueError("Frequency must be positive")

        if self.input is None:
            raise ValueError("Receiver input must be set before processing")

        input_signal = self.input.output(frequency)
        if not isinstance(input_signal, Signal):
            raise TypeError("Input must provide a Signal")

        # Apply gain to input power
        output_power = input_signal.power * 10 ** (
            self.gain(frequency).to(u.dB).value / 10
        )

        # Add noise temperatures
        output_noise_temp = input_signal.noise_temperature + self.noise_temperature

        self._processed_signal = Signal(
            power=output_power, noise_temperature=output_noise_temp
        )

    def output(self, frequency: Quantity) -> Signal:
        """Get the output signal at the given frequency.

        Args:
            frequency: The frequency at which to get the output signal.

        Returns:
            The output signal.

        Raises:
            ValueError: If input is not set
        """
        if self.input is None:
            raise ValueError("Receiver input must be set before calculating output")

        if self._processed_signal is None:
            self.process_input(frequency)
        return self._processed_signal
