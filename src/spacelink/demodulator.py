"""
Demodulator module for RF demodulators.

This module provides the Demodulator class that represents a complete RF demodulator
chain, including mixers, filters, and other components that convert RF signals
to baseband.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable
import astropy.units as u

from spacelink.sink import Sink
from spacelink.signal import Signal
from spacelink.units import Frequency, Watts, Temperature, Decibels, enforce_units


class Demodulator(Sink):
    """
    A class representing an RF demodulator.

    A demodulator is a sink that processes input signals to extract baseband
    information, typically including components like mixers, filters, and
    other RF stages.

    Attributes:
        conversion_loss: Conversion loss in dB
        noise_temperature: Noise temperature in Kelvin
    """
    def __init__(self, conversion_loss: Decibels, noise_temperature: Temperature):
        """
        Initialize a Demodulator.

        Args:
            conversion_loss: Conversion loss in dB
            noise_temperature: Noise temperature in Kelvin

        Raises:
            TypeError: If conversion_loss is not a Quantity with dB units
            TypeError: If noise_temperature is not a Quantity with temperature units
            ValueError: If conversion_loss is negative
            ValueError: If noise_temperature is negative
        """
        super().__init__()
        
        # Validate conversion loss
        if not isinstance(conversion_loss, u.Quantity):
            raise TypeError("Conversion loss must be a Quantity with dB units")
        if not conversion_loss.unit.is_equivalent(u.dB):
            raise u.UnitConversionError(f"Cannot convert {conversion_loss.unit} to dB")
        if conversion_loss < 0 * u.dB:
            raise ValueError("Conversion loss must be non-negative")
        self._conversion_loss = conversion_loss
        
        # Validate noise temperature
        if not isinstance(noise_temperature, u.Quantity):
            raise TypeError("Noise temperature must be a Quantity with temperature units")
        if not noise_temperature.unit.is_equivalent(u.K):
            raise u.UnitConversionError(f"Cannot convert {noise_temperature.unit} to K")
        if noise_temperature < 0 * u.K:
            raise ValueError("Noise temperature must be non-negative")
        self._noise_temperature = noise_temperature

    @property
    def conversion_loss(self) -> Decibels:
        """Get the conversion loss in dB."""
        return self._conversion_loss

    @conversion_loss.setter
    def conversion_loss(self, value: Decibels) -> None:
        """Set the conversion loss in dB."""
        if not isinstance(value, u.Quantity):
            raise TypeError("Conversion loss must be a Quantity with dB units")
        if not value.unit.is_equivalent(u.dB):
            raise u.UnitConversionError(f"Cannot convert {value.unit} to dB")
        if value < 0 * u.dB:
            raise ValueError("Conversion loss must be non-negative")
        self._conversion_loss = value

    @property
    def noise_temperature(self) -> Temperature:
        """Get the noise temperature in Kelvin."""
        return self._noise_temperature

    @noise_temperature.setter
    def noise_temperature(self, value: Temperature) -> None:
        """Set the noise temperature in Kelvin."""
        if not isinstance(value, u.Quantity):
            raise TypeError("Noise temperature must be a Quantity with temperature units")
        if not value.unit.is_equivalent(u.K):
            raise u.UnitConversionError(f"Cannot convert {value.unit} to K")
        if value < 0 * u.K:
            raise ValueError("Noise temperature must be non-negative")
        self._noise_temperature = value

    def process_input(self, frequency: Frequency) -> None:
        """
        Process the input signal at the given frequency.

        Args:
            frequency: The frequency at which to process the input

        Raises:
            ValueError: If input is not set

        This method simulates the demodulation process by calculating the
        output power and noise temperature after conversion loss and
        adding the demodulator's noise temperature.
        """
        if self.input is None:
            raise ValueError("Demodulator input must be set before processing")

        # Get input signal
        input_signal = self.input.output(frequency)

        # Apply conversion loss to power
        output_power = input_signal.power / (10 ** (self.conversion_loss.value / 10))

        # Add demodulator noise temperature to input noise temperature
        output_noise_temp = input_signal.noise_temperature + self.noise_temperature

        # Store the processed signal (in a real implementation, this would
        # be used for further processing or output)
        self._processed_signal = Signal(power=output_power, noise_temperature=output_noise_temp) 