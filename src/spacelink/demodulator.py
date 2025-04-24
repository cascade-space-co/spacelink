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
    @enforce_units
    def __init__(self, conversion_loss: Decibels, noise_temperature: Temperature):
        """
        Initialize a Demodulator.

        Args:
            conversion_loss: Conversion loss in dB
            noise_temperature: Noise temperature in Kelvin
        """
        super().__init__()
        self.conversion_loss = conversion_loss
        self.noise_temperature = noise_temperature

    @enforce_units
    def process_input(self, frequency: Frequency) -> None:
        """
        Process the input signal at the given frequency.

        Args:
            frequency: The frequency at which to process the input

        This method simulates the demodulation process by calculating the
        output power and noise temperature after conversion loss and
        adding the demodulator's noise temperature.
        """
        if self.input is None:
            raise ValueError("Demodulator input must be set before processing")

        # Get input signal
        input_signal = self.input()

        # Apply conversion loss to power
        output_power = input_signal.power / (10 ** (self.conversion_loss.value / 10))

        # Add demodulator noise temperature to input noise temperature
        output_noise_temp = input_signal.noise_temperature + self.noise_temperature

        # Store the processed signal (in a real implementation, this would
        # be used for further processing or output)
        self._processed_signal = Signal(power=output_power, noise_temperature=output_noise_temp) 