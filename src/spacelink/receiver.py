"""
Receiver module for RF receivers.

This module provides the Receiver class that represents a complete RF receiver
chain, including low-noise amplifiers, filters, and other components.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable
import astropy.units as u

from spacelink.stage import Stage
from spacelink.signal import Signal
from spacelink.units import Frequency, Watts, Temperature, Decibels, enforce_units


class Receiver(Stage):
    """
    A class representing an RF receiver.

    A receiver is a stage that processes input signals, typically including
    components like low-noise amplifiers, filters, and other RF stages.

    Attributes:
        gain: Gain in dB
        noise_temperature: Noise temperature in Kelvin
    """
    @enforce_units
    def __init__(self, gain: Decibels, noise_temperature: Temperature):
        """
        Initialize a Receiver.

        Args:
            gain: Gain in dB
            noise_temperature: Noise temperature in Kelvin
        """
        super().__init__()
        self.gain = gain
        self.noise_temperature = noise_temperature

    @enforce_units
    def output(self, frequency: Frequency) -> Signal:
        """
        Calculate the output signal for this receiver.

        Args:
            frequency: The frequency at which to calculate the output

        Returns:
            Signal: The output signal from this receiver

        Raises:
            ValueError: If input is not set
        """
        if self.input is None:
            raise ValueError("Receiver input must be set before calculating output")

        # Get input signal
        input_signal = self.input()

        # Apply gain to power
        output_power = input_signal.power * (10 ** (self.gain.value / 10))

        # Add receiver noise temperature to input noise temperature
        output_noise_temp = input_signal.noise_temperature + self.noise_temperature

        return Signal(power=output_power, noise_temperature=output_noise_temp) 