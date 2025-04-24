"""
Stage module for RF stage analysis.

This module provides the Stage base class that defines common functionality for
RF stages in a cascade. Each stage can be connected to other stages through its
input property, allowing for flexible chaining of RF components.
"""

from typing import Optional, Union
import astropy.units as u
import numpy as np

from spacelink.signal import Signal
from spacelink.units import Frequency, Decibels, Temperature, Dimensionless, enforce_units
from spacelink.validation import non_negative
from spacelink.units import to_linear, to_dB
from spacelink.noise import noise_figure_to_temperature, temperature_to_noise_figure, noise_factor_to_temperature, temperature_to_noise_factor
from spacelink.source import Source


class Stage:
    """
    Base class for stages.

    A stage represents a single component in a cascade of RF stages (e.g., amplifier,
    attenuator, filter). Each stage has an input that can be connected to another
    stage's output or a source, allowing for chaining of stages.

    Attributes:
        input: Reference to the input stage or source. This is typically set to another
               stage or source, allowing for chaining of stages.
    """
    def __init__(self):
        self._input: Optional[Union['Stage', Source]] = None
        self._noise_temperature: Temperature = 0 * u.K

    @property
    def input(self) -> Optional[Union['Stage', Source]]:
        """Get the input stage or source for this stage."""
        return self._input

    @input.setter
    def input(self, value: Optional[Union['Stage', Source]]) -> None:
        """Set the input stage or source for this stage."""
        if value is not None and not isinstance(value, (Stage, Source)):
            raise ValueError("Input must be a Stage or Source instance")
        self._input = value

    @property
    def noise_temperature(self) -> Temperature:
        """
        Get the noise temperature of this stage.

        Returns:
            Temperature: The noise temperature in Kelvin
        """
        return self._noise_temperature

    @noise_temperature.setter
    @enforce_units
    def noise_temperature(self, value: Temperature) -> None:
        """
        Set the noise temperature of this stage.

        Args:
            value: The noise temperature in Kelvin
        """
        self._noise_temperature = value

    @property
    def noise_factor(self) -> Dimensionless:
        """
        Get the noise factor of this stage.

        The noise factor is the linear ratio (not dB) defined as:
        F = 1 + T/T0
        where:
        - F is the noise factor
        - T is the noise temperature
        - T0 is the reference temperature (290K)

        Returns:
            Dimensionless: The noise factor (linear ratio)
        """
        return temperature_to_noise_factor(self.noise_temperature)

    @noise_factor.setter
    @enforce_units
    def noise_factor(self, value: Dimensionless) -> None:
        """
        Set the noise factor of this stage.

        Args:
            value: The noise factor (linear ratio, not dB)

        Raises:
            ValueError: If noise factor is less than 1
        """
        if value < 1:
            raise ValueError("Noise factor must be greater than or equal to 1")
        self.noise_temperature = noise_factor_to_temperature(value)

    @property
    def noise_figure(self) -> Decibels:
        """
        Calculate the noise figure in dB based on the noise temperature.

        The noise figure is the noise factor expressed in dB:
        NF = 10 * log10(F)
        where F is the noise factor.

        Returns:
            Decibels: The noise figure in dB
        """
        return temperature_to_noise_figure(self.noise_temperature)

    @noise_figure.setter
    @enforce_units
    def noise_figure(self, value: Decibels) -> None:
        """
        Set the noise figure of this stage.

        Args:
            value: The noise figure in dB

        Raises:
            ValueError: If noise figure is negative
        """
        if value < 0 * u.dB:
            raise ValueError("Noise figure must be non-negative")
        self.noise_temperature = noise_figure_to_temperature(value)

    def output(self, frequency: Frequency) -> Signal:
        """
        Calculate the output signal for this stage.

        Args:
            frequency: The frequency at which to calculate the output

        Returns:
            Signal: The output signal from this stage

        Raises:
            ValueError: If input is not set
        """
        if self.input is None:
            raise ValueError("Stage input must be set before calculating output")
        return self.input.output(frequency)


class GainBlock(Stage):
    """
    A class representing a gain block stage.

    A gain block is a stage that provides gain and adds noise. It can be used to
    model amplifiers, attenuators, and other gain-providing components.

    Attributes:
        gain: Gain in dB (stored internally as linear ratio)
        noise_temperature: Noise temperature in Kelvin
        input_return_loss: Input return loss in dB (stored internally as linear ratio)
    """
    def __init__(
        self,
        gain: Decibels,
        noise_temperature: Optional[Temperature] = None,
        noise_figure: Optional[Decibels] = None,
        input_return_loss: Optional[Decibels] = None,
    ):
        """
        Initialize a GainBlock.

        Args:
            gain: The gain of the block in dB
            noise_temperature: The noise temperature in K
            noise_figure: The noise figure in dB
            input_return_loss: The input return loss in dB

        Raises:
            ValueError: If both noise_temperature and noise_figure are specified
            ValueError: If noise_temperature is negative
            ValueError: If input_return_loss is negative
        """
        super().__init__()
        self.gain = gain

        # Validate noise parameters
        if noise_temperature is not None and noise_figure is not None:
            raise ValueError("Cannot specify both noise_temperature and noise_figure")
        
        if noise_temperature is not None:
            if noise_temperature < 0 * u.K:
                raise ValueError("noise_temperature must be non-negative")
            self.noise_temperature = noise_temperature
        elif noise_figure is not None:
            self.noise_temperature = noise_figure_to_temperature(noise_figure)
        else:
            self.noise_temperature = 0 * u.K

        if input_return_loss is not None:
            if input_return_loss < 0 * u.dB:
                raise ValueError("input_return_loss must be non-negative")
            self._input_return_loss = input_return_loss
        else:
            self._input_return_loss = None

    @property
    @enforce_units
    def gain(self) -> Decibels:
        """Get the gain in dB."""
        return self._gain

    @gain.setter
    @enforce_units
    def gain(self, value: Decibels) -> None:
        """Set the gain in dB."""
        self._gain = value
        self._gain_linear = 10 ** (value.to(u.dB).value / 10)

    @property
    @enforce_units
    def noise_temperature(self) -> Temperature:
        """Get the noise temperature in Kelvin."""
        return self._noise_temperature

    @noise_temperature.setter
    @enforce_units
    def noise_temperature(self, value: Temperature) -> None:
        """Set the noise temperature in Kelvin."""
        if value < 0 * u.K:
            raise ValueError("noise_temperature must be non-negative")
        self._noise_temperature = value

    @property
    @enforce_units
    def input_return_loss(self) -> Optional[Decibels]:
        """Get the input return loss in dB."""
        return self._input_return_loss

    @input_return_loss.setter
    @enforce_units
    def input_return_loss(self, value: Optional[Decibels]) -> None:
        """Set the input return loss in dB."""
        if value is not None and value < 0 * u.dB:
            raise ValueError("input_return_loss must be non-negative")
        self._input_return_loss = value

    @property
    def noise_factor(self) -> Dimensionless:
        """
        Get the noise factor of this stage.

        The noise factor is the linear ratio (not dB) defined as:
        F = 1 + T/T0
        where:
        - F is the noise factor
        - T is the noise temperature
        - T0 is the reference temperature (290K)

        Returns:
            Dimensionless: The noise factor (linear ratio)
        """
        return temperature_to_noise_factor(self.noise_temperature)

    @noise_factor.setter
    @enforce_units
    def noise_factor(self, value: Dimensionless) -> None:
        """
        Set the noise factor of this stage.

        Args:
            value: The noise factor (linear ratio, not dB)

        Raises:
            ValueError: If noise factor is less than 1
        """
        if value < 1:
            raise ValueError("Noise factor must be greater than or equal to 1")
        self.noise_temperature = noise_factor_to_temperature(value)

    @property
    def noise_figure(self) -> Decibels:
        """
        Calculate the noise figure in dB based on the noise temperature.

        The noise figure is the noise factor expressed in dB:
        NF = 10 * log10(F)
        where F is the noise factor.

        Returns:
            Decibels: The noise figure in dB
        """
        return temperature_to_noise_figure(self.noise_temperature)

    @noise_figure.setter
    @enforce_units
    def noise_figure(self, value: Decibels) -> None:
        """
        Set the noise figure of this stage.

        Args:
            value: The noise figure in dB

        Raises:
            ValueError: If noise figure is negative
        """
        if value < 0 * u.dB:
            raise ValueError("Noise figure must be non-negative")
        self.noise_temperature = noise_figure_to_temperature(value)

    def output(self, frequency: Frequency) -> Signal:
        """
        Calculate the output signal for this gain block.

        Args:
            frequency: The frequency at which to calculate the output

        Returns:
            Signal: The output signal from this gain block

        Raises:
            ValueError: If input is not set
        """
        if self.input is None:
            raise ValueError("GainBlock input must be set before calculating output")

        # Get input signal from the input stage
        input_signal = self.input.output(frequency)

        # Apply gain to power (using linear gain directly)
        output_power = input_signal.power * self._gain_linear

        # Add noise temperature to input noise temperature
        output_noise_temp = self.output_noise_temperature(input_signal.noise_temperature)

        return Signal(power=output_power, noise_temperature=output_noise_temp)

    @enforce_units
    def output_noise_temperature(self, input_noise_temp: Temperature) -> Temperature:
        """
        Calculate the output noise temperature based on input noise temperature.

        For a gain block, the output noise temperature is:
        T_out = (T_in + T_e) * G
        where:
        - T_in is the input noise temperature
        - T_e is the block's input-referred noise temperature
        - G is the linear gain

        Both the input noise and the block's input-referred noise are added first,
        then amplified by the gain.

        Args:
            input_noise_temp: Input noise temperature in Kelvin

        Returns:
            Temperature: Output noise temperature in Kelvin
        """
        return (input_noise_temp + self.noise_temperature) * self._gain_linear


class Attenuator(GainBlock):
    """
    A class representing an attenuator stage.

    An attenuator is a special case of a gain block where the gain is less than 1 (negative dB).
    The noise temperature is automatically calculated based on the attenuation value.

    Attributes:
        attenuation: Attenuation in dB (stored internally as negative gain)
        noise_temperature: Noise temperature in Kelvin (calculated from attenuation)
        input_return_loss: Input return loss in dB (stored internally as linear ratio)
    """
    def __init__(
        self,
        attenuation: Decibels,
        input_return_loss: Decibels = float("inf") * u.dB,
    ):
        """
        Initialize an Attenuator.

        Args:
            attenuation: Attenuation in dB (positive value)
            input_return_loss: Input return loss in dB (default: infinite, perfect match)

        Raises:
            ValueError: If attenuation is negative or input_return_loss is negative
        """
        # Validate attenuation is non-negative
        if attenuation < 0 * u.dB:
            raise ValueError("attenuation must be non-negative")

        # Calculate noise temperature from attenuation
        # For an attenuator, noise factor equals linear attenuation
        noise_factor = to_linear(attenuation)
        noise_temp = noise_factor_to_temperature(noise_factor)
        
        # Initialize parent GainBlock with reciprocal gain and calculated noise temperature
        super().__init__(gain=-attenuation, noise_temperature=noise_temp, input_return_loss=input_return_loss)

    @property
    def attenuation(self) -> Decibels:
        """Get the attenuation in dB."""
        return -self.gain

    @attenuation.setter
    @enforce_units
    def attenuation(self, value: Decibels) -> None:
        """Set the attenuation in dB."""
        if value < 0 * u.dB:
            raise ValueError("attenuation must be non-negative")
        # Set gain to negative of attenuation
        self.gain = -value
        self.noise_temperature = noise_factor_to_temperature(to_linear(value))

    