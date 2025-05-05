"""
Stage module for RF stage analysis.

This module provides the Stage base class that defines common functionality for
RF stages in a cascade. Each stage can be connected to other stages through its
input property, allowing for flexible chaining of RF components.
"""

from typing import Optional, Union
import astropy.units as u


# from astropy import constants as const # Unused
from ..core.path import free_space_path_loss
from ..core.antenna import polarization_loss
from .signal import Signal
from ..core.units import (
    Frequency,
    Decibels,
    Temperature,
    Dimensionless,
    Length,
    enforce_units,
    to_linear,
    to_dB,
)

# from ..core.validation import non_negative # Unused
from ..core.noise import (
    noise_figure_to_temperature,
    temperature_to_noise_figure,
    noise_factor_to_temperature,
    temperature_to_noise_factor,
)

from .antenna import Antenna
from .source import Source


class Stage:
    """ """

    def __init__(self):
        self._input = None
        self._gain_linear = 1
        self._gain_value = 0 * u.dB
        self._noise_temperature = 0 * u.K
        self._input_return_loss = 30 * u.dB

    @property
    def input(self) -> Optional[Union["Stage", "Source"]]:
        """Get the input stage or source for this stage."""
        return self._input

    @input.setter
    def input(self, value: Optional[Union["Stage", "Source"]]) -> None:
        """Set the input stage or source for this stage."""
        # Import locally only if needed, otherwise TYPE_CHECKING handles it
        from .source import Source

        if value is not None and not isinstance(value, (Stage, Source)):
            raise ValueError("Input must be a Stage or Source instance")
        self._input = value

    def cascaded_gain(self, frequency: Frequency) -> Decibels:
        """
        Get the cascaded gain up to this point.

        The cascaded gain is the sum of all gains in the chain up to this point.

        Args:
            frequency: The frequency at which to calculate the gain

        Returns:
            Decibels: The cascaded gain in dB
        """
        if self.input is None:
            # If no input, cascaded gain is just this stage's gain
            return self.gain(frequency)
        elif isinstance(self.input, Source):
            # If input is a Source, cascaded gain starts at this stage
            return self.gain(frequency)
        else:
            # Otherwise, add this stage's gain to the input's cascaded gain
            return self.input.cascaded_gain(frequency) + self.gain(frequency)

    def cascaded_noise_figure(self, frequency: Frequency) -> Decibels:
        """
        Get the cascaded noise figure up to this point.

        The cascaded noise figure is calculated using Friis' formula:
        F_total = F1 + (F2-1)/G1 + (F3-1)/(G1*G2) + ...
        where:
        - F1, F2, etc. are the noise factors (linear) of each stage
        - G1, G2, etc. are the gains (linear) of each stage

        Args:
            frequency: The frequency at which to calculate the noise figure

        Returns:
            Decibels: The cascaded noise figure in dB
        """
        # Import locally only if needed, otherwise TYPE_CHECKING handles it
        from .source import Source

        if self.input is None:
            return self.noise_figure(frequency)

        # Handle Source as input - cascaded NF starts at the first *Stage*
        if isinstance(self.input, Source):
            return self.noise_figure(frequency)

        # Get input stage's cascaded values
        input_gain_lin = to_linear(self.input.cascaded_gain(frequency))
        input_nf_lin = to_linear(self.input.cascaded_noise_figure(frequency))

        # Convert this stage's noise figure to linear
        stage_nf_lin = to_linear(self.noise_figure(frequency))

        # Calculate total noise factor using Friis' formula
        total_nf_lin = input_nf_lin + (stage_nf_lin - 1) / input_gain_lin

        # Convert back to dB
        return to_dB(total_nf_lin)

    def gain(self, frequency: Frequency) -> Decibels:
        """Get the gain at a specific frequency."""
        return self._gain_value

    def set_gain(self, value: Decibels) -> None:
        """Set the gain in dB."""
        self._gain_value = value
        self._gain_linear = to_linear(value)

    def noise_figure(self, frequency: Frequency) -> Decibels:
        """Get the noise figure in dB."""
        return temperature_to_noise_figure(self._noise_temperature)

    @property
    def noise_temperature(self) -> Temperature:
        """Get the noise temperature in Kelvin."""
        return self._noise_temperature

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

    @noise_temperature.setter
    @enforce_units
    def noise_temperature(self, value: Temperature) -> None:
        """Set the noise temperature in Kelvin."""
        if value < 0 * u.K:
            raise ValueError("noise_temperature must be non-negative")
        self._noise_temperature = value

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

        # Get input signal from the input stage
        input_signal = self.input.output(frequency)

        # Apply gain to power (using linear gain directly)
        output_power = input_signal.power * to_linear(self.gain(frequency))

        # Add noise temperature to input noise temperature
        output_noise_temp = self.output_noise_temperature(
            input_signal.noise_temperature, frequency
        )

        return Signal(power=output_power, noise_temperature=output_noise_temp)

    def output_noise_temperature(
        self, input_noise_temp: Temperature, frequency: Frequency
    ) -> Temperature:
        """
        Calculate the output noise temperature based on input noise temperature.

        For a stage, the output noise temperature is:
        T_out = (T_in + T_e) * G
        where:
        - T_in is the input noise temperature
        - T_e is the stage's input-referred noise temperature
        - G is the linear gain

        Both the input noise and the stage's input-referred noise are added first,
        then amplified by the gain.

        Args:
            input_noise_temp: Input noise temperature in Kelvin
            frequency: The frequency at which to calculate the noise temperature

        Returns:
            Temperature: Output noise temperature in Kelvin
        """
        return (input_noise_temp + self.noise_temperature) * to_linear(
            self.gain(frequency)
        )


class GainBlock(Stage):
    """
    A class representing a gain block stage.

    A gain block is a stage that provides gain and adds noise. It can be used to
    model amplifiers, attenuators, and other gain-providing components.

    Attributes:
        noise_temperature: Noise temperature in Kelvin
        input_return_loss: Input return loss in dB (stored internally as linear ratio)
    """

    def __init__(
        self,
        gain_value: Decibels,
        noise_temperature: Optional[Temperature] = None,
        noise_figure: Optional[Decibels] = None,
        input_return_loss: Optional[Decibels] = None,
    ):
        """
        Initialize a GainBlock.

        Args:
            gain_value: The gain of the block in dB (can be positive or negative)
            noise_temperature: The noise temperature in K
            noise_figure: The noise figure in dB
            input_return_loss: The input return loss in dB

        Raises:
            ValueError: If both noise_temperature and noise_figure are specified
            ValueError: If noise_temperature is negative
            ValueError: If input_return_loss is negative
        """
        super().__init__()
        self._gain_value = gain_value
        self._gain_linear = to_linear(gain_value)

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
        super().__init__(
            gain_value=-attenuation,
            noise_temperature=noise_temp,
            input_return_loss=input_return_loss,
        )

    @property
    def attenuation(self) -> Decibels:
        """Get the attenuation in dB."""
        return -self._gain_value

    @attenuation.setter
    @enforce_units
    def attenuation(self, value: Decibels) -> None:
        """Set the attenuation in dB."""
        if value < 0 * u.dB:
            raise ValueError("attenuation must be non-negative")
        # Set gain to negative of attenuation
        self.set_gain(-value)
        # Update noise temperature when attenuation changes
        noise_factor = to_linear(value)
        self.noise_temperature = noise_factor_to_temperature(noise_factor)


class TransmitAntenna(Stage):
    """Stage representing a transmitting antenna."""

    def __init__(self, antenna: Antenna):
        super().__init__()
        self.antenna = antenna
        self._noise_temperature = antenna.noise_temperature

    def gain(self, frequency: Frequency) -> Decibels:
        return self.antenna.gain(frequency)

    def axial_ratio(self, frequency: Frequency) -> Decibels:
        """Axial ratio as function of frequency is not implemented yet"""
        return self.antenna.axial_ratio


class ReceiveAntenna(Stage):
    """Stage representing a receiving antenna."""

    def __init__(self, antenna: Antenna):
        super().__init__()
        self.antenna = antenna
        self._noise_temperature = antenna.noise_temperature

    def gain(self, frequency: Frequency) -> Decibels:
        return self.antenna.gain(frequency) - self.polarization_loss(frequency)

    def polarization_loss(self, frequency: Frequency) -> Decibels:
        """Calculate polarization loss based on input stage's axial ratio."""
        if self.input is None:
            return 0 * u.dB

        # Get axial ratio of previous stage
        prev_axial_ratio = self.input.axial_ratio(frequency)

        # Calculate polarization loss based on both axial ratios
        return polarization_loss(prev_axial_ratio, self.antenna.axial_ratio)

    def output_noise_temperature(
        self, input_noise_temp: Temperature, frequency: Frequency
    ) -> Temperature:
        return input_noise_temp + self.noise_temperature


class Path(Stage):
    """Stage representing a free space path."""

    def __init__(self, distance: Length):
        super().__init__()
        self._distance = distance

    def axial_ratio(self, frequency: Frequency) -> Decibels:
        """Pass through the axial ratio from the transmit antenna."""
        if self.input is None:
            return 0 * u.dB
        return self.input.axial_ratio(frequency)

    def gain(self, frequency: Frequency) -> Decibels:
        """
        Get the gain of the path.

        This is the negative of the free space path loss, since loss
        reduces the signal power.
        """
        return -free_space_path_loss(self._distance, frequency)

    def loss(self, frequency: Frequency) -> Decibels:
        return -self.gain(frequency)
