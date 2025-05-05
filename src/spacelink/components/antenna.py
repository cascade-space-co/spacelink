"""
Antenna component models.

This module provides the Antenna base class and concrete implementations
like FixedGain and Dish antennas.
"""

from abc import ABC, abstractmethod
import astropy.units as u

# Use relative imports for core modules
from ..core.units import (
    Decibels,
    Dimensionless,
    Temperature,
    Frequency,
    Length,
    enforce_units,
)

# Update import to use renamed core function
from ..core.antenna import dish_gain


class Antenna(ABC):
    """
    Abstract base class for antennas.

    This class defines the interface that all antenna implementations must follow.
    The gain method must be implemented by all concrete antenna classes.
    """

    @enforce_units
    def __init__(
        self,
        axial_ratio: Decibels = 0.0 * u.dB,
        noise_temperature: Temperature = 0.0 * u.K,
        return_loss: Decibels = float("inf") * u.dB,
    ):
        """
        Initialize an antenna with polarization, noise temperature, and return loss.

        Args:
            axial_ratio: Axial ratio in dB (default: 0.0 dB for perfect circular polarization)
                         0 dB represents perfect circular polarization
                         >40 dB represents linear polarization
            noise_temperature: Antenna noise temperature in Kelvin (default: 0.0 K)
            return_loss: Return loss in dB (>=0, float('inf') for perfect match)

        Raises:
            ValueError: If axial_ratio, noise_temperature, or return_loss is negative
        """
        # Validate axial ratio
        if axial_ratio < 0.0 * u.dB:
            raise ValueError("Axial ratio must be non-negative")
        # Validate noise temperature
        if noise_temperature < 0.0 * u.K:
            raise ValueError("Noise temperature must be non-negative")
        # Validate return loss
        if return_loss < 0.0 * u.dB:
            raise ValueError("Return loss must be non-negative")
        self.axial_ratio = axial_ratio
        self.noise_temperature = noise_temperature
        self.return_loss = return_loss

    @abstractmethod
    @enforce_units
    def gain(self, frequency: Frequency) -> Decibels:
        """
        Calculate the antenna gain at a given frequency.

        Args:
            frequency: Frequency

        Returns:
            Antenna gain in dB

        Raises:
            ValueError: If frequency is not positive
        """


class FixedGain(Antenna):
    """
    A class representing an antenna with fixed gain.

    This class implements the Antenna interface for an antenna with a constant gain
    regardless of frequency. The gain can be positive or negative, as some
    antennas (like small whip antennas or patch antennas) can have negative gain.

    Attributes:
        gain_: Fixed antenna gain in dB
        axial_ratio: Axial ratio in dB (0 dB for perfect circular, >40 dB for linear)
    """

    def __init__(
        self,
        gain: Decibels,
        axial_ratio: Decibels = 0.0 * u.dB,
        noise_temperature: Temperature = 0.0 * u.K,
        return_loss: Decibels = float("inf") * u.dB,
    ):
        """
        Initialize a FixedGain antenna.

        Args:
            gain_: Fixed antenna gain in dB (can be positive or negative)
            axial_ratio: Axial ratio in dB (default: 0.0 dB for perfect circular polarization)
                         0 dB represents perfect circular polarization
                         >40 dB represents linear polarization
        """
        super().__init__(axial_ratio, noise_temperature, return_loss)
        self._gain = gain

    def gain(self, frequency: Frequency) -> Decibels:
        """
        Return the fixed antenna gain.

        Args:
            frequency: Frequency in Hz (ignored)

        Returns:
            Fixed antenna gain in dB
        """
        return self._gain


class Dish(Antenna):
    """
    A class representing a parabolic dish antenna.

    This class implements the Antenna interface for a parabolic dish antenna,
    calculating gain based on the dish diameter and frequency.

    Attributes:
        diameter: Dish diameter in meters
        efficiency: Antenna efficiency (default: 0.65)
        axial_ratio: Axial ratio in dB (0 dB for perfect circular, >40 dB for linear)
    """

    @enforce_units
    def __init__(
        self,
        diameter: Length,
        efficiency: Dimensionless = 0.65 * u.dimensionless,
        axial_ratio: Decibels = 0.0 * u.dB,
        noise_temperature: Temperature = 0.0 * u.K,
        return_loss: Decibels = float("inf") * u.dB,
    ):
        """
        Initialize a Dish antenna.

        Args:
            diameter: Dish diameter in meters
            efficiency: Antenna efficiency as a dimensionless quantity (default: 0.65)
            axial_ratio: Axial ratio in dB (default: 0.0 dB for perfect circular polarization)
                         0 dB represents perfect circular polarization
                         >40 dB represents linear polarization

        Raises:
            ValueError: If diameter is not positive or efficiency is not between 0 and 1
        """
        super().__init__(axial_ratio, noise_temperature, return_loss)
        if diameter <= 0 * u.m:
            raise ValueError("Dish diameter must be positive")
        if not 0 < efficiency.to_value(u.dimensionless) <= 1:
            raise ValueError("Efficiency must be between 0 and 1")

        self.diameter = diameter
        self.efficiency = efficiency

    @enforce_units
    def gain(self, frequency: Frequency) -> Decibels:
        """
        Calculate the dish antenna gain at a given frequency using the core function.

        Args:
            frequency: Frequency in Hz

        Returns:
            Antenna gain in dB

        Raises:
            ValueError: If frequency is not positive (handled by core function)
        """
        # Directly return the dB value from the core function
        return dish_gain(
            diameter=self.diameter, frequency=frequency, efficiency=self.efficiency
        )
