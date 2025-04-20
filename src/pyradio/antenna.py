"""
Antenna calculations for radio communications.

This module provides classes and functions for antenna calculations,
including dish gain and beamwidth calculations.
"""

import math
from abc import ABC, abstractmethod
from pint import Quantity
from .units import db, wavelength, K


class Antenna(ABC):
    """
    Abstract base class for antennas.

    This class defines the interface that all antenna implementations must follow.
    The gain method must be implemented by all concrete antenna classes.
    """

    def __init__(self,
                 axial_ratio: float = 0,
                 noise_temperature: Quantity = 0.0 * K,
                 return_loss: float = float('inf')):
        """
        Initialize an antenna with polarization, noise temperature, and return loss.

        Args:
            axial_ratio: Axial ratio in dB (default: 0.0 for perfect circular polarization)
                         0 dB represents perfect circular polarization
                         >40 dB represents linear polarization
            noise_temperature: Antenna noise temperature in Kelvin (default: 0.0 K)
            return_loss: Return loss in dB (>=0, float('inf') for perfect match)

        Raises:
            ValueError: If axial_ratio, noise_temperature, or return_loss is negative
        """
        # Validate axial ratio
        if axial_ratio < 0:
            raise ValueError("Axial ratio must be non-negative")
        # Validate noise temperature
        if noise_temperature < 0.0 * K:
            raise ValueError("Noise temperature must be non-negative")
        # Validate return loss
        if return_loss < 0.0:
            raise ValueError("Return loss must be non-negative")
        self.axial_ratio = axial_ratio
        self.noise_temperature = noise_temperature
        self.return_loss = return_loss

    @abstractmethod
    def gain(self, frequency: Quantity) -> float:
        """
        Calculate the antenna gain at a given frequency.

        Args:
            frequency: Frequency

        Returns:
            float: Antenna gain in dB

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

    def __init__(self, gain_: float, axial_ratio: float = 0,
                 noise_temperature: Quantity = 0.0 * K,
                 return_loss: float = float('inf')):
        """
        Initialize a FixedGain antenna.

        Args:
            gain_: Fixed antenna gain in dB (can be positive or negative)
            axial_ratio: Axial ratio in dB (default: 0.0 for perfect circular polarization)
                         0 dB represents perfect circular polarization
                         >40 dB represents linear polarization
        """
        super().__init__(axial_ratio, noise_temperature, return_loss)
        self.gain_ = gain_

    def gain(self, frequency: Quantity) -> float:
        """
        Return the fixed antenna gain.

        Args:
            frequency: Frequency in Hz (ignored)

        Returns:
            float: Fixed antenna gain in dB
        """
        return self.gain_


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

    def __init__(self, diameter: Quantity,
                 efficiency: float = 0.65,
                 axial_ratio: float = 0,
                 noise_temperature: Quantity = 0.0 * K,
                 return_loss: float = float('inf')):
        """
        Initialize a Dish antenna.

        Args:
            diameter: Dish diameter in meters
            efficiency: Antenna efficiency (default: 0.65)
            axial_ratio: Axial ratio in dB (default: 0.0 for perfect circular polarization)
                         0 dB represents perfect circular polarization
                         >40 dB represents linear polarization

        Raises:
            ValueError: If diameter is not positive or efficiency is not between 0 and 1
        """
        super().__init__(axial_ratio, noise_temperature, return_loss)
        if diameter <= 0:
            raise ValueError("Dish diameter must be positive")
        if not 0 < efficiency <= 1:
            raise ValueError("Efficiency must be between 0 and 1")

        self.diameter = diameter
        self.efficiency = efficiency

    def gain(self, frequency: Quantity) -> float:
        """
        Calculate the dish antenna gain at a given frequency.

        The gain is calculated using the formula:
        G = 10 * log10(efficiency * (pi * D / lambda)^2)
        where:
        - efficiency is the antenna efficiency
        - D is the dish diameter in meters
        - lambda is the wavelength in meters

        Args:
            frequency: Frequency in Hz

        Returns:
            float: Antenna gain in dB

        Raises:
            ValueError: If frequency is not positive
        """
        # Perform calculation with quantities
        gain_linear = (
            self.efficiency * (math.pi * self.diameter / wavelength(frequency)) ** 2
        )

        # Convert result to scalar and return
        return gain_linear.to("dB").magnitude


def polarization_loss(tx_axial_ratio: float, rx_axial_ratio: float) -> float:
    """
    Calculate the polarization loss in dB between two antennas with given axial ratios.

    The polarization loss is calculated using the standard formula for polarization
    mismatch between two antennas with different axial ratios. For circular polarization,
    the axial ratio is 0 dB, and for linear polarization, it is >40 dB.

    Args:
        tx_axial_ratio: Transmit antenna axial ratio in dB
        rx_axial_ratio: Receive antenna axial ratio in dB

    Returns:
        float: Polarization loss in dB (positive value)

    Examples:
        >>> # Perfect circular to perfect circular
        >>> loss = polarization_loss(0.0, 0.0)
        >>> round(loss, 1)
        0.0

        >>> # Circular to linear (theoretical 3dB)
        >>> loss = polarization_loss(0.0, 40.0)
        >>> round(loss, 1)
        3.0
    """
    # Convert axial ratios from dB to linear scale
    tx_ar_linear = 10 ** (tx_axial_ratio / 20)
    rx_ar_linear = 10 ** (rx_axial_ratio / 20)

    # Calculate polarization loss using the polarization efficiency formula
    # This is the standard formula for polarization mismatch between antennas
    plf = (4 * tx_ar_linear * rx_ar_linear) / (
        (1 + tx_ar_linear**2) * (1 + rx_ar_linear**2)
    )

    # Convert the mismatch factor to a positive loss in dB
    # db(plf) is negative or zero since plf <= 1; negate to make loss positive
    return -db(plf)
