"""
Link budget calculations for radio communications.

This module provides a Link class that encapsulates all parameters needed
for a complete link budget calculation, including transmitter and receiver
characteristics, path loss, and noise calculations.
"""

import math
from typing import Callable
from pint import Quantity
from .antenna import Antenna, polarization_loss
from .path import free_space_path_loss
from .units import Q_, Hz, K, db, m, km, W
from . import noise


class Link:
    """
    A class representing a radio communication link.

    This class encapsulates all parameters needed for a complete link budget
    calculation, including transmitter and receiver characteristics, path loss,
    and noise calculations.

    Attributes:
        tx_power: Transmitter power in W
        tx_antenna: Transmitter antenna
        rx_antenna: Receiver antenna
        rx_system_noise_temp: Receiver system noise temperature in Kelvin
        rx_antenna_noise_temp: Receiver antenna noise temperature in Kelvin
        distance_fn: Callable that returns the distance in meters
        frequency: Carrier frequency in Hz
        bandwidth: Signal bandwidth in Hz
        required_ebno: Required Eb/N0 in dB for the desired BER
        implementation_loss: Implementation loss in dB (default: 0)
    """

    def __init__(
        self,
        frequency: Quantity,
        tx_antenna: Antenna,
        rx_antenna: Antenna,
        tx_power: Quantity,
        rx_system_noise_temp: Quantity,
        rx_antenna_noise_temp: Quantity,
        distance_fn: Callable[[], Quantity],
        bandwidth: Quantity,
        required_ebno: float,
        implementation_loss: float = 0.0,
    ):
        """
        Initialize a Link object with all necessary parameters.

        Args:
            frequency: Carrier frequency in Hz
            tx_antenna: Transmitter antenna
            rx_antenna: Receiver antenna
            tx_power: Transmitter power in W
            rx_system_noise_temp: Receiver system noise temperature in Kelvin
            rx_antenna_noise_temp: Receiver antenna noise temperature in Kelvin
            distance_fn: Callable that returns the distance in meters
            bandwidth: Signal bandwidth in Hz
            required_ebno: Required Eb/N0 in dB for the desired BER
            implementation_loss: Implementation loss in dB (default: 0 dB)

        Raises:
            ValueError: If any parameter is invalid
        """
        # Validate inputs
        if tx_power <= 0.0 * W:
            raise ValueError("Transmitter power must be positive")
        if not isinstance(tx_antenna, Antenna):
            raise ValueError("tx_antenna must be an Antenna instance")
        if not isinstance(rx_antenna, Antenna):
            raise ValueError("rx_antenna must be an Antenna instance")
        if rx_system_noise_temp <= 0.0 * K:
            raise ValueError("System noise temperature must be positive")
        if rx_antenna_noise_temp < 0.0 * K:
            raise ValueError("Antenna noise temperature cannot be negative")
        if not callable(distance_fn):
            raise ValueError("distance_fn must be a callable")
        if frequency <= 0.0 * Hz:
            raise ValueError("Frequency must be positive")
        if bandwidth <= 0.0 * Hz:
            raise ValueError("Bandwidth must be positive")

        # Store parameters
        self.tx_power = tx_power
        self.tx_antenna = tx_antenna
        self.rx_antenna = rx_antenna
        self.rx_system_noise_temp = rx_system_noise_temp
        self.rx_antenna_noise_temp = rx_antenna_noise_temp
        self.distance_fn = distance_fn
        self.frequency = frequency
        self.bandwidth = bandwidth
        self.implementation_loss = implementation_loss
        self.required_ebno = required_ebno

    @property
    def distance(self) -> Quantity:
        """
        Get the current distance in meters.

        Returns:
            Quantity: Current distance in meters

        Raises:
            ValueError: If the distance is not positive
        """
        distance = self.distance_fn()
        if distance <= 0 * m:
            raise ValueError("Distance must be positive")
        return distance

    @property
    def system_noise_temperature(self) -> Quantity:
        """
        Calculate the total system noise temperature in Kelvin.

        The total system noise temperature is the sum of the receiver system
        noise temperature and the antenna noise temperature.

        Returns:
            Quantity: Total system noise temperature in Kelvin
        """
        return self.rx_system_noise_temp + self.rx_antenna_noise_temp

    @property
    def noise_power(self) -> float:
        """
        Calculate the noise power in dBW.

        Noise Power = k * T * B
        where:
        - k is Boltzmann's constant
        - T is the system noise temperature in Kelvin
        - B is the bandwidth in Hz

        Returns:
            float: Noise power in dBW
        """
        power_quantity = noise.power(self.bandwidth, self.system_noise_temperature)
        return power_quantity.to('dBW').magnitude

    @property
    def eirp(self) -> float:
        """
        Calculate the Effective Isotropic Radiated Power (EIRP) in dBW.

        EIRP = Transmitter Power + Transmitter Antenna Gain

        Note: Implementation loss is accounted for in the link margin calculation,
        not in the EIRP calculation.

        Returns:
            float: EIRP in dBW
        """
        return self.tx_power.to('dBW').magnitude + self.tx_antenna.gain(self.frequency)

    @property
    def path_loss(self) -> float:
        """
        Calculate the free space path loss in dB.

        Path Loss = Free Space Path Loss

        Returns:
            float: Free space path loss in dB (positive value)
        """
        fspl = free_space_path_loss(self.distance, self.frequency).to('dB').magnitude
        return fspl

    @property
    def polarization_loss(self) -> float:
        """
        Calculate the polarization loss in dB based on the axial ratios of the antennas.

        The polarization loss is calculated using the standard formula for polarization
        mismatch between two antennas with different axial ratios.
        Note: Returns a negative value, as it's a loss

        Returns:
            float: Polarization loss in dB (negative value)
        """
        return -polarization_loss(self.tx_antenna.axial_ratio, self.rx_antenna.axial_ratio)

    @property
    def received_power(self) -> float:
        """
        Calculate the received power in dBW.

        Received Power = EIRP + Receiver Antenna Gain + Path Loss + Polarization Loss
        Where Path Loss and Polarization Loss are negative values

        Returns:
            float: Received power in dBW
        """
        return (
            self.eirp
            + self.rx_antenna.gain(self.frequency)
            + self.path_loss
            + self.polarization_loss
        )

    @property
    def carrier_to_noise_ratio(self) -> float:
        """
        Calculate the carrier-to-noise ratio in dB.

        C/N = Received Power - Noise Power

        Returns:
            float: Carrier-to-noise ratio in dB
        """
        return self.received_power - self.noise_power

    @property
    def ebno(self) -> float:
        """
        Calculate the energy per bit to noise power spectral density ratio (Eb/N0) in dB.

        Eb/N0 = C/N + 10*log10(B/R)
        where:
        - C/N is the carrier-to-noise ratio in dB
        - B is the bandwidth in Hz
        - R is the data rate in bits per second

        Returns:
            float: Eb/N0 in dB
        """
        return self.carrier_to_noise_ratio

    @property
    def margin(self) -> float:
        """
        Calculate the link margin in dB.

        Link Margin = Eb/N0 - Required Eb/N0 - Implementation Loss

        Implementation loss is subtracted here rather than in the EIRP calculation,
        as it represents losses in the communication system that degrade performance.

        Returns:
            float: Link margin in dB
        """
        return self.ebno - self.required_ebno - self.implementation_loss

    def __str__(self) -> str:
        """
        Return a string representation of the link budget.

        Returns:
            str: String representation of the link budget
        """
        return (
            f"Link Budget:\n"
            f"  EIRP: {self.eirp:.1f} dBW\n"
            f"  Path Loss: {self.path_loss:.1f} dB\n"
            f"  Polarization Loss: {self.polarization_loss:.1f} dB\n"
            f"  Received Power: {self.received_power:.1f} dBW\n"
            f"  System Noise Temperature: {self.system_noise_temperature.magnitude:.1f} K\n"
            f"  Noise Power: {self.noise_power:.1f} dBW\n"
            f"  C/N: {self.carrier_to_noise_ratio:.1f} dB\n"
            f"  Margin: {self.margin:.1f} dB"
        )
