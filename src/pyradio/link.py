"""
Link budget calculations for radio communications.

This module provides a Link class that encapsulates all parameters needed
for a complete link budget calculation, including transmitter and receiver
characteristics, path loss, and noise calculations.
"""

from typing import Callable
from .path import free_space_path_loss
from .units import ureg, Quantity
from .antenna import Antenna, polarization_loss
import .noise as noise


class Link:
    """
    A class representing a radio communication link.

    This class encapsulates all parameters needed for a complete link budget
    calculation, including transmitter and receiver characteristics, path loss,
    and noise calculations.

    Attributes:
        tx_power: Transmitter power in dBW
        tx_antenna: Transmitter antenna
        rx_antenna: Receiver antenna
        rx_system_noise_temp: Receiver system noise temperature in Kelvin
        rx_antenna_noise_temp: Receiver antenna noise temperature in Kelvin
        distance_fn: Callable that returns the distance in meters
        frequency: Carrier frequency in Hz
        bandwidth: Signal bandwidth in Hz
        required_ebno: Required Eb/N0 in dB for the desired BER
        implementation_loss: Implementation loss in dB (default: 0)
        polarization_loss: Polarization loss in dB (default: 0)
        pointing_loss: Antenna pointing loss in dB (default: 0)
        atmospheric_loss: Atmospheric loss in dB (default: 0)
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
        required_ebno: Quantity,
        implementation_loss: Q_[float, 'dB'] = Q_(0.0, 'dB'),
        atmospheric_loss: Callable[[], Quantity],
    ):
        """
        Initialize a Link object with all necessary parameters.

        Args:
            tx_power: Transmitter power in dBW
            tx_antenna: Transmitter antenna
            rx_antenna: Receiver antenna
            rx_system_noise_temp: Receiver system noise temperature in Kelvin
            rx_antenna_noise_temp: Receiver antenna noise temperature in Kelvin
            distance_fn: Callable that returns the distance in meters
            frequency: Carrier frequency in Hz
            bandwidth: Signal bandwidth in Hz
            required_ebno: Required Eb/N0 in dB for the desired BER
            implementation_loss: Implementation loss in dB (default: 0)
            pointing_loss: Antenna pointing loss in dB (default: 0)
            atmospheric_loss: Atmospheric loss in dB (default: 0)

        Raises:
            ValueError: If any parameter is invalid
        """
        # Validate inputs
        if tx_power.magnitude <= 0:
            raise ValueError("Transmitter power must be positive")
        if not isinstance(tx_antenna, Antenna):
            raise ValueError("tx_antenna must be an Antenna instance")
        if not isinstance(rx_antenna, Antenna):
            raise ValueError("rx_antenna must be an Antenna instance")
        if rx_system_noise_temp.magnitude <= 0:
            raise ValueError("System noise temperature must be positive")
        if rx_antenna_noise_temp.magnitude < 0:
            raise ValueError("Antenna noise temperature cannot be negative")
        if not callable(distance_fn):
            raise ValueError("distance_fn must be a callable")
        if frequency.magnitude <= 0:
            raise ValueError("Frequency must be positive")
        if bandwidth.magnitude <= 0:
            raise ValueError("Bandwidth must be positive")
        if implementation_loss.magnitude < 0:
            raise ValueError("Implementation loss cannot be negative")
        if pointing_loss.magnitude < 0:
            raise ValueError("Pointing loss cannot be negative")
        if atmospheric_loss.magnitude < 0:
            raise ValueError("Atmospheric loss cannot be negative")
        if required_ebno.magnitude < 0:
            raise ValueError("Required Eb/N0 cannot be negative")

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
        self.pointing_loss = pointing_loss
        self.atmospheric_loss = atmospheric_loss
        self.required_ebno = required_ebno

    @property
    def distance(self) -> Q_[float, 'm']:
        """
        Get the current distance in meters.

        Returns:
            Q_[float, 'm']: Current distance in meters

        Raises:
            ValueError: If the distance is not positive
        """
        distance = self.distance_fn()
        if distance.magnitude <= 0:
            raise ValueError("Distance must be positive")
        return distance

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
        return self.tx_power + Q_(self.tx_antenna.gain(self.frequency), 'dB')

    @property
    def path_loss(self) -> float:
        """
        Calculate the total path loss in dB.

        Path Loss = Free Space Path Loss + Atmospheric Loss + Pointing Loss

        Returns:
            Q_[float, 'dB']: Total path loss in dB
        """
        return (
            free_space_path_loss(self.distance.magnitude, self.frequency.magnitude)
            + self.atmospheric_loss()
        )

    @property
    def polarization_loss(self) -> float:
        """
        Calculate the polarization loss in dB based on the axial ratios of the antennas.

        The polarization loss is calculated using the standard formula for polarization
        mismatch between two antennas with different axial ratios.

        Returns:
            float: Polarization loss in dB
        """
        return polarization_loss(self.tx_antenna.axial_ratio, self.rx_antenna.axial_ratio)

    @property
    def received_power(self) -> Q_[float, 'dBW']:
        """
        Calculate the received power in dBW.

        Received Power = EIRP + Receiver Antenna Gain - Path Loss - Polarization Loss

        Returns:
            Q_[float, 'dBW']: Received power in dBW
        """
        return (
            self.eirp
            + Q_(self.rx_antenna.gain(self.frequency), 'dB')
            - self.path_loss
            - self.polarization_loss
        )

    @property
    def carrier_to_noise_ratio(self) -> Q_[float, 'dB']:
        """
        Calculate the carrier-to-noise ratio in dB.

        C/N = Received Power - Noise Power

        Returns:
            Q_[float, 'dB']: Carrier-to-noise ratio in dB
        """
        return self.received_power - self.noise_power

    def ebno(self) -> Q_[float, 'dB']:
        """
        Calculate the energy per bit to noise power spectral density ratio (Eb/N0) in dB.

        Eb/N0 = C/N + 10*log10(B/R)
        where:
        - C/N is the carrier-to-noise ratio in dB
        - B is the bandwidth in Hz
        - R is the data rate in bits per second

        Returns:
            Q_[float, 'dB']: Eb/N0 in dB

        Raises:
            ValueError: If data rate is not positive
        """
        return self.carrier_to_noise_ratio

    def margin(self) -> float:
        """
        Calculate the link margin in dB.

        Link Margin = Eb/N0 - Required Eb/N0 - Implementation Loss

        Implementation loss is subtracted here rather than in the EIRP calculation,
        as it represents losses in the communication system that degrade performance.

        Returns:
           float: Link margin in dB
        """
        return self.ebno() - self.required_ebno - self.implementation_loss

    def __str__(self) -> str:
        """
        Return a string representation of the link budget.

        Returns:
            str: String representation of the link budget
        """
        return (
            f"Link Budget:\n"
            f"  EIRP: {self.eirp.magnitude:.1f} dBW\n"
            f"  Path Loss: {self.path_loss.magnitude:.1f} dB\n"
            f"  Received Power: {self.received_power.magnitude:.1f} dBW\n"
            f"  System Noise Temperature: {self.system_noise_temperature.magnitude:.1f} K\n"
            f"  Noise Power: {self.noise_power.magnitude:.1f} dBW\n"
            f"  C/N: {self.carrier_to_noise_ratio.magnitude:.1f} dB"
        )
