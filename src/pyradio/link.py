"""
Link budget calculations for radio communications.

This module provides a Link class that encapsulates all parameters needed
for a complete link budget calculation, including transmitter and receiver
characteristics, path loss, and noise calculations.
"""

from typing import Callable
from pint import Quantity
from .antenna import Antenna, polarization_loss
from .mode import Mode
from .path import free_space_path_loss
from .units import Hz, K, m, W
from . import noise
from .cascade import Cascade


class Link:
    """
    A class representing a radio communication link.

    This class encapsulates all parameters needed for a complete link budget
    calculation, including transmitter and receiver characteristics, path loss,
    and noise calculations.

    Attributes:
        frequency: Carrier frequency in Hz
        tx_power: Transmitter power in W
        tx_antenna: Transmitter antenna
        rx_antenna: Receiver antenna
        tx_front_end: Cascade of transmit front end stages
        rx_front_end: Cascade of receive front end stages
        rx_antenna_noise_temp: Receiver antenna noise temperature in Kelvin
        distance_fn: Callable that returns the distance in meters
        mode: Mode object with modulation and channel coding settings
    """

    def __init__(
        self,
        frequency: Quantity,
        tx_antenna: Antenna,
        rx_antenna: Antenna,
        tx_power: Quantity,
        tx_front_end: Cascade,
        rx_front_end: Cascade,
        rx_antenna_noise_temp: Quantity,
        distance_fn: Callable[[], Quantity],
        mode: Mode,
    ):
        """
        Initialize a Link object with all necessary parameters.

        Args:
            frequency: Carrier frequency in Hz
            tx_antenna: Transmitter antenna
            rx_antenna: Receiver antenna
            tx_power: Transmitter power in W
            tx_front_end: Cascade of transmit front end stages
            rx_front_end: Cascade of receive front end stages
            rx_antenna_noise_temp: Receiver antenna noise temperature in Kelvin
            distance_fn: Callable that returns the distance in meters
            mode: Mode object with modulation and channel coding settings

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
        if not isinstance(tx_front_end, Cascade):
            raise ValueError("tx_front_end must be a Cascade instance")
        if not isinstance(rx_front_end, Cascade):
            raise ValueError("rx_front_end must be a Cascade instance")
        if rx_antenna_noise_temp < 0.0 * K:
            raise ValueError("Antenna noise temperature cannot be negative")
        if not callable(distance_fn):
            raise ValueError("distance_fn must be a callable")
        if frequency <= 0.0 * Hz:
            raise ValueError("Frequency must be positive")

        # Store parameters
        self.tx_power = tx_power
        self.tx_antenna = tx_antenna
        self.rx_antenna = rx_antenna
        self.tx_front_end = tx_front_end
        self.rx_front_end = rx_front_end
        self.rx_antenna_noise_temp = rx_antenna_noise_temp
        self.distance_fn = distance_fn
        self.frequency = frequency
        self.mode = mode

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
        # Total system noise temperature = antenna noise temp + input-referred noise of receive front end
        # If no stages in front end, treat front end noise as 0
        try:
            fe_noise_temp = (
                self.rx_front_end.cascaded_noise_temperature_k()
                if len(self.rx_front_end) > 0
                else 0.0 * K
            )
        except ValueError:
            fe_noise_temp = 0.0 * K
        return self.rx_antenna_noise_temp + fe_noise_temp

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
        power_quantity = noise.power(self.mode.bandwidth, self.system_noise_temperature)
        return power_quantity.to("dBW").magnitude

    @property
    def eirp(self) -> float:
        """
        Calculate the Effective Isotropic Radiated Power (EIRP) in dBW.

        EIRP = Transmitter Power + Cascaded gain of transmit front end + Transmitter Antenna Gain

        Returns:
            float: EIRP in dBW
        """
        tx_power_dbw = self.tx_power.to("dBW").magnitude
        fe_gain_db = self.tx_front_end.cascaded_gain().to("dB").magnitude
        ant_gain_db = self.tx_antenna.gain(self.frequency)
        return tx_power_dbw + fe_gain_db + ant_gain_db

    @property
    def path_loss(self) -> float:
        """
        Calculate the free space path loss in dB (positive value).

        Path Loss = Free Space Path Loss (positive dB loss)

        Returns:
            float: Free space path loss in dB
        """
        return free_space_path_loss(self.distance, self.frequency)

    @property
    def polarization_loss(self) -> float:
        """
        Calculate the polarization loss in dB (positive value) based on the antennas' axial ratios.

        Returns:
            float: Polarization loss in dB
        """
        return polarization_loss(
            self.tx_antenna.axial_ratio, self.rx_antenna.axial_ratio
        )

    @property
    def received_power(self) -> float:
        """
        Calculate the received power in dBW.

        Received Power = EIRP + Receiver Antenna Gain - Path Loss - Polarization Loss

        Returns:
            float: Received power in dBW
        """
        return (
            self.eirp
            + self.rx_antenna.gain(self.frequency)
            - self.path_loss
            - self.polarization_loss
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
        return self.mode.ebno(self.carrier_to_noise_ratio)

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
        return self.mode.margin(self.carrier_to_noise_ratio)

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
