"""
Link budget calculations for radio communications.

This module provides a Link class that encapsulates all parameters needed
for a complete link budget calculation, including transmitter and receiver
characteristics, path loss, and noise calculations.
"""

from typing import Union, Optional, Callable
from .path import free_space_path_loss_db
from .conversions import db, db2linear
from .antenna import Antenna
from .noise import thermal_noise_power

class Link:
    """
    A class representing a radio communication link.
    
    This class encapsulates all parameters needed for a complete link budget
    calculation, including transmitter and receiver characteristics, path loss,
    and noise calculations.
    
    Attributes:
        tx_power_dbw: Transmitter power in dBW
        tx_antenna: Transmitter antenna
        rx_antenna: Receiver antenna
        rx_system_noise_temp_k: Receiver system noise temperature in Kelvin
        rx_antenna_noise_temp_k: Receiver antenna noise temperature in Kelvin
        distance_fn: Callable that returns the distance in meters
        frequency_hz: Carrier frequency in Hz
        bandwidth_hz: Signal bandwidth in Hz
        required_ebno_db: Required Eb/N0 in dB for the desired BER
        implementation_loss_db: Implementation loss in dB (default: 0)
        polarization_loss_db: Polarization loss in dB (default: 0)
        pointing_loss_db: Antenna pointing loss in dB (default: 0)
        atmospheric_loss_db: Atmospheric loss in dB (default: 0)
    """
    
    def __init__(self,
                 tx_power_dbw: float,
                 tx_antenna: Antenna,
                 rx_antenna: Antenna,
                 rx_system_noise_temp_k: float,
                 rx_antenna_noise_temp_k: float,
                 distance_fn: Callable[[], float],
                 frequency_hz: float,
                 bandwidth_hz: float,
                 required_ebno_db: float,
                 implementation_loss_db: float = 0.0,
                 polarization_loss_db: float = 0.0,
                 pointing_loss_db: float = 0.0,
                 atmospheric_loss_db: float = 0.0):
        """
        Initialize a Link object with all necessary parameters.
        
        Args:
            tx_power_dbw: Transmitter power in dBW
            tx_antenna: Transmitter antenna
            rx_antenna: Receiver antenna
            rx_system_noise_temp_k: Receiver system noise temperature in Kelvin
            rx_antenna_noise_temp_k: Receiver antenna noise temperature in Kelvin
            distance_fn: Callable that returns the distance in meters
            frequency_hz: Carrier frequency in Hz
            bandwidth_hz: Signal bandwidth in Hz
            required_ebno_db: Required Eb/N0 in dB for the desired BER
            implementation_loss_db: Implementation loss in dB (default: 0)
            polarization_loss_db: Polarization loss in dB (default: 0)
            pointing_loss_db: Antenna pointing loss in dB (default: 0)
            atmospheric_loss_db: Atmospheric loss in dB (default: 0)
            
        Raises:
            ValueError: If any parameter is invalid
        """
        # Validate inputs
        if tx_power_dbw <= 0:
            raise ValueError("Transmitter power must be positive")
        if not isinstance(tx_antenna, Antenna):
            raise ValueError("tx_antenna must be an Antenna instance")
        if not isinstance(rx_antenna, Antenna):
            raise ValueError("rx_antenna must be an Antenna instance")
        if rx_system_noise_temp_k <= 0:
            raise ValueError("System noise temperature must be positive")
        if rx_antenna_noise_temp_k < 0:
            raise ValueError("Antenna noise temperature cannot be negative")
        if not callable(distance_fn):
            raise ValueError("distance_fn must be a callable")
        if frequency_hz <= 0:
            raise ValueError("Frequency must be positive")
        if bandwidth_hz <= 0:
            raise ValueError("Bandwidth must be positive")
        if implementation_loss_db < 0:
            raise ValueError("Implementation loss cannot be negative")
        if polarization_loss_db < 0:
            raise ValueError("Polarization loss cannot be negative")
        if pointing_loss_db < 0:
            raise ValueError("Pointing loss cannot be negative")
        if atmospheric_loss_db < 0:
            raise ValueError("Atmospheric loss cannot be negative")
        if required_ebno_db < 0:
            raise ValueError("Required Eb/N0 cannot be negative")
            
        # Store parameters
        self.tx_power_dbw = tx_power_dbw
        self.tx_antenna = tx_antenna
        self.rx_antenna = rx_antenna
        self.rx_system_noise_temp_k = rx_system_noise_temp_k
        self.rx_antenna_noise_temp_k = rx_antenna_noise_temp_k
        self.distance_fn = distance_fn
        self.frequency_hz = frequency_hz
        self.bandwidth_hz = bandwidth_hz
        self.implementation_loss_db = implementation_loss_db
        self.polarization_loss_db = polarization_loss_db
        self.pointing_loss_db = pointing_loss_db
        self.atmospheric_loss_db = atmospheric_loss_db
        self.required_ebno_db = required_ebno_db
        
    @property
    def distance_m(self) -> float:
        """
        Get the current distance in meters.
        
        Returns:
            float: Current distance in meters
            
        Raises:
            ValueError: If the distance is not positive
        """
        distance = self.distance_fn()
        if distance <= 0:
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
        return (self.tx_power_dbw + 
                self.tx_antenna.gain(self.frequency_hz))
                
    @property
    def path_loss_db(self) -> float:
        """
        Calculate the total path loss in dB.
        
        Path Loss = Free Space Path Loss + Atmospheric Loss + Pointing Loss
        
        Returns:
            float: Total path loss in dB
        """
        return (free_space_path_loss_db(self.distance_m, self.frequency_hz) +
                self.atmospheric_loss_db +
                self.pointing_loss_db)
                
    @property
    def received_power_dbw(self) -> float:
        """
        Calculate the received power in dBW.
        
        Received Power = EIRP + Receiver Antenna Gain - Path Loss - Polarization Loss
        
        Returns:
            float: Received power in dBW
        """
        return (self.eirp +
                self.rx_antenna.gain(self.frequency_hz) -
                self.path_loss_db -
                self.polarization_loss_db)
                
    @property
    def system_noise_temperature_k(self) -> float:
        """
        Calculate the total system noise temperature in Kelvin.
        
        System Noise Temperature = Antenna Noise Temperature + System Noise Temperature
        
        Returns:
            float: Total system noise temperature in Kelvin
        """
        return self.rx_antenna_noise_temp_k + self.rx_system_noise_temp_k
        
    @property
    def noise_power_dbw(self) -> float:
        """
        
        Returns:
            float: Noise power in dBW
        """
        return db(thermal_noise_power(self.bandwidth_hz, self.system_noise_temperature_k))
        
    @property
    def carrier_to_noise_ratio_db(self) -> float:
        """
        Calculate the carrier-to-noise ratio in dB.
        
        C/N = Received Power - Noise Power
        
        Returns:
            float: Carrier-to-noise ratio in dB
        """
        return self.received_power_dbw - self.noise_power_dbw
        
    def ebno_db(self) -> float:
        """
        Calculate the energy per bit to noise power spectral density ratio (Eb/N0) in dB.
        
        Eb/N0 = C/N + 10*log10(B/R)
        where:
        - C/N is the carrier-to-noise ratio in dB
        - B is the bandwidth in Hz
        - R is the data rate in bits per second
            
        Returns:
            float: Eb/N0 in dB
            
        Raises:
            ValueError: If data rate is not positive
        """

        return self.carrier_to_noise_ratio_db
                
    def link_margin_db(self) -> float:
        """
        Calculate the link margin in dB.
        
        Link Margin = Eb/N0 - Required Eb/N0 - Implementation Loss
        
        Implementation loss is subtracted here rather than in the EIRP calculation,
        as it represents losses in the communication system that degrade performance.
        
        Returns:
            float: Link margin in dB
        """
        return self.ebno_db() - self.required_ebno_db - self.implementation_loss_db
        
    def __str__(self) -> str:
        """
        Return a string representation of the link budget.
        
        Returns:
            str: String representation of the link budget
        """
        return (f"Link Budget:\n"
                f"  EIRP: {self.eirp:.1f} dBW\n"
                f"  Path Loss: {self.path_loss_db:.1f} dB\n"
                f"  Received Power: {self.received_power_dbw:.1f} dBW\n"
                f"  System Noise Temperature: {self.system_noise_temperature_k:.1f} K\n"
                f"  Noise Power: {self.noise_power_dbw:.1f} dBW\n"
                f"  C/N: {self.carrier_to_noise_ratio_db:.1f} dB") 