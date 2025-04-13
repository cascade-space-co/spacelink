"""Satellite dish model with gain calculation functionality."""

import numpy as np
from .conversions import db, wavelength

class Dish:
    """A class representing a satellite dish antenna.
    
    This class models a parabolic dish antenna with configurable diameter and efficiency,
    and provides methods to calculate its gain at a given frequency.
    
    Attributes:
        diameter (float): Diameter of the dish in meters
        efficiency (float): Antenna efficiency (0.0 to 1.0)
        lna_gain (float): Low Noise Amplifier gain in dB
        system_temperature (float): System noise temperature in Kelvin
        sky_temperature (float): Sky noise temperature in Kelvin
    """
    
    def __init__(self, diameter: float, efficiency: float = 0.65, lna_gain: float = 0.0,
                 system_temperature: float = 290.0, sky_temperature: float = 20.0):
        """Initialize a satellite dish.
        
        Args:
            diameter (float): Diameter of the dish in meters
            efficiency (float, optional): Antenna efficiency. Defaults to 0.65.
            lna_gain (float, optional): LNA gain in dB. Defaults to 0.0.
            system_temperature (float, optional): System noise temperature in Kelvin. Defaults to 290.0.
            sky_temperature (float, optional): Sky noise temperature in Kelvin. Defaults to 20.0.
        
        Raises:
            ValueError: If diameter is negative, efficiency is not between 0 and 1,
                       or temperatures are not positive
        """
        self._diameter = None
        self._efficiency = None
        self._lna_gain = None
        self._system_temperature = None
        self._sky_temperature = None
        
        # Use property setters to validate initial values
        self.diameter = diameter
        self.efficiency = efficiency
        self.lna_gain = lna_gain
        self.system_temperature = system_temperature
        self.sky_temperature = sky_temperature
        
    @property
    def diameter(self) -> float:
        """Get the diameter of the dish in meters."""
        return self._diameter
    
    @diameter.setter
    def diameter(self, value: float) -> None:
        """Set the diameter of the dish in meters.
        
        Args:
            value (float): Diameter in meters
            
        Raises:
            ValueError: If diameter is not positive
        """
        if value <= 0:
            raise ValueError("Diameter must be positive")
        self._diameter = value
    
    @property
    def efficiency(self) -> float:
        """Get the antenna efficiency."""
        return self._efficiency
    
    @efficiency.setter
    def efficiency(self, value: float) -> None:
        """Set the antenna efficiency.
        
        Args:
            value (float): Efficiency value between 0 and 1
            
        Raises:
            ValueError: If efficiency is not between 0 and 1
        """
        if not 0 <= value <= 1:
            raise ValueError("Efficiency must be between 0 and 1")
        self._efficiency = value
    
    @property
    def lna_gain(self) -> float:
        """Get the LNA gain in dB."""
        return self._lna_gain
    
    @lna_gain.setter
    def lna_gain(self, value: float) -> None:
        """Set the LNA gain in dB.
        
        Args:
            value (float): LNA gain in dB
        """
        self._lna_gain = value
    
    @property
    def system_temperature(self) -> float:
        """Get the system temperature in Kelvin."""
        return self._system_temperature
    
    @system_temperature.setter
    def system_temperature(self, value: float) -> None:
        """Set the system temperature in Kelvin.
        
        Args:
            value (float): System temperature in Kelvin
            
        Raises:
            ValueError: If temperature is not positive
        """
        if value <= 0:
            raise ValueError("System temperature must be positive")
        self._system_temperature = value
    
    @property
    def sky_temperature(self) -> float:
        """Get the sky temperature in Kelvin."""
        return self._sky_temperature
    
    @sky_temperature.setter
    def sky_temperature(self, value: float) -> None:
        """Set the sky temperature in Kelvin.
        
        Args:
            value (float): Sky temperature in Kelvin
            
        Raises:
            ValueError: If temperature is not positive
        """
        if value <= 0:
            raise ValueError("Sky temperature must be positive")
        self._sky_temperature = value
    
    def dish_gain(self, frequency: float) -> float:
        """Calculate the antenna gain at a given frequency.
        
        The gain is calculated using the formula:
        G = η * (π * D / λ)²
        where:
        - η is the antenna efficiency
        - D is the antenna diameter
        - λ is the wavelength (c/frequency)
        
        Args:
            frequency (float): Frequency in Hz
            
        Returns:
            float: Gain in dB
            
        Raises:
            ValueError: If frequency is not positive
        """
        if frequency <= 0:
            raise ValueError("Frequency must be positive")
            
        wavelength_m = wavelength(frequency)
        gain_linear = self.efficiency * (np.pi * self.diameter / wavelength_m)**2
        return db(gain_linear)
    
    def gain(self, frequency: float) -> float:
        """Calculate the total system gain (dish + LNA) at a given frequency.
        
        The total gain is the sum of the dish gain and LNA gain in dB.
        
        Args:
            frequency (float): Frequency in Hz
            
        Returns:
            float: Total gain in dB
            
        Raises:
            ValueError: If frequency is not positive
        """
        if frequency <= 0:
            raise ValueError("Frequency must be positive")
            
        return self.dish_gain(frequency) + self.lna_gain
    
    def half_power_beamwidth(self, frequency: float) -> float:
        """Calculate the half-power beamwidth (HPBW) at a given frequency.
        
        The HPBW is approximately given by:
        HPBW ≈ 70° * (λ/D)
        where:
        - λ is the wavelength (c/frequency)
        - D is the antenna diameter
        
        Args:
            frequency (float): Frequency in Hz
            
        Returns:
            float: Half-power beamwidth in degrees
            
        Raises:
            ValueError: If frequency is not positive
        """
        if frequency <= 0:
            raise ValueError("Frequency must be positive")
            
        wavelength_m = wavelength(frequency)
        return 70 * (wavelength_m / self.diameter)
    
    def __str__(self) -> str:
        """Return a string representation of the dish."""
        return (f"Dish(diameter={self.diameter:.2f}m, efficiency={self.efficiency:.2f}, "
                f"lna_gain={self.lna_gain:.1f}dB, system_temp={self.system_temperature:.1f}K, "
                f"sky_temp={self.sky_temperature:.1f}K)")

    def gt_ratio(self, frequency: float) -> float:
        """Calculate the G/T ratio at a given frequency.
        
        The G/T ratio is calculated as:
        G/T = G_ant + G_lna - 10*log10(T_sys + T_sky)
        where:
        - G_ant is the antenna gain in dB
        - G_lna is the LNA gain in dB
        - T_sys is the system temperature in Kelvin
        - T_sky is the sky temperature in Kelvin
        
        Args:
            frequency (float): Frequency in Hz
            
        Returns:
            float: G/T ratio in dB/K
            
        Raises:
            ValueError: If frequency is not positive
        """
        if frequency <= 0:
            raise ValueError("Frequency must be positive")
            
        # Calculate total system temperature
        total_temp = self.system_temperature + self.sky_temperature
        
        # Calculate G/T ratio in dB/K
        return self.gain(frequency) - db(total_temp) 