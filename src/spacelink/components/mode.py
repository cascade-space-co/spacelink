"""
Communication mode definitions for radio systems.

This module provides a Mode class that encapsulates parameters for different
communication modes, including modulation schemes, channel coding, and required
signal quality parameters.
"""

import astropy.units as u
from dataclasses import dataclass

# Update imports
from ..core.units import (
    Dimensionless,
    Decibels,
    to_dB,
)

"""TODO: change this to an abstract base class

    Each class of modes will have unique functions for computing things like
    spectral efficiency based on rolloff etc etc.

    This class should only have parameters that are fairly static, and that don't
    change during a mission
"""


@dataclass
class Mode:
    """
    A class representing a communication mode with specific coding and modulation.

    This class encapsulates parameters for different communication modes, including
    channel coding scheme, required Eb/N0 for a target BER, and implementation loss.

    Attributes:
        name: Human-readable name of the mode
        coding_scheme: Channel coding scheme used (e.g., "Uncoded", "Reed-Solomon", "LDPC")
        modulation: Modulation scheme used (e.g., "BPSK", "QPSK", "8PSK")
        spectral_efficiency: Bits per symbol (accounting for coding overhead)
        required_ebno: Required Eb/N0 in dB for the target BER
        target_ber: Target bit error rate (e.g., 1e-5)
        implementation_loss: Implementation loss in dB
        code_rate: The channel code rate (data bits / coded bits)
    """

    name: str
    coding_scheme: str
    modulation: str
    bits_per_symbol: Dimensionless
    code_rate: float
    spectral_efficiency: float
    required_ebno: float
    implementation_loss: float = 0.0

    def __post_init__(self):
        """Validate parameters after initialization."""
        if not self.name:
            raise ValueError("Name must not be empty")
        if not self.coding_scheme:
            raise ValueError("Coding scheme must not be empty")
        if not self.modulation:
            raise ValueError("Modulation must not be empty")
        if self.spectral_efficiency <= 0.0:
            raise ValueError("Spectral efficiency must be positive")
        if self.code_rate <= 0.0 or self.code_rate > 1.0:
            raise ValueError("Code rate must be between 0 and 1")
        if self.implementation_loss < 0.0:
            raise ValueError("Implementation loss must be non-negative")
        # Ensure bits_per_symbol is a Quantity
        if not isinstance(self.bits_per_symbol, u.Quantity):
            # Attempt conversion if it's a plain number
            if isinstance(self.bits_per_symbol, (int, float)):
                self.bits_per_symbol = self.bits_per_symbol * u.dimensionless
            else:
                raise TypeError("bits_per_symbol must be a dimensionless Quantity")
        elif not self.bits_per_symbol.unit.is_equivalent(u.dimensionless):
            raise u.UnitConversionError("bits_per_symbol must be dimensionless")

    def ebno(self, c_over_n: Decibels) -> Decibels:
        """Eb/N0 for given carrier to noise ratio"""
        bits_per_symbol_db = to_dB(self.bits_per_symbol)
        return c_over_n - bits_per_symbol_db

    def margin(self, c_over_n: Decibels) -> Decibels:
        """
        Calculate the link margin in dB for a given carrier-to-noise ratio.

        The margin is calculated by converting C/N to Eb/N0 and comparing with the required Eb/N0.
        The conversion accounts for the modulation and coding scheme encapsulated in this mode.

        Eb/N0 = C/N + 10*log10(bandwidth/symbol_rate) - 10*log10(spectral_efficiency)
        Margin = Eb/N0 - Required Eb/N0 - Implementation Loss

        Returns:
            Link margin in dB
        """

        # Calculate margin (subtract positive implementation loss)
        return (
            self.ebno(c_over_n)
            - self.required_ebno * u.dB
            - self.implementation_loss * u.dB
        )

    def __str__(self) -> str:
        """
        Return a string representation of the mode.

        Returns:
            String representation of the mode
        """
        code_rate_str = (
            f", Code Rate: {self.code_rate:.3f}" if self.code_rate is not None else ""
        )
        return (
            f"{self.name}: {self.modulation} with {self.coding_scheme}\n"
            f"  Spectral Efficiency: {self.spectral_efficiency:.3f} bits/symbol{code_rate_str}\n"
            f"  Required Eb/N0: {self.required_ebno:.2f} dB\n"
            f"  Implementation Loss: {self.implementation_loss:.2f} dB"
        )
