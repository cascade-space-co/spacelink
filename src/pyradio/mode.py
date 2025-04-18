"""
Communication mode definitions for radio systems.

This module provides a Mode class that encapsulates parameters for different
communication modes, including modulation schemes, channel coding, and required
signal quality parameters.
"""

from typing import Optional
from pint import Quantity
from .units import db

"""TODO: change this to an abstract base class

    Each class of modes will have unique functions for computing things like
    spectral efficiency based on rolloff etc etc.

    This class should only have parameters that are fairly static, and that don't
    change during a mission
"""
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

    def __init__(
        self,
        name: str,
        coding_scheme: str,
        modulation: str,
        bits_per_symbol: Quantity,
        symbol_rate: Quantity,
        code_rate: float,
        spectral_efficiency: float,
        required_ebno: float,
        implementation_loss: float = 0.0,
    ):
        """
        Initialize a Mode object with all necessary parameters.

        Args:
            name: Human-readable name of the mode
            coding_scheme: Channel coding scheme used (e.g., "Uncoded", "Reed-Solomon", "LDPC")
            modulation: Modulation scheme used (e.g., "BPSK", "QPSK", "8PSK")
            spectral_efficiency: Bits per symbol (accounting for coding overhead)
            required_ebno: Required Eb/N0 in dB for the target BER
            implementation_loss: Implementation loss in dB (default: 0.0)
            code_rate: The channel code rate (data bits / coded bits) (default: None)

        Raises:
            ValueError: If any parameter is invalid
        """
        # Validate inputs
        if not name:
            raise ValueError("Name must not be empty")
        if not coding_scheme:
            raise ValueError("Coding scheme must not be empty")
        if not modulation:
            raise ValueError("Modulation must not be empty")
        if spectral_efficiency <= 0.0:
            raise ValueError("Spectral efficiency must be positive")
        if (code_rate <= 0.0 or code_rate > 1.0):
            raise ValueError("Code rate must be between 0 and 1")
        if implementation_loss > 0.0:
            raise ValueError("Implementation loss must be negative")

        # Store parameters
        self.name = name
        self.coding_scheme = coding_scheme
        self.modulation = modulation
        self.bits_per_symbol = bits_per_symbol
        self.symbol_rate = symbol_rate
        self.spectral_efficiency = spectral_efficiency
        self.required_ebno = required_ebno
        self.implementation_loss = implementation_loss
        self.code_rate = code_rate

    @property
    def data_rate(self) -> Quantity:
        """
        Calculate the data rate for this mode given a bandwidth.

        The data rate is calculated based on the spectral efficiency and bandwidth.
        Returns:
            Data rate in bits per second (bps)

        Raises:
            ValueError: If bandwidth is not positive
        """
        return self.symbol_rate * self.bits_per_symbol * self.code_rate

    @property
    def bandwidth(self) -> Quantity:
        """
        Calculate the required bandwidth for this mode given a symbol rate.

        The bandwidth is calculated based on the spectral efficiency and data rate.
        Returns:
            Required bandwidth in Hz

        Raises:
            ValueError: If data rate is not positive
        """
        # Calculate required bandwidth in Hz
        return self.symbol_rate.to('Hz') / self.spectral_efficiency

    def ebno(self, c_over_n: float) -> float:
        """Eb/N0 for given carrier to noise ratio"""
        return c_over_n - db(self.bits_per_symbol.magnitude)

    def margin(self, c_over_n: float) -> float:
        """
        Calculate the link margin in dB for a given carrier-to-noise ratio.

        The margin is calculated by converting C/N to Eb/N0 and comparing with the required Eb/N0.
        The conversion accounts for the modulation and coding scheme encapsulated in this mode.

        Eb/N0 = C/N + 10*log10(bandwidth/symbol_rate) - 10*log10(spectral_efficiency)
        Margin = Eb/N0 - Required Eb/N0 - Implementation Loss

        Returns:
            Link margin in dB
        """

        # Calculate margin
        return self.ebno(c_over_n) - self.required_ebno + self.implementation_loss

    def __str__(self) -> str:
        """
        Return a string representation of the mode.

        Returns:
            String representation of the mode
        """
        code_rate_str = f", Code Rate: {self.code_rate:.3f}" if self.code_rate is not None else ""
        return (
            f"{self.name}: {self.modulation} with {self.coding_scheme}\n"
            f"  Spectral Efficiency: {self.spectral_efficiency:.3f} bits/symbol{code_rate_str}\n"
            f"  Required Eb/N0: {self.required_ebno:.2f} dB\n"
            f"  Implementation Loss: {self.implementation_loss:.2f} dB"
        )
