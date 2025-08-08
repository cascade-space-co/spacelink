import math
from fractions import Fraction

import pydantic

from spacelink.core.units import Frequency, enforce_units


class Modulation(pydantic.BaseModel):
    r"""
    Represents a single modulation scheme.

    Parameters
    ----------
    name : str
        Name of the modulation scheme.
    bits_per_symbol : int
        Number of bits per symbol.
    """

    name: str
    bits_per_symbol: int


class Code(pydantic.BaseModel):
    r"""
    Represents a single forward error correction code.

    Parameters
    ----------
    name : str
        Name of the code.
    rate : Fraction
        Code rate.
    interleaver_depth : int | None
        Interleaver depth for codes such as Reed Solomon that are commonly paired with
        an interleaver.
    """

    name: str
    rate: Fraction
    interleaver_depth: int | None = None


class CodeChain(pydantic.BaseModel):
    r"""
    Represents a chain of forward error correction codes.

    This provides the flexibility to handle concatenated codes such as Reed Solomon
    paired with convolutional code but also single codes and even the absence of forward
    error correction.

    Parameters
    ----------
    codes : list[Code]
        List of codes in in encoding order from outermost to innermost code. If no error
        correction is used the list will be empty.
    """

    codes: list[Code]

    @property
    def rate(self) -> Fraction:
        r"""
        The overall code rate of the chain, which is the product of the individual code
        rates or 1 if no error correction is used.

        Returns
        -------
        Fraction
            Code rate.
        """
        return math.prod((code.rate for code in self.codes), start=Fraction(1))


class LinkMode(pydantic.BaseModel):
    r"""
    Represents a specific combination of modulation and coding.

    Parameters
    ----------
    id : str
        Unique identifier for the link mode.
    modulation : Modulation
        Modulation scheme.
    coding : CodeChain
        Forward error correction code chain.
    ref : str, optional
        Reference or source of the mode definition.
    """

    id: str
    modulation: Modulation
    coding: CodeChain
    ref: str = ""

    @property
    def info_bits_per_symbol(self) -> Fraction:
        return self.modulation.bits_per_symbol * self.coding.rate

    @property
    def channel_bits_per_symbol(self) -> int:
        return self.modulation.bits_per_symbol

    @enforce_units
    def info_bit_rate(self, symbol_rate_hz: Frequency) -> Frequency:
        r"""
        Calculate the information bit rate as a function of the symbol rate.

        The information bit rate refers to the rate of information bits, which are the
        input to the first stage of the encoding chain on the transmit end of the link
        or the output of the last decoding stage on the receive end of the link. This is
        sometimes referred to as the "net bit rate."

        Parameters
        ----------
        symbol_rate_hz : Frequency
            Symbol rate.

        Returns
        -------
        Frequency
            Bit rate in Hertz.
        """
        return symbol_rate_hz * self.info_bits_per_symbol

    @enforce_units
    def symbol_rate(self, info_bit_rate: Frequency) -> Frequency:
        r"""
        Calculate the symbol rate as a function of the information bit rate.

        Parameters
        ----------
        info_bit_rate : Frequency
            Information bit rate.

        Returns
        -------
        Frequency
            Symbol rate in Hertz.
        """
        return info_bit_rate / self.info_bits_per_symbol
