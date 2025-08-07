import math
from fractions import Fraction

import pydantic

from spacelink.core.units import Frequency, enforce_units


class Modulation(pydantic.BaseModel):
    name: str
    bits_per_symbol: int


class Code(pydantic.BaseModel):
    name: str
    rate: Fraction
    interleaver_depth: int | None = None


class CodeChain(pydantic.BaseModel):
    codes: list[Code]

    @property
    def rate(self) -> Fraction:
        r"""
        The overall code rate of the code chain.

        Returns
        -------
        Fraction
            Code rate.
        """
        return math.prod((code.rate for code in self.codes), start=Fraction(1))


class LinkMode(pydantic.BaseModel):
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
