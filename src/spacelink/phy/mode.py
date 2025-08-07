import math
from fractions import Fraction

from pydantic import BaseModel

from spacelink.core.units import Frequency, enforce_units


class Modulation(BaseModel):
    name: str
    bits_per_symbol: int


class Code(BaseModel):
    name: str
    rate: Fraction
    interleaver_depth: int | None = None


class CodeChain(BaseModel):
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


class LinkMode(BaseModel):
    id: str
    modulation: Modulation
    coding: CodeChain
    ref: str = ""

    @enforce_units
    def bit_rate(self, symbol_rate_hz: Frequency) -> Frequency:
        r"""
        Calculate the bit rate as a function of the symbol rate.

        Parameters
        ----------
        symbol_rate_hz : Frequency
            Symbol rate.

        Returns
        -------
        Frequency
            Bit rate in Hertz.
        """
        return symbol_rate_hz * float(self.coding.rate)

    @enforce_units
    def symbol_rate(self, bit_rate: Frequency) -> Frequency:
        r"""
        Calculate the symbol rate as a function of the bit rate.

        Parameters
        ----------
        bit_rate : Frequency
            Bit rate.

        Returns
        -------
        Frequency
            Symbol rate in Hertz.
        """
        return bit_rate / float(self.coding.rate)
