"""
Communication mode definitions for radio systems.

This module provides a Mode class that encapsulates parameters for different
communication modes, including modulation schemes, channel coding, and required
signal quality parameters.
"""

import astropy.units as u

from spacelink.core.ranging import (
    power_fractions_sine,
)

from spacelink.core.modcod import (
    ErrorRate,
    required_ebno_for_psk_ber,
    get_code_rate_from_scheme,
)

# Update imports
from ..core.units import (
    Dimensionless,
    Decibels,
    to_dB,
    Frequency,
)


class RangingMode:
    def __init__(self, ranging_mod_idx: u.dimensionless, data_mod_idx: u.dimensionless):
        self._ranging_mod_idx = ranging_mod_idx
        self._data_mod_idx = data_mod_idx

    @property
    def power_fractions(self):
        return power_fractions_sine(self._ranging_mod_idx, self._data_mod_idx)

    @property
    def ranging_power_fraction(self) -> Decibels:
        carrier, ranging, data = self.power_fractions
        return to_dB(ranging)

    @property
    def data_power_fraction(self) -> Decibels:
        carrier, ranging, data = self.power_fractions
        return to_dB(data)


class DataMode:
    """
    Modulation scheme and channel coding for the data (sub) carrier.
    """

    def __init__(
        self,
        coding_scheme: str,
        bits_per_symbol: Dimensionless,
        error_rate: ErrorRate = ErrorRate.E_NEG_5,
    ):
        self.coding_scheme = coding_scheme
        self.error_rate = error_rate
        self.bits_per_symbol = bits_per_symbol

    @property
    def required_ebno(self) -> Decibels:
        return required_ebno_for_psk_ber(self.error_rate, self.coding_scheme)

    @property
    def code_rate(self) -> Dimensionless:
        return get_code_rate_from_scheme(self.coding_scheme)

    def bit_rate(self, symbol_rate: Frequency) -> Frequency:
        return symbol_rate.to(u.Hz) * self.bits_per_symbol * self.code_rate

    def ebno(self, cn0: Decibels, symbol_rate: Frequency) -> Decibels:
        return cn0 - to_dB(self.bit_rate(symbol_rate))

    def margin(self, cn0: Decibels, symbol_rate: Frequency) -> Decibels:
        ebn0 = self.ebno(cn0, symbol_rate)
        return ebn0 - self.required_ebno


# class Carrier:
#     def __init__(self, ranging_mode: RangingMode, data_mode: DataMode):
#         self.ranging_mode = ranging_mode
#         self.data_mode = data_mode
