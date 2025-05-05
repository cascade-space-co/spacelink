"""
Demodulator component model.

This stage represents the end of a receive chain, responsible for calculating
link performance metrics like C/N0 and margins based on the received signal
and the expected communication mode.
"""

import astropy.units as u
from typing import Optional

from .stage import Stage
from .signal import Signal
from .mode import DataMode, RangingMode
from ..core.units import (
    Frequency,
    Decibels,
    enforce_units,
    to_dB,
)  # Add necessary units
from ..core.noise import noise_power


class Demodulator(Stage):
    """
    Represents a demodulator at the end of a receive chain.

    Calculates key performance metrics like C/N0, Eb/N0, and link margins
    for data and/or ranging signals, accounting for power sharing when both
    modes are present.
    """

    @enforce_units
    def __init__(
        self,
        carrier_frequency: Frequency,
        symbol_rate: Frequency,
        data_mode: Optional[DataMode] = None,
        ranging_mode: Optional[RangingMode] = None,
        implementation_loss: Decibels = 0 * u.dB,
    ):
        """
        Initialize the Demodulator.

        Args:
            symbol_rate: The symbol rate of the signal in Hz or equivalent.
                         (Primarily applies to the data mode).
            data_mode: Optional DataMode object defining data modulation, coding, etc.
            ranging_mode: Optional RangingMode object defining ranging modulation indices.
            implementation_loss: Implementation loss of the demodulator in dB.
        """
        super().__init__()
        if data_mode is None and ranging_mode is None:
            raise ValueError(
                "At least one of data_mode or ranging_mode must be provided."
            )
        if data_mode is not None and not isinstance(data_mode, DataMode):
            raise TypeError("data_mode must be an instance of DataMode or None")
        if ranging_mode is not None and not isinstance(ranging_mode, RangingMode):
            raise TypeError("ranging_mode must be an instance of RangingMode or None")

        self.data_mode = data_mode
        self.ranging_mode = ranging_mode
        self.symbol_rate = symbol_rate
        self.implementation_loss = implementation_loss
        self.carrier_frequency = carrier_frequency

    @enforce_units
    def cn0(self) -> Decibels:
        input_signal = self.input.output(self.carrier_frequency)
        # Calculate noise power in 1 Hz bandwidth to get N0
        n0 = noise_power(bandwidth=1 * u.Hz, temperature=input_signal.noise_temperature)
        return to_dB(input_signal.power / n0)

    @property
    def data_cn0(self) -> Decibels:
        """
        Calculates the C/N0 for the data mode in dB-Hz.
        """
        if self.ranging_mode is not None:
            return self.cn0() + self.ranging_mode.data_power_fraction
        else:
            return self.cn0()

    @property
    def ranging_prn0(self) -> Decibels:
        """
        Calculates the PRN0 for the ranging mode in dB-Hz.
        """
        return self.cn0() + self.ranging_mode.ranging_power_fraction

    @property
    def ebno(self) -> Decibels:
        """
        Calculates the Energy per bit to Noise Density ratio (Eb/N0) in dB.

        If ranging mode is present, C/N0 used is adjusted by the data power fraction.
        Eb/N0 = (C_data / N0) / BitRate

        Returns:
            Eb/N0 in dB. Returns -inf dB if not a DataMode.
        """
        return (
            self.data_mode.ebno(self.data_cn0, self.symbol_rate)
            - self.implementation_loss
        )

    @property
    def data_margin(self) -> Decibels:
        """
        Calculates the data link margin in dB by delegating to the DataMode object.

        Margin = Actual_Eb/N0 - Required_Eb/N0

        Returns:
            Data link margin in dB.
        """
        return self.data_mode.margin(self.data_cn0, self.symbol_rate)

    # --- Ranging Placeholder ---
    # TODO: Implement ranging calculations based on core.ranging module

    # Override output method - Demodulator consumes the signal
    def output(self, frequency: Frequency) -> Signal:
        raise NotImplementedError(
            "Demodulator does not produce an output signal in the chain."
        )
