"""
Sink module for RF signal sinks.

This module provides the Sink base class that represents components that consume
signals without producing an output, such as power meters or spectrum analyzers.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union
import astropy.units as u

from spacelink.signal import Signal
from spacelink.units import Frequency
# Removed: from spacelink.stage import Stage
# Removed: from spacelink.source import Source


class Sink(ABC):
    """
    Base class for signal sinks.

    A sink represents the endpoint of an RF chain, processing the final signal.
    Examples include demodulators or data recorders.
    """
    def __init__(self):
        # super().__init__() # Removed Stage super call
        self._input = None
        self._processed_signal = None

    @property
    def input(self) -> Optional[Union['Stage', 'Source']]: # Kept hint broad for now
        """Get the input stage or source for this sink."""
        return self._input

    @input.setter
    def input(self, value: Optional[Union['Stage', 'Source']]) -> None:
        """Set the input stage or source for this sink."""
        # Need to import locally to avoid circular if Stage imports Sink
        from spacelink.stage import Stage
        from spacelink.source import Source
        if value is not None and not isinstance(value, (Stage, Source)):
            raise ValueError("Input must be a Stage or Source instance")
        self._input = value
        self._processed_signal = None # Reset processing when input changes

    # Removed Stage methods: gain, noise_figure, cascaded_gain, cascaded_noise_figure, output, output_noise_temperature

    @abstractmethod
    def process_input(self, frequency: Frequency) -> None:
        """
        Process the input signal and store the result internaly or perform an action.
        This method should be called after setting the input.

        Args:
            frequency: The frequency at which to process the input.

        Raises:
            ValueError: If input is not set.
            NotImplementedError: If the sink does not implement this method.
        """
        pass

    # Potentially add methods to retrieve processed data if needed, e.g.:
    # def get_snr(self) -> float: ...
    # def get_data_rate(self) -> u.Quantity: ... 