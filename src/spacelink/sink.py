"""
Sink module for RF signal sinks.

This module provides the Sink base class that represents components that consume
signals without producing an output, such as power meters or spectrum analyzers.
"""

from typing import Optional, Union
import astropy.units as u

from spacelink.signal import Signal
from spacelink.units import Frequency
from spacelink.stage import Stage
from spacelink.source import Source


class Sink:
    """
    Base class for signal sinks.

    A sink represents a component that consumes signals without producing an output,
    such as a power meter or spectrum analyzer. Sinks are the end points of RF chains.
    """
    def __init__(self):
        self._input: Optional[Union[Stage, Source]] = None

    @property
    def input(self) -> Optional[Union[Stage, Source]]:
        """Get the input for this sink."""
        return self._input

    @input.setter
    def input(self, value: Optional[Union[Stage, Source]]) -> None:
        """Set the input for this sink."""
        if value is not None and not isinstance(value, (Stage, Source)):
            raise ValueError("Input must be a Stage or Source instance")
        self._input = value

    def output(self, frequency: Frequency) -> Signal:
        """
        Calculate the output signal for this sink.

        Args:
            frequency: The frequency at which to calculate the output

        Returns:
            Signal: The output signal from this sink

        Raises:
            ValueError: If input is not set
        """
        if self.input is None:
            raise ValueError("Sink input must be set before calculating output")
        return self.input.output(frequency) 