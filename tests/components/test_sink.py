"""Tests for the Sink module."""

import pytest
import astropy.units as u
# from astropy.tests.helper import assert_quantity_allclose # Unused
from typing import Optional

from spacelink.components.sink import Sink
from spacelink.components.signal import Signal
from spacelink.components.source import Source
# from spacelink.components.stage import Stage # Unused


class MockSource(Source):
    """Concrete implementation of Source for testing."""

    def __init__(self):
        super().__init__()
        self._power = 1 * u.W
        self.noise_temperature = 290 * u.K

    def output(self, frequency):
        return Signal(power=self._power, noise_temperature=self.noise_temperature)


class MockSink(Sink):
    """Concrete implementation of Sink for testing."""

    def process_input(self, frequency):
        """Process the input signal."""
        if self.input is None:
            raise ValueError("Sink input must be set before processing")
        self._processed_signal = self.input.output(frequency)

    def get_processed_signal(self) -> Optional[Signal]:
        """Retrieve the signal processed by the sink."""
        return self._processed_signal


def test_sink_input_property():
    """Test the input property getter and setter."""
    sink = MockSink()

    # Test initial state
    assert sink.input is None

    # Test setting valid input
    source = MockSource()
    sink.input = source
    assert sink.input == source

    # Test setting None
    sink.input = None
    assert sink.input is None

    # Test setting invalid input
    with pytest.raises(ValueError, match="Input must be a Stage or Source instance"):
        sink.input = "not a stage"


def test_sink_abstract():
    """Test that Sink is an abstract base class."""
    with pytest.raises(TypeError):
        Sink()
