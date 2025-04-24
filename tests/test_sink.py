"""Tests for the Sink class."""

import pytest
import astropy.units as u

from spacelink.sink import Sink
from spacelink.signal import Signal


class TestSink(Sink):
    """Concrete implementation of Sink for testing."""
    def process_input(self, frequency):
        """Test implementation of abstract method."""
        pass


def test_sink_input_property():
    """Test the input property getter and setter."""
    sink = TestSink()
    
    # Test initial state
    assert sink.input is None
    
    # Test setting valid input
    def valid_input():
        return Signal(power=1 * u.W, noise_temperature=290 * u.K)
    sink.input = valid_input
    assert sink.input == valid_input
    
    # Test setting None
    sink.input = None
    assert sink.input is None
    
    # Test setting invalid input
    with pytest.raises(ValueError, match="Input must be a callable that returns a Signal"):
        sink.input = "not a callable"


def test_sink_abstract():
    """Test that Sink is an abstract base class."""
    with pytest.raises(TypeError):
        Sink()  # Should raise TypeError as Sink is abstract 