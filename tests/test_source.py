"""Tests for the Source class."""

import pytest
import astropy.units as u

from spacelink.source import Source
from spacelink.signal import Signal


class MockSource(Source):
    """Concrete implementation of Source for testing."""
    def __init__(self):
        super().__init__()
        self._power = 1 * u.W
        self.noise_temperature = 290 * u.K

    def output(self, frequency):
        return Signal(power=self._power, noise_temperature=self.noise_temperature)


def test_source_output():
    """Test that a concrete source implementation works."""
    source = MockSource()
    signal = source.output(1 * u.GHz)
    
    assert isinstance(signal, Signal)
    assert signal.power == 1 * u.W
    assert signal.noise_temperature == 290 * u.K


def test_source_abstract():
    """Test that Source is an abstract base class."""
    with pytest.raises(TypeError):
        Source()  # Should raise TypeError as Source is abstract 