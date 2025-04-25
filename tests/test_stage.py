"""Tests for the Stage module."""

import pytest
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose

from spacelink.stage import Stage, GainBlock, Attenuator
from spacelink.source import Source
from spacelink.sink import Sink
from spacelink.signal import Signal

from spacelink.noise import temperature_to_noise_figure
class TestSource(Source):
    """Concrete implementation of Source for testing."""
    def output(self, frequency):
        return Signal(power=1 * u.W, noise_temperature=290 * u.K)

    @property
    def gain(self) -> u.Quantity:
        """Get the gain of this source (0 dB)."""
        return 0 * u.dB

    @property
    def noise_figure(self) -> u.Quantity:
        """Get the noise figure of this source (0 dB)."""
        return 0 * u.dB


def test_source_input_property():
    """Test that Source doesn't accept inputs."""
    source = TestSource()
    
    # Test that source has no input property
    with pytest.raises(AttributeError):
        _ = source.input


def test_sink_input_property():
    """Test that Sink accepts Stage inputs."""
    sink = Sink()
    source = TestSource()
    
    # Test initial state
    assert sink.input is None
    
    # Test setting valid input
    sink.input = source
    assert sink.input == source
    
    # Test setting None
    sink.input = None
    assert sink.input is None
    
    # Test setting invalid input
    with pytest.raises(ValueError, match="Input must be a Stage or Source instance"):
        sink.input = "not a stage"


def test_gain_block_creation():
    """Test creating a GainBlock with valid parameters."""
    gain_block = GainBlock(
        gain=20 * u.dB,
        noise_temperature=290 * u.K,
        input_return_loss=20 * u.dB
    )
    
    assert gain_block.gain.value == pytest.approx(20.0)
    assert gain_block.noise_temperature.value == pytest.approx(290.0)
    assert gain_block.input_return_loss.value == pytest.approx(20.0)


def test_gain_block_invalid_params():
    """Test creating a GainBlock with invalid parameters."""
    # Test negative noise temperature
    with pytest.raises(ValueError, match="noise_temperature must be non-negative"):
        GainBlock(gain=20 * u.dB, noise_temperature=-1 * u.K)
    
    # Test negative return loss
    with pytest.raises(ValueError, match="input_return_loss must be non-negative"):
        GainBlock(gain=20 * u.dB, noise_temperature=290 * u.K, input_return_loss=-1 * u.dB)


def test_gain_block_output():
    """Test GainBlock output calculation."""
    source = TestSource()
    gain_block = GainBlock(gain=10 * u.dB, noise_temperature=290 * u.K)
    gain_block.input = source
    
    output = gain_block.output(1 * u.GHz)
    
    # 10 dB gain = 10x power
    assert_quantity_allclose(output.power, 10 * u.W)
    # Noise temperature should be sum of source and gain block
    assert_quantity_allclose(output.noise_temperature, 5800 * u.K)


def test_attenuator_creation():
    """Test creating an Attenuator with valid parameters."""
    attenuator = Attenuator(
        attenuation=10 * u.dB,
        input_return_loss=20 * u.dB
    )
    
    assert_quantity_allclose(attenuator.attenuation, 10 * u.dB)
    assert_quantity_allclose(attenuator.input_return_loss, 20 * u.dB)


def test_attenuator_invalid_params():
    """Test creating an Attenuator with invalid parameters."""
    # Test negative attenuation
    with pytest.raises(ValueError, match="attenuation must be non-negative"):
        Attenuator(attenuation=-1 * u.dB)
    
    # Test negative return loss
    with pytest.raises(ValueError, match="input_return_loss must be non-negative"):
        Attenuator(attenuation=10 * u.dB, input_return_loss=-1 * u.dB)


def test_attenuator_output():
    """Test Attenuator output calculation."""
    source = TestSource()
    attenuator = Attenuator(attenuation=10 * u.dB)
    attenuator.input = source
    
    output = attenuator.output(1 * u.GHz)
    
    # 10 dB attenuation = 1/10 power
    assert_quantity_allclose(output.power, 0.1 * u.W)
    assert_quantity_allclose(output.noise_temperature, 290 * u.K)


def test_stage_chain():
    """Test chaining multiple stages together."""
    source = TestSource()
    gain_block1 = GainBlock(gain=20 * u.dB, noise_figure=2 * u.dB)
    gain_block2 = GainBlock(gain=20 * u.dB, noise_figure=4 * u.dB)
    gain_block3 = GainBlock(gain=20 * u.dB, noise_figure=10 * u.dB)
    
    # Connect the chain
    gain_block1.input = source
    gain_block2.input = gain_block1
    gain_block3.input = gain_block2

    # Get output at sink
    output = gain_block3.output(1 * u.GHz)

    assert_quantity_allclose(output.power.to(u.dBW), 60 * u.dBW)
    # The cascaded input referred noise temperature is 2.044 dB but we are
    # meauring the ouput referred noise temperature here so we add the gain of 
    # all the stages together
    assert_quantity_allclose(temperature_to_noise_figure(output.noise_temperature), 62.044 * u.dB, atol=0.01 * u.dB)

    # Cascaded noise figure should be 2.044
    assert_quantity_allclose(gain_block3.cascaded_noise_figure, 2.044 * u.dB, atol=0.01 * u.dB)
    assert_quantity_allclose(gain_block3.cascaded_gain, 60 * u.dB, atol=0.01 * u.dB)

def test_cascaded_noise_figure():
    """Test cascaded noise figure calculation."""
    source = TestSource()
    gain_block1 = GainBlock(gain=20 * u.dB, noise_figure=2 * u.dB)
    gain_block2 = GainBlock(gain=20 * u.dB, noise_figure=4 * u.dB)


def test_attenuator_noise_calculations():
    """Test that attenuator noise temperature and noise figure are calculated correctly."""
    # Test 3 dB attenuator
    atten = Attenuator(attenuation=3 * u.dB)
    # For 3 dB attenuation (L = 2):
    # noise_temperature = (2-1)*290K = 290K
    # noise_figure = 3 dB (equal to attenuation)
    assert_quantity_allclose(atten.noise_temperature, 290 * u.K, atol=1.5 * u.K)
    assert_quantity_allclose(atten.noise_figure, 3 * u.dB, atol=0.01 * u.dB)

    # Test 6 dB attenuator
    atten.attenuation = 6 * u.dB
    # For 6 dB attenuation (L = 4):
    # noise_temperature = (4-1)*290K = 870K
    # noise_figure = 6 dB (equal to attenuation)
    assert_quantity_allclose(atten.noise_temperature, 864.51 * u.K, atol=0.01 * u.K)
    assert_quantity_allclose(atten.noise_figure, 6 * u.dB, atol=0.01 * u.dB)

    # Test 10 dB attenuator
    atten.attenuation = 10 * u.dB
    # For 10 dB attenuation (L = 10):
    # noise_temperature = (10-1)*290K = 2610K
    # noise_figure = 10 dB (equal to attenuation)
    assert_quantity_allclose(atten.noise_temperature, 2610 * u.K, atol=0.01 * u.K)
    assert_quantity_allclose(atten.noise_figure, 10 * u.dB, atol=0.01 * u.dB)

    # Test 20 dB attenuator
    atten.attenuation = 20 * u.dB
    # For 20 dB attenuation (L = 100):
    # noise_temperature = (100-1)*290K = 28710K
    # noise_figure = 20 dB (equal to attenuation)
    assert_quantity_allclose(atten.noise_temperature, 28710 * u.K, atol=0.01 * u.K)
    assert_quantity_allclose(atten.noise_figure, 20 * u.dB, atol=0.01 * u.dB)