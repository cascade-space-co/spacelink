"""Tests for the noise module."""

import pytest
from pyradio.noise import (
    thermal_noise_power,
    noise_figure_to_temperature,
    temperature_to_noise_figure,
)


def test_thermal_noise_power():
    """Test thermal noise power calculation."""
    # Test with standard room temperature (290K)
    power = thermal_noise_power(1e6)  # 1 MHz bandwidth
    expected = 1.380649e-23 * 290.0 * 1e6
    assert power == pytest.approx(expected)

    # Test with different temperature
    power = thermal_noise_power(1e6, temperature=100.0)
    expected = 1.380649e-23 * 100.0 * 1e6
    assert power == pytest.approx(expected)

    # Test with zero bandwidth
    assert thermal_noise_power(0.0) == 0.0

    # Test with negative bandwidth
    with pytest.raises(ValueError):
        thermal_noise_power(-1.0)


def test_noise_figure_to_temperature():
    """Test conversion from noise figure to temperature."""
    # Test with 3 dB noise figure
    temp = noise_figure_to_temperature(3.0)
    expected = 290.0 * (10 ** (3.0 / 10) - 1)
    assert temp == pytest.approx(expected, rel=1e-3)

    # Test with 0 dB noise figure (ideal case)
    assert noise_figure_to_temperature(0.0) == 0.0

    # Test with negative noise figure
    with pytest.raises(ValueError):
        noise_figure_to_temperature(-1.0)


def test_temperature_to_noise_figure():
    """Test conversion from temperature to noise figure."""
    # Test with temperature corresponding to 3 dB noise figure
    noise_fig = temperature_to_noise_figure(288.6)
    assert noise_fig == pytest.approx(3.0, rel=1e-2)

    # Test with 0K temperature (ideal case)
    assert temperature_to_noise_figure(0.0) == 0.0

    # Test with negative temperature
    with pytest.raises(ValueError):
        temperature_to_noise_figure(-1.0)


def test_noise_figure_temperature_roundtrip():
    """Test that converting between noise figure and temperature is reversible."""
    # Test with various noise figures
    noise_figures = [0.0, 1.0, 3.0, 6.0, 10.0]
    for nf in noise_figures:
        temp = noise_figure_to_temperature(nf)
        nf_back = temperature_to_noise_figure(temp)
        assert nf == pytest.approx(nf_back, rel=1e-3)
