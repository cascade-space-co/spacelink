"""
SpaceLink - A Python library for radio frequency calculations.

This package provides utilities for radio frequency calculations, including:
- Antenna gain and beamwidth calculations
- RF conversions between dB and linear scales
- Noise calculations and conversions
- Communication mode definitions for coding and modulation
"""

from spacelink.stage import (
    Stage,
    GainBlock,
    Attenuator,
    Antenna,
    TransmitAntenna,
    ReceiveAntenna,
    Path,
)
from spacelink.source import Source
from spacelink.sink import Sink
from spacelink.signal import Signal

from spacelink.units import (
    Frequency,
    Decibels,
    Temperature,
    Dimensionless,
    Length,
    Distance,
    enforce_units,
    to_dB,
    to_linear,
    wavelength,
)

from spacelink.noise import (
    noise_figure_to_temperature,
    temperature_to_noise_figure,
    noise_factor_to_temperature,
    temperature_to_noise_factor,
)

from spacelink.path import (
    spreading_loss,
    aperture_loss,
    free_space_path_loss,
)

from spacelink.antenna import polarization_loss

__all__ = [
    # Core classes
    'Stage',
    'Source',
    'Sink',
    'Signal',
    'GainBlock',
    'Attenuator',
    'Antenna',
    'TransmitAntenna',
    'ReceiveAntenna',
    'Path',
    
    # Units and type hints
    'Frequency',
    'Decibels',
    'Temperature',
    'Dimensionless',
    'Length',
    'Distance',
    'enforce_units',
    'to_dB',
    'to_linear',
    'wavelength',
    
    # Noise functions
    'noise_figure_to_temperature',
    'temperature_to_noise_figure',
    'noise_factor_to_temperature',
    'temperature_to_noise_factor',
    
    # Path loss functions
    'spreading_loss',
    'aperture_loss',
    'free_space_path_loss',
    'polarization_loss',
]
