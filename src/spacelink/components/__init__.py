"""
Spacelink Components Submodule.

Contains object-oriented models for RF components and systems.
"""

from .stage import (
    Stage,
    GainBlock,
    Attenuator,
    # Antenna as StageAntenna, # Removed unused import
    TransmitAntenna,
    ReceiveAntenna,
    Path,
)
from .source import Source
from .sink import Sink
from .signal import Signal
from .antenna import (
    Antenna,  # Base Antenna class
    FixedGain,
    Dish,
)
from .demodulator import Demodulator
from .transmitter import Transmitter
from .mode import RangingMode, DataMode

__all__ = [
    # Base classes
    "Stage",
    "Source",
    "Sink",
    "Antenna",  # Base Antenna class
    # Concrete Stage Components
    "GainBlock",
    "Attenuator",
    "TransmitAntenna",
    "ReceiveAntenna",
    "Path",
    # Concrete Antenna Components
    "FixedGain",
    "Dish",
    # End-to-end Components
    "Demodulator",
    "Transmitter",
    # Data Structures
    "Signal",
    "RangingMode",
    "DataMode",
    # StageAntenna? - Revisit if needed
]
