"""
Spacelink Core Submodule.

Contains fundamental units, constants, validation logic,
and calculation functions.
"""

from .units import (
    Frequency,
    Decibels,
    Temperature,
    Dimensionless,
    Length,
    Distance,
    DecibelWatts,
    DecibelMilliwatts,
    Watts,
    Wavelength,
    enforce_units,
    to_dB,
    to_linear,
    wavelength,
    frequency,
    return_loss_to_vswr,
    vswr_to_return_loss,
    mismatch_loss,
)
from .noise import (
    noise_figure_to_temperature,
    temperature_to_noise_figure,
    noise_factor_to_temperature,
    temperature_to_noise_factor,
)
from .path import (
    spreading_loss,
    aperture_loss,
    free_space_path_loss,
)
from .antenna import (
    polarization_loss,
    dish_gain,
)
# Add back import from validation if non_negative exists
# from .validation import (
#     non_negative,
# )

__all__ = [
    # Units & Types
    "Frequency",
    "Decibels",
    "Temperature",
    "Dimensionless",
    "Length",
    "Distance",
    "DecibelWatts",
    "DecibelMilliwatts",
    "Watts",
    "Wavelength",
    # Unit Conversion/Helpers
    "enforce_units",
    "to_dB",
    "to_linear",
    "wavelength",
    "frequency",
    # Noise
    "noise_figure_to_temperature",
    "temperature_to_noise_figure",
    "noise_factor_to_temperature",
    "temperature_to_noise_factor",
    # Path Loss
    "spreading_loss",
    "aperture_loss",
    "free_space_path_loss",
    # Antenna Calcs
    "polarization_loss",
    "dish_gain",
    # Validation
    # 'non_negative', # Keep commented if not defined in validation.py
    # VSWR/Return Loss
    "return_loss_to_vswr",
    "vswr_to_return_loss",
    "mismatch_loss",
]
