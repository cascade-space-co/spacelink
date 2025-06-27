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
    Power,
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
    noise_power,
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
from .modcod import (
    coding_gain_from_ebno,
    required_ebno_for_psk_ber,
    get_code_rate_from_scheme,
    psk_bandwidth,
    required_c_n0,
    delta_c_n0,
    ErrorRate,
)
from .ranging import (
    pn_sequence_range_ambiguity,
    chip_snr,
    uplink_carrier_to_total_power,
    uplink_ranging_to_total_power,
    uplink_data_to_total_power,
    CommandModulation,
)


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
    "Power",
    "Wavelength",
    # Unit Conversion/Helpers
    "enforce_units",
    "to_dB",
    "to_linear",
    "wavelength",
    "frequency",
    # Noise
    "noise_power",
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
    # Channel Coding
    "coding_gain_from_ebno",
    "required_ebno_for_psk_ber",
    "get_code_rate_from_scheme",
    "psk_bandwidth",
    "required_c_n0",
    "delta_c_n0",
    "ErrorRate",
    "return_loss_to_vswr",
    "vswr_to_return_loss",
    "mismatch_loss",
    # Ranging
    "pn_sequence_range_ambiguity",
    "chip_snr",
    "uplink_carrier_to_total_power",
    "uplink_ranging_to_total_power",
    "uplink_data_to_total_power",
    "CommandModulation",
]
