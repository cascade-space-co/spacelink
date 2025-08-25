import datetime
import importlib.metadata
import json
import pathlib
import typing

import astropy.units as u
import numpy as np
import pandas as pd

from . import antenna as antenna
from . import units as units

# Serialization format identifiers (kept stable across code refactors)
_FORMAT_NAME = "spacelink.RadiationPattern"
_FORMAT_VERSION = 1
_CONVENTIONS = (
    "theta[rad], phi[rad], frequency[Hz], e_* dimensionless; handedness=enum.name"
)

# Resolve the package version without importing the top-level package
_SPACELINK_VERSION = importlib.metadata.version("spacelink")


def load_radiation_pattern_npz(
    source: pathlib.Path | typing.BinaryIO,
) -> antenna.RadiationPattern:
    """
    Load a radiation pattern from a NumPy NPZ file or file-like object.

    Parameters
    ----------
    source : pathlib.Path or file-like object
        Path to the NPZ file containing the radiation pattern data, or a file-like
        object (such as BytesIO) containing NPZ data. This allows loading from
        files, databases, or in-memory buffers.

    Returns
    -------
    RadiationPattern
        A new RadiationPattern object reconstructed from the saved data.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist (when ``source`` is a path).
    KeyError
        If required keys are missing from the NPZ file.
    ValueError
        If the file's ``format_name`` or ``format_version`` are unsupported.
    """
    data = np.load(source)

    # Validate format identity and version. These keys are required.
    format_name = str(np.asarray(data["format_name"]).item())
    if format_name != _FORMAT_NAME:
        raise ValueError(
            f"Unsupported format_name '{format_name}'; expected '{_FORMAT_NAME}'"
        )
    format_version = int(np.asarray(data["format_version"]))
    if format_version != _FORMAT_VERSION:
        raise ValueError(
            f"Unsupported format_version {format_version}; expected {_FORMAT_VERSION}"
        )

    default_polarization = None
    if data.get("has_default_polarization", False):
        default_polarization = antenna.Polarization(
            tilt_angle=data["default_pol_tilt_angle"] * u.rad,
            axial_ratio=data["default_pol_axial_ratio"] * u.dimensionless,
            handedness=antenna.Handedness[str(data["default_pol_handedness"])],
        )

    frequency = None
    if data.get("has_frequency", False):
        frequency = data["frequency"] * u.Hz

    default_frequency = None
    if data.get("has_default_frequency", False):
        default_frequency = data["default_frequency"] * u.Hz

    return antenna.RadiationPattern(
        theta=data["theta"] * u.rad,
        phi=data["phi"] * u.rad,
        frequency=frequency,
        e_theta=data["e_theta"] * u.dimensionless,
        e_phi=data["e_phi"] * u.dimensionless,
        rad_efficiency=data["rad_efficiency"] * u.dimensionless,
        default_polarization=default_polarization,
        default_frequency=default_frequency,
    )


def save_radiation_pattern_npz(
    pattern: antenna.RadiationPattern, destination: pathlib.Path | typing.BinaryIO
) -> None:
    """
    Save the radiation pattern data to a NumPy NPZ file or file-like object.

    Parameters
    ----------
    pattern : RadiationPattern
        The radiation pattern to save.
    destination : pathlib.Path or file-like object
        Path to the output NPZ file, or a file-like object (such as BytesIO)
        to write NPZ data to. This allows saving to files, databases, or
        in-memory buffers.
    """
    theta_vals = pattern.theta.to(u.rad).value
    phi_vals = pattern.phi.to(u.rad).value
    e_theta_vals = pattern.e_theta.value
    e_phi_vals = pattern.e_phi.value
    rad_eff_vals = pattern.rad_efficiency.value

    # Optional frequency
    freq_vals = None
    if pattern.frequency is not None:
        freq_vals = pattern.frequency.to(u.Hz).value

    # Human/provenance metadata that requires no pickling
    created_ts = datetime.datetime.now(datetime.UTC).isoformat()
    producer_str = f"spacelink {_SPACELINK_VERSION}"

    # Summarize dtypes and shapes for quick inspection
    dtype_summary: dict[str, typing.Any] = {
        "theta": {
            "dtype": np.asarray(theta_vals).dtype.name,
            "shape": list(np.asarray(theta_vals).shape),
        },
        "phi": {
            "dtype": np.asarray(phi_vals).dtype.name,
            "shape": list(np.asarray(phi_vals).shape),
        },
        "e_theta": {
            "dtype": np.asarray(e_theta_vals).dtype.name,
            "shape": list(np.asarray(e_theta_vals).shape),
        },
        "e_phi": {
            "dtype": np.asarray(e_phi_vals).dtype.name,
            "shape": list(np.asarray(e_phi_vals).shape),
        },
        "rad_efficiency": {
            "dtype": np.asarray(rad_eff_vals).dtype.name,
            "shape": list(np.asarray(rad_eff_vals).shape),
        },
    }
    if freq_vals is not None:
        dtype_summary["frequency"] = {
            "dtype": np.asarray(freq_vals).dtype.name,
            "shape": list(np.asarray(freq_vals).shape),
        }

    data_dict = {
        # Data arrays
        "theta": theta_vals,
        "phi": phi_vals,
        "e_theta": e_theta_vals,
        "e_phi": e_phi_vals,
        "rad_efficiency": rad_eff_vals,
        # Presence flags
        "has_default_polarization": pattern.default_polarization is not None,
        "has_frequency": pattern.frequency is not None,
        "has_default_frequency": pattern.default_frequency is not None,
        # Format identity and provenance
        "format_name": _FORMAT_NAME,
        "format_version": _FORMAT_VERSION,
        "conventions": _CONVENTIONS,
        "producer": producer_str,
        "created_utc": created_ts,
        "dtype_info": json.dumps(dtype_summary),
    }

    if pattern.default_polarization is not None:
        pol = pattern.default_polarization
        data_dict.update(
            {
                "default_pol_tilt_angle": pol.tilt_angle.to(u.rad).value,
                "default_pol_axial_ratio": pol.axial_ratio.value,
                "default_pol_handedness": pol.handedness.name,
            }
        )

    if freq_vals is not None:
        data_dict["frequency"] = freq_vals

    if pattern.default_frequency is not None:
        data_dict["default_frequency"] = pattern.default_frequency.to(u.Hz).value

    np.savez_compressed(
        destination,
        allow_pickle=False,
        **data_dict,
    )


@units.enforce_units
def import_hfss_csv(
    hfss_csv_path: pathlib.Path,
    *,
    rad_efficiency: units.Dimensionless,
) -> antenna.RadiationPattern:
    r"""
    Create a radiation pattern from an HFSS exported CSV file.

    This expects the CSV file to contain the following columns in any order:
    - Freq [GHz]
    - Theta [deg]
    - Phi [deg]
    - dB(RealizedGainLHCP) []
    - dB(RealizedGainRHCP) []
    - ang_deg(rELHCP) [deg]
    - ang_deg(rERHCP) [deg]

    Any other columns will be ignored. There must be exactly one header row with the
    column names.

    The Theta and Phi values must form a regular grid.

    Parameters
    ----------
    hfss_csv_path: pathlib.Path
        Path to the HFSS CSV file.
    rad_efficiency: Dimensionless
        Radiation efficiency :math:`\eta` in (0, 1].

    Returns
    -------
    RadiationPattern
        Radiation pattern constructed from the CSV. If the CSV contains only a single
        frequency the pattern will be created without a frequency axis (it will be
        frequency-invariant).

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    KeyError
        If required columns are missing from the CSV.
    ValueError
        If duplicate (Freq, Theta, Phi) grid rows are present or if the CSV does not
        form a complete regular grid (NaNs detected after pivoting and reindexing).
    """
    # Define column name constants
    freq_col = "Freq [GHz]"
    theta_col = "Theta [deg]"
    phi_col = "Phi [deg]"
    gain_lhcp_col = "dB(RealizedGainLHCP) []"
    gain_rhcp_col = "dB(RealizedGainRHCP) []"
    phase_lhcp_col = "ang_deg(rELHCP) [deg]"
    phase_rhcp_col = "ang_deg(rERHCP) [deg]"

    df = pd.read_csv(hfss_csv_path)
    df = df.sort_values([freq_col, theta_col, phi_col])

    # Validate that there are no duplicate grid rows for the same
    # (frequency, theta, phi) coordinate triple. Duplicates would be silently
    # collapsed by the pivot operation; fail fast instead.
    if df.duplicated(subset=[freq_col, theta_col, phi_col]).any():
        raise ValueError(
            "Duplicate rows detected for the same (Freq, Theta, Phi) grid point"
        )

    # Axes
    theta = np.sort(df[theta_col].unique()) * u.deg
    phi = np.sort(df[phi_col].unique()) * u.deg
    # Save original frequency values to avoid precision issues when matching pivot table
    # columns later
    freq_values_original = np.sort(df[freq_col].unique())
    frequencies = (freq_values_original * u.GHz).to(u.Hz)

    n_theta = theta.size
    n_phi = phi.size
    n_freq = frequencies.size

    # Single pivot across all value columns, then reindex once
    index_target = pd.MultiIndex.from_product(
        [theta.to_value(u.deg), phi.to_value(u.deg)], names=[theta_col, phi_col]
    )
    value_cols = [gain_lhcp_col, gain_rhcp_col, phase_lhcp_col, phase_rhcp_col]
    # Use original frequency values to match pivot table columns exactly
    columns_target = pd.MultiIndex.from_product([value_cols, freq_values_original])

    df_pivoted = pd.pivot_table(
        df,
        index=[theta_col, phi_col],
        columns=freq_col,
        values=value_cols,
        aggfunc="first",
    )
    df_pivoted = df_pivoted.reindex(index=index_target, columns=columns_target)

    # After reindexing, any NaNs indicate that the CSV did not form a complete
    # regular grid across theta, phi, and frequency. Fail to avoid propagating
    # invalid values into the radiation pattern tensors.
    if df_pivoted.isna().any().any():
        raise ValueError(
            "CSV does not form a complete regular grid; missing theta/phi/frequency "
            "samples detected"
        )

    # Reshape to (n_theta, n_phi, n_freq)
    gain_lhcp = units.to_linear(
        df_pivoted[gain_lhcp_col].to_numpy().reshape(n_theta, n_phi, n_freq) * u.dB
    )
    gain_rhcp = units.to_linear(
        df_pivoted[gain_rhcp_col].to_numpy().reshape(n_theta, n_phi, n_freq) * u.dB
    )
    angle_lhcp = (
        df_pivoted[phase_lhcp_col].to_numpy().reshape(n_theta, n_phi, n_freq) * u.deg
    )
    angle_rhcp = (
        df_pivoted[phase_rhcp_col].to_numpy().reshape(n_theta, n_phi, n_freq) * u.deg
    )

    # If CSV contains a single frequency, emit a frequency-invariant pattern
    if frequencies.size == 1:
        return antenna.RadiationPattern.from_circular_gain(
            theta=theta,
            phi=phi,
            frequency=None,
            gain_lhcp=gain_lhcp[..., 0],
            gain_rhcp=gain_rhcp[..., 0],
            phase_lhcp=angle_lhcp[..., 0],
            phase_rhcp=angle_rhcp[..., 0],
            rad_efficiency=rad_efficiency,
        )

    # Otherwise, return full 3D frequency-aware pattern
    return antenna.RadiationPattern.from_circular_gain(
        theta=theta,
        phi=phi,
        frequency=frequencies,
        gain_lhcp=gain_lhcp,
        gain_rhcp=gain_rhcp,
        phase_lhcp=angle_lhcp,
        phase_rhcp=angle_rhcp,
        rad_efficiency=rad_efficiency,
    )
