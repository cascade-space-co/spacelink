import pathlib
import typing

import astropy.units as u
import numpy as np
import pandas as pd

from . import antenna as antenna
from . import units as units


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
        If the specified file does not exist (when source is a path).
    KeyError
        If required keys are missing from the NPZ file.
    """
    data = np.load(source)

    # Reconstruct arrays with proper units
    theta = data["theta"] * u.rad
    phi = data["phi"] * u.rad
    e_theta = data["e_theta"] * u.dimensionless
    e_phi = data["e_phi"] * u.dimensionless
    rad_efficiency = data["rad_efficiency"] * u.dimensionless

    return antenna.RadiationPattern(
        theta=theta,
        phi=phi,
        e_theta=e_theta,
        e_phi=e_phi,
        rad_efficiency=rad_efficiency,
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
    np.savez_compressed(
        destination,
        theta=pattern.theta.to(u.rad).value,
        phi=pattern.phi.to(u.rad).value,
        e_theta=pattern.e_theta.value,
        e_phi=pattern.e_phi.value,
        rad_efficiency=pattern.rad_efficiency.value,
        allow_pickle=False,
    )


def import_hfss_csv(
    hfss_csv_path: pathlib.Path,
    *,
    carrier_frequency: units.Frequency,
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
    carrier_frequency: Frequency
        Pattern data corresponding to this carrier frequency will be imported from
        the CSV file. Only a single frequency is currently supported.
    rad_efficiency: Dimensionless
        Radiation efficiency :math:`\eta` in [0, 1].

    Returns
    -------
    RadiationPattern
        Radiation pattern constructed from the CSV.
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
    target_freq = carrier_frequency.to(u.GHz).value
    df_one_freq = df[np.isclose(df[freq_col], target_freq)]

    if df_one_freq.empty:
        raise ValueError(f"No data found for frequency {carrier_frequency}")

    df_one_freq = df_one_freq.sort_values([theta_col, phi_col])

    # Extract arrays of unique theta and phi values
    theta = np.sort(df_one_freq[theta_col].unique()) * u.deg
    phi = np.sort(df_one_freq[phi_col].unique()) * u.deg

    # Create 2D arrays with shape (N_theta, N_phi)
    gain_lhcp = units.to_linear(
        df_one_freq.pivot(
            index=theta_col,
            columns=phi_col,
            values=gain_lhcp_col,
        ).values
        * u.dB
    )
    gain_rhcp = units.to_linear(
        df_one_freq.pivot(
            index=theta_col,
            columns=phi_col,
            values=gain_rhcp_col,
        ).values
        * u.dB
    )
    angle_lhcp = (
        df_one_freq.pivot(
            index=theta_col, columns=phi_col, values=phase_lhcp_col
        ).values
        * u.deg
    )
    angle_rhcp = (
        df_one_freq.pivot(
            index=theta_col, columns=phi_col, values=phase_rhcp_col
        ).values
        * u.deg
    )

    # HFSS exports often have phi = 0 and 360 degrees which means the last phi
    # value is redundant with the first. In that case we drop the redundant phi
    # values.
    if np.isclose(phi[-1] - phi[0], 360 * u.deg):
        phi_last_idx: typing.Optional[int] = -1
    else:
        phi_last_idx = None

    return antenna.RadiationPattern.from_circular_gain(
        theta,
        phi[:phi_last_idx],
        gain_lhcp[:, :phi_last_idx],
        gain_rhcp[:, :phi_last_idx],
        angle_lhcp[:, :phi_last_idx],
        angle_rhcp[:, :phi_last_idx],
        rad_efficiency=rad_efficiency,
    )
