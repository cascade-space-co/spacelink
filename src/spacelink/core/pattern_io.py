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

    default_polarization = None
    if data["has_default_polarization"]:
        default_polarization = antenna.Polarization(
            tilt_angle=data["default_pol_tilt_angle"] * u.rad,
            axial_ratio=data["default_pol_axial_ratio"] * u.dimensionless,
            handedness=antenna.Handedness[str(data["default_pol_handedness"])],
        )

    return antenna.RadiationPattern(
        theta=data["theta"] * u.rad,
        phi=data["phi"] * u.rad,
        e_theta=data["e_theta"] * u.dimensionless,
        e_phi=data["e_phi"] * u.dimensionless,
        rad_efficiency=data["rad_efficiency"] * u.dimensionless,
        default_polarization=default_polarization,
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
    data_dict = {
        "theta": pattern.theta.to(u.rad).value,
        "phi": pattern.phi.to(u.rad).value,
        "e_theta": pattern.e_theta.value,
        "e_phi": pattern.e_phi.value,
        "rad_efficiency": pattern.rad_efficiency.value,
        "has_default_polarization": pattern.default_polarization is not None,
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

    np.savez_compressed(
        destination,
        allow_pickle=False,
        **data_dict,
    )


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
    df = df.sort_values([freq_col, theta_col, phi_col])

    # Axes
    theta = np.sort(df[theta_col].unique()) * u.deg
    phi = np.sort(df[phi_col].unique()) * u.deg
    frequencies = (np.sort(df[freq_col].unique()) * u.GHz).to(u.Hz)

    # HFSS exports often have phi = 0 and 360 degrees which means the last phi
    # value is redundant with the first. In that case we drop the redundant phi
    # values.
    if np.isclose(phi[-1] - phi[0], 360 * u.deg):
        phi = phi[:-1]

    # Drop redundant phi rows before pivoting
    df = df[df[phi_col].isin(phi.to_value(u.deg))]

    n_theta = theta.size
    n_phi = phi.size
    n_freq = frequencies.size

    # Single pivot across all value columns, then reindex once
    index_target = pd.MultiIndex.from_product(
        [theta.to_value(u.deg), phi.to_value(u.deg)], names=[theta_col, phi_col]
    )
    value_cols = [gain_lhcp_col, gain_rhcp_col, phase_lhcp_col, phase_rhcp_col]
    columns_target = pd.MultiIndex.from_product(
        [value_cols, frequencies.to_value(u.GHz)]
    )

    df_pivoted = pd.pivot_table(
        df,
        index=[theta_col, phi_col],
        columns=freq_col,
        values=value_cols,
        aggfunc="first",
    )
    df_pivoted = df_pivoted.reindex(index=index_target, columns=columns_target)

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

    return antenna.RadiationPattern.from_circular_gain(
        theta,
        phi,
        frequencies,
        gain_lhcp,
        gain_rhcp,
        angle_lhcp,
        angle_rhcp,
        rad_efficiency,
    )
