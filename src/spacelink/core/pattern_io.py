import pathlib
import typing

import astropy.units as u
import numpy as np

from . import antenna as antenna


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
