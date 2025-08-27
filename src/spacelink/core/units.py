r"""
Wavelength
----------

The relationship between wavelength and frequency is given by:

.. math::
   \lambda = \frac{c}{f}

where:

* :math:`c` is the speed of light (299,792,458 m/s)
* :math:`f` is the frequency in Hz


to_dB
-----

The conversion to decibels is done using:

.. math::
   \text{X}_{\text{dB}} = \text{factor} \cdot \log_{10}(x)

The result will have units of dB(input_unit), e.g. dBW, dBK, dBHz, etc.
For dimensionless input, the result will have unit dB.

to_linear
---------

The conversion from decibels to a linear (dimensionless) ratio is done using:

.. math::
   x = 10^{\frac{\text{X}_{\text{dB}}}{\text{factor}}}

where:

* :math:`\text{X}_{\text{dB}}` is the value in decibels
* :math:`\text{factor}` is 10 for power quantities, 20 for field quantities

Return Loss to VSWR
-------------------

The conversion from return loss in decibels to voltage standing wave ratio (VSWR) is
done using:

.. math::
   \text{VSWR} = \frac{1 + |\Gamma|}{1 - |\Gamma|}

where:

* :math:`|\Gamma|` is the magnitude of the reflection coefficient
* :math:`|\Gamma| = 10^{-\frac{\text{RL}}{20}}`
* :math:`\text{RL}` is the return loss in dB

VSWR to Return Loss
-------------------

The conversion from voltage standing wave ratio (VSWR) to return loss in decibels is
done using:

.. math::
   \text{RL} = -20 \log_{10}\left(\frac{\text{VSWR} - 1}{\text{VSWR} + 1}\right)

where:

* :math:`\text{VSWR}` is the voltage standing wave ratio
* :math:`\text{RL}` is the return loss in dB
"""

import types
from functools import wraps
from inspect import signature
from typing import Annotated, Any, Union, get_args, get_origin, get_type_hints

import astropy.constants as constants
import astropy.units as u
import numpy as np
from astropy.units import Quantity

if not hasattr(u, "dBHz"):  # pragma: no cover
    u.dBHz = u.dB(u.Hz)
if not hasattr(u, "dBW"):  # pragma: no cover
    u.dBW = u.dB(u.W)
if not hasattr(u, "dBm"):  # pragma: no cover
    u.dBm = u.dB(u.mW)
if not hasattr(u, "dBK"):  # pragma: no cover
    u.dBK = u.dB(u.K)
if not hasattr(u, "dB_per_K"):  # pragma: no cover
    u.dB_per_K = u.dB(1 / u.K)

if not hasattr(u, "dimensionless"):  # pragma: no cover
    u.dimensionless = u.dimensionless_unscaled

Decibels = Annotated[Quantity, u.dB]
DecibelWatts = Annotated[Quantity, u.dB(u.W)]
DecibelMilliwatts = Annotated[Quantity, u.dB(u.mW)]
DecibelKelvins = Annotated[Quantity, u.dB(u.K)]
DecibelPerKelvin = Annotated[Quantity, u.dB(1 / u.K)]
Power = Annotated[Quantity, u.W]
PowerDensity = Annotated[Quantity, u.W / u.Hz]
Frequency = Annotated[Quantity, u.Hz]
Wavelength = Annotated[Quantity, u.m]
Dimensionless = Annotated[Quantity, u.dimensionless_unscaled]
Distance = Annotated[Quantity, u.m]
Temperature = Annotated[Quantity, u.K]
Length = Annotated[Quantity, u.m]
DecibelHertz = Annotated[Quantity, u.dB(u.Hz)]
Angle = Annotated[Quantity, u.rad]
SolidAngle = Annotated[Quantity, u.sr]
Time = Annotated[Quantity, u.s]


def _extract_annotated_from_hint(hint: Any) -> tuple[type, u.Unit] | None:
    """
    Extract Annotated type and unit from a type hint, handling optional parameters.

    Parameters
    ----------
    hint : Any
        Type hint that may be Annotated directly or a Union containing Annotated

    Returns
    -------
    tuple[type, u.Unit] | None
        (quantity_type, unit) if Annotated type found, None otherwise
    """
    if hint is None:
        return None

    # Check if hint is directly Annotated
    if get_origin(hint) is Annotated:
        args = get_args(hint)
        if len(args) >= 2:
            return args[0], args[1]

    # Check if hint is a Union (including PEP 604 X | Y syntax)
    origin = get_origin(hint)
    if origin is Union or (hasattr(types, "UnionType") and origin is types.UnionType):
        # Look through union arguments for Annotated types
        for arg in get_args(hint):
            if get_origin(arg) is Annotated:
                annotated_args = get_args(arg)
                if len(annotated_args) >= 2:
                    return annotated_args[0], annotated_args[1]

    return None


def enforce_units(func):
    """
    Decorator to enforce the units specified in function parameter type annotations.

    This decorator enforces some unit consistency rules for function parameters that
    annotated with one of the ``Annotated`` types in this module:

    * The argument must be a ``Quantity`` object.
    * The argument must be provided with a compatible unit. For example, a ``Frequency``
      argument's units can be ``u.Hz``, ``u.MHz``, ``u.GHz``, etc. but not ``u.m``,
      ``u.K``, or any other non-frequency unit.

    In addition to the above, the value of any ``Annotated`` argument will be converted
    automatically to the unit specified in for that type. For example, the ``Angle``
    type will be converted to ``u.rad``, even if the argument is provided with a unit of
    ``u.deg``. This allows functions to flexibly handle compatible units while keeping
    tedious unit conversion logic out of the function body.

    Parameters
    ----------
    func : callable
        The function to wrap.

    Returns
    -------
    callable
        The wrapped function with unit enforcement.

    Raises
    ------
    UnitConversionError
        If any argument has incompatible units.
    TypeError
        If an ``Annotated`` argument is not an Astropy ``Quantity`` object.
    """
    sig = signature(func)
    hints = get_type_hints(func, include_extras=True)

    @wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        for name, value in bound.arguments.items():
            hint = hints.get(name)
            annotated_info = _extract_annotated_from_hint(hint)

            if annotated_info is not None:
                _, unit = annotated_info

                # Handle None values for optional parameters
                if value is None:
                    continue

                if isinstance(value, Quantity):
                    # Convert to expected unit
                    try:
                        if unit.is_equivalent(u.K):
                            converted_value = value.to(
                                unit, equivalencies=u.temperature()
                            )
                        else:
                            converted_value = value.to(unit)
                    except u.UnitConversionError as e:
                        raise u.UnitConversionError(
                            f"Parameter '{name}' requires unit compatible with {unit}, "
                            f"but got {value.unit}. Original error: {e}"
                        ) from e

                    # Unit conversion successful
                    bound.arguments[name] = converted_value

                else:
                    # Handle non-Quantity inputs
                    raise TypeError(
                        f"Parameter '{name}' must be provided as an astropy Quantity, "
                        f"not a raw number."
                    )
        return func(*bound.args, **bound.kwargs)

    return wrapper


@enforce_units
def wavelength(frequency: Frequency) -> Wavelength:
    r"""
    Convert frequency to wavelength.

    Parameters
    ----------
    frequency : Quantity
        Frequency quantity (e.g., in Hz)

    Returns
    -------
    Quantity
        Wavelength in meters

    Raises
    ------
    UnitConversionError
        If the input quantity has incompatible units
    """
    return constants.c / frequency.to(u.Hz)


@enforce_units
def frequency(wavelength: Wavelength) -> Frequency:
    r"""
    Convert wavelength to frequency.

    Parameters
    ----------
    wavelength : Quantity
        Wavelength quantity (e.g., in meters)

    Returns
    -------
    Quantity
        Frequency in hertz

    Raises
    ------
    UnitConversionError
        If the input quantity has incompatible units
    """
    return constants.c / wavelength.to(u.m)


@enforce_units
def to_dB(x: Dimensionless, *, factor: float = 10.0) -> Decibels:
    r"""
    Convert dimensionless quantity to decibels.

    Note that for referenced logarithmic units, conversions should
    be done using the .to(unit) method.
    Parameters
    ----------
    x : Dimensionless
        value to be converted
    factor : float, optional
        10 for power quantities, 20 for field quantities

    Returns
    -------
    Decibels
    """
    with np.errstate(divide="ignore"):  # Suppress warnings for np.log10(0)
        return factor * np.log10(x.value) * u.dB


@enforce_units
def to_linear(x: Decibels, *, factor: float = 10.0) -> Dimensionless:
    """
    Convert decibels to a linear (dimensionless) ratio.

    Parameters
    ----------
    x : Decibels
        A quantity in decibels
    factor : float, optional
        10 for power quantities, 20 for field quantities

    Returns
    -------
    Dimensionless
    """
    linear_value = np.power(10, x.value / factor)
    return linear_value * u.dimensionless


@enforce_units
def return_loss_to_vswr(return_loss: Decibels) -> Dimensionless:
    r"""
    Convert a return loss in decibels to voltage standing wave ratio (VSWR).

    Parameters
    ----------
    return_loss : Quantity
        Return loss in decibels (>= 0). Use float('inf') for a perfect match

    Returns
    -------
    Dimensionless
        VSWR (>= 1)

    Raises
    ------
    ValueError
        If return_loss is negative
    """
    if return_loss.value < 0:
        raise ValueError(f"return loss must be >= 0 ({return_loss}).")
    if return_loss.value == float("inf"):
        return 1.0 * u.dimensionless
    gamma = to_linear(-return_loss, factor=20)
    return ((1 + gamma) / (1 - gamma)) * u.dimensionless


@enforce_units
def vswr_to_return_loss(vswr: Dimensionless) -> Decibels:
    r"""
    Convert voltage standing wave ratio (VSWR) to return loss in decibels.

    Parameters
    ----------
    vswr : Quantity
        VSWR value (> 1). Use 1 for a perfect match (infinite return loss)

    Returns
    -------
    Quantity
        Return loss in decibels

    Raises
    ------
    ValueError
        If vswr is less than 1
    """
    if vswr < 1.0:
        raise ValueError(f"VSWR must be >= 1 ({vswr}).")
    gamma = (vswr - 1) / (vswr + 1)
    return safe_negate(to_dB(gamma, factor=20))


def safe_negate(quantity: Quantity) -> Quantity:
    """
    Safely negate a dB or function unit quantity, preserving the unit.
    Astropy does not allow direct negation of function units (like dB).
    """
    return (-1 * quantity.value) * quantity.unit
