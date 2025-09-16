r"""
Wavelength
----------

The relationship between wavelength and frequency is given by:

.. math::
   \lambda = \frac{c}{f}

where:

* :math:`c` is the speed of light (299,792,458 m/s)
* :math:`f` is the frequency in Hz

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

import dataclasses
import types
from collections.abc import Callable
from functools import wraps
from inspect import signature
from typing import (
    Annotated,
    Any,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import astropy.constants as constants
import astropy.units as u
import numpy as np
from astropy.units import Quantity

# Type variable for enforce_units decorator - accepts only functions or classes
FuncOrClass = TypeVar("FuncOrClass", Callable[..., Any], type)


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

# Using u.dB(1) allows conversion from decibels to u.dimensionless_unscaled. The (1)
# informs Astropy that the value is decibels relative to 1; without it a bare u.dB has
# no defined reference point.
Decibels = Annotated[Quantity, u.dB(1)]
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

# Module-level flag to enable return unit checking (for tests)
_RETURN_UNITS_CHECK_ENABLED = False


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
    if hint is None:  # pragma: no cover
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


def _extract_tuple_annotations(
    hint: Any,
) -> list[tuple[tuple[type, u.Unit] | None, Any]] | None:
    """
    Extract annotations from tuple type hints.

    Parameters
    ----------
    hint : Any
        Type hint that may be a tuple containing Annotated types

    Returns
    -------
    list[tuple[tuple[type, u.Unit] | None, Any]] | None
        List of ((quantity_type, unit), original_hint) for each tuple element,
        where the first element is None if not annotated. Returns None if hint is not a
        tuple.
    """
    origin = get_origin(hint)
    if origin is tuple:
        args = get_args(hint)
        annotations = []
        for arg in args:
            annotated_info = _extract_annotated_from_hint(arg)
            annotations.append((annotated_info, arg))
        return annotations
    return None


def _validate_tuple_return(result, expected_annotations):
    """
    Validate tuple return values against their type annotations.

    Parameters
    ----------
    result : Any
        The actual return value (should be a tuple)
    expected_annotations : list[tuple[tuple[type, u.Unit] | None, Any]]
        List of ((quantity_type, unit), original_hint) for each tuple element
    """
    if not isinstance(result, tuple):
        raise TypeError("Expected tuple return value.")

    if len(result) != len(expected_annotations):
        raise TypeError(
            f"Expected tuple with {len(expected_annotations)} elements, "
            f"got {len(result)} elements."
        )

    for i, (value, (annotation, original_hint)) in enumerate(
        zip(result, expected_annotations, strict=False)
    ):
        if annotation is not None:  # Only check annotated elements
            _, expected_unit = annotation

            if value is None:
                # Check if None is allowed (Optional type)
                origin = get_origin(original_hint)
                if not (
                    origin in (Union, getattr(types, "UnionType", ()))
                    and type(None) in get_args(original_hint)
                ):
                    raise TypeError(
                        f"tuple[{i}] is None but not annotated as Optional."
                    )
                continue

            if not isinstance(value, Quantity):
                raise TypeError(f"tuple[{i}] must be an astropy Quantity.")

            if value.unit != expected_unit:
                raise u.UnitConversionError(
                    f"tuple[{i}] unit {value.unit} != annotated {expected_unit}."
                )


def _convert_parameter_units(name: str, value: Any, expected_unit: u.Unit) -> Quantity:
    """
    Convert a parameter value to the expected unit.

    Parameters
    ----------
    name : str
        Parameter name for error messages
    value : Any
        Parameter value to convert
    expected_unit : u.Unit
        Expected unit for the parameter

    Returns
    -------
    Quantity
        Converted quantity with the expected unit

    Raises
    ------
    TypeError
        If value is not a Quantity
    UnitConversionError
        If units are incompatible
    """
    if not isinstance(value, Quantity):
        raise TypeError(
            f"Parameter '{name}' must be provided as an astropy Quantity with unit "
            f"compatible with {expected_unit}, not a raw number."
        )

    # Units like deg_C are not automatically convertible to/from K
    if expected_unit.is_equivalent(u.K, equivalencies=u.temperature()):
        equivalencies = u.temperature()
    elif value.unit == u.dB:
        # Allows conversion from u.dB to u.dimensionless_unscaled as if
        # value.unit was u.dB(1)
        equivalencies = u.logarithmic()
    else:
        equivalencies = []

    try:
        return value.to(expected_unit, equivalencies=equivalencies)
    except u.UnitConversionError as e:
        raise u.UnitConversionError(
            f"Parameter '{name}' requires unit compatible with {expected_unit}, "
            f"but got {value.unit}. Original error: {e}"
        ) from e


def _validate_single_return(result: Any, expected_unit: u.Unit, ret_hint: Any) -> None:
    """
    Validate a single return value against its expected unit.

    Parameters
    ----------
    result : Any
        The actual return value
    expected_unit : u.Unit
        Expected unit for the return value
    ret_hint : Any
        Original return type hint for Optional checking

    Raises
    ------
    TypeError
        If result is None when not Optional, or not a Quantity
    UnitConversionError
        If units don't match exactly
    """
    if result is None:
        # Check if None is allowed (Optional type)
        origin = get_origin(ret_hint)
        if not (
            origin in (Union, getattr(types, "UnionType", ()))
            and type(None) in get_args(ret_hint)
        ):
            raise TypeError("Return value is None but not annotated as Optional.")
        return

    if not isinstance(result, Quantity):
        raise TypeError("Return value must be an astropy Quantity.")

    if result.unit != expected_unit:
        raise u.UnitConversionError(
            f"Return unit {result.unit} != annotated {expected_unit}."
        )


def _validate_return_units(result: Any, ret_hint: Any) -> None:
    """
    Validate return value units based on type hint.

    Parameters
    ----------
    result : Any
        The actual return value
    ret_hint : Any
        Return type hint
    """
    if not _RETURN_UNITS_CHECK_ENABLED or ret_hint is None:
        return

    # Try tuple support first
    tuple_annotations = _extract_tuple_annotations(ret_hint)
    if tuple_annotations is not None:
        _validate_tuple_return(result, tuple_annotations)
        return

    # Single annotated quantity
    annotated_info = _extract_annotated_from_hint(ret_hint)
    if annotated_info is not None:
        _, expected_unit = annotated_info
        _validate_single_return(result, expected_unit, ret_hint)


def _process_parameter(name: str, value: Any, hint: Any) -> Quantity | Any:
    """
    Process a single parameter, converting units if needed.

    Parameters
    ----------
    name : str
        Parameter name
    value : Any
        Parameter value
    hint : Any
        Type hint for the parameter

    Returns
    -------
    Quantity | Any
        Converted parameter value, or original value if not annotated
    """
    annotated_info = _extract_annotated_from_hint(hint)
    if annotated_info is None:
        return value  # Not an annotated parameter

    _, expected_unit = annotated_info

    # Handle None values for optional parameters
    if value is None:
        return value

    return _convert_parameter_units(name, value, expected_unit)


def _wrap_function_with_unit_enforcement(
    func: Callable[..., Any],
) -> Callable[..., Any]:
    """
    Internal helper to wrap a function with unit enforcement logic.

    This is the core unit enforcement logic extracted to be reusable
    for both regular functions and dataclass __init__ methods.
    """
    sig = signature(func)
    hints = get_type_hints(func, include_extras=True)

    @wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        # Process all parameters
        for name, value in bound.arguments.items():
            hint = hints.get(name)
            bound.arguments[name] = _process_parameter(name, value, hint)

        # Execute the function
        result = func(*bound.args, **bound.kwargs)

        # Validate return value units
        _validate_return_units(result, hints.get("return"))

        return result

    return wrapper


def enforce_units(func_or_class: FuncOrClass) -> FuncOrClass:
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

    When applied to a dataclass, this decorator will wrap the ``__init__`` method
    to enforce units on dataclass field assignments.

    Parameters
    ----------
    func_or_class : callable or class
        The function or dataclass to wrap.

    Returns
    -------
    callable or class
        The wrapped function or modified dataclass with unit enforcement.

    Raises
    ------
    UnitConversionError
        If any argument has incompatible units.
    TypeError
        If an ``Annotated`` argument is not an Astropy ``Quantity`` object.
    """
    # Check if this is a class
    if isinstance(func_or_class, type):
        if dataclasses.is_dataclass(func_or_class):
            # Handle dataclass case: wrap the __init__ method
            original_init = func_or_class.__init__
            wrapped_init = _wrap_function_with_unit_enforcement(original_init)
            func_or_class.__init__ = wrapped_init

            return func_or_class
        else:
            # Regular class - this is probably a mistake
            raise TypeError(
                f"@enforce_units should not be applied to regular classes. "
                f"Apply it directly to the __init__ method instead:\n\n"
                f"class {func_or_class.__name__}:\n"
                f"    @enforce_units\n"
                f"    def __init__(self, ...):\n"
                f"        ..."
            )
    else:
        # Handle regular function case
        return _wrap_function_with_unit_enforcement(func_or_class)


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
def return_loss_to_vswr(return_loss: Dimensionless) -> Dimensionless:
    r"""
    Convert a return loss in decibels to voltage standing wave ratio (VSWR).

    Parameters
    ----------
    return_loss : Dimensionless
        Return loss. Must be >= 1 if provided as dimensionless or >= 0 if provided in
        decibels. Use np.inf for a perfect match.

    Returns
    -------
    Dimensionless
        VSWR (>= 1)

    Raises
    ------
    ValueError
        If return_loss is < 0 dB
    """
    if np.any(return_loss.value < 1):
        raise ValueError("Return loss must be >= 1.")

    gamma = 1 / np.sqrt(return_loss)
    return (1 + gamma) / (1 - gamma)


@enforce_units
def vswr_to_return_loss(vswr: Dimensionless) -> Decibels:
    r"""
    Convert voltage standing wave ratio (VSWR) to return loss in decibels.

    Parameters
    ----------
    vswr : Quantity
        VSWR value (>= 1). Use 1 for a perfect match (infinite return loss)

    Returns
    -------
    Quantity
        Return loss in decibels

    Raises
    ------
    ValueError
        If vswr is less than 1
    """
    if np.any(vswr < 1.0):
        raise ValueError("VSWR must be >= 1.")
    gamma = (vswr - 1) / (vswr + 1)
    return (1 / np.abs(gamma) ** 2).to(u.dB(1))


def safe_negate(quantity: Quantity) -> Quantity:
    """
    Safely negate a dB or function unit quantity, preserving the unit.
    Astropy does not allow direct negation of function units (like dB).
    """
    return (-1 * quantity.value) * quantity.unit
