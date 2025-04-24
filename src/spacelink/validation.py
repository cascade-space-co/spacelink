"""
Validation decorators for RF components.

This module provides reusable decorators for validating parameters in RF components,
such as bounds checking for physical quantities.
"""

from functools import wraps
import astropy.units as u
from typing import Callable, TypeVar, Any, Optional

T = TypeVar('T')

def non_negative(unit=None):
    """
    Decorator to enforce non-negative values.

    Args:
        unit: Optional unit to include in error message

    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(self, value, *args, **kwargs):
            if value < 0:
                param_name = func.__name__.replace("_", " ")
                if unit:
                    msg = f"{param_name} must be non-negative"
                else:
                    msg = f"{param_name} must be non-negative"
                raise ValueError(msg)
            return func(self, value, *args, **kwargs)
        return wrapper
    return decorator

def positive(unit: u.Unit) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to ensure a parameter is positive.

    Args:
        unit: The unit to check against (e.g., u.Hz for frequency)

    Returns:
        Decorator function that validates the parameter is positive

    Example:
        @positive(u.Hz)
        def set_frequency(self, freq: u.Quantity) -> None:
            self.freq = freq
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Get the first argument (self) and the parameter name from the function
            self = args[0]
            param_name = func.__code__.co_varnames[1]  # Skip 'self'
            
            # Get the value from kwargs or args
            value = kwargs.get(param_name, args[1] if len(args) > 1 else None)
            
            if value is not None and value <= 0 * unit:
                raise ValueError(f"{param_name} must be positive")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def in_range(unit: u.Unit, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to ensure a parameter is within a specified range.

    Args:
        unit: The unit to check against
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)

    Returns:
        Decorator function that validates the parameter is within range

    Example:
        @in_range(u.dimensionless, 0, 1)
        def set_efficiency(self, eff: u.Quantity) -> None:
            self.eff = eff
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Get the first argument (self) and the parameter name from the function
            self = args[0]
            param_name = func.__code__.co_varnames[1]  # Skip 'self'
            
            # Get the value from kwargs or args
            value = kwargs.get(param_name, args[1] if len(args) > 1 else None)
            
            if value is not None:
                if min_val is not None and value < min_val * unit:
                    raise ValueError(f"{param_name} must be >= {min_val} {unit}")
                if max_val is not None and value > max_val * unit:
                    raise ValueError(f"{param_name} must be <= {max_val} {unit}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator 