"""
Validation decorators.

This module provides reusable decorators for validating function arguments,
primarily physical quantities.
"""

from functools import wraps
import astropy.units as u # Keep u for Quantity check
from astropy.units import Quantity
from typing import Callable, TypeVar, Any

T = TypeVar("T")
