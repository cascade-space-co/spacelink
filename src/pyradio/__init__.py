"""
PyRadio - A Python library for radio frequency calculations.

This package provides utilities for radio frequency calculations, including:
- Satellite dish modeling and gain calculations
- RF conversions between dB and linear scales
"""

from .conversions import db, db2linear
from .dish import Dish

__all__ = ['db', 'db2linear', 'Dish'] 