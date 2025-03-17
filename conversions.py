"""
Functions for converting between decibels (dB) and linear scale values.

This module provides utility functions for converting values between decibel
and linear scales, commonly used in radio, acoustics, and signal processing.
"""

import math
from typing import Union

def db2linear(db_value: Union[float, int]) -> float:
    """
    Convert a value from decibels (dB) to linear scale.
    
    Args:
        db_value: The value in decibels to convert
        
    Returns:
        float: The equivalent value in linear scale
        
    Examples:
        >>> db2linear(0)
        1.0
        >>> round(db2linear(3), 4)
        1.9953
        >>> round(db2linear(-6), 4)
        0.5012
    """
    return 10 ** (db_value / 10)

def db(linear_value: Union[float, int]) -> float:
    """
    Convert a value from linear scale to decibels (dB).
    
    Args:
        linear_value: The linear value to convert (must be positive)
        
    Returns:
        float: The equivalent value in decibels
        
    Raises:
        ValueError: If linear_value is less than or equal to 0
        
    Examples:
        >>> db(1)
        0.0
        >>> round(db(2), 4)
        3.0103
        >>> round(db(0.5), 4)
        -3.0103
    """
    if linear_value <= 0:
        raise ValueError("Linear value must be positive")
    return 10 * math.log10(linear_value) 