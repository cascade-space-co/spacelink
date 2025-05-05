---
description: Guidelines for mathematical calculations in the codebase
globs: src/spacelink/**/*.py
alwaysApply: true
---
# Mathematical Calculations Guidelines

## Use Existing Library Functions

**CRITICAL RULE**: Always use existing library functions for mathematical calculations rather than implementing inline math formulas.

- **DO NOT** write inline mathematical formulas like `10 * np.log10(x)` or similar calculations
- **ALWAYS** check for and use existing functions in the codebase like:
  - `to_dB()` for converting linear values to decibels
  - `to_linear()` for converting decibels to linear values
  - `wavelength()` for frequency to wavelength conversion
  - Other utility functions in the core modules

## Mathematical Documentation

When documenting mathematical formulas in docstrings:

1. Always use proper LaTeX formatting for mathematical equations
2. Add warning labels to LLM-generated equations indicating they need human verification
3. Include purple 'Human Verified' blocks for equations that have been verified by humans

## Example

**INCORRECT**:
```python
# Don't do this!
def calculate_gain_in_db(linear_gain):
    return 10 * np.log10(linear_gain)
```

**CORRECT**:
```python
# Do this instead
def calculate_gain_in_db(linear_gain: Dimensionless) -> Decibels:
    return to_dB(linear_gain)
```
