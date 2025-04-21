# SpaceLink Development Guide
This guide is primarily for prompting LLM coding agents, however it also
serves as a reference for human editors

## Overview

SpaceLink is a Python module for space mission planning and spacecraft design, specializing in:
- Link budget modeling
- Receiver/transmitter lineup analysis
- Cascaded noise figure analysis
- Gain/saturation analysis
- Isolation analysis

## RF Engineering Guidelines

### Core Concepts
- RF (Radio Frequency) calculations follow specific conventions
- All gains/losses use decibels (dB, dB, db are equivalent)
- Use fundamental units (meters, Hz, dBW) for computations
- Total gain = sum of all gains minus losses (gains can be negative, losses must be positive)

### Unit Handling
1. Always use conversion functions for unit transformations
2. Never perform inline unit conversions (e.g., avoid `meters = 1000 * kilometers`)
3. Use pint for all non-dB units
4. Keep variables in fundamental units
5. Express numerical values in human-readable engineering units
6. Represent dB using pint units for all inputs
6. Represent dB values as floats for function return values

## Development Workflow

### Testing
```bash
poetry run pytest -v                                                # Run all tests
poetry run pytest tests/test_file.py::test_name -v                 # Run specific test
poetry run pytest --cov-report term-missing                        # Test coverage
```

### Poetry Configuration
Before committing any changes to pyproject.toml or poetry.lock:
```bash
poetry check                     # Validate poetry configuration
poetry lock                     # Update lock file if needed
poetry install                  # Install dependencies
```

### Code Quality
```bash
poetry run black .       # Format code
poetry run flake8 .      # Lint code
```

### Dependency Management
- Add dependencies to pyproject.toml
- Update environment: `poetry lock && poetry install`

## Coding Standards

### Python Style
- Follow PEP 8 guidelines
- Use snake_case for functions/variables
- Use PascalCase for classes
- Use 4 spaces for indentation
- Apply type hints for all functions

### Documentation
- Use Google-style docstrings
- Include for each function:
  - Args
  - Returns
  - Raises
  - Examples
- Add doctests where applicable

### Import Order
1. Standard library
2. Third-party packages
3. Local modules

### Error Handling
- Use explicit exception types
- Provide clear error messages
- Include context in error descriptions

### Testing Requirements
- Maintain 90% code coverage minimum
- Use pytest.approx for float comparisons
  - Wrap expected values
  - Use abs=0.01 for dB comparisons
- when comparing pint units, use `assert_allclose`
- Apply pytest parametrization where suitable

### Architecture
- Follow single responsibility principle
- Use dependency injection where appropriate
