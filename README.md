# pyradio
Python module for radio computation

## Installation

This project uses Poetry for dependency management. To install:

```bash
# Install project dependencies
poetry install

# Install with test dependencies
poetry install --with test
```

## Development

### Running Tests

Run tests using Poetry:

```bash
# Run all tests
poetry run pytest

# Run with verbose output
poetry run pytest -v

# Show coverage report in terminal
poetry run pytest --cov-report term-missing

# Generate HTML coverage report
poetry run pytest --cov-report html
```

The test suite is configured to:
- Run all tests in the project root and `tests` directory
- Generate coverage reports
- Require 90% code coverage
- Show test summary and execution report

### Dependencies

- Python >= 3.13
- NumPy >= 2.2.4
- SciPy >= 1.15.2

## Features

- Conversion between decibel (dB) and linear scales
- More features coming soon...
