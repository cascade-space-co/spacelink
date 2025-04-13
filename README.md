# PyRadio

A Python library for radio frequency calculations, including satellite dish modeling and RF conversions.

## Installation

```bash
# Install using poetry
poetry install

# Or install directly with pip
pip install .
```

## Development

### Setup

```bash
# Install dependencies and the package in development mode
poetry install

```

### Project Structure
```
pyradio/
├── src/                    # Source code
│   └── pyradio/           # Main package
│       ├── __init__.py    # Package initialization and public API
│       ├── dish.py        # Dish class implementation
│       └── conversions.py # Conversion utilities
├── tests/                 # Test directory
│   ├── __init__.py
│   ├── test_dish.py
│   └── test_conversions.py
├── pyproject.toml         # Project configuration
├── poetry.lock           # Dependency lock file
└── README.md             # Project documentation
```

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with verbose output
poetry run pytest -v

# Run with print statement output
poetry run pytest -s

# Run with coverage report
poetry run pytest --cov=pyradio

# Run a specific test file
poetry run pytest tests/test_dish.py
```

### Building Documentation

```bash
# Build HTML documentation
poetry run sphinx-build -b html docs/source docs/build
```

## Features

- Satellite dish modeling
  - Gain calculation
  - Half-power beamwidth
  - G/T ratio
  - System temperature modeling
- RF conversions
  - dB to linear power ratio
  - Frequency to wavelength
  - And more...

## License

MIT License - see LICENSE file for details
