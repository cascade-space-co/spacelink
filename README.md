# SpaceLink

A Python library for radio frequency calculations, including antenna modeling, RF conversions, and noise calculations.

Created and maintained by the Heliosphere Network Corporation

https://www.heliospherenetwork.com

## Installation

```bash
# Install using poetry
poetry install

# Or install directly with pip
pip install .
```

## Usage

## Development

### Setup
#### Mac OS X
```bash
# Install dependencies and the package in development mode
poetry install
```

### Project Structure

```
spacelink/
├── src/
│   └── spacelink/
│       ├── __init__.py
│       ├── antenna.py     # Antenna gain and beamwidth calculations
│       ├── conversions.py # dB/linear conversions
│       └── noise.py      # Noise calculations
├── tests/
│   ├── __init__.py
│   ├── test_antenna.py
│   ├── test_conversions.py
│   └── test_noise.py
├── pyproject.toml
├── poetry.lock
└── README.md
```

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=spacelink --cov-report=term-missing

# Run a specific test file
poetry run pytest tests/test_antenna.py

# Run tests with verbose output
poetry run pytest -v
```

### Code Formatting and Linting

```bash
# Format code with Black
poetry run black .

# Lint code with Flake8
poetry run flake8 .
```


### Building Documentation

```bash
# Build HTML documentation
poetry run sphinx-build -b html docs/source docs/build
```

## License

MIT License
