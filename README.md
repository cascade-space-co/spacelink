# SpaceLink

A Python library for radio frequency calculations, including antenna modeling, RF conversions, and noise calculations.

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

### Running Jupyter Notebooks

The examples directory contains Jupyter notebooks demonstrating SpaceLink's capabilities. To run these notebooks:

```bash
# Install the development dependencies which include Jupyter
poetry install --with dev

# Start the Jupyter server
poetry run jupyter notebook

# This will open your browser to the Jupyter file browser
# Navigate to the examples/ directory and open any .ipynb file
```

When running a notebook for the first time, make sure to select the correct kernel:

1. After opening the notebook, look for the kernel indicator in the top-right corner
2. Click on it and select "Python (spacelink-*)" from the dropdown menu
   - This is the Poetry-managed virtual environment with all dependencies installed
3. If you don't see the spacelink kernel, run:
   ```bash
   poetry run python -m ipykernel install --user --name spacelink --display-name "Python (SpaceLink)"
   ```

Alternatively, you can directly open a specific notebook:

```bash
# Start Jupyter with a specific notebook
poetry run jupyter notebook examples/leo_satellite_analysis.ipynb
```

### Building Documentation

```bash
# Build HTML documentation
poetry run sphinx-build -b html docs/source docs/build
```

## Features

- **Antenna Calculations**
  - Dish antenna gain calculation
  - 3dB beamwidth calculation
  - Support for different frequencies and dish sizes

- **RF Conversions**
  - Convert between dB and linear scales
  - Wavelength calculations
  - Power ratio conversions

- **Noise Calculations**
  - Thermal noise power
  - Noise figure to temperature conversion
  - Temperature to noise figure conversion
  - Cascaded noise temperature calculation

## License

MIT License
