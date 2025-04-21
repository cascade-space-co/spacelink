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
1. Click on it and select "Python (spacelink-\*)" from the dropdown menu
   - This is the Poetry-managed virtual environment with all dependencies installed
1. If you don't see the spacelink kernel, run:
   ```bash
   poetry run python -m ipykernel install --user --name spacelink --display-name "Python (SpaceLink)"
   ```

### Building Documentation

```bash
# Build HTML documentation
poetry run sphinx-build -b html docs/source docs/build
```

## License

Copyright 2025 Heliosphere Network Corp

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
