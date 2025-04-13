# PyRadio

A Python library for radio frequency calculations, including antenna modeling, RF conversions, and noise calculations.

## Installation

```bash
# Install using poetry
poetry install

# Or install directly with pip
pip install .
```

## Usage

```python
# Import specific functions from their respective modules
from pyradio.antenna import dish_gain, dish_3db_beamwidth
from pyradio.conversions import db, db2linear
from pyradio.noise import noise_figure_to_temperature, temperature_to_noise_figure

# Calculate dish gain
gain_db = dish_gain(diameter=1.0, frequency=2.4e9)  # 1m dish at 2.4 GHz
beamwidth = dish_3db_beamwidth(diameter=1.0, frequency=2.4e9)

# Convert between dB and linear scales
power_ratio = db2linear(3.0)  # Convert 3 dB to linear scale
power_db = db(2.0)  # Convert power ratio 2.0 to dB

# Convert between noise figure and temperature
temp_k = noise_figure_to_temperature(3.0)  # Convert 3 dB noise figure to temperature
noise_fig_db = temperature_to_noise_figure(288.6)  # Convert temperature to noise figure
```

## Development

### Setup

```bash
# Install dependencies and the package in development mode
poetry install

# Or with pip
pip install -e .
```

### Project Structure

```
pyradio/
├── src/                    
│   └── pyradio/           
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
poetry run pytest --cov=pyradio --cov-report=term-missing

# Run a specific test file
poetry run pytest tests/test_antenna.py

# Run tests with verbose output
poetry run pytest -v
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
