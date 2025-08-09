# SpaceLink

A Python library for radio frequency calculations, including antenna modeling, RF 
conversions, and noise calculations.

Created and maintained by [Cascade Space](https://cascade.space).

Published documentation for the latest released version can be found here: 
https://cascade-space-co.github.io/spacelink/

## Features

- **Antenna Modeling**: Calculate antenna gain, beamwidth, and polarization effects
- **RF System Analysis**: Model complete RF chains with cascaded elements
- **Link Budget Calculations**: Comprehensive analysis of radio communication links
- **Noise Calculations**: System noise temperature and related parameters
- **Space Communications**: Built-in support for satellite link analysis
- **Unit-Aware Calculations**: Integrated unit handling for RF parameters

## Installation

### Quick Install

For users who want to import a released version of the package:
```bash
pip install spacelink
```

### Development Setup

Aside from modifying the source code you may want to install from source in order to:

* Run the provided Jupyter notebooks
* Generate the documentation locally for a specific version


#### Prerequisites

1. Python 3.11 or higher
2. Poetry package manager ([Install Poetry](https://python-poetry.org/docs/))

#### Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/cascade-space-co/spacelink.git
   cd spacelink
   ```

2. Install it to a Poetry virtual environment using one of these options:
   
   * Production mode without developer tools:
     ```bash
     poetry install
     ```
   
   * With developer tools:
     ```bash
     poetry install --with dev
     ```

   * With Jupyter notebook dependencies:
     ```bash
     poetry install --with demo
     ```

   * With developer tools and Jupyter notebook dependencies:
     ```bash
     poetry install --with dev,demo
     ```

## Documentation

The documentation includes API references and technical guides.

To build the documentation locally:
```bash
poetry run sphinx-build -b html docs/source docs/build/html
```

Then open `docs/build/html/index.html` in your browser.

## Contributing

We welcome contributions to the SpaceLink project! See 
[CONTRIBUTING.md](https://github.com/cascade-space-co/spacelink/blob/main/CONTRIBUTING.md) for detailed instructions and guidelines.

## License

[MIT License](https://github.com/cascade-space-co/spacelink/blob/main/LICENSE)

