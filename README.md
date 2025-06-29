# SpaceLink

A Python library for radio frequency calculations, including antenna modeling, RF conversions, and noise calculations.

Created and maintained by [Cascade Space](https://www.cascade.space)

## Features

- **Antenna Modeling**: Calculate antenna gain, beamwidth, and polarization effects
- **RF System Analysis**: Model complete RF chains with cascaded elements
- **Link Budget Calculations**: Comprehensive analysis of radio communication links
- **Noise Calculations**: System noise temperature and related parameters
- **Space Communications**: Built-in support for satellite link analysis
- **Unit-Aware Calculations**: Integrated unit handling for RF parameters

## Installation

### Quick Install

For users who want to use the package:
```bash
pip install spacelink
```

### Development Setup

#### Prerequisites

1. Python 3.11 or higher
2. Poetry package manager ([Install Poetry](https://python-poetry.org/docs/))

#### Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/cascade-space-co/spacelink.git
   cd spacelink
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

For production use without development tools:
   ```bash
   poetry install --without dev
   ```

## Documentation

The documentation includes API references and technical guides.

To build the documentation locally:
```bash
poetry run sphinx-build -b html docs/source docs/build
```

## Contributing

We welcome contributions to the SpaceLink project! To get started, please follow these steps:

1. **Read the Guidelines**: Review the [CONTRIBUTING.md](../CONTRIBUTING.md) file for detailed instructions on coding style, testing, and project conventions.

2. **Set Up Your Environment**:
   - Install dependencies using Poetry:
     ```bash
     poetry install
     ```

3. **Run Tests**:
   - Ensure all tests pass before submitting your changes:
     ```bash
     poetry run pytest
     ```
   - Run tests with coverage:
     ```bash
     poetry run pytest --cov=spacelink --cov-report=term-missing
     ```
   - Run a specific test file:
     ```bash
     poetry run pytest tests/core/test_antenna.py
     ```
   - Run tests with verbose output:
     ```bash
     poetry run pytest -v
     ```

4. **Follow Code Style**:
   - Format your code with Black:
     ```bash
     poetry run black .
     ```
   - Lint your code with Flake8:
     ```bash
     poetry run flake8 .
     ```

5. **Submit a Pull Request**:
   - Push your changes to a feature branch and open a pull request on GitHub.
   - Provide a clear description of your changes and link any related issues.

Thank you for contributing to SpaceLink!

[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

## License

[MIT License](LICENSE)

