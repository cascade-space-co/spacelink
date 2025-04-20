# PyRadio Development Guide
## Purpose
This python module is used for planning space missions and designing spacecraft. It Contains
code for modeling link budgets and doing receiver and transmitter lineup analysis, such as
cascaded noise figure analysis, gain / saturation analysis, and isolation analysis.

## RF + Engineering
This section is to inform you of some basic RF knowledge that is important context for editing
this code
- RF = radio frequency
- All gains and losses are reported in units of decibels. decibels, dB, and db are equivalent
- All computations are to be done in fundamental units such as meters, Hz, dBW, etc.
- Always do conversions with conversion functions, never convert engineering units in place eg meters = 1000 x kilometers
- There is no difference between gain and loss in terms of representation
- Total gain is computed by adding all the gains and losses (do not subtract losses)
- Never use negative losses, although negative gain is fine and expected in many cases

## Commands
- Build: `poetry build`
- Install: `poetry install --with test`
- Run all tests: `poetry run pytest -v`
- Run single test: `poetry run pytest test_conversions.py::test_db2linear_common_values -v`
- Test with coverage: `poetry run pytest --cov-report term-missing`
- Lint code: `poetry run flake8 .`
- Format code: `poetry run black .`
- Add dependencies by adding them to pyproject.toml and using `poetry lock & poetry install` to install them

## Code Style
- Follow PEP 8 guidelines.
- Use snake_case for variable and function names.
- Class names should be in PascalCase.
- Use 4 spaces for indentation.
- Use Python type hints for all function parameters and return values
- Document functions with docstrings (Google style) including Args, Returns, Raises, Examples
- Format imports: standard library first, then third-party, then local
- Handle errors with appropriate exceptions and explicit messages
- Meet 90% code coverage minimum requirement
- Include doctests in function docstrings where possible
- do not append units to variables
- Use pint for units except those in dB. variables should always be in fundamental units
- Numerical values should be in human readable engineering units.
- dB (decibels) shall be floats

## How to work with dB + Pint
In some special cases we need to work with dB values in Pint
- path_loss is a unitless ratio, so you can convert using: `path_loss.to('dB').magnitude`

## Testing
- Use pytest.approx for float comparisons, always wrap the expected value
- When comparing values in dB (decibels) always use abs=0.01
- Use pytest for testing with parametrization where appropriate

## Architecture and Design Patterns
<!-- - `main.py`: Entry point of the application.
- `utils.py`: Utility functions.
- `models/`: Data models.
- `services/`: External API interactions. -->
- Use async/await for network calls.
- Keep functions focused on a single responsibility
- Use dependency injection where appropriate

# `examples/` Directory
Contains jupyter notebooks with examples of how to use this library

## Custom Commands
- /refactor: Improve code structure in `src/`.
- /doc: Generate documentation for modules.
- /review: Request code review for recent changes.
