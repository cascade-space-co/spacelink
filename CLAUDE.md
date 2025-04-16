# PyRadio Development Guide

## Commands
- Build: `poetry build`
- Install: `poetry install --with test`
- Run all tests: `poetry run pytest`
- Run single test: `poetry run pytest test_conversions.py::test_db2linear_common_values -v`
- Test with coverage: `poetry run pytest --cov-report term-missing` 

## Code Style
- Use Python type hints for all function parameters and return values
- Document functions with docstrings (Google style) including Args, Returns, Raises, Examples
- Use pytest for testing with parametrization where appropriate
- Format imports: standard library first, then third-party, then local
- Naming: snake_case for functions/variables, PascalCase for classes
- Handle errors with appropriate exceptions and explicit messages
- Meet 90% code coverage minimum requirement
- Include doctests in function docstrings where possible
- Keep functions focused on a single responsibility
- Use pytest.approx for float comparisons, always wrap the expected value
- When comparing values in dB (decibels) always use abs=0.01