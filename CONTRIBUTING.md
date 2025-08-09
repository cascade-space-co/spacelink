# Contributing to SpaceLink

Thank you for your interest in contributing! Please read these guidelines to help us 
review your changes quickly and smoothly.

### Getting Help
If you have a question, start a 
[discussion](https://github.com/cascade-space-co/spacelink/discussions).

### Project Scope
We welcome bugfixes, new features, documentation, and tests.

### Development Setup
1. Install Python 3.11+ and [Poetry](https://python-poetry.org/docs/).
2. Clone the repo and run `poetry install --with dev`.
3. Run tests with `poetry run pytest`.

### Building Documentation
- Build the Sphinx docs locally:

  ```bash
  poetry run sphinx-build -b html docs/source docs/build/html
  ```

- Then open `docs/build/html/index.html` in a browser.

### Coding Standards
- Use Black for formatting: `poetry run black .`
- Use Flake8 for linting: `poetry run flake8 .`
- Write docstrings for public functions/classes.
- Write unit tests.
  - Run the tests with `poetry run pytest`
  - Strive for 100% coverage: 
`poetry run pytest --cov=spacelink --cov-report=term-missing`

### Making a Pull Request
1. Fork and branch from `main`.
2. Make your changes with tests and docs.
3. Ensure all tests pass and code is formatted and linted.
4. Open a PR with a clear description.

### Code of Conduct
By participating, you agree to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

### License
By contributing, you agree your contributions will be licensed under the 
[MIT License](LICENSE).

# Release Process

This section describes how to publish a new release of SpaceLink to PyPI. Only maintainers should perform these steps.

### 1. Bump the Version
- Update the version number in `pyproject.toml`.
- Follow [semantic versioning](https://semver.org/).
- Commit the version bump (e.g., `git commit -am 'Bump version to X.Y.Z'`).

### 2. Test the Package

Check the [tests GitHub Actions workflow](https://github.com/cascade-space-co/spacelink/actions/workflows/python-tests.yml) to ensure that all tests (unit tests, formatting, and lint check) passed on the latest commit to `main`.

This can also be checked locally with the following commands:
```bash
poetry run pytest
poetry run black --check .
poetry run flake8 .
```

### 3. Build the Distribution
Build the package locally:
   ```bash
   poetry build
   ```

Check the distribution files:
```bash
poetry run twine check dist/*
```

### 4. (Optional) Test Upload to TestPyPI

You can test the upload process using TestPyPI by manually running the 
`.github/workflows/publish-testpypi.yml` [workflow](https://github.com/cascade-space-co/spacelink/actions/workflows/publish-testpypi.yml) from the Actions tab in GitHub.

Alternatively this can be done locally by running
```bash
poetry run twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

After publishing, visit https://test.pypi.org/project/spacelink/ to verify that the
update worked as expected.

### 5. Create a GitHub Release
1. Ensure all changes to be released are on the `main` branch.
2. Go to GitHub and 
[create a new release](https://github.com/cascade-space-co/spacelink/releases/new) (with a 
tag matching the version, e.g., `vX.Y.Z`). This will trigger the GitHub Actions workflow
to build and publish the package.

### 6. Publishing to PyPI
The `.github/workflows/publish-pypi.yml` 
[workflow](https://github.com/cascade-space-co/spacelink/actions/workflows/publish-pypi.yml)
will automatically build and upload the package to PyPI when a release is published.

### 7. Verify the Release
- Check https://pypi.org/project/spacelink/ for the new version.
- Install the package with pip to verify:
  ```bash
  pip install spacelink==X.Y.Z
  ```

---

For questions or help, open a discussion or contact a maintainer.