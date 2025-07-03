## Contributing to SpaceLink

Thank you for your interest in contributing! Please read these guidelines to help us review your changes quickly and smoothly.

### Getting Help
- For questions, open an issue or start a discussion.

### Project Scope
- We welcome bugfixes, new features, documentation, and tests.

### Development Setup
1. Install Python 3.11+ and Poetry.
2. Clone the repo and run `poetry install --with dev`.
3. Run tests with `poetry run pytest`.
4. Lint with `poetry run flake8 .` and format with `poetry run black .`.

### Coding Standards
- Use Black for formatting.
- Use Flake8 for linting.
- Write docstrings for public functions/classes.

### Making a Pull Request
1. Fork and branch from `main`.
2. Make your changes with tests and docs.
3. Ensure all tests pass and code is linted.
4. Open a PR with a clear description.

### Code of Conduct
By participating, you agree to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

### License
By contributing, you agree your contributions will be licensed under the MIT License.

## Release Process

This section describes how to publish a new release of SpaceLink to PyPI. Only maintainers should perform these steps.

### 1. Bump the Version
- Update the version number in `pyproject.toml` (and `src/spacelink/__init__.py` if present).
- Follow [semantic versioning](https://semver.org/).
- Commit the version bump (e.g., `git commit -am 'Bump version to X.Y.Z'`).

### 2. Test the Package
- Ensure all tests pass:
  ```bash
  poetry run pytest
  ```
- Optionally, check code style:
  ```bash
  poetry run black .
  poetry run flake8 .
  ```

### 3. Build the Distribution
- Build the package locally:
  ```bash
  poetry build
  ```
- Check the distribution files:
  ```bash
  poetry run twine check dist/*
  ```

### 4. (Optional) Test Upload to TestPyPI
- You can test the upload process using TestPyPI:
  ```bash
  poetry run twine upload --repository-url https://test.pypi.org/legacy/ dist/*
  ```
- Visit https://test.pypi.org/project/spacelink/ to verify.

### 5. Create a GitHub Release
- Push your changes to `main`.
- Go to GitHub and create a new release (with a tag matching the version, e.g., `vX.Y.Z`).
- This will trigger the GitHub Actions workflow to build and publish the package.

### 6. Publishing to PyPI
- The `.github/workflows/publish-pypi.yml` workflow will automatically build and upload the package to PyPI when a release is published.

### 7. (Optional) Manual TestPyPI Publish
- You can manually run the `.github/workflows/publish-testpypi.yml` workflow from the Actions tab to test the process without publishing to the real PyPI.

### 8. Verify the Release
- Check https://pypi.org/project/spacelink/ for the new version.
- Install the package with pip to verify:
  ```bash
  pip install spacelink==X.Y.Z
  ```

---

For questions or help, open an issue or contact a maintainer.