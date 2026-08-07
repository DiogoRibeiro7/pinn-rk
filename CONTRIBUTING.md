# Contributing to pinn-rk

Thank you for your interest in contributing to pinn-rk! We welcome contributions from the community.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/DiogoRibeiro7/pinn-rk.git
cd pinn-rk

# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Install pre-commit hooks
poetry run pre-commit install
```

## Code Standards

We maintain high code quality standards:

- **Style**: Follow PEP 8 (automatically enforced by ruff)
- **Type hints**: Add type annotations to all functions (checked by mypy)
- **Tests**: Write tests for new features (we aim for >85% coverage)
- **Documentation**: Update docstrings and README for changes
- **Security**: Run bandit security checks

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 2. Make Changes

Write your code following our standards:

```python
def your_function(x: Tensor, t: Tensor) -> Tensor:
    """
    Brief description.

    Parameters
    ----------
    x : Tensor
        Description of x.
    t : Tensor
        Description of t.

    Returns
    -------
    Tensor
        Description of return value.
    """
    # Your implementation
    pass
```

### 3. Run Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=pinn_rk --cov-report=term-missing

# Run specific test file
poetry run pytest tests/test_specific.py
```

### 4. Check Code Quality

```bash
# Lint with ruff
poetry run ruff check .

# Type check with mypy
poetry run mypy src

# Security check with bandit
poetry run bandit -r src
```

### 5. Format Code

```bash
# Auto-fix linting issues
poetry run ruff check . --fix

# The pre-commit hook will also run these checks
```

### 6. Commit Changes

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
git commit -m "feat: add support for higher-order RK methods"
git commit -m "fix: correct Gauss-Legendre coefficients"
git commit -m "docs: update installation instructions"
git commit -m "test: add convergence tests"
git commit -m "refactor: simplify loss calculation"
```

**Commit Types:**
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation only changes
- `test:` - Adding or updating tests
- `refactor:` - Code changes that neither fix bugs nor add features
- `perf:` - Performance improvements
- `chore:` - Maintenance tasks

### 7. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then open a Pull Request on GitHub with:
- Clear title following conventional commits format
- Description of what changed and why
- References to related issues (if any)
- Screenshots or examples (if applicable)

## Pull Request Process

1. **Ensure CI passes**: All tests, lints, and type checks must pass
2. **Update documentation**: Add/update docstrings and README if needed
3. **Add tests**: New features should include tests
4. **Update CHANGELOG**: Add entry under "Unreleased" section
5. **Request review**: Tag maintainers for review
6. **Address feedback**: Respond to review comments
7. **Squash commits**: We may ask you to squash commits before merging

## Testing Guidelines

### Unit Tests

Test individual components in isolation:

```python
def test_butcher_tableau_validation() -> None:
    """Test that invalid tableaux raise errors."""
    T = ButcherTableau(A=invalid_A, b=b, c=c)
    with pytest.raises(ValueError):
        T.validate()
```

### Integration Tests

Test components working together:

```python
def test_loss_decreases_during_training() -> None:
    """Test that loss decreases after optimization steps."""
    # Setup model, loss, optimizer
    loss0 = loss_fn()
    # Train for a few steps
    loss1 = loss_fn()
    assert loss1 < loss0
```

### Property Tests

Test mathematical properties:

```python
def test_lagrange_partition_of_unity() -> None:
    """Test that Lagrange basis functions sum to 1."""
    L = lagrange_eval(t, nodes, weights)
    assert torch.allclose(L.sum(dim=-1), torch.ones(...))
```

## Documentation Guidelines

We use NumPy-style docstrings:

```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """
    Short one-line summary.

    Longer description if needed. Can span multiple lines and include
    mathematical notation using LaTeX in docstrings.

    Parameters
    ----------
    param1 : Type1
        Description of param1.
    param2 : Type2
        Description of param2.

    Returns
    -------
    ReturnType
        Description of return value.

    Raises
    ------
    ValueError
        When parameter validation fails.

    Examples
    --------
    >>> result = function_name(arg1, arg2)
    >>> print(result)
    expected_output

    Notes
    -----
    Additional information about the implementation or mathematical
    background.

    References
    ----------
    .. [1] Author, "Paper Title", Journal, Year.
    """
    pass
```

## Adding New Features

### New RK Methods

To add a new Runge-Kutta method:

1. Add factory function in `src/pinn_rk/tableau.py`
2. Include validation in the tableau
3. Add tests in `tests/test_tableau.py`
4. Update documentation with method properties (order, stability)
5. Add example usage

### New PDE Operators

To add a new operator:

1. Create class implementing `EllipticOperator` protocol in `src/pinn_rk/operators.py`
2. Implement `__call__` and `requires_hessian` methods
3. Add tests in `tests/test_operators.py`
4. Document the operator and its mathematical form
5. Provide example PDE using the operator

### New Examples

Add examples to `src/pinn_rk/examples/`:

1. Create standalone script with clear comments
2. Include convergence study if applicable
3. Add visualization of results
4. Document expected runtime and accuracy
5. Update examples README

## Project Structure

```
pinn-rk/
├── src/pinn_rk/          # Source code
│   ├── __init__.py       # Public API
│   ├── config.py         # Configuration dataclass
│   ├── tableau.py        # RK tableaux
│   ├── mesh.py           # Time discretization
│   ├── interpolants.py   # Lagrange interpolation
│   ├── operators.py      # PDE operators
│   ├── model.py          # Neural network models
│   ├── loss.py           # Loss function
│   └── examples/         # Example scripts
├── tests/                # Test suite
├── docs/                 # Documentation (future)
├── .github/              # GitHub configs
└── pyproject.toml        # Dependencies and config
```

## Releasing

Releases are cut **manually**. There is deliberately no release workflow: an
automated one previously ran on every push to `main` and failed on each of them,
and tagging is infrequent enough that the automation was not worth the
maintenance. Every release therefore goes through the steps below.

Publishing a release triggers the Zenodo webhook, which archives the tag and
mints a DOI. That is not reversible, so make sure `main` is green first.

1. **Bump the version in all four places.** They must agree; a mismatch ships a
   package whose reported version is wrong.

   - `pyproject.toml` → `[tool.poetry] version`
   - `src/pinn_rk/__about__.py` → `__version__`
   - `CITATION.cff` → top-level `version`
   - `CITATION.cff` → `preferred-citation.version`

   Verify:

   ```bash
   poetry run python - <<'PY'
   import tomllib, yaml, pinn_rk
   pv = tomllib.load(open("pyproject.toml", "rb"))["tool"]["poetry"]["version"]
   c = yaml.safe_load(open("CITATION.cff", encoding="utf-8"))
   assert pv == pinn_rk.__version__ == str(c["version"]) == str(c["preferred-citation"]["version"])
   print("versions agree:", pv)
   PY
   ```

   Do **not** add a `version` field to `.zenodo.json`. It is omitted on purpose so
   Zenodo takes the version from the git tag; pinning it there would archive every
   future release under the pinned value.

2. **Promote the changelog.** Move `[Unreleased]` into a new `## [X.Y.Z] - YYYY-MM-DD`
   section, leave `[Unreleased]` empty, and update the link definitions at the
   bottom of the file.

3. **Choose the version honestly.** While the project is `0.x`, bump the minor
   version for anything that changes what the loss computes, even when no public
   signature changes — users' trained weights will differ, and that deserves more
   than a patch bump.

4. **Confirm CI is green on the commit you intend to tag**, then create the release.
   `gh` creates the tag as part of this.

   ```bash
   gh release create vX.Y.Z --target main --title "vX.Y.Z — short summary" --notes-file NOTES.md --latest
   git fetch --tags
   ```

5. **Record the DOI.** Once Zenodo has archived the release, add the new
   per-version DOI to the `identifiers` list in `CITATION.cff`. The concept DOI
   (`10.5281/zenodo.21839391`) never changes and always resolves to the latest
   version, so the README badge and BibTeX entry need no update.

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bug Reports**: Use the bug report issue template
- **Feature Requests**: Use the feature request template
- **Security Issues**: Email dfr@esmad.ipp.pt privately

## Recognition

Contributors will be acknowledged in:
- CHANGELOG.md for each release
- GitHub contributors page
- Future academic publications citing the software

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:

- Be respectful and considerate
- Welcome newcomers and help them learn
- Focus on what is best for the community
- Show empathy towards other community members

Unacceptable behavior will not be tolerated.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to open an issue or discussion if you have questions about contributing!
