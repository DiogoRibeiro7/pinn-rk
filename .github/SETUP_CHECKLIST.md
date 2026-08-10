# Professional Repository Setup Checklist

This checklist guides you through setting up all the professional components for pinn-rk.

## ✅ Phase 1: Essential Documentation (HIGH PRIORITY)

### Core Documentation Files

- [ ] `CONTRIBUTING.md` - Created ✓
  - Development setup instructions
  - Code standards and style guide
  - Pull request process
  - Testing guidelines

- [ ] `SECURITY.md` - Created ✓
  - Supported versions table
  - Vulnerability reporting process
  - Security best practices
  - Contact information

- [ ] `CHANGELOG.md` - Created ✓
  - Version history
  - Release notes for v0.1.0
  - Keep a Changelog format
  - Links to releases

- [ ] `CITATION.cff` - Created ✓
  - Structured citation data
  - Author information
  - Update your ORCID ID

### GitHub Issue Templates

- [ ] `.github/ISSUE_TEMPLATE/bug_report.md` - Created ✓
  - Bug report template
  - Environment information fields
  - Reproducible example section

- [ ] `.github/ISSUE_TEMPLATE/feature_request.md` - Created ✓
  - Feature request template
  - Use case descriptions
  - Alternative approaches section

### GitHub PR Template

- [ ] `.github/pull_request_template.md` - Created ✓
  - PR description template
  - Type of change checklist
  - Testing requirements
  - Documentation checklist

### Repository Configuration

- [ ] `.github/CODEOWNERS` - Created ✓
  - Set code ownership
  - Auto-assign reviewers

## ✅ Phase 2: CI/CD Enhancement (HIGH PRIORITY)

### Enhanced CI Workflow

- [ ] `.github/workflows/ci.yml` - Updated ✓
  - Multi-platform testing (Linux, macOS, Windows)
  - Python 3.10, 3.11, 3.12
  - Coverage reporting to Codecov
  - Security scanning with Bandit
  - Build and install testing
  - Artifact uploads

### Setup Codecov

- [x] Sign up at <https://codecov.io> — done; the repository reports 91.42%
- [x] Add repository to Codecov — done
- [ ] Get Codecov token — not needed. Public repositories upload tokenlessly from
      GitHub Actions, and the workflow already sets `fail_ci_if_error: false`.
      Add one only if uploads start being rejected:
  - Go to repository Settings → Secrets and variables → Actions
  - Click "New repository secret"
  - Name: `CODECOV_TOKEN`
  - Value: [your token from Codecov]

### README Badges

- [ ] Update `README.md` with badges - Created ✓
  - CI status badge
  - Coverage badge
  - PyPI version badge
  - Python versions badge
  - License badge
  - Code style badge

## ✅ Phase 3: Examples (MEDIUM PRIORITY)

### Examples Directory

- [ ] `examples/README.md` - Created ✓
  - Overview of all examples
  - Running instructions
  - Expected results

- [ ] `examples/01_basic_heat_equation.py` - Created ✓
  - Simple heat equation example
  - Command-line arguments
  - Model saving option

- [ ] `examples/02_different_rk_methods.py` - Created ✓
  - Compare Gauss, Radau, Lobatto
  - Performance benchmarking
  - Results table

- [x] `examples/03_custom_operator.py` - Created ✓
  - Custom `EllipticOperator` for reaction-diffusion
  - Non-zero right-hand side from a manufactured solution
  - Validates the operator against its closed form before training

- [x] `examples/04_convergence_study.py` - Created ✓
  - Systematic convergence analysis with fitted rates
  - Plot generation (`--save-plots`)
  - CSV export (`--export-csv`)

### Jupyter Notebooks

- [x] `examples/notebooks/visualization.ipynb` - Created ✓
  - Interactive visualizations (Plotly time slider)
  - Solution evolution plots against the exact solution
  - Space-time error analysis
  - Committed with executed outputs; guarded by `tests/test_notebooks.py`

- [x] `examples/notebooks/benchmarking.ipynb` - Created ✓
  - Performance profiling, forward and backward split
  - Memory usage analysis
  - Scaling studies in slabs and spatial batch
  - Accuracy per unit cost: measured convergence orders per tableau

## ✅ Phase 4: Documentation Website (LOWER PRIORITY)

### MkDocs Setup

- [ ] `mkdocs.yml` - Created ✓
  - Site configuration
  - Navigation structure
  - Theme settings (Material)
  - Plugins configuration

- [ ] `docs/index.md` - Created ✓
  - Landing page
  - Quick start
  - Feature highlights

### Documentation Pages

- [x] `docs/getting-started/installation.md` - Created ✓
- [x] `docs/getting-started/quickstart.md` - Created ✓
- [x] `docs/theory/rk-pinn-formulation.md` - Created ✓
- [x] `docs/api/*.md` - Created ✓ (all seven modules, via mkdocstrings)
- [x] Site deployed to GitHub Pages ✓
- [ ] `docs/guide/` pages on configuration and training

The nav lists only pages that exist. Adding an entry without a page fails
`mkdocs build --strict`, which is the intended behaviour.

### Deploy Documentation

- [ ] Install docs dependencies:
  ```bash
  poetry install --with docs
  ```

- [ ] Build locally:
  ```bash
  poetry run mkdocs serve
  # Visit http://localhost:8000
  ```

- [ ] Deploy to GitHub Pages:
  ```bash
  poetry run mkdocs gh-deploy
  ```

## ✅ Phase 5: PyPI Publishing (WHEN READY)

### Prepare for PyPI

- [ ] Update `pyproject.toml` - Updated ✓
  - Homepage, repository, documentation URLs
  - Proper classifiers
  - All metadata fields

- [ ] Test build locally:
  ```bash
  poetry build
  ls dist/
  # Should see .whl and .tar.gz files
  ```

- [ ] Test install from wheel:
  ```bash
  pip install dist/pinn_rk-0.1.0-py3-none-any.whl
  python -c "import pinn_rk; print(pinn_rk.__version__)"
  ```

### Publish to TestPyPI (First)

- [ ] Get TestPyPI token:
  - Register at https://test.pypi.org
  - Create API token

- [ ] Configure Poetry:
  ```bash
  poetry config repositories.testpypi https://test.pypi.org/legacy/
  poetry config pypi-token.testpypi [your-token]
  ```

- [ ] Publish to TestPyPI:
  ```bash
  poetry publish -r testpypi
  ```

- [ ] Test install from TestPyPI:
  ```bash
  pip install --index-url https://test.pypi.org/simple/ pinn-rk
  ```

### Publish to PyPI (Production)

- [x] Published. Uploads happen automatically from
      `.github/workflows/publish-pypi.yml` when a GitHub release is published, or
      manually via `workflow_dispatch`.
- [x] **No API token needed.** Authentication is PyPI Trusted Publishing over OIDC:
      GitHub mints a short-lived identity for the workflow and PyPI verifies it, so
      there is no long-lived secret to leak or rotate.
- [ ] TestPyPI dry runs — would need a separate trusted publisher registered there.

The registration on PyPI names the workflow file, so renaming `publish-pypi.yml`
breaks publishing until PyPI is updated to match. It is registered **without** a
GitHub environment; declaring one in the workflow would add an environment claim to
the OIDC token that the registration does not expect, and the upload would be
rejected. To gate uploads behind an approval later, change both sides together.

<details><summary>Superseded: API token setup</summary>

- [ ] Get PyPI token:
  - Register at https://pypi.org
  - Create API token

- [ ] Configure Poetry:
  ```bash
  poetry config pypi-token.pypi [your-token]
  ```

- [ ] Publish to PyPI:
  ```bash
  poetry publish
  ```

- [ ] Verify on PyPI:
  - Visit https://pypi.org/project/pinn-rk/

- [ ] Test install:
  ```bash
  pip install pinn-rk
  ```

</details>

## 🔧 Immediate Action Items

### Do These First (Next 1-2 Hours)

1. **Update CITATION.cff with your ORCID**
   - Register at https://orcid.org if you don't have one
   - Replace placeholder in CITATION.cff

2. **Setup Codecov**
   - Sign up and add repository
   - Add token to GitHub Secrets
   - Push changes and verify CI runs

3. **Test Examples Locally**
   ```bash
   poetry run python examples/01_basic_heat_equation.py
   poetry run python examples/02_different_rk_methods.py
   ```

4. **Update README.md Badges**
   - After first CI run, verify badges work
   - Update URLs if repository name changes

### Do These This Week

5. **Create Remaining Examples**
   - Custom operator example
   - Convergence study
   - At least one Jupyter notebook

6. **Start Documentation**
   - Write getting-started guide
   - Document at least one API module
   - Test mkdocs locally

7. **Community Setup**
   - Enable GitHub Discussions
   - Pin important issues/discussions
   - Create welcome message

### Do These This Month

8. **Complete Documentation**
   - All API reference pages
   - Theory documentation
   - User guides

9. **Publish to PyPI**
   - Test on TestPyPI first
   - Create GitHub release
   - Announce in discussions

10. **Gather Feedback**
    - Share with colleagues
    - Post in relevant communities
    - Iterate based on feedback

## 📋 Verification Commands

Run these to verify everything is set up correctly:

```bash
# 1. Check all files exist
ls -la CONTRIBUTING.md SECURITY.md CHANGELOG.md CITATION.cff
ls -la .github/ISSUE_TEMPLATE/
ls -la .github/CODEOWNERS

# 2. Verify pre-commit works
git add .
poetry run pre-commit run --all-files

# 3. Run full test suite
poetry run pytest --cov=pinn_rk --cov-report=term

# 4. Check linting
poetry run ruff check .
poetry run mypy src

# 5. Test examples
poetry run python examples/01_basic_heat_equation.py --steps 100 --quiet

# 6. Build package
poetry build

# 7. Test documentation (if setup)
poetry run mkdocs build --strict

# 8. Verify package metadata
poetry show pinn-rk
```

## 📊 Progress Tracking

Update this table as you complete items:

| Category | Items Done | Total Items | Progress |
|----------|------------|-------------|----------|
| Documentation | 4 | 4 | ✅ 100% |
| GitHub Templates | 3 | 3 | ✅ 100% |
| CI/CD | 2 | 2 | ✅ 100% |
| Examples | 6 | 6 | ✅ 100% |
| Docs Website | 9 | 10 | 🔄 90% |
| PyPI Publishing | 4 | 5 | 🔄 80% |

## 🎯 Success Criteria

Current state, checked rather than assumed:

- [x] All high-priority items completed
- [x] CI passing on all platforms (9 matrix jobs: Linux, macOS, Windows x 3.10-3.12)
- [x] Code coverage > 85% — **91%**, and Codecov reports 91.42%
- [x] Documentation website live — <https://diogoribeiro7.github.io/pinn-rk/>
- [x] At least 3 working examples — four scripts and two executed notebooks
- [x] Package published to PyPI — `pip install pinn-rk` installs 0.6.0
- [x] All badges in README resolve, including the PyPI version, Python-version and
      download badges, which are back now that the package exists

## 🆘 Getting Help

If you encounter issues:

1. **CI Failures**: Check GitHub Actions logs
2. **Codecov Issues**: Verify token is set correctly
3. **Poetry Problems**: `poetry lock --no-update`
4. **MkDocs Errors**: `poetry run mkdocs build --verbose`

## 📚 Resources

- [Keep a Changelog](https://keepachangelog.com/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Poetry Documentation](https://python-poetry.org/docs/)
- [MkDocs Material](https://squidfunk.github.io/mkdocs-material/)
- [GitHub Actions](https://docs.github.com/en/actions)
- [PyPI Publishing Guide](https://packaging.python.org/tutorials/packaging-projects/)

---

**Last Updated:** October 2025
**Next Review:** After completing Phase 1
