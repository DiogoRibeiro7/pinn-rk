# Description 

<!-- Provide a clear and concise description of what this PR does -->

 # Related Issues 

<!-- Link related issues using #issue_number or "Fixes #123" to auto-close -->

 - Fixes #
- Related to #

# Type of Change 

<!-- Mark the relevant option with an "x" -->

 - [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring (no functional changes)
- [ ] Test updates
- [ ] CI/CD changes
- [ ] Dependency updates

# Changes Made 

<!-- Detailed list of changes -->

 - Change 1
- Change 2
- Change 3

# Testing 

<!-- Describe the tests you ran and how to reproduce them -->

 **Test Configuration**:

- Python version:
- PyTorch version:
- OS:

**Tests performed:**

- [ ] Existing tests pass
- [ ] Added new tests for the changes
- [ ] Manual testing performed

```bash
# Commands used for testing
poetry run pytest
poetry run pytest tests/test_specific.py -v
```

**Test results:**

```
# Paste relevant test output or coverage report
```

# Code Quality Checklist 

<!-- Ensure all checks pass before requesting review -->

 - [ ] Code follows the project's style guidelines (ruff)
- [ ] Type hints added/updated (mypy passes)
- [ ] Security checks pass (bandit)
- [ ] All tests pass locally
- [ ] Added tests that prove the fix is effective or feature works
- [ ] New and existing unit tests pass
- [ ] No decrease in code coverage

```bash
# Run these commands to verify
poetry run ruff check .
poetry run mypy src
poetry run bandit -r src
poetry run pytest --cov=pinn_rk
```

# Documentation Checklist

- [ ] Updated docstrings for new/modified functions
- [ ] Updated README.md (if needed)
- [ ] Updated CHANGELOG.md under "Unreleased" section
- [ ] Added/updated type hints
- [ ] Added/updated examples (if applicable)

# Screenshots / Examples 

<!-- If applicable, add screenshots, plots, or example outputs -->

 **Before:**

```
# Code/output before changes
```

**After:**

```
# Code/output after changes
```

# Performance Impact 

<!-- Describe any performance implications -->

 - [ ] No performance impact
- [ ] Performance improvement: [describe]
- [ ] Potential performance regression: [describe and justify]

**Benchmarks** (if applicable):

```
# Performance measurements
```

# Breaking Changes 

<!-- If this introduces breaking changes, describe them -->

 **Does this PR introduce breaking changes?**

- [ ] Yes
- [ ] No

**If yes, describe:**

- What breaks:
- Migration guide for users:
- Deprecation warnings added:

# Backward Compatibility

- [ ] This change is backward compatible
- [ ] This change requires version bump: [ ] patch [ ] minor [ ] major
- [ ] Added deprecation warnings for old functionality

# Additional Context 

<!-- Add any other context about the PR here -->

 # Reviewer Notes 

<!-- Anything specific you want reviewers to focus on? -->

 - Please pay special attention to: [area of concern]
- Known limitations: [if any]
- Future work: [what's left for later]

# Pre-Merge Checklist 

<!-- Final checks before merge -->

 - [ ] All CI checks pass
- [ ] Code has been reviewed and approved
- [ ] All review comments addressed
- [ ] Changelog updated
- [ ] Branch is up to date with main
- [ ] Commits are squashed/cleaned up (if requested)

# Post-Merge Actions 

<!-- Any actions needed after merging? -->

 - [ ] None
- [ ] Update documentation site
- [ ] Announce in discussions
- [ ] Create release notes
- [ ] Other: [specify]

-------------------------------------------------------------------------------- 

<!-- Thank you for contributing to pinn-rk! Please ensure you've read CONTRIBUTING.md before submitting. -->
