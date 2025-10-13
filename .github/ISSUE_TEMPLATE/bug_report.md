---
name: Bug report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''
---

# Bug Description

A clear and concise description of what the bug is.

# To Reproduce

Steps to reproduce the behavior:

```python
import torch
from pinn_rk import ...

# Your minimal reproducible code here
```

**Expected behavior:** A clear and concise description of what you expected to happen.

**Actual behavior:** What actually happened, including error messages.

# Error Message

If applicable, paste the full error traceback:

```
Paste error traceback here
```

# Environment

Please provide the following information:

- **OS**: [e.g., Ubuntu 22.04, macOS 14.1, Windows 11]
- **Python version**: [e.g., 3.11.5]
- **pinn-rk version**: [e.g., 0.1.0]
- **PyTorch version**: [e.g., 2.3.0]
- **CUDA version** (if applicable): [e.g., 12.1]
- **GPU model** (if applicable): [e.g., NVIDIA RTX 4090]

You can get version info by running:

```python
import sys
import torch
import pinn_rk

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"pinn-rk: {pinn_rk.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
```

# Additional Context

Add any other context about the problem here, such as:

- Does this happen consistently or intermittently?
- Did this work in a previous version?
- Are there any workarounds you've found?
- Screenshots or plots (if relevant)

# Possible Solution

If you have ideas about what might be causing the issue or how to fix it, please share them here (optional).

# Checklist

Before submitting, please check:

- [ ] I have searched existing issues to ensure this is not a duplicate
- [ ] I have provided a minimal reproducible example
- [ ] I have included version information
- [ ] I have checked the documentation for related information
