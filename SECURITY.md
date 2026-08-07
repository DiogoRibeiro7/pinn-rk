# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take the security of pinn-rk seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Please do NOT:

- Open a public GitHub issue for security vulnerabilities
- Disclose the vulnerability publicly before it has been addressed

### Please DO:

**Report security vulnerabilities to: [dfr@esmad.ipp.pt](mailto:dfr@esmad.ipp.pt)**

Include the following information in your report:

1. **Type of vulnerability** (e.g., code injection, denial of service, etc.)
2. **Full paths of source file(s)** related to the vulnerability
3. **Location of the affected source code** (tag/branch/commit or direct URL)
4. **Step-by-step instructions** to reproduce the issue
5. **Proof-of-concept or exploit code** (if possible)
6. **Impact of the vulnerability** - what can an attacker do?
7. **Your contact information** for follow-up questions

### What to expect:

- **Confirmation**: We will acknowledge receipt of your report within **48 hours**
- **Assessment**: We will assess the vulnerability and determine its impact and severity
- **Updates**: We will keep you informed about the progress toward fixing the vulnerability
- **Fix**: We will work on a fix and prepare a security advisory
- **Disclosure**: Once fixed, we will:
  - Release a patched version
  - Publish a security advisory
  - Credit you for the discovery (unless you prefer to remain anonymous)

### Timeline:

- **48 hours**: Initial response
- **7 days**: Assessment and severity classification
- **30 days**: Target for releasing a fix (may vary based on complexity)

## Security Best Practices

When using pinn-rk:

### 1. Keep Dependencies Updated

```bash
poetry update
```

Regularly update PyTorch and other dependencies to get security patches.

### 2. Validate User Inputs

If you're building applications on top of pinn-rk:

```python
# Always validate tensor shapes and ranges
def validate_input(x: Tensor) -> None:
    if x.ndim != 2 or x.shape[1] != 1:
        raise ValueError("Invalid input shape")
    if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
        raise ValueError("Input contains NaN or Inf")
```

### 3. Resource Limits

Set appropriate limits when running on untrusted inputs:

```python
# Limit batch size
MAX_BATCH_SIZE = 10000
if x.shape[0] > MAX_BATCH_SIZE:
    raise ValueError(f"Batch size exceeds limit of {MAX_BATCH_SIZE}")

# Use timeouts for training
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Training exceeded time limit")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(3600)  # 1 hour limit
```

### 4. Sandboxing

If running user-provided code or models:

- Use containers (Docker) with resource limits
- Run with minimal privileges
- Isolate from sensitive data
- Monitor resource usage

### 5. Model Serialization

Be cautious when loading models from untrusted sources:

```python
# Use weights_only=True to prevent code execution
model = torch.load('model.pth', weights_only=True)

# Validate model architecture matches expectations
if not isinstance(model, ExpectedModelClass):
    raise ValueError("Unexpected model type")
```

## Known Issues

Currently, there are no known security vulnerabilities.

### Dependency Security

We use:

- **Dependabot** for automated dependency updates
- **Bandit** for Python security linting
- **pip-audit** (optional) for vulnerability scanning

To run security checks locally:

```bash
# Security linting
poetry run bandit -r src

# Check for known vulnerabilities (requires pip-audit)
pip install pip-audit
poetry export -f requirements.txt | pip-audit -r -
```

## Security-Related Configuration

### Disable Debug Features in Production

```python
# In production, ensure:
torch.autograd.set_detect_anomaly(False)  # Disable anomaly detection
torch.backends.cudnn.benchmark = True     # Enable cuDNN benchmarking
```

### Numerical Stability

The library includes checks for numerical issues:

```python
# Automatic detection of non-finite values
if not torch.isfinite(loss):
    raise FloatingPointError("Non-finite loss encountered")
```

## Scope

This security policy applies to:

- ✅ The pinn-rk library code in `src/pinn_rk/`
- ✅ Example scripts in `src/pinn_rk/examples/`
- ✅ Dependencies specified in `pyproject.toml`
- ❌ User code built on top of pinn-rk
- ❌ Third-party integrations not maintained by this project

## Contact

For security concerns: **[dfr@esmad.ipp.pt](mailto:dfr@esmad.ipp.pt)**

For general questions: Open a GitHub Discussion

## Recognition

We appreciate responsible disclosure and will credit security researchers who report valid vulnerabilities (unless they prefer to remain anonymous).

## Updates to This Policy

This security policy may be updated from time to time. Check the [repository](https://github.com/DiogoRibeiro7/pinn-rk) for the latest version.

---

Last updated: October 2025
