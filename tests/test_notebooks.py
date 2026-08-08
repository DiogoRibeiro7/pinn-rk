"""
Structural checks on the shipped example notebooks.

The notebooks are committed with their outputs so they render on GitHub without being
run. That is only worth doing if the stored outputs are real, so these tests assert the
notebooks were actually executed and that nothing in them raised.

Notebooks are plain JSON, so this needs no notebook tooling and runs in CI, where the
optional ``examples`` dependency group is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "examples" / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))


def test_notebook_directory_is_populated() -> None:
    """The README advertises these; a rename or deletion should fail loudly."""
    names = {p.name for p in NOTEBOOKS}
    assert {"visualization.ipynb", "benchmarking.ipynb"} <= names, names


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_well_formed(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb.get("nbformat", 0) >= 4
    assert nb.get("cells"), "notebook has no cells"
    for cell in nb["cells"]:
        assert cell["cell_type"] in {"code", "markdown", "raw"}


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_stores_no_errors(path: Path) -> None:
    """A committed traceback means the notebook was published broken."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    failures = [
        (out.get("ename"), out.get("evalue"))
        for cell in nb["cells"]
        for out in cell.get("outputs", [])
        if out.get("output_type") == "error"
    ]
    assert not failures, f"{path.name} contains error outputs: {failures}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_was_actually_executed(path: Path) -> None:
    """
    Guard against committing a notebook that was written but never run.

    Every code cell should carry an execution count, and the notebook should contain at
    least one rendered figure -- both absent if the file was authored by hand.
    """
    nb = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code_cells, "notebook has no code cells"

    unexecuted = [i for i, c in enumerate(code_cells) if c.get("execution_count") is None]
    assert not unexecuted, f"{path.name}: code cells never executed at indices {unexecuted}"

    figures = sum(
        1 for c in code_cells for out in c.get("outputs", []) if "image/png" in out.get("data", {})
    )
    assert figures > 0, f"{path.name} stores no rendered figures"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_imports_the_installed_package(path: Path) -> None:
    """Notebooks must use the public API, not a sys.path shim into src/."""
    source = "\n".join(
        "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        for c in json.loads(path.read_text(encoding="utf-8"))["cells"]
        if c["cell_type"] == "code"
    )
    assert "from pinn_rk import" in source or "import pinn_rk" in source
    assert "sys.path" not in source, "notebook manipulates sys.path instead of importing"
