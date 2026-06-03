# pytest: e2e, qualitative
"""Examples for python_tool() — arithmetic, CSV summary, and matplotlib plot.

Each example uses the ``local_unsafe`` tier so no Docker daemon is required.

Package-install story
---------------------
Pass ``packages=["pkg1", "pkg2"]`` to ``python_tool()`` for on-demand installs.
The tool runs ``uv pip install`` when uv is on PATH (the default in uv-managed
projects), falling back to ``python -m pip`` otherwise.  Installs happen once
per tool lifetime (the shared install cache is keyed to the ``python_tool``
instance), so repeated ``tool.run()`` calls only pay the pip cost once.

Artifact directory
------------------
Pass ``artifact_dir=<Path>`` to collect files the code writes.  Only files
produced by a **successful** execution are surfaced.  When ``artifact_dir`` is
omitted, a per-call tempdir is created; it is kept alive as long as the
returned ``ExecutionResult`` holds artifacts, and cleaned up otherwise.
"""

import tempfile
from pathlib import Path

from mellea.stdlib.tools import python_tool


def example_arithmetic():
    """Run a simple arithmetic expression and print the result."""
    tool = python_tool()
    result = tool.run(code="print(2 ** 10)")
    print("stdout:", result.stdout)
    assert result.success
    assert result.stdout == "1024"


def example_csv_summary():
    """Write a pandas CSV summary and surface it as a structured artifact.

    Installs pandas on demand via packages= — no pre-installed deps required.
    Uses a TemporaryDirectory context manager so the tempdir is cleaned up
    after the example completes.
    """
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = Path(tmp)
        tool = python_tool(packages=["pandas"], artifact_dir=artifact_dir)

        # Use a relative filename — the tool sets CWD to artifact_dir so
        # 'summary.csv' lands directly inside it.
        code = """
import pandas as pd

df = pd.DataFrame({"name": ["alice", "bob", "carol"], "score": [95, 87, 92]})
df.to_csv("summary.csv", index=False)
print(f"Wrote {len(df)} rows")
"""

        result = tool.run(code=code)
        print("stdout:", result.stdout)
        assert result.success, result.stderr
        assert len(result.artifacts) == 1
        artifact = result.artifacts[0]
        print(f"  path: {artifact.path}")
        print(f"  size_bytes: {artifact.size_bytes}")
        print(f"  content_type: {artifact.content_type}")


def example_matplotlib_plot():
    """Generate a y=x² plot, save it to a tempdir, and inspect the artifact.

    Installs matplotlib and numpy on demand via packages= — no pre-installed
    deps required.  matplotlib.use('Agg') is injected automatically so the
    plot renders to a file without requiring a display.
    Uses a TemporaryDirectory context manager so the tempdir is cleaned up
    after the example completes.
    """
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = Path(tmp)
        tool = python_tool(packages=["matplotlib", "numpy"], artifact_dir=artifact_dir)

        # Relative filename — CWD is artifact_dir.
        code = """
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 200)
plt.plot(x, x ** 2)
plt.title('y = x^2')
plt.savefig("plot.png", dpi=72)
print("Plot saved.")
"""

        result = tool.run(code=code)
        print("stdout:", result.stdout)
        assert result.success, result.stderr
        assert result.artifacts, "Expected at least one artifact"
        artifact = result.artifacts[0]
        print(f"  path: {artifact.path}")
        print(f"  size_bytes: {artifact.size_bytes}")
        print(f"  content_type: {artifact.content_type}")
        assert artifact.content_type == "image/png"


if __name__ == "__main__":
    print("=== arithmetic ===")
    example_arithmetic()
    print("\n=== CSV summary ===")
    example_csv_summary()
    print("\n=== matplotlib plot ===")
    example_matplotlib_plot()
