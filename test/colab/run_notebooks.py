# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute the example notebooks with their Colab setup cells enabled.

The notebooks under `docs/examples/notebooks/` tag their Colab-only setup cells
`skip-execution`, so the regular nbmake run exercises the working tree rather than
installing the released package. That leaves the Colab onboarding path itself untested:
the ollama install, and `uv pip install mellea` resolving against Colab's preinstalled
package set. This runner covers that gap. It is invoked identically on a plain CI runner
and inside Colab's published runtime image.

Two mechanisms make that work without editing a single notebook:

- `skip_cells_with_tag` is pointed at a sentinel tag that matches nothing, so the
  `skip-execution` cells execute.
- `UV_OVERRIDE` names the checked-out tree, so the notebooks' own unmodified
  `!uv pip install mellea` installs this working copy instead of the release from PyPI.

`jupyter execute` is deliberately not used. Its `--output` applies `.with_suffix()` to the
pattern, so `{notebook_name}.executed` resolves back to the input filename and silently
overwrites the source notebook, and it saves nothing at all when a cell raises, which is
exactly when the executed copy is worth reading.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

# A tag no cell carries, so pointing nbclient at it makes every `skip-execution` cell run.
SENTINEL_TAG = "colab-harness-never-executes"

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

# Keeps a failure summary readable in a markdown table cell; the exception name and the
# start of the message are the parts worth seeing, and the executed notebook artifact
# carries the full traceback.
SUMMARY_WIDTH = 200

# Matches notebooks.yml's --nbmake-timeout: CI inference is slow enough to need it.
DEFAULT_CELL_TIMEOUT = 600


def notebook_markers(path: Path) -> list[str]:
    """Read a notebook's declared pytest markers.

    Args:
        path: Notebook to inspect.

    Returns:
        The `metadata.mellea.markers` list, or an empty list when the notebook does not
        opt in to collection.
    """
    metadata = json.loads(path.read_text()).get("metadata", {})
    return list(metadata.get("mellea", {}).get("markers", []))


def select_notebooks(
    notebooks_dir: Path, *, include_slow: bool = False, only: list[str] | None = None
) -> list[Path]:
    """Choose which notebooks to execute.

    Selection is derived from each notebook's own metadata rather than hardcoded, so it
    tracks the `mellea` metadata block that `test/test_example_collection.py` enforces.

    Args:
        notebooks_dir: Directory holding the example notebooks.
        include_slow: Include notebooks declaring the `slow` marker. They cost far more
            and re-test the same setup path, so they are excluded by default.
        only: Restrict to these filenames or stems, ignoring markers.

    Returns:
        Sorted notebook paths.
    """
    candidates = sorted(
        p for p in notebooks_dir.glob("*.ipynb") if ".executed" not in p.suffixes
    )
    if only:
        wanted = {Path(name).stem for name in only}
        return [p for p in candidates if p.stem in wanted]
    selected = []
    for path in candidates:
        markers = notebook_markers(path)
        if not markers:
            continue  # Opted out of collection entirely.
        if "slow" in markers and not include_slow:
            continue
        selected.append(path)
    return selected


def write_uv_override(repo_root: Path, dest: Path) -> Path:
    """Point `uv` at the working tree for any install of `mellea`.

    uv applies `--overrides` to top-level requirements as well as transitive ones, so the
    notebooks' `!uv pip install mellea` picks up this tree. Extras such as
    `mellea[docling]` override on the base name.

    Args:
        repo_root: Repository root to install from.
        dest: File to write the override requirement to.

    Returns:
        The written path.
    """
    dest.write_text(f"mellea @ file://{repo_root.resolve()}\n")
    return dest


def execute_notebook(path: Path, out_dir: Path, timeout: int) -> str | None:
    """Execute one notebook with its Colab setup cells enabled.

    The executed copy is written whether or not execution succeeded, so a failure leaves
    behind the outputs and traceback that explain it.

    Args:
        path: Notebook to execute.
        out_dir: Directory to write the executed copy into.
        timeout: Per-cell timeout in seconds.

    Returns:
        None on success, or a short failure description.
    """
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        skip_cells_with_tag=SENTINEL_TAG,
        allow_errors=False,
        # Kernel cwd, so cells resolving paths relative to the notebook behave as they do
        # in Colab.
        resources={"metadata": {"path": str(path.parent)}},
    )
    failure: str | None = None
    try:
        client.execute()
    except CellExecutionError as err:
        failure = _summarize_cell_error(err)
    except Exception as err:  # Dead kernel, startup failure, timeout.
        failure = f"{type(err).__name__}: {err}"
    finally:
        out_dir.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, out_dir / f"{path.stem}.executed.ipynb")
    return failure


def _summarize_cell_error(err: CellExecutionError) -> str:
    """Reduce a cell failure to a one-line `Ename: message` summary.

    Args:
        err: The raised execution error.

    Returns:
        A one-line summary suitable for a job summary table. The structured `ename` and
        `evalue` are used rather than the rendered traceback: IPython appends a dashed
        separator and a "NOTE: If your import is failing..." advisory *after* the exception
        line, so the traceback's last line is punctuation rather than the cause. ANSI colour
        codes are stripped and pipes escaped, since either would corrupt the markdown row
        this lands in.
    """
    detail = " ".join(ANSI_ESCAPE.sub("", err.evalue).split()).replace("|", r"\|")
    summary = f"{err.ename}: {detail}" if detail else err.ename
    if len(summary) > SUMMARY_WIDTH:
        summary = summary[: SUMMARY_WIDTH - 3].rstrip() + "..."
    return summary


def _report(results: dict[Path, str | None], out_dir: Path) -> None:
    """Print a summary and append it to the GitHub job summary when running in CI.

    Args:
        results: Notebook to failure description (None when it passed).
        out_dir: Directory the executed notebooks were written to.
    """
    lines = ["", "| Notebook | Result |", "| --- | --- |"]
    for path, failure in results.items():
        lines.append(f"| `{path.name}` | {'FAILED: ' + failure if failure else 'ok'} |")
    failed = [p.name for p, f in results.items() if f]
    lines.append("")
    lines.append(
        f"{len(results) - len(failed)}/{len(results)} notebooks passed. "
        f"Executed copies written to `{out_dir}`."
    )
    if failed:
        lines.append("")
        lines.append(
            "A failure here means the Colab onboarding path is broken for these "
            "notebooks: either the setup cells no longer run, or `uv pip install mellea` "
            "does not resolve against Colab's preinstalled packages."
        )
    report = "\n".join(lines)
    print(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as handle:
            handle.write(report + "\n")


def main() -> int:
    """Run the selected notebooks and report the outcome.

    Returns:
        Process exit status: 0 when every notebook passed.
    """
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebooks-dir", type=Path, default=repo_root / "docs/examples/notebooks"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/tmp/colab-executed"),
        help="Where to write executed notebooks. Outside the repo by default so the "
        "source notebooks are never modified.",
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_CELL_TIMEOUT)
    parser.add_argument("--include-slow", action="store_true")
    parser.add_argument(
        "--only", nargs="+", help="Run only these notebooks, by filename or stem."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the selection and exit without executing anything.",
    )
    args = parser.parse_args()

    notebooks = select_notebooks(
        args.notebooks_dir, include_slow=args.include_slow, only=args.only
    )
    if not notebooks:
        print(f"No notebooks selected under {args.notebooks_dir}", file=sys.stderr)
        return 1
    if args.list:
        for path in notebooks:
            print(f"{path.relative_to(repo_root)}  markers={notebook_markers(path)}")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    override = write_uv_override(repo_root, args.out_dir / "uv-override.txt")
    # The kernel inherits this process's environment, so `!uv pip install` in a cell sees
    # it. UV_CONSTRAINT is left alone: the constraints job sets it, and inside the Colab
    # image it must stay as Colab ships it.
    os.environ["UV_OVERRIDE"] = str(override)
    print(f"UV_OVERRIDE={override} -> {override.read_text().strip()}")
    print(
        f"Executing {len(notebooks)} notebook(s) with '{SENTINEL_TAG}' as the skip tag"
    )

    results: dict[Path, str | None] = {}
    for path in notebooks:
        print(f"\n=== {path.name} ===", flush=True)
        results[path] = execute_notebook(path, args.out_dir, args.timeout)
        print(f"--- {path.name}: {results[path] or 'ok'}", flush=True)
    _report(results, args.out_dir)
    return 1 if any(results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
