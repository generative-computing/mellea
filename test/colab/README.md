# Colab environment notebook testing

The example notebooks tag their Colab setup cells `skip-execution`, so
`.github/workflows/notebooks.yml` never exercises the part most likely to break in Colab:
the ollama install, and `uv pip install mellea` resolving against Colab's preinstalled
package set. This directory covers that gap, via
[`.github/workflows/colab-notebooks.yml`](../../.github/workflows/colab-notebooks.yml).

`run_notebooks.py` points nbclient's `skip_cells_with_tag` at a sentinel tag that matches
nothing, so the tagged cells run, and sets `UV_OVERRIDE` so the notebooks' own unmodified
`!uv pip install mellea` installs the checked-out tree instead of the PyPI release. No
notebook is edited. Selection comes from each notebook's `metadata.mellea.markers` block,
so a new notebook is picked up automatically and a `slow` one is excluded by default.

## `colab-constraints.txt` is generated, do not edit it

It is Colab's package set, extracted from the pristine runtime image by the nightly
`colab-image` job and committed so the per-PR job has something cheap to resolve against. A
conflict against it is a real signal about mellea's dependencies, so hand-editing a pin
hides the finding rather than fixing it.

To refresh it after the nightly job warns about drift:

```bash
BRANCH="$(git branch --show-current)"
gh workflow run colab-notebooks.yml --ref "$BRANCH"      # ~1 h: 23 GB image pull
RUN_ID="$(gh run list --workflow=colab-notebooks.yml --branch="$BRANCH" \
  --limit=1 --json databaseId --jq '.[0].databaseId')"
gh run watch "$RUN_ID"
gh run download "$RUN_ID" --name colab-constraints --dir test/colab/
head -6 test/colab/colab-constraints.txt   # expect "# python: 3.12.x" plus tag/digest headers
```

Then commit it. This stays a manual, reviewed step on purpose.

## Running it locally

```bash
uv run poe colabtest
```

Two local side effects, both intentional:

- ollama is installed if missing. An existing install, a running daemon, and an
  already-clamped model are all left alone.
- The notebooks' `!uv pip install mellea` installs the working tree into the active
  environment, replacing an editable install with a regular one. Run
  `uv sync --all-extras --all-groups` afterwards to restore it.

Executed notebooks land in `/tmp/colab-executed/`, outside the repo, so the source notebooks
are never touched. Useful flags:

```bash
uv run python test/colab/run_notebooks.py --list            # show the selection, run nothing
uv run python test/colab/run_notebooks.py --only example    # one notebook
uv run python test/colab/run_notebooks.py --include-slow    # add the nightly-only notebooks
```

## Reading a failure

Both jobs are **advisory**, and deliberately not part of `ci.yml`'s `code-checks` aggregate:
`publish-release.yml` depends on that aggregate, so a Colab-side change must not be able to
block a release.

The per-PR job is also *stricter* than real Colab. `uv pip install -c` makes a version
conflict an error, whereas Colab would upgrade the preinstalled package instead. A red X
there means "mellea forces upgrades in Colab, check what moved", not "Colab is broken".
