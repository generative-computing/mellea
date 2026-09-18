# Jupyter Notebook Examples

This directory contains interactive Jupyter notebooks demonstrating various Mellea features.

## Notebooks

### example.ipynb
General introduction to Mellea with basic examples.

### compositionality_with_generative_stubs.ipynb
Interactive tutorial on composing generative functions.

### context_example.ipynb
Working with contexts and context management.

### document_mobject.ipynb
Using document MObjects for text processing.

### georgia_tech.ipynb
Domain-specific example (possibly academic/research use case).

### instruct_validate_repair.ipynb
Interactive walkthrough of the instruct-validate-repair paradigm.

### m_serve_example.ipynb
Deploying Mellea programs as services.

### mcp_example.ipynb
Model Context Protocol integration examples.

### model_options_example.ipynb
Configuring model options and parameters.

### sentiment_classifier.ipynb
Building a sentiment classification system.

### simple_email.ipynb
Email generation with requirements.

### table_mobject.ipynb
Working with table data structures.

## Running the Notebooks

```bash
# Install Jupyter if needed
uv pip install jupyter

# Start Jupyter
jupyter notebook docs/examples/notebooks/

# Or use JupyterLab
jupyter lab docs/examples/notebooks/
```

## Testing These Notebooks

CI executes these notebooks so that the Colab onboarding path cannot silently
break. Locally:

```bash
uv run poe nbtest                                        # the subset PR CI runs
uv run pytest --nbmake docs/examples/notebooks -m e2e    # everything, incl. slow
```

Six notebooks are nightly-only, because they cost far more than the rest:

- `simple_email.ipynb` — rejection sampling with LLM-validated requirements;
  blows past nbmake's default 300s per-cell budget.
- `document_mobject.ipynb` — downloads docling model weights plus a PDF from
  arxiv, then loops over 5 seeds.
- `georgia_tech.ipynb` — two Ollama models, docling weights, and a long
  multi-step pipeline.
- `compositionality_with_generative_stubs.ipynb` (417s), `context_example.ipynb`
  (369s), `instruct_validate_repair.ipynb` (296s) — timings from the CPU-only PR
  runner, which generates at ~13.5 tok/s. Their prompts are open-ended, so a
  thinking model's output length varies by a factor of five across runs on the
  same prompt (835 to 4699 tokens for one turn of `context_example.ipynb`). Past
  ~4000 tokens a single turn exceeds the Ollama backend's 300s per-request read
  timeout and the notebook fails; the nightly host is fast enough that the same
  turn finishes well inside that budget.

Three things to know before editing or adding one:

- **Every notebook declares what it needs in its own metadata.** A `mellea` block
  in the notebook's top-level metadata carries its markers and any optional
  packages, which is the notebook equivalent of the `# pytest:` comment a `.py`
  example carries. Without it the notebook is skipped and a unit test fails.
  Notebooks are collected only when `--nbmake` is passed.

  ```json
  "metadata": {
   "mellea": {
    "markers": ["e2e", "ollama", "slow"],
    "packages": ["docling"]
   },
   ...
  }
  ```

- **The Colab setup cells are tagged `skip-execution`** (installing ollama,
  `uv pip install mellea`), so a test run exercises the working tree rather than
  the released package. Keep that tag on any new setup cell, and keep such cells
  free of anything the rest of the notebook depends on. The tagged cells are not
  untested: `.github/workflows/colab-notebooks.yml` runs them in a Colab-like
  environment, so a setup cell that breaks is still caught.
- **Prefer the default session model** (`mellea.start_session()`), or a model CI
  already pulls. Naming another model makes the Ollama backend download it
  mid-cell, which usually just exhausts the per-cell timeout.

Full detail: [test/README.md → Notebooks](../../../test/README.md#notebooks).

## Benefits of Notebooks

- **Interactive Learning**: Experiment with code in real-time
- **Visualization**: See results immediately
- **Documentation**: Combine code, output, and explanations
- **Experimentation**: Try different parameters and approaches
- **Sharing**: Easy to share complete examples with outputs

## Corresponding Python Files

Most notebooks have corresponding Python files in the `tutorial/` directory for non-interactive use.

## Tips

- Run cells in order for proper context building
- Restart kernel if you encounter state issues
- Use `Shift+Enter` to run cells
- Check cell outputs for errors before proceeding

## Related Documentation

- See `tutorial/` for Python script versions
- See individual example directories for more details on each topic