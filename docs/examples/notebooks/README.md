# Jupyter Notebook Examples

This directory contains interactive Jupyter notebooks demonstrating various Mellea features.

## Notebooks

### example.ipynb
General introduction to Mellea with basic examples.

### compositionality_with_generative_slots.ipynb
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