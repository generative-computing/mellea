# Requirements Examples

This directory contains examples for using Mellea's requirement validation system, including specialized requirements for RAG (Retrieval-Augmented Generation) workflows and code generation tasks like plotting.

## Files

### matplotlib_plotting.py
Demonstrates how to use matplotlib-specific requirements to validate code that generates plots.

**Key Features:**
- Validating headless backend configuration (Agg, Cairo, pdf, etc.)
- Ensuring plots are explicitly saved to files
- Checking that required dependencies (matplotlib, numpy) are available
- Multiple code patterns: `plt.savefig()`, `fig.savefig()`, `ax.savefig()`
- Supporting both positional and keyword arguments

**Examples Included:**
1. Headless backend validation (valid Agg backend)
2. Headless backend failure (interactive TkAgg backend)
3. Plot file saved validation
4. Plot file saved failure (no savefig)
5. Dependencies available check
6. Multiple savefig calls
7. Figure savefig() pattern
8. Keyword arguments in savefig

### hallucination_requirement.py
Demonstrates how to use `HallucinationRequirement` to validate that RAG responses are faithful to retrieved documents.

**Key Features:**
- Detecting hallucinated content in RAG responses
- Configurable faithfulness thresholds
- Lenient vs. strict validation modes
- Single-document and multi-document validation
- Integration with validation workflows

**Examples Included:**
1. Single-document faithful response validation
2. Single-document hallucination detection
3. Lenient validation with partial hallucinations
4. Multi-document faithful response (3 documents)
5. Multi-document with partial hallucination
6. Multi-document with lenient validation

## HallucinationRequirement

The `HallucinationRequirement` class validates RAG responses for hallucinated content by checking if sentences in an assistant's response are faithful to the retrieved documents.

### Basic Usage

```python
from mellea.stdlib.requirements import HallucinationRequirement
from mellea.stdlib.components import Document, Message
from mellea.stdlib.context import ChatContext
import mellea.stdlib.functional as mfuncs

# Create requirement
req = HallucinationRequirement(
    threshold=0.5,  # Faithfulness threshold (0.0-1.0)
    max_hallucinated_ratio=0.0  # Maximum allowed hallucination ratio
)

# Single document example
documents = [Document(doc_id="1", text="The sky is blue.")]
context = (
    ChatContext()
    .add(Message("user", "What color is the sky?"))
    .add(Message("assistant", "The sky is blue.", documents=documents))
)

# Multi-document example
multi_docs = [
    Document(doc_id="1", text="The sky appears blue during the day."),
    Document(doc_id="2", text="This is due to Rayleigh scattering."),
    Document(doc_id="3", text="At sunset, the sky can appear red or orange."),
]
multi_context = (
    ChatContext()
    .add(Message("user", "Why is the sky blue?"))
    .add(Message("assistant",
                 "The sky appears blue due to Rayleigh scattering.",
                 documents=multi_docs))
)

# Validate
results = mfuncs.validate(reqs=[req], context=context, backend=backend)
print(f"Passed: {results[0].as_bool()}")
print(f"Reason: {results[0].reason}")
```

### Configuration Options

#### Threshold (0.0-1.0)
Controls per-sentence faithfulness detection:
- **0.3**: Strict - flag uncertain content (high-stakes applications)
- **0.5**: Balanced - flag likely hallucinations (default, general use)
- **0.7**: Lenient - only flag clear hallucinations (exploratory use)

#### Max Hallucinated Ratio (0.0-1.0)
Controls overall response quality gate:
- **0.0**: Zero tolerance - any hallucination fails (default, critical accuracy)
- **0.1**: Allow minor issues - up to 10% hallucinated (production RAG)
- **0.3**: Lenient - up to 30% hallucinated (draft/brainstorming)

### Multi-Document Validation

The requirement works seamlessly with multiple retrieved documents:

```python
# Multiple documents from different sources
documents = [
    Document(doc_id="wiki", text="Purple bumble fish are tropical species."),
    Document(doc_id="study", text="They grow to 15-20 cm in length."),
    Document(doc_id="conservation", text="Populations have stabilized recently."),
]

# Response synthesizing information from multiple documents
response = "Purple bumble fish are tropical fish that grow to 15-20 cm."
context = ChatContext().add(
    Message("user", "Tell me about purple bumble fish.")
).add(
    Message("assistant", response, documents=documents)
)

# Validates against all documents
results = mfuncs.validate(reqs=[req], context=context, backend=backend)
```

**Benefits of Multi-Document Validation:**
- Validates synthesis across multiple sources
- Detects when response contradicts any document
- Identifies unsupported claims even with rich context
- Useful for complex RAG pipelines with multiple retrievers

### Use Cases

**When to Use:**
- ✅ RAG applications where factual accuracy is critical
- ✅ When you have retrieved documents to validate against
- ✅ For quality monitoring and logging
- ✅ In validation/repair workflows
- ✅ Multi-document synthesis and summarization
- ✅ Complex RAG pipelines with multiple retrievers

**When NOT to Use:**
- ❌ Non-RAG applications (no documents to validate against)
- ❌ Creative/generative tasks where "hallucination" is desired
- ❌ When documents are not available at validation time
- ❌ With backends that don't support adapters (OpenAI, Anthropic)

### Integration Patterns

#### Pattern 1: Post-Generation Validation
```python
# Generate response
response = session.instruct("Answer the question", 
                           grounding_context={"docs": documents})

# Validate after generation
req = HallucinationRequirement(threshold=0.5)
results = session.validate(req)
```

#### Pattern 2: With Rejection Sampling
```python
# Use with sampling strategy for automatic retry
from mellea.stdlib.sampling import RejectionSamplingStrategy

result = session.instruct(
    "Answer using only the documents",
    requirements=[HallucinationRequirement()],
    strategy=RejectionSamplingStrategy(loop_budget=3)
)
```

#### Pattern 3: Lenient Monitoring
```python
# Monitor but don't fail on minor hallucinations
monitor_req = HallucinationRequirement(
    threshold=0.5,
    max_hallucinated_ratio=0.2  # Allow up to 20%
)
```

## MatplotlibHeadlessBackend

The `MatplotlibHeadlessBackend` requirement validates that matplotlib code uses a headless backend suitable for server environments.

### Basic Usage

```python
from mellea.stdlib.requirements.plotting import MatplotlibHeadlessBackend
from mellea.stdlib.context import ChatContext
from mellea.core import ModelOutputThunk

code = """```python
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
plt.plot([1, 2, 3])
```"""

context = ChatContext().add(ModelOutputThunk(value=code))
req = MatplotlibHeadlessBackend()
result = req.validation_fn(context)
print(result.as_bool())  # True
```

### Supported Backends
- `Agg` — Raster output (most common)
- `Cairo` — Vector output with Cairo
- `pdf`, `svg`, `pgf` — File formats
- `nbAgg` — Jupyter notebooks
- `module://gr.matplotlib.backend_gr` — GR graphics library

## PlotFileSaved

The `PlotFileSaved` requirement validates that plots are explicitly saved to a specific file path.

### Basic Usage

```python
from mellea.stdlib.requirements.plotting import PlotFileSaved
from mellea.stdlib.context import ChatContext
from mellea.core import ModelOutputThunk

code = """```python
import matplotlib.pyplot as plt
plt.plot([1, 2, 3])
plt.savefig('/tmp/plot.png')
```"""

context = ChatContext().add(ModelOutputThunk(value=code))
req = PlotFileSaved(output_path="/tmp/plot.png")
result = req.validation_fn(context)
print(result.as_bool())  # True
```

### Supported Patterns
- `plt.savefig('/tmp/plot.png')`
- `fig.savefig('/tmp/plot.png')`
- `ax.savefig('/tmp/plot.png')`
- Keyword arguments: `fig.savefig(fname='/tmp/plot.png', dpi=300)`

## PlotDependenciesAvailable

The `PlotDependenciesAvailable` requirement validates that matplotlib and numpy are importable.

### Basic Usage

```python
from mellea.stdlib.requirements.plotting import PlotDependenciesAvailable

req = PlotDependenciesAvailable()
result = req.validation_fn(context)
print(result.as_bool())  # True if matplotlib and numpy are available
```

## Use Cases

**When to Use:**
- ✅ Generating plotting code that must run on servers (headless)
- ✅ Validating plots are saved to files (not displayed interactively)
- ✅ Ensuring code can run in CI/CD environments
- ✅ Verifying required data science libraries are available
- ✅ Code quality gates for machine learning notebooks

**When NOT to Use:**
- ❌ Interactive plotting applications
- ❌ Jupyter notebooks meant for display (use `nbAgg` backend instead)
- ❌ Desktop applications with display servers

## Related Documentation

- See `mellea/stdlib/requirements/plotting/matplotlib.py` for implementation
- See `test/stdlib/requirements/plotting/test_matplotlib.py` for tests
- See `mellea/stdlib/requirements/rag.py` for RAG requirements
- See `mellea/stdlib/components/intrinsic/rag.py` for underlying intrinsic
- See `test/stdlib/requirements/test_rag_requirements.py` for RAG tests
- See `docs/examples/intrinsics/hallucination_detection.py` for intrinsic usage

## Requirements

### For Matplotlib Examples
- matplotlib and numpy installed (auto-checked by requirements)
- Any code analysis backend (requirements use AST analysis)

### For RAG Examples
- Backend with adapter support (HuggingFace, vLLM)
- Documents must be attached to assistant messages
- Hallucination detection adapter (auto-loaded)