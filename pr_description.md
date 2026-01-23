# Friendly Error Messages for Optional Dependencies

## Summary
Improves the user experience for optional backends (`hf`, `watsonx`, `litellm`) by wrapping their imports in `try/except ImportError` blocks.

## Problem
Previously, if a user ran `start_session(backend="hf")` without installing `mellea[hf]`, they received a raw `ModuleNotFoundError: No module named 'outlines'`. This was confusing for new users who didn't know `outlines` is an internal dependency of the HuggingFace backend.

## Solution
This PR catches the `ImportError` during the backend resolution phase and raises a cleaner `ImportError` that explicitly tells the user which extra to install.

**New Error Message:**
> The 'hf' backend requires extra dependencies. Please install them with: pip install 'mellea[hf]'

## Verification
Verified in a clean virtual environment (Python 3.12) with **no** extra dependencies installed.

**Test Script Output:**
<details>
<summary>Click to see reproduction script (verify_fixes.py)</summary>

```python
import mellea
import sys

def test_backend_import(backend_name):
    print(f"\n--- Testing backend: {backend_name} ---")
    try:
        # start_session internally calls backend_name_to_class which triggers the import
        mellea.start_session(backend_name=backend_name)
    except ImportError as e:
        msg = str(e)
        print(f"Caught Expected ImportError: {msg}")
        if "requires extra dependencies" in msg and f"mellea[{backend_name}]" in msg:
             print("SUCCESS: Friendly error message detected.")
        else:
             print("FAILURE: ImportError caught but message format is wrong!")
    except Exception as e:
        print(f"FAILURE: Caught unexpected exception type: {type(e).__name__}: {e}")
    else:
        print("FAILURE: No exception raised! (Did you install the extras by mistake?)")

if __name__ == "__main__":
    test_backend_import("hf")
    test_backend_import("watsonx")
    test_backend_import("litellm")
```
</details>

```text
--- Testing backend: hf ---
Caught Expected ImportError: The 'hf' backend requires extra dependencies. Please install them with: pip install 'mellea[hf]'
SUCCESS: Friendly error message detected.

--- Testing backend: watsonx ---
Caught Expected ImportError: The 'watsonx' backend requires extra dependencies. Please install them with: pip install 'mellea[watsonx]'
SUCCESS: Friendly error message detected.

--- Testing backend: litellm ---
Caught Expected ImportError: The 'litellm' backend requires extra dependencies. Please install them with: pip install 'mellea[litellm]'
SUCCESS: Friendly error message detected.
```

## Note on Other Dependencies
*   **vLLM**: Not used in `start_session()`, so users cannot trigger this crash. To fix this, we would first need to add "vllm" support to `backend_name_to_class`, and then wrap that new import in the same `try/except` block. That feels like a new feature so do it when adding?
*   **Docling**: Usage is isolated to `RichDocument`, but requires refactoring top-level imports (lazy loading) to fix properly. Out of scope for this quick fix. Can add here for completeness but would mean a bigger change?
*   **Dev dependencies**: Missing dev tools (e.g. `pytest`) will still raise standard errors.
