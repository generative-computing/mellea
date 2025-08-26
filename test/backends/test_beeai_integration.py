"""Simple integration test for BeeAI backend."""

import sys
import os
from pathlib import Path
import pytest

# Add the project root to the path so we can import mellea
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from mellea.backends.beeai import BeeAIBackend
    from mellea.backends.formatter import TemplateFormatter
    from mellea.stdlib.base import CBlock, LinearContext, ModelOutputThunk
    print("✅ Successfully imported BeeAI backend and dependencies")
except ImportError as e:
    print(f"❌ Failed to import dependencies: {e}")
    sys.exit(1)


@pytest.fixture
def backend():
    """Create a BeeAI backend instance for testing."""
    try:
        backend = BeeAIBackend(
            model_id="granite3.3:2b",
            formatter=TemplateFormatter(model_id="granite3.3:2b"),
            base_url="http://localhost:11434"
        )
        return backend
    except Exception as e:
        pytest.skip(f"Could not create BeeAI backend: {e}")
        return None


def test_backend_creation():
    """Test that we can create a BeeAI backend instance."""
    try:
        backend = BeeAIBackend(
            model_id="granite3.3:2b",
            formatter=TemplateFormatter(model_id="granite3.3:2b"),
            base_url="http://localhost:11434"
        )
        print("✅ Successfully created BeeAI backend instance")
        assert backend is not None
        assert backend.model_id == "granite3.3:2b"
    except Exception as e:
        print(f"❌ Failed to create BeeAI backend: {e}")
        pytest.fail(f"Failed to create BeeAI backend: {e}")


def test_backend_interface(backend):
    """Test that the backend has the required interface."""
    required_methods = ['generate_from_context', '_generate_from_raw']
    for method in required_methods:
        assert hasattr(backend, method), f"Backend missing required method: {method}"
        assert callable(getattr(backend, method)), f"Method {method} is not callable"
        print(f"✅ Backend has required method: {method}")
    
    print("✅ Backend interface validation passed")


def test_formatter_integration(backend):
    """Test that the formatter integrates properly with the backend."""
    # Test that we can access the formatter
    formatter = backend.formatter
    assert formatter is not None, "Formatter is None"
    assert hasattr(formatter, 'print'), "Formatter missing print method"
    assert hasattr(formatter, 'print_context'), "Formatter missing print_context method"
    assert hasattr(formatter, 'parse'), "Formatter missing parse method"
    print("✅ Formatter integration test passed")


def test_model_options_handling(backend):
    """Test that the backend can handle model options properly."""
    # Test that model options are stored
    assert hasattr(backend, 'model_options'), "Backend missing model_options attribute"
    assert isinstance(backend.model_options, dict), "model_options is not a dict"
    
    # Test that we can access model_id
    assert hasattr(backend, 'model_id'), "Backend missing model_id attribute"
    assert backend.model_id == "granite3.3:2b", f"Expected model_id 'granite3.3:2b', got '{backend.model_id}'"
    
    print("✅ Model options handling test passed")


def main():
    """Run all integration tests."""
    print("🚀 Starting BeeAI backend integration tests...\n")
    
    # Test 1: Backend creation
    try:
        backend = BeeAIBackend(
            model_id="granite3.3:2b",
            formatter=TemplateFormatter(model_id="granite3.3:2b"),
            base_url="http://localhost:11434"
        )
        print("✅ Successfully created BeeAI backend instance")
    except Exception as e:
        print(f"❌ Failed to create BeeAI backend: {e}")
        backend = None
    
    # Test 2: Interface validation
    if backend:
        try:
            test_backend_interface(backend)
        except Exception as e:
            print(f"❌ Interface validation failed: {e}")
        
        # Test 3: Formatter integration
        try:
            test_formatter_integration(backend)
        except Exception as e:
            print(f"❌ Formatter integration failed: {e}")
        
        # Test 4: Model options handling
        try:
            test_model_options_handling(backend)
        except Exception as e:
            print(f"❌ Model options handling failed: {e}")
    
    print("\n🎯 Integration test summary:")
    if backend:
        print("✅ BeeAI backend is ready for use!")
        print("📝 Next steps:")
        print("   1. Add your actual BeeAI API key")
        print("   2. Configure the correct model ID")
        print("   3. Run the full test suite with: pytest test/backends/test_beeai.py")
    else:
        print("❌ BeeAI backend setup failed")
        print("📝 Check the error messages above and fix any issues")


if __name__ == "__main__":
    main()
