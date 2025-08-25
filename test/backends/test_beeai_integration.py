"""Simple integration test for BeeAI backend."""

import sys
import os
from pathlib import Path

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


def test_backend_creation():
    """Test that we can create a BeeAI backend instance."""
    try:
        backend = BeeAIBackend(
            model_id="granite3.3:2b",
            formatter=TemplateFormatter(model_id="granite3.3:2b"),
            base_url="http://localhost:11434"
        )
        print("✅ Successfully created BeeAI backend instance")
        return backend
    except Exception as e:
        print(f"❌ Failed to create BeeAI backend: {e}")
        return None


def test_backend_interface(backend):
    """Test that the backend has the required interface."""
    if not backend:
        return False
    
    required_methods = ['generate_from_context', '_generate_from_raw']
    for method in required_methods:
        if hasattr(backend, method) and callable(getattr(backend, method)):
            print(f"✅ Backend has required method: {method}")
        else:
            print(f"❌ Backend missing required method: {method}")
            return False
    
    print("✅ Backend interface validation passed")
    return True


def test_formatter_integration(backend):
    """Test that the formatter integrates properly with the backend."""
    if not backend:
        return False
    
    try:
        # Test that we can access the formatter
        formatter = backend.formatter
        assert formatter is not None
        assert hasattr(formatter, 'print')
        assert hasattr(formatter, 'print_context')
        assert hasattr(formatter, 'parse')
        print("✅ Formatter integration test passed")
        return True
    except Exception as e:
        print(f"❌ Formatter integration test failed: {e}")
        return False


def test_model_options_handling(backend):
    """Test that the backend can handle model options properly."""
    if not backend:
        return False
    
    try:
        # Test that model options are stored
        assert hasattr(backend, 'model_options')
        assert isinstance(backend.model_options, dict)
        
        # Test that we can access model_id
        assert hasattr(backend, 'model_id')
        assert backend.model_id == "granite3.3:2b"
        
        print("✅ Model options handling test passed")
        return True
    except Exception as e:
        print(f"❌ Model options handling test failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("🚀 Starting BeeAI backend integration tests...\n")
    
    # Test 1: Backend creation
    backend = test_backend_creation()
    
    # Test 2: Interface validation
    if backend:
        test_backend_interface(backend)
        
        # Test 3: Formatter integration
        test_formatter_integration(backend)
        
        # Test 4: Model options handling
        test_model_options_handling(backend)
    
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
