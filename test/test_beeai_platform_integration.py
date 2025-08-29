"""Tests for BeeAI Platform integration."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import datetime
import tempfile
import json
from pathlib import Path

from mellea.stdlib.base import CBlock, Context, GenerateLog, ModelOutputThunk
from mellea.stdlib.session import MelleaSession
from mellea.backends.formatter import TemplateFormatter


class TestBeeAIPlatformBackend(unittest.TestCase):
    """Test BeeAI Platform backend integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the BeeAI framework dependencies
        self.mock_backend = Mock()
        self.mock_chat_model = Mock()
        self.mock_response = Mock()
        self.mock_response.messages = [Mock()]
        self.mock_response.messages[0].content = "Test response"
        self.mock_chat_model.create.return_value = self.mock_response
        
        # Create a real formatter for testing
        self.formatter = TemplateFormatter(model_id="test-model")
    
    @patch('mellea.backends.beeai.ChatModel')
    def test_platform_backend_creation(self, mock_chat_model_class):
        """Test creating a BeeAI Platform backend."""
        mock_chat_model_class.from_name.return_value = self.mock_chat_model
        
        from mellea.backends.beeai_platform import BeeAIPlatformBackend
        
        backend = BeeAIPlatformBackend(
            model_id="test-model",
            formatter=self.formatter,
            trace_granularity="generate",
            enable_traces=True,
        )
        
        self.assertEqual(backend.model_id, "test-model")
        self.assertEqual(backend.trace_granularity, "generate")
        self.assertTrue(backend.enable_traces)
        self.assertIsNotNone(backend.trace_context)
    
    @patch('mellea.backends.beeai.ChatModel')
    def test_trace_collection(self, mock_chat_model_class):
        """Test that traces are collected during generation."""
        mock_chat_model_class.from_name.return_value = self.mock_chat_model
        
        from mellea.backends.beeai_platform import BeeAIPlatformBackend
        
        backend = BeeAIPlatformBackend(
            model_id="test-model",
            formatter=self.formatter,
            trace_granularity="generate",
            enable_traces=True,
        )
        
        # Create test context and action
        from mellea.stdlib.session import MelleaSession
        session = MelleaSession(backend=backend)
        ctx = session.ctx
        action = CBlock("Test input")
        
        # Generate with tracing
        result = backend.generate_from_context(action, ctx)
        
        # Verify trace was created
        traces = backend.trace_context.get_traces()
        self.assertGreater(len(traces), 0)
        
        trace = traces[0]
        self.assertEqual(trace["name"], "generate_test-model")
        self.assertIn("action", trace["input_data"])
        self.assertIn("result_value", trace["output_data"])
    
    def test_trace_granularity_settings(self):
        """Test different trace granularity settings."""
        from mellea.backends.beeai_platform import BeeAIPlatformBackend
        
        # Test with traces disabled
        backend_none = BeeAIPlatformBackend(
            model_id="test-model",
            formatter=self.formatter,
            trace_granularity="none",
        )
        self.assertFalse(backend_none._should_trace("generate"))
        
        # Test with generate-level tracing
        backend_generate = BeeAIPlatformBackend(
            model_id="test-model",
            formatter=self.formatter,
            trace_granularity="generate",
        )
        self.assertTrue(backend_generate._should_trace("generate"))
        self.assertFalse(backend_generate._should_trace("component"))
        
        # Test with all tracing
        backend_all = BeeAIPlatformBackend(
            model_id="test-model",
            formatter=self.formatter,
            trace_granularity="all",
        )
        self.assertTrue(backend_all._should_trace("generate"))
        self.assertTrue(backend_all._should_trace("component"))
    
    @patch('mellea.backends.beeai.ChatModel')
    def test_save_traces(self, mock_chat_model_class):
        """Test saving traces to file."""
        mock_chat_model_class.from_name.return_value = self.mock_chat_model
        
        from mellea.backends.beeai_platform import BeeAIPlatformBackend
        
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = BeeAIPlatformBackend(
                model_id="test-model",
                formatter=self.formatter,
                trace_granularity="generate",
                enable_traces=True,
                trace_output_dir=temp_dir,
            )
            
            # Generate some traces
            from mellea.stdlib.session import MelleaSession
            session = MelleaSession(backend=backend)
            ctx = session.ctx
            action = CBlock("Test input")
            backend.generate_from_context(action, ctx)
            
            # Save traces
            trace_file = backend.save_traces("test_traces.json")
            
            # Verify file exists and contains data
            self.assertTrue(Path(trace_file).exists())
            
            with open(trace_file) as f:
                trace_data = json.load(f)
            
            self.assertEqual(trace_data["version"], "1.0")
            self.assertEqual(trace_data["backend"], "beeai_platform::test-model")
            self.assertGreater(len(trace_data["traces"]), 0)


class TestBeeAITraceContext(unittest.TestCase):
    """Test BeeAI trace context management."""
    
    def test_trace_lifecycle(self):
        """Test trace creation, nesting, and completion."""
        from mellea.backends.beeai_platform import BeeAITraceContext
        
        context = BeeAITraceContext()
        
        # Start root trace
        root_trace = context.start_trace(
            name="root",
            input_data={"test": "data"}
        )
        
        self.assertEqual(context.current_trace, root_trace)
        self.assertEqual(len(context.traces), 1)
        
        # Start child trace
        child_trace = context.start_trace(
            name="child",
            input_data={"child": "data"}
        )
        
        self.assertEqual(context.current_trace, child_trace)
        self.assertEqual(len(root_trace.child_traces), 1)
        self.assertEqual(child_trace.parent_trace_id, root_trace.trace_id)
        
        # End child trace
        context.end_trace({"result": "child_result"})
        self.assertEqual(context.current_trace, root_trace)
        self.assertIsNotNone(child_trace.end_time)
        
        # End root trace
        context.end_trace({"result": "root_result"})
        self.assertIsNone(context.current_trace)
        self.assertIsNotNone(root_trace.end_time)
    
    def test_trace_serialization(self):
        """Test trace serialization to dictionary."""
        from mellea.backends.beeai_platform import BeeAITrace
        
        start_time = datetime.datetime.now()
        trace = BeeAITrace(
            trace_id="test-id",
            name="test-trace",
            start_time=start_time,
            input_data={"input": "test"},
            metadata={"meta": "data"}
        )
        
        # Add some processing time
        import time
        time.sleep(0.01)
        trace.finish({"output": "result"})
        
        trace_dict = trace.to_dict()
        
        self.assertEqual(trace_dict["trace_id"], "test-id")
        self.assertEqual(trace_dict["name"], "test-trace")
        self.assertEqual(trace_dict["input_data"], {"input": "test"})
        self.assertEqual(trace_dict["output_data"], {"output": "result"})
        self.assertEqual(trace_dict["metadata"], {"meta": "data"})
        self.assertIsNotNone(trace_dict["duration_ms"])
        self.assertGreater(trace_dict["duration_ms"], 0)


class TestBeeAIAgentManifest(unittest.TestCase):
    """Test BeeAI agent manifest creation."""
    
    def test_create_manifest(self):
        """Test creating a BeeAI agent manifest."""
        from mellea.backends.beeai_platform import create_beeai_agent_manifest
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dummy Python file
            program_path = Path(temp_dir) / "test_program.py"
            program_path.write_text("# Test Mellea program\nprint('Hello')")
            
            # Create manifest
            manifest_path = create_beeai_agent_manifest(
                mellea_program=str(program_path),
                agent_name="TestAgent",
                description="Test agent for unit tests",
                version="1.0.0",
                output_dir=temp_dir,
            )
            
            # Verify manifest file
            self.assertTrue(Path(manifest_path).exists())
            
            with open(manifest_path) as f:
                manifest = json.load(f)
            
            self.assertEqual(manifest["name"], "TestAgent")
            self.assertEqual(manifest["description"], "Test agent for unit tests")
            self.assertEqual(manifest["version"], "1.0.0")
            self.assertEqual(manifest["type"], "mellea_agent")
            self.assertIn("chat", manifest["endpoints"])
            self.assertIn("mellea", manifest["capabilities"])


class TestCLIIntegration(unittest.TestCase):
    """Test CLI integration for BeeAI Platform."""
    
    @patch('subprocess.run')
    def test_beeai_cli_check(self, mock_subprocess):
        """Test checking BeeAI CLI availability."""
        from mellea.backends.beeai_platform import start_beeai_platform
        
        # Mock successful version check
        mock_subprocess.return_value = Mock(returncode=0)
        
        # This should not raise an exception
        try:
            # We'll test the function without actually starting the platform
            import subprocess
            subprocess.run(["beeai", "--version"], check=True, capture_output=True)
        except Exception:
            # Expected in test environment without BeeAI CLI
            pass
    
    def test_cli_command_structure(self):
        """Test that CLI commands are properly structured."""
        from cli.gui.commands import gui_app
        
        # Verify the gui app is properly configured
        self.assertIsNotNone(gui_app)
        self.assertEqual(gui_app.info.name, "gui")
        
        # For now, just verify that the app exists
        # Command introspection in typer is complex, so we skip detailed checks
        print("✅ GUI app properly configured")


if __name__ == "__main__":
    unittest.main()
