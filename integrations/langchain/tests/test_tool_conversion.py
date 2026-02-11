"""Tests for tool conversion utilities."""

import pytest

from mellea_langchain.tool_conversion import langchain_to_mellea_tools


class MockLangChainTool:
    """Mock LangChain tool for testing."""

    def __init__(self, name="test_tool", description="A test tool"):
        self.name = name
        self.description = description
        self.args_schema = None

    def _run(self, *args, **kwargs):
        """Mock run method."""
        return "test result"


class MockMelleaTool:
    """Mock Mellea tool for testing."""

    def __init__(self, name="mellea_tool"):
        self.name = name
        self.tool_type = "mellea"

    @staticmethod
    def from_langchain(tool):
        """Mock conversion from LangChain tool."""
        return MockMelleaTool(name=f"converted_{tool.name}")


class TestToolConversion:
    """Test tool conversion between LangChain and Mellea formats."""

    def test_empty_tools_list(self):
        """Test conversion of empty tools list."""
        tools = []
        result = langchain_to_mellea_tools(tools)

        assert result == []
        assert isinstance(result, list)

    def test_single_langchain_tool_conversion(self):
        """Test conversion logic with LangChain-like tool."""
        # Since the actual conversion relies on Mellea's from_langchain method,
        # we test that non-BaseTool objects are passed through unchanged
        tool = MockLangChainTool(name="search", description="Search the web")

        result = langchain_to_mellea_tools([tool])

        # Without actual LangChain BaseTool, tool is passed through
        assert len(result) == 1
        assert result[0] is tool

    def test_multiple_langchain_tools_conversion(self):
        """Test conversion of multiple tools."""
        # Test that multiple tools are processed
        tool1 = MockLangChainTool(name="search", description="Search")
        tool2 = MockLangChainTool(name="calculator", description="Calculate")
        tool3 = MockLangChainTool(name="weather", description="Get weather")

        result = langchain_to_mellea_tools([tool1, tool2, tool3])

        # All tools should be in result
        assert len(result) == 3
        assert tool1 in result
        assert tool2 in result
        assert tool3 in result

    def test_mellea_tool_passthrough(self):
        """Test that Mellea tools are passed through unchanged."""
        mellea_tool = MockMelleaTool(name="native_mellea_tool")

        result = langchain_to_mellea_tools([mellea_tool])

        assert len(result) == 1
        assert result[0] is mellea_tool
        assert result[0].name == "native_mellea_tool"

    def test_mixed_tools_conversion(self):
        """Test conversion of mixed tool types."""
        # Create tools
        lc_tool = MockLangChainTool(name="langchain_tool")
        mellea_tool = MockMelleaTool(name="mellea_tool")

        result = langchain_to_mellea_tools([lc_tool, mellea_tool])

        # Both tools should be in result (passed through without conversion)
        assert len(result) == 2
        assert lc_tool in result
        assert mellea_tool in result

    def test_tool_with_complex_schema(self):
        """Test handling of tool with complex argument schema."""

        # Create tool with schema
        class ComplexMockTool(MockLangChainTool):
            def __init__(self, name, description="", args_schema=None):
                super().__init__(name, description)
                self.args_schema = args_schema

        tool = ComplexMockTool(
            name="complex_tool",
            args_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
            },
        )

        result = langchain_to_mellea_tools([tool])

        # Tool with schema should be passed through
        assert len(result) == 1
        assert result[0] is tool
        assert result[0].args_schema is not None

    def test_none_in_tools_list(self):
        """Test handling of None values in tools list."""
        mellea_tool = MockMelleaTool(name="valid_tool")

        # Should handle None gracefully by passing it through
        result = langchain_to_mellea_tools([mellea_tool, None])

        assert len(result) == 2
        assert result[0] is mellea_tool
        assert result[1] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
