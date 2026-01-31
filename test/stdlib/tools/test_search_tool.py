import pytest

from mellea.core import Context, CBlock
from mellea.stdlib.tools import MelleaSearchTool, SearchResultList


def test_search_tool():
    tool = MelleaSearchTool()
    tool_result = tool.run("Who is Nathan Fulton?")
    tool_result_parsed_repr: SearchResultList = tool.parsed_repr(tool_result)
    print(tool_result_parsed_repr.search_results)


if __name__ == "__main__":
    pytest.main([__file__])
