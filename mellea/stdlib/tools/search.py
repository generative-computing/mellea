"""Search tool for querying the web and fetching content."""

from typing import Any

import requests

from mellea.backends.tools import MelleaTool
from mellea.core import (
    CBlock,
    Component,
    ModelOutputThunk,
    TemplateRepresentation,
    parts_contract,
)


class SearchResult(Component[ModelOutputThunk]):
    """A search result with optional fetched content."""

    def __init__(self, title: str, summary: str, url: str, content: str | None = None):
        """Initialize a search result.

        Args:
            title: The title of the search result
            summary: A summary/snippet of the search result
            url: The URL of the search result
            content: Optional fetched content from the URL
        """
        self._title = CBlock(title)
        self._summary = CBlock(summary)
        self._url = CBlock(url)
        self._content = CBlock(content) if content is not None else None

    def with_contents(self, to_md: bool = False) -> "SearchResult":
        """Returns a new SearchResult component with the content instantiated."""
        # Fetch the content from the URL and store it in _content.
        try:
            assert self.url.value is not None
            assert self.title.value is not None
            assert self.summary.value is not None
            response = requests.get(self.url.value, timeout=10)
            response_as_str = response.content.decode()
            if to_md:
                raise Exception("unimplemented. add markdownify call here.")
            response.raise_for_status()
            return SearchResult(
                title=self.title.value,
                summary=self.summary.value,
                url=self.url.value,
                content=str(response_as_str),
            )
        except requests.exceptions.RequestException:
            raise Exception(f"Could not get content for {self.url.value}")

    @property
    def title(self) -> CBlock:
        """The title of the search result."""
        return self._title

    @property
    def summary(self) -> CBlock:
        """The summary/snippet of the search result."""
        return self._summary

    @property
    def url(self) -> CBlock:
        """The URL of the search result."""
        return self._url

    @property
    def content(self) -> CBlock | None:
        """The fetched content from the URL, if available."""
        return self._content

    def parts(self):
        """Return the parts of this search result."""
        parts = [self._title, self._summary, self._url]
        if self._content is not None:
            parts.append(self._content)
        assert parts_contract(parts)
        return parts

    def format_for_llm(self):
        """Format the search result for LLM consumption."""
        content_value = ""
        if self._content is not None:
            content_value = self._content.value or ""

        return TemplateRepresentation(
            obj=self,
            template_order=["*", "SearchResult"],
            args={
                "title": self._title,
                "summary": self._summary,
                "url": self._url,
                "content": CBlock(content_value),
            },
        )

    def _parse(self, computed):
        """Parse the model output."""
        assert computed is not None
        return computed


class SearchResultList(Component[ModelOutputThunk]):
    """A list of search results."""

    def __init__(self, results: list[SearchResult]):
        """Initialize a list of search results.

        Args:
            results: List of SearchResult objects
        """
        self._results = results

    def with_contents(self, to_md: bool = False) -> "SearchResultList":
        """Returns a new SearchResultList with the contents of each page fetched."""
        new_results = [sr.with_contents(to_md) for sr in self.search_results]
        return SearchResultList(new_results)

    @property
    def search_results(self) -> list[SearchResult]:
        """The list of search results."""
        return self._results

    def parts(self):
        """Return all search results as parts."""
        rv = self._results
        assert parts_contract(rv)
        return rv

    def format_for_llm(self):
        """Format the search result list for LLM consumption."""
        return TemplateRepresentation(
            obj=self,
            template_order=["*", "SearchResultList"],
            args={"results": self._results},
        )

    def _parse(self, computed):
        """Parse the model output."""
        assert computed is not None
        return computed


class MelleaSearchTool(MelleaTool):
    """A search tool built on DuckDuckGo for web searching."""

    def __init__(self):
        """Initialize the search tool with DuckDuckGo."""
        try:
            from langchain_community.tools import DuckDuckGoSearchResults

            lc_ddg_search = DuckDuckGoSearchResults(output_format="list")
        except ImportError:
            raise ImportError(
                "You must install langchain-community to use MelleaSearchTool. "
                "Run: pip install langchain-community"
            )

        # Convert langchain tool to MelleaTool
        self._langchain_tool = lc_ddg_search

        def _search_wrapper(*args, **kwargs) -> list[dict[str, str]]:
            results = lc_ddg_search.run(*args, **kwargs)
            return results

        super().__init__(
            name=lc_ddg_search.name,
            tool_call=_search_wrapper,
            as_json_tool=self._get_json_tool(lc_ddg_search),
        )

    def _get_json_tool(self, lc_tool: Any) -> dict[str, Any]:
        """Get the JSON tool definition from langchain tool."""
        try:
            from langchain_core.utils.function_calling import convert_to_openai_tool

            return convert_to_openai_tool(lc_tool)
        except ImportError:
            raise ImportError(
                "langchain-core is required for MelleaSearchTool. "
                "Run: pip install langchain-core"
            )

    def parsed_repr(
        self, search_results: list[dict[str, str]] | None = None
    ) -> SearchResultList:
        """Convert search results to a SearchResultList component.

        Args:
            search_results: List of search results from the tool call.
                If None, returns an empty SearchResultList.
                Expected format: [{"title": str, "snippet": str, "link": str}, ...]

        Returns:
            SearchResultList component with SearchResult objects
        """
        if search_results is None:
            return SearchResultList([])

        results = []
        for item in search_results:
            result = SearchResult(
                title=item.get("title", ""),
                summary=item.get("snippet", ""),
                url=item.get("link", ""),
            )
            results.append(result)

        return SearchResultList(results)
