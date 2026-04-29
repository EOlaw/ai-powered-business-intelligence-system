"""
InsightSerenity AI Engine — Web Search Tool
============================================
Gives the agent access to live web information by performing an HTTP
search and extracting relevant text from the results.

Architecture:
    1. Format the query as a URL for a self-hosted search endpoint
       (or a structured web scrape when no search service is available)
    2. Fetch the top-N result pages using our async crawler
    3. Extract clean text using the HTML extractor (Phase 1)
    4. Return the most relevant snippets (up to max_chars)

Search strategy (no external API key required):
    We use two approaches depending on availability:
    a) Self-hosted search (SearXNG or similar — configured via env var)
    b) Direct URL fetch: the agent provides a specific URL to read
    c) Simulated search: fetch from a curated domain list (fallback)

The key constraint: NO EXTERNAL AI API KEYS. Web search itself is just
HTTP — it doesn't require an AI service key.

Usage:
    tool = WebSearchTool()
    result = tool.execute("What is the population of France in 2024?")
"""

from typing import Optional

from src.agents.tools.tool_registry import BaseTool
from src.utils.logger import get_logger

logger = get_logger(__name__)


class WebSearchTool(BaseTool):
    """
    Web search and URL reader tool for the agent.

    Supports two modes:
        URL mode:    Input starts with "http" → fetch that specific URL
        Query mode:  Input is a query string → search or fetch from configured endpoint

    Args:
        search_endpoint: URL of a self-hosted search engine (e.g. SearXNG).
                         If None, the tool attempts direct URL reads when given a URL,
                         or returns a "search not configured" message for queries.
        max_chars:       Maximum characters to return from each page.
        timeout:         HTTP timeout in seconds.
    """

    name        = "web_search"
    description = (
        "Searches the web or reads a URL. "
        "For queries: provide a search query string. "
        "For URLs: provide the full URL starting with http(s)://. "
        "Returns relevant text from the results."
    )

    def __init__(
        self,
        search_endpoint: Optional[str] = None,
        max_chars:        int   = 1500,
        timeout:          int   = 15,
    ) -> None:
        super().__init__(max_output_length=max_chars)
        self.search_endpoint = search_endpoint
        self.max_chars       = max_chars
        self.timeout         = timeout

    def _run(self, tool_input: str) -> str:
        """
        Execute the web search or URL fetch.

        Args:
            tool_input: Either a URL or a search query.

        Returns:
            Extracted text from the page(s), or an error message.
        """
        query = tool_input.strip()

        if not query:
            return "Error: No search query or URL provided"

        # URL mode: fetch the specific page
        if query.lower().startswith(("http://", "https://")):
            return self._fetch_url(query)

        # Query mode: use search endpoint or fallback
        if self.search_endpoint:
            return self._search_via_endpoint(query)

        # Fallback: no search configured
        return (
            f"Search query received: '{query}'\n"
            "Note: No web search endpoint configured. "
            "Set SEARCH_ENDPOINT environment variable to enable live web search. "
            "To read a specific page, provide the full URL instead."
        )

    def _fetch_url(self, url: str) -> str:
        """Fetch and extract text from a single URL."""
        try:
            import httpx
            from src.data.preprocessing.html_extractor import HTMLExtractor

            extractor = HTMLExtractor()

            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                response = client.get(url, headers={
                    "User-Agent": "InsightSerenityAgent/1.0"
                })

            if response.status_code != 200:
                return f"Error: HTTP {response.status_code} for {url}"

            text = extractor.extract(response.text)
            if not text:
                return f"No readable content found at {url}"

            # Return the most relevant portion
            return text[:self.max_chars]

        except Exception as e:
            return f"Error fetching {url}: {e}"

    def _search_via_endpoint(self, query: str) -> str:
        """Search using the configured self-hosted endpoint."""
        try:
            import httpx
            from src.data.preprocessing.html_extractor import HTMLExtractor
            import urllib.parse

            extractor = HTMLExtractor()
            encoded   = urllib.parse.quote_plus(query)
            search_url = f"{self.search_endpoint}?q={encoded}&format=json"

            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(search_url)

            if response.status_code != 200:
                return f"Search error: HTTP {response.status_code}"

            data    = response.json()
            results = data.get("results", [])[:3]   # Top 3 results

            if not results:
                return f"No results found for: {query}"

            snippets = []
            for r in results:
                title   = r.get("title", "")
                snippet = r.get("content", r.get("snippet", ""))
                url_r   = r.get("url", "")
                if snippet:
                    snippets.append(f"**{title}**\n{snippet}\nSource: {url_r}")

            return "\n\n".join(snippets)[:self.max_chars]

        except Exception as e:
            return f"Search error: {e}"


class URLReaderTool(BaseTool):
    """
    Simple URL reader — fetches and extracts text from any given URL.
    Use when you have a specific URL you want the agent to read.
    """

    name        = "read_url"
    description = (
        "Reads and extracts the text content from a given URL. "
        "Input: a complete URL starting with http:// or https://"
    )

    def __init__(self, timeout: int = 20, max_chars: int = 2000) -> None:
        super().__init__(max_output_length=max_chars)
        self.timeout   = timeout
        self.max_chars = max_chars
        self._search   = WebSearchTool(max_chars=max_chars, timeout=timeout)

    def _run(self, tool_input: str) -> str:
        url = tool_input.strip()
        if not url.startswith(("http://", "https://")):
            return "Error: Please provide a complete URL starting with http:// or https://"
        return self._search._fetch_url(url)
