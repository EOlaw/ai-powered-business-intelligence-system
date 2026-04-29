"""
InsightSerenity AI Engine — HTML → Clean Text Extractor
========================================================
Converts raw HTML pages (as produced by the crawler) into clean, readable
plain text suitable for language model training.

The challenge: a web page's HTML contains enormous amounts of noise that has
no training value — navigation menus, cookie banners, JavaScript code, CSS,
ads, footers, boilerplate legal text. A model trained on this noise learns
to repeat it. This extractor aggressively strips everything that is not the
main article content.

Strategy:
    1. Parse HTML with lxml (fast) via BeautifulSoup
    2. Remove all non-content tags entirely (script, style, nav, footer, …)
    3. Decode HTML entities (&amp; → &, &lt; → <, …)
    4. Extract visible text, preserving paragraph structure
    5. Detect and score main content area (largest text block heuristic)

Usage:
    from src.data.preprocessing.html_extractor import HTMLExtractor

    extractor = HTMLExtractor()
    text = extractor.extract("raw html string")
    # Returns clean text or empty string if no usable content found
"""

import re
import html as html_lib
from typing import List, Optional

from bs4 import BeautifulSoup, Comment, Tag

from src.utils.logger import get_logger

logger = get_logger(__name__)


# Tags whose entire subtree should be removed (no text value at all)
_REMOVE_TAGS = frozenset({
    "script", "style", "noscript", "iframe", "object", "embed",
    "svg", "math", "canvas", "template",
    # Navigation chrome
    "nav", "header", "footer", "aside",
    # Forms and interactive UI
    "form", "button", "select", "option", "input", "textarea",
    # Media
    "figure", "picture", "audio", "video", "source", "track",
    # Ads and tracking (common class names handled separately)
    "ins",
})

# CSS class / id substrings that strongly indicate non-content elements
_NOISE_CLASS_PATTERNS = re.compile(
    r"(nav|menu|sidebar|footer|header|banner|ad-|advertisement|"
    r"cookie|popup|modal|social|share|comment|disqus|widget|"
    r"breadcrumb|pagination|related|recommend|newsletter|subscribe)",
    re.IGNORECASE,
)


class HTMLExtractor:
    """
    Extracts clean readable text from raw HTML documents.

    The extractor is stateless and thread-safe — one instance can be shared
    across worker processes (no mutable instance state).

    Args:
        min_paragraph_length: Minimum character length for a paragraph to be
                               kept. Very short paragraphs (navigation items,
                               button labels) are discarded.
        extract_title:        Whether to prepend the page <title> as the first
                               line of extracted text.
    """

    def __init__(
        self,
        min_paragraph_length: int = 30,
        extract_title: bool = True,
    ) -> None:
        self._min_para_len  = min_paragraph_length
        self._extract_title = extract_title

    def extract(self, html: str) -> str:
        """
        Convert a raw HTML string to clean plain text.

        Returns an empty string if the HTML contains no usable content
        (e.g. 404 error pages, redirect stubs, JavaScript-only pages).

        Args:
            html: Raw HTML source code as a string.

        Returns:
            Extracted text with paragraphs separated by double newlines.
        """
        if not html or not html.strip():
            return ""

        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception as e:
            logger.debug("HTML parse error", error=str(e))
            return ""

        # Remove all comment nodes (<!-- ... -->) — these add no training value
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            comment.extract()

        # Remove noise tags entirely
        self._remove_noise_tags(soup)

        # Extract title
        title_text = ""
        if self._extract_title:
            title_tag = soup.find("title")
            if title_tag:
                title_text = title_tag.get_text(strip=True)

        # Try to find the main content body
        main_content = self._find_main_content(soup)
        if main_content is None:
            main_content = soup.find("body") or soup

        # Extract paragraphs from the main content area
        paragraphs = self._extract_paragraphs(main_content)

        if not paragraphs:
            return ""

        # Assemble final text
        parts: List[str] = []
        if title_text:
            parts.append(title_text)
        parts.extend(paragraphs)

        return "\n\n".join(parts)

    def extract_metadata(self, html: str) -> dict:
        """
        Extract lightweight metadata from HTML without full text extraction.

        Returns:
            dict with keys: title, description, language, canonical_url
        """
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            return {}

        meta: dict = {}

        # Title
        title_tag = soup.find("title")
        if title_tag:
            meta["title"] = title_tag.get_text(strip=True)

        # Meta description
        desc = soup.find("meta", attrs={"name": "description"})
        if desc and desc.get("content"):
            meta["description"] = desc["content"]

        # Language
        html_tag = soup.find("html")
        if html_tag and html_tag.get("lang"):
            meta["language"] = html_tag["lang"][:5]   # "en-US" → "en-US"

        # Canonical URL
        canonical = soup.find("link", attrs={"rel": "canonical"})
        if canonical and canonical.get("href"):
            meta["canonical_url"] = canonical["href"]

        return meta

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _remove_noise_tags(self, soup: BeautifulSoup) -> None:
        """
        Remove all elements that are known to contain no training-worthy text.

        Two passes:
          1. Remove by tag name (script, style, nav, etc.)
          2. Remove by CSS class/id containing noise keywords
        """
        # Pass 1: tag name removal
        for tag in _REMOVE_TAGS:
            for el in soup.find_all(tag):
                el.decompose()

        # Pass 2: class/id pattern removal
        for el in soup.find_all(True):
            if el.attrs is None:
                continue

            class_attr = el.get("class", [])
            if isinstance(class_attr, str):
                classes = class_attr
            else:
                classes = " ".join(class_attr)
            element_id = el.get("id", "")
            if _NOISE_CLASS_PATTERNS.search(classes) or _NOISE_CLASS_PATTERNS.search(element_id):
                el.decompose()

    def _find_main_content(self, soup: BeautifulSoup) -> Optional[Tag]:
        """
        Heuristically identify the main content container.

        Priority:
          1. <main> tag
          2. <article> tag
          3. Element with id="content", id="main", id="article"
          4. Largest <div> block by total text length
        """
        # Semantic HTML5 elements
        for tag_name in ("main", "article"):
            el = soup.find(tag_name)
            if el:
                return el

        # ID-based heuristics
        for content_id in ("content", "main", "article", "post", "entry", "story"):
            el = soup.find(id=content_id)
            if el:
                return el
            # Also try class
            el = soup.find(class_=content_id)
            if el:
                return el

        # Largest div by text length
        best_div   = None
        best_len   = 0
        for div in soup.find_all("div"):
            text_len = len(div.get_text())
            if text_len > best_len:
                best_len = text_len
                best_div = div

        return best_div

    def _extract_paragraphs(self, root: Tag) -> List[str]:
        """
        Walk the content tree and collect meaningful text paragraphs.

        Treats block-level elements as paragraph separators. Inline elements
        are collapsed into the surrounding paragraph.

        Args:
            root: The BeautifulSoup element to extract text from.

        Returns:
            List of clean paragraph strings, each longer than min_para_len.
        """
        paragraphs: List[str] = []

        # Block-level elements that create natural paragraph breaks
        block_tags = frozenset({
            "p", "h1", "h2", "h3", "h4", "h5", "h6",
            "li", "blockquote", "dd", "dt", "pre", "td", "th",
        })

        for el in root.find_all(block_tags):
            raw_text = el.get_text(separator=" ", strip=True)

            # Decode HTML entities (e.g. &amp; → &, &nbsp; → space)
            text = html_lib.unescape(raw_text)

            # Collapse whitespace
            text = re.sub(r"\s+", " ", text).strip()

            if len(text) >= self._min_para_len:
                paragraphs.append(text)

        return paragraphs
