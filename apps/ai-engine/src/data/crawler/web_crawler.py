"""
InsightSerenity AI Engine — Async Web Crawler
=============================================
A production-grade, polite, breadth-first web crawler that collects raw HTML
pages as input to the data preprocessing pipeline.

Architecture:
    - asyncio + httpx for non-blocking I/O (handles hundreds of concurrent
      requests across many domains with minimal thread overhead)
    - BFS frontier queue (asyncio.Queue) — pages are processed in breadth-first
      order so seed pages are always fully explored before following outlinks
    - Per-domain rate limiting (DomainRateLimiter) — respects each server
    - robots.txt compliance (RobotsParser) — checks before every new URL
    - URL deduplication via a visited set (thread-safe in the event loop)
    - Configurable depth limit, page cap, and concurrency ceiling
    - Progress checkpointing — the crawler can resume from a partial run

Output:
    JSONL file with one record per crawled page:
    {
        "url":          "https://example.com/page",
        "domain":       "example.com",
        "depth":        2,
        "status_code":  200,
        "content_type": "text/html; charset=utf-8",
        "html":         "<html>...</html>",
        "crawled_at":   "2025-01-01T12:00:00Z"
    }

Usage:
    from src.data.crawler.web_crawler import WebCrawler

    crawler = WebCrawler(output_path="storage/datasets/raw_html.jsonl")
    await crawler.crawl(seed_urls=["https://example.com", "https://other.com"])
"""

import asyncio
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Set
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup

from src.config.settings import settings
from src.data.crawler.robots_parser import RobotsParser
from src.data.crawler.rate_limiter import DomainRateLimiter
from src.utils.file_io import append_jsonl, read_jsonl, ensure_dir
from src.utils.logger import get_logger, LogTimer

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# URL utilities
# ─────────────────────────────────────────────────────────────────────────────

def normalise_url(url: str) -> Optional[str]:
    """
    Canonicalise a URL by removing fragment identifiers, default ports,
    and trailing slashes on root paths.

    Returns None if the URL is not a valid http/https address.
    """
    try:
        parsed = urlparse(url.strip())
    except Exception:
        return None

    # Only allow http and https
    if parsed.scheme not in ("http", "https"):
        return None

    # Strip fragment — fragments are client-side only, same document
    normalised = parsed._replace(fragment="")

    # Strip default ports
    netloc = normalised.netloc
    netloc = re.sub(r":80$", "", netloc)    # http default
    netloc = re.sub(r":443$", "", netloc)   # https default
    normalised = normalised._replace(netloc=netloc)

    return urlunparse(normalised)


def extract_domain(url: str) -> str:
    """Return the hostname (netloc) of a URL."""
    return urlparse(url).netloc


def is_same_domain(url: str, allowed_domains: Set[str]) -> bool:
    """Return True only if the URL's domain is in `allowed_domains`."""
    if not allowed_domains:
        return True   # No restriction → all domains allowed
    return extract_domain(url) in allowed_domains


def should_skip_url(url: str) -> bool:
    """
    Quickly reject URLs that are obviously not useful text documents.
    Avoids wasting HTTP connections on images, PDFs, JavaScript, etc.
    """
    cfg          = settings.crawler
    lower        = url.lower()
    parsed_path  = urlparse(url).path.lower()

    return any(pattern in lower for pattern in cfg.excluded_url_patterns)


# ─────────────────────────────────────────────────────────────────────────────
# Crawl result
# ─────────────────────────────────────────────────────────────────────────────

class CrawlResult:
    """
    Represents the outcome of fetching a single URL.
    Carries either the raw HTML or an error reason.
    """

    __slots__ = (
        "url", "domain", "depth", "status_code",
        "content_type", "html", "error", "crawled_at",
    )

    def __init__(
        self,
        url: str,
        domain: str,
        depth: int,
        status_code: int = 0,
        content_type: str = "",
        html: str = "",
        error: Optional[str] = None,
    ) -> None:
        self.url          = url
        self.domain       = domain
        self.depth        = depth
        self.status_code  = status_code
        self.content_type = content_type
        self.html         = html
        self.error        = error
        self.crawled_at   = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "url":          self.url,
            "domain":       self.domain,
            "depth":        self.depth,
            "status_code":  self.status_code,
            "content_type": self.content_type,
            "html":         self.html,
            "crawled_at":   self.crawled_at,
        }

    @property
    def is_success(self) -> bool:
        return self.status_code == 200 and not self.error


# ─────────────────────────────────────────────────────────────────────────────
# Web Crawler
# ─────────────────────────────────────────────────────────────────────────────

class WebCrawler:
    """
    Asynchronous, polite, BFS web crawler.

    The crawler writes each successfully crawled page to `output_path` as a
    JSONL record in real time — results are not buffered in memory. If the
    process is interrupted, the output file contains all pages crawled so far
    and the run can be resumed.

    Args:
        output_path:      Where to write raw HTML JSONL records.
        restrict_domains: If non-empty, only follow links within these domains.
                          If empty, the crawler follows links to any domain.
        checkpoint_path:  Optional path to persist/restore the visited-URL set
                          for resumable crawls.
    """

    def __init__(
        self,
        output_path: str,
        restrict_domains: Optional[List[str]] = None,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        cfg = settings.crawler

        self._output_path       = Path(output_path)
        self._checkpoint_path   = Path(checkpoint_path) if checkpoint_path else None
        self._restrict_domains: Set[str] = set(restrict_domains or [])

        self._max_depth        = cfg.max_depth
        self._max_pages        = cfg.max_pages
        self._max_concurrent   = cfg.max_concurrent
        self._request_timeout  = cfg.request_timeout
        self._user_agent       = cfg.user_agent
        self._max_response_b   = cfg.max_response_bytes
        self._allowed_ct       = set(cfg.allowed_content_types)

        # Frontier: (url, depth) pairs waiting to be fetched
        self._queue: asyncio.Queue  = asyncio.Queue()

        # Visited set: prevents re-crawling the same URL
        self._visited: Set[str] = set()

        # Thread-safe counters
        self._pages_crawled: int = 0
        self._pages_failed:  int = 0

        # Politeness components
        self._robots  = RobotsParser(user_agent=self._user_agent)
        self._limiter = DomainRateLimiter(default_delay=cfg.delay_per_domain)

    # ── Public API ─────────────────────────────────────────────────────────────

    async def crawl(self, seed_urls: List[str]) -> None:
        """
        Begin crawling from the given seed URLs.

        Seeds are enqueued at depth 0. Workers fan out following links up to
        `max_depth`, stopping when `max_pages` total pages are collected.

        This method blocks until crawling is complete or limits are reached.

        Args:
            seed_urls: Starting URLs. At least one is required.
        """
        if not seed_urls:
            raise ValueError("At least one seed URL must be provided.")

        ensure_dir(self._output_path.parent)

        # Restore checkpoint if available
        self._restore_checkpoint()

        # Enqueue seeds (skip already-visited if resuming)
        for url in seed_urls:
            norm = normalise_url(url)
            if norm and norm not in self._visited:
                await self._queue.put((norm, 0))
                logger.info("Seed enqueued", url=norm)

        with LogTimer(logger, "Web crawl", seeds=len(seed_urls)):
            # Spawn worker coroutines
            workers = [
                asyncio.create_task(self._worker(worker_id=i))
                for i in range(self._max_concurrent)
            ]

            # Wait for the queue to be fully processed
            await self._queue.join()

            # Cancel idle workers
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

        # Save checkpoint of visited URLs for future resumption
        self._save_checkpoint()

        logger.info(
            "Crawl finished",
            pages_crawled=self._pages_crawled,
            pages_failed=self._pages_failed,
            output=str(self._output_path),
        )

    # ── Worker ─────────────────────────────────────────────────────────────────

    async def _worker(self, worker_id: int) -> None:
        """
        Single worker coroutine. Pulls URLs from the frontier, fetches them,
        extracts links, and writes successful results to the output file.

        Runs indefinitely until cancelled by the crawl coordinator.
        """
        logger.debug("Worker started", worker_id=worker_id)

        async with httpx.AsyncClient(
            timeout=self._request_timeout,
            headers={
                "User-Agent": self._user_agent,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            },
            follow_redirects=True,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        ) as client:
            while True:
                try:
                    url, depth = await asyncio.wait_for(
                        self._queue.get(), timeout=5.0
                    )
                except asyncio.TimeoutError:
                    # Queue has been empty for 5s; keep waiting for new work.
                    continue
                except asyncio.CancelledError:
                    break

                try:
                    await self._process_url(client, url, depth)
                except Exception as e:
                    logger.error(
                        "Unexpected error processing URL",
                        url=url,
                        worker_id=worker_id,
                        error=str(e),
                    )
                finally:
                    self._queue.task_done()

    async def _process_url(
        self, client: httpx.AsyncClient, url: str, depth: int
    ) -> None:
        """
        Fetch one URL, extract outlinks, and persist the result.

        Steps:
          1. Skip if already visited or over page limit
          2. Check robots.txt
          3. Acquire rate limit token
          4. Fetch HTTP response
          5. Parse HTML for outlinks
          6. Write result to JSONL
          7. Enqueue new links
        """
        domain = extract_domain(url)

        # ── Guard: skip if over limits ──────────────────────────────────────
        if self._pages_crawled >= self._max_pages:
            return

        if url in self._visited:
            return

        # Mark as visited immediately to prevent duplicate enqueueing
        self._visited.add(url)

        # ── Guard: URL exclusion patterns ───────────────────────────────────
        if should_skip_url(url):
            logger.debug("URL excluded by pattern", url=url)
            return

        # ── Guard: domain restriction ───────────────────────────────────────
        if not is_same_domain(url, self._restrict_domains):
            return

        # ── robots.txt check ────────────────────────────────────────────────
        if settings.crawler.respect_robots_txt:
            allowed = await self._robots.is_allowed(url)
            if not allowed:
                logger.debug("robots.txt disallowed", url=url)
                return

            # Apply Crawl-delay from robots.txt (overrides our default)
            robots_delay = await self._robots.crawl_delay(domain)
            if robots_delay:
                await self._limiter.set_delay(domain, robots_delay)

        # ── Rate limiter ─────────────────────────────────────────────────────
        await self._limiter.acquire(domain)

        # ── HTTP fetch ───────────────────────────────────────────────────────
        result = await self._fetch(client, url, depth)

        if not result.is_success:
            self._pages_failed += 1
            logger.warning(
                "Crawl fetch failed",
                url=url,
                status_code=result.status_code,
                error=result.error,
                content_type=result.content_type,
            )
            return

        # ── Write output ─────────────────────────────────────────────────────
        append_jsonl(str(self._output_path), result.to_dict())
        self._pages_crawled += 1

        if self._pages_crawled % 100 == 0:
            logger.info(
                "Crawl progress",
                pages_crawled=self._pages_crawled,
                queue_size=self._queue.qsize(),
                domains=self._limiter.domain_count(),
            )

        # ── Extract and enqueue links ─────────────────────────────────────
        if depth < self._max_depth:
            outlinks = self._extract_links(result.html, url)
            for link in outlinks:
                norm = normalise_url(link)
                if norm and norm not in self._visited:
                    await self._queue.put((norm, depth + 1))

    async def _fetch(
        self, client: httpx.AsyncClient, url: str, depth: int
    ) -> CrawlResult:
        """Send an HTTP GET request and return a CrawlResult."""
        domain = extract_domain(url)

        try:
            response = await client.get(url)

            content_type = response.headers.get("content-type", "")

            # Check content type before reading body
            if not self._is_allowed_content_type(content_type):
                logger.debug(
                    "Skipped non-text content type",
                    url=url,
                    content_type=content_type,
                )
                return CrawlResult(url, domain, depth, error="content_type_rejected")

            # Check body size
            if len(response.content) > self._max_response_b:
                logger.debug("Response too large", url=url, size=len(response.content))
                return CrawlResult(url, domain, depth, error="response_too_large")

            html = response.text

            return CrawlResult(
                url=url,
                domain=domain,
                depth=depth,
                status_code=response.status_code,
                content_type=content_type,
                html=html,
            )

        except httpx.TimeoutException:
            logger.debug("Request timed out", url=url)
            return CrawlResult(url, domain, depth, error="timeout")

        except httpx.TooManyRedirects:
            logger.debug("Too many redirects", url=url)
            return CrawlResult(url, domain, depth, error="too_many_redirects")

        except Exception as e:
            logger.debug("Fetch error", url=url, error=str(e))
            return CrawlResult(url, domain, depth, error=str(e))

    # ── Link extraction ────────────────────────────────────────────────────────

    def _extract_links(self, html: str, base_url: str) -> List[str]:
        """
        Parse HTML with BeautifulSoup and extract all unique <a href> links,
        resolved to absolute URLs against the base URL.

        Args:
            html:     Raw HTML content of the page.
            base_url: The URL the page was fetched from, used to resolve
                      relative hrefs (e.g. "/about" → "https://example.com/about").

        Returns:
            List of absolute, deduplicated URLs found on the page.
        """
        try:
            soup  = BeautifulSoup(html, "lxml")
            links = []
            seen  = set()

            for tag in soup.find_all("a", href=True):
                href = tag["href"].strip()
                if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
                    continue

                absolute = urljoin(base_url, href)
                norm = normalise_url(absolute)
                if norm and norm not in seen:
                    seen.add(norm)
                    links.append(norm)

            return links

        except Exception as e:
            logger.debug("Link extraction failed", url=base_url, error=str(e))
            return []

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _is_allowed_content_type(self, content_type: str) -> bool:
        """Check if the Content-Type header matches our allowed set."""
        ct_lower = content_type.lower()
        return any(allowed in ct_lower for allowed in self._allowed_ct)

    def _save_checkpoint(self) -> None:
        """Persist the visited-URL set to disk for resumable crawls."""
        if not self._checkpoint_path:
            return
        try:
            from src.utils.file_io import write_jsonl
            write_jsonl(
                str(self._checkpoint_path),
                [{"url": url} for url in self._visited],
            )
            logger.info(
                "Checkpoint saved",
                path=str(self._checkpoint_path),
                visited=len(self._visited),
            )
        except Exception as e:
            logger.warning("Failed to save checkpoint", error=str(e))

    def _restore_checkpoint(self) -> None:
        """Reload the visited-URL set from a previous checkpoint file."""
        if not self._checkpoint_path or not self._checkpoint_path.exists():
            return
        try:
            from src.utils.file_io import read_jsonl
            records = read_jsonl(str(self._checkpoint_path))
            self._visited = {r["url"] for r in records}
            logger.info(
                "Checkpoint restored",
                path=str(self._checkpoint_path),
                visited=len(self._visited),
            )
        except Exception as e:
            logger.warning("Failed to restore checkpoint", error=str(e))

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def pages_crawled(self) -> int:
        return self._pages_crawled

    @property
    def pages_failed(self) -> int:
        return self._pages_failed

    @property
    def visited_count(self) -> int:
        return len(self._visited)
