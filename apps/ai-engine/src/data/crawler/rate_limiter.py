"""
InsightSerenity AI Engine — Per-Domain Rate Limiter
====================================================
Implements a token-bucket rate limiter that enforces a minimum delay between
consecutive HTTP requests to the same domain. This is the core politeness
mechanism of the web crawler.

Why per-domain? Because rate limits should restrict *our load on a single
server*, not our total crawl throughput. We can crawl 50 different domains
in parallel while still being respectful to each individual server.

Design: Token Bucket
- Each domain has one bucket that fills at a rate of 1 token per `delay` seconds
- A request may only proceed when the bucket has ≥ 1 token
- Tokens do not accumulate past 1 (i.e. we don't build up credit)
- asyncio.Lock per domain ensures single-threaded access to each bucket

Usage:
    from src.data.crawler.rate_limiter import DomainRateLimiter

    limiter = DomainRateLimiter(default_delay=1.0)
    await limiter.acquire("example.com")        # waits if needed
    await limiter.acquire("example.com")        # waits again
    await limiter.set_delay("slow-site.com", 3.0)  # custom delay
"""

import asyncio
import time
from typing import Dict, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class _DomainBucket:
    """
    Internal token-bucket state for a single domain.

    Tracks when the next request to this domain may be issued.
    `next_allowed_at` is a monotonic timestamp (from time.monotonic()).
    """

    __slots__ = ("delay", "next_allowed_at", "lock", "total_requests")

    def __init__(self, delay: float) -> None:
        self.delay: float            = delay
        self.next_allowed_at: float  = 0.0        # 0 = immediately available
        self.lock: asyncio.Lock      = asyncio.Lock()
        self.total_requests: int     = 0           # Telemetry counter


class DomainRateLimiter:
    """
    Coordinates per-domain request timing for the async web crawler.

    Creates one token bucket per domain on first access. The bucket
    serialises concurrent coroutines that want to crawl the same domain,
    ensuring the configured inter-request delay is always respected.

    Args:
        default_delay: Minimum seconds between requests to the same domain.
                       Can be overridden per-domain via set_delay().
    """

    def __init__(self, default_delay: float = 1.0) -> None:
        if default_delay < 0.1:
            raise ValueError("default_delay must be at least 0.1 seconds.")

        self._default_delay = default_delay
        # Maps domain hostname → _DomainBucket
        self._buckets: Dict[str, _DomainBucket] = {}
        # Protects concurrent creation of new bucket objects
        self._registry_lock = asyncio.Lock()

    # ── Public API ─────────────────────────────────────────────────────────────

    async def acquire(self, domain: str) -> float:
        """
        Block until we are permitted to send a request to `domain`.

        If the domain is being crawled by another coroutine right now, this
        method will wait until that request has completed its minimum delay
        before returning.

        Args:
            domain: Hostname only (e.g. "example.com"), no scheme.

        Returns:
            Actual wall-clock seconds waited (useful for telemetry).
        """
        bucket = await self._get_or_create_bucket(domain)
        wait_start = time.monotonic()

        async with bucket.lock:
            now = time.monotonic()
            wait_needed = bucket.next_allowed_at - now

            if wait_needed > 0:
                logger.debug(
                    "Rate limiter waiting",
                    domain=domain,
                    wait_s=round(wait_needed, 3),
                )
                await asyncio.sleep(wait_needed)

            # Advance the token refill time — next request must wait `delay` seconds
            bucket.next_allowed_at = time.monotonic() + bucket.delay
            bucket.total_requests += 1

        return max(0.0, time.monotonic() - wait_start)

    async def set_delay(self, domain: str, delay: float) -> None:
        """
        Override the inter-request delay for a specific domain.

        This is called by the crawler when it discovers a Crawl-delay
        directive in the domain's robots.txt.

        Args:
            domain: Hostname to configure.
            delay:  Minimum seconds between requests. Must be >= 0.1.
        """
        if delay < 0.1:
            logger.warning(
                "Crawl-delay too small; using minimum 0.1s",
                domain=domain,
                requested_delay=delay,
            )
            delay = 0.1

        bucket = await self._get_or_create_bucket(domain)
        async with bucket.lock:
            if bucket.delay != delay:
                logger.debug(
                    "Rate limit updated",
                    domain=domain,
                    old_delay=bucket.delay,
                    new_delay=delay,
                )
                bucket.delay = delay

    def get_stats(self) -> Dict[str, Dict[str, object]]:
        """
        Return telemetry statistics for all tracked domains.

        Useful for monitoring crawl health and diagnosing slow domains.
        """
        return {
            domain: {
                "delay_s":        bucket.delay,
                "total_requests": bucket.total_requests,
            }
            for domain, bucket in self._buckets.items()
        }

    def domain_count(self) -> int:
        """Return the number of unique domains seen so far."""
        return len(self._buckets)

    # ── Internal ───────────────────────────────────────────────────────────────

    async def _get_or_create_bucket(self, domain: str) -> _DomainBucket:
        """
        Return the bucket for `domain`, creating it atomically if needed.

        The registry lock ensures two coroutines discovering a new domain
        simultaneously will not create duplicate bucket objects.
        """
        # Fast path: no lock needed for existing buckets
        if domain in self._buckets:
            return self._buckets[domain]

        async with self._registry_lock:
            # Re-check after acquiring the lock (another coroutine may have
            # created the bucket while we were waiting)
            if domain not in self._buckets:
                self._buckets[domain] = _DomainBucket(delay=self._default_delay)
                logger.debug("Rate limiter bucket created", domain=domain)

        return self._buckets[domain]
