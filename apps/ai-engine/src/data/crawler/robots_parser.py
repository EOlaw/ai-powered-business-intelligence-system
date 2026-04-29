"""
InsightSerenity AI Engine — robots.txt Parser
==============================================
Fetches and parses robots.txt files so the crawler can respect exclusion rules
before requesting any page. This is both an ethical requirement and a legal
precaution — crawling pages explicitly excluded by robots.txt can create
liability and damage the platform's reputation.

The parser implements the full Robots Exclusion Protocol (REP) specification:
  - Multiple User-agent groups (we match our own bot name AND the wildcard '*')
  - Allow and Disallow directives (Allow takes precedence for equal-length paths)
  - Crawl-delay directive
  - Sitemap discovery

Caching: each domain's robots.txt is cached in memory for the lifetime of the
crawler run to avoid redundant HTTP requests.

Usage:
    from src.data.crawler.robots_parser import RobotsParser

    parser = RobotsParser(user_agent="InsightSerenityBot/1.0")
    allowed = await parser.is_allowed("https://example.com/some/path")
    delay   = await parser.crawl_delay("example.com")
"""

import asyncio
import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, urljoin

import httpx

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

class RobotsRules:
    """
    Parsed rules from a single robots.txt file.

    Stores disallow/allow directive pairs keyed by user-agent group name.
    Rule lookup resolves in this order:
      1. Rules for our specific bot name
      2. Rules for the wildcard agent '*'
      3. Default: allow everything

    The REP specifies that Allow beats Disallow when both match a path with
    equal specificity (path length). We implement this correctly.
    """

    def __init__(self) -> None:
        # Maps agent name → list of (directive, path) tuples
        # directive is either "allow" or "disallow"
        self._rules: Dict[str, List[Tuple[str, str]]] = {}
        self._crawl_delays: Dict[str, float] = {}
        self._sitemaps: List[str] = []

    def add_rule(self, agent: str, directive: str, path: str) -> None:
        """Record an Allow or Disallow rule for the given agent."""
        agent = agent.strip().lower()
        if agent not in self._rules:
            self._rules[agent] = []
        self._rules[agent].append((directive.lower(), path))

    def add_crawl_delay(self, agent: str, delay: float) -> None:
        """Record the Crawl-delay for the given agent."""
        self._crawl_delays[agent.strip().lower()] = delay

    def add_sitemap(self, url: str) -> None:
        """Record a Sitemap URL discovered in robots.txt."""
        self._sitemaps.append(url)

    @property
    def sitemaps(self) -> List[str]:
        return self._sitemaps

    def is_allowed(self, path: str, bot_name: str) -> bool:
        """
        Determine whether `path` is allowed for our bot.

        Matching algorithm (REP-compliant):
          1. Collect all Disallow and Allow rules that match `path` using
             prefix matching.
          2. Among matching rules, the most specific (longest path) wins.
          3. If two rules have equal length, Allow beats Disallow.
          4. If no rules match, the path is allowed.

        Args:
            path:     The URL path component (e.g. "/some/page").
            bot_name: Our bot's user-agent name (lower-cased).

        Returns:
            True if crawling is permitted, False if excluded.
        """
        # Gather applicable rule sets: our bot first, then wildcard
        applicable_rules: List[Tuple[str, str]] = []

        for agent_key in [bot_name.lower(), "*"]:
            if agent_key in self._rules:
                applicable_rules.extend(self._rules[agent_key])
                break  # Use the most specific matching group only

        if not applicable_rules:
            return True  # No rules → allowed by default

        # Find all rules whose path is a prefix of the requested path
        matching: List[Tuple[str, str]] = []
        for directive, rule_path in applicable_rules:
            if self._path_matches(path, rule_path):
                matching.append((directive, rule_path))

        if not matching:
            return True  # No matching rules → allowed

        # The most specific rule (longest path) takes precedence
        # Among ties, Allow wins over Disallow
        best_directive, _ = max(
            matching,
            key=lambda item: (len(item[1]), item[0] == "allow"),
        )
        return best_directive == "allow"

    def crawl_delay_for(self, bot_name: str) -> Optional[float]:
        """Return the Crawl-delay for our bot (or the wildcard), or None."""
        for agent_key in [bot_name.lower(), "*"]:
            if agent_key in self._crawl_delays:
                return self._crawl_delays[agent_key]
        return None

    @staticmethod
    def _path_matches(request_path: str, rule_path: str) -> bool:
        """
        Check if `rule_path` matches `request_path` using REP prefix semantics.

        Handles the wildcard '*' and end-of-path '$' metacharacters.
        """
        if not rule_path:
            return True  # Empty rule path matches everything

        # Replace glob wildcards with regex equivalents
        pattern = re.escape(rule_path).replace(r"\*", ".*").replace(r"\$", "$")
        if not pattern.endswith("$"):
            pattern += ".*"  # Prefix match: rule must match the start of the path

        return bool(re.match(pattern, request_path))


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────

class RobotsParser:
    """
    Fetches robots.txt files over HTTP and caches the parsed rules per domain.

    Thread/task-safe: concurrent callers asking about the same domain will
    only issue one HTTP request (coordinated via asyncio.Lock per domain).

    Args:
        user_agent: The User-Agent string our crawler sends. This name is also
                    used to look up matching rules in robots.txt.
        timeout:    HTTP request timeout in seconds for fetching robots.txt.
    """

    def __init__(self, user_agent: str, timeout: int = 10) -> None:
        self._user_agent = user_agent
        self._bot_name   = self._extract_bot_name(user_agent)
        self._timeout    = timeout

        # Cache: domain → RobotsRules (or None if fetch failed)
        self._cache: Dict[str, Optional[RobotsRules]] = {}

        # Per-domain lock prevents duplicate in-flight fetch requests
        self._locks: Dict[str, asyncio.Lock] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    async def is_allowed(self, url: str) -> bool:
        """
        Check whether our bot is permitted to fetch the given URL.

        Args:
            url: The full URL to check (e.g. "https://example.com/page").

        Returns:
            True if crawling is allowed or robots.txt is unreachable,
            False if explicitly disallowed.
        """
        parsed = urlparse(url)
        domain = parsed.netloc
        path   = parsed.path or "/"

        rules = await self._get_rules(domain, parsed.scheme)
        if rules is None:
            # If robots.txt is unreachable, we default to allowing the crawl
            return True

        return rules.is_allowed(path, self._bot_name)

    async def crawl_delay(self, domain: str, scheme: str = "https") -> Optional[float]:
        """
        Return the Crawl-delay specified for our bot on this domain, or None.
        None means: use the default delay from CrawlerSettings.
        """
        rules = await self._get_rules(domain, scheme)
        if rules is None:
            return None
        return rules.crawl_delay_for(self._bot_name)

    async def get_sitemaps(self, domain: str, scheme: str = "https") -> List[str]:
        """Return sitemap URLs discovered in the domain's robots.txt."""
        rules = await self._get_rules(domain, scheme)
        return rules.sitemaps if rules else []

    # ── Internal ───────────────────────────────────────────────────────────────

    async def _get_rules(
        self, domain: str, scheme: str
    ) -> Optional[RobotsRules]:
        """Fetch and cache the robots.txt for `domain`, with per-domain locking."""
        if domain in self._cache:
            return self._cache[domain]

        # Ensure only one coroutine fetches per domain concurrently
        if domain not in self._locks:
            self._locks[domain] = asyncio.Lock()

        async with self._locks[domain]:
            # Double-check after acquiring the lock (another coroutine may have
            # populated the cache while we waited)
            if domain in self._cache:
                return self._cache[domain]

            rules = await self._fetch_and_parse(domain, scheme)
            self._cache[domain] = rules
            return rules

    async def _fetch_and_parse(
        self, domain: str, scheme: str
    ) -> Optional[RobotsRules]:
        """Fetch robots.txt via HTTP and parse it into a RobotsRules object."""
        robots_url = f"{scheme}://{domain}/robots.txt"

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                headers={"User-Agent": self._user_agent},
                follow_redirects=True,
            ) as client:
                response = await client.get(robots_url)

            if response.status_code == 404:
                # No robots.txt → no restrictions
                logger.debug("No robots.txt found", domain=domain)
                return RobotsRules()   # Empty rules = allow all

            if response.status_code != 200:
                logger.warning(
                    "Unexpected robots.txt status",
                    domain=domain,
                    status=response.status_code,
                )
                return None

            return self._parse(response.text)

        except Exception as e:
            logger.warning(
                "Failed to fetch robots.txt — defaulting to allow",
                domain=domain,
                error=str(e),
            )
            return None

    def _parse(self, content: str) -> RobotsRules:
        """
        Parse the text content of a robots.txt file into a RobotsRules object.

        Handles:
          - User-agent groups
          - Disallow / Allow directives
          - Crawl-delay
          - Sitemap declarations
          - Comments (lines starting with #)
          - Blank lines (group separator)
        """
        rules                = RobotsRules()
        current_agents: List[str] = []
        in_group             = False

        for raw_line in content.splitlines():
            # Strip comments and surrounding whitespace
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                # Blank line ends the current agent group
                if in_group:
                    current_agents = []
                    in_group = False
                continue

            # Parse "field: value" pairs
            if ":" not in line:
                continue

            field, _, value = line.partition(":")
            field = field.strip().lower()
            value = value.strip()

            if field == "user-agent":
                current_agents.append(value)
                in_group = True

            elif field == "disallow":
                for agent in current_agents:
                    rules.add_rule(agent, "disallow", value)

            elif field == "allow":
                for agent in current_agents:
                    rules.add_rule(agent, "allow", value)

            elif field == "crawl-delay":
                try:
                    delay = float(value)
                    for agent in current_agents:
                        rules.add_crawl_delay(agent, delay)
                except ValueError:
                    pass

            elif field == "sitemap":
                rules.add_sitemap(value)

        return rules

    @staticmethod
    def _extract_bot_name(user_agent: str) -> str:
        """
        Extract the identifying name from a User-Agent string for rule matching.

        "InsightSerenityBot/1.0 (+https://...)" → "insightserenitybot"
        """
        # Take the part before the first '/' or whitespace and lower-case it
        name = re.split(r"[/\s]", user_agent)[0]
        return name.lower()
